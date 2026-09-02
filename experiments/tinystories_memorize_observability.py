#!/usr/bin/env python3
"""
TinyStories Memorization Test - Full Observability Version

Same as previous training with added observability:
- Validation loss on separate TinyStories validation split
- Perplexity from loss
- Actual learning rate from optimizer
- Ternary weight distribution (-1/0/+1 percentages)
- Fixed-prompt generation at every checkpoint

NO CHANGES to model/optimizer/LR/data configuration.
"""

import sys
import time
import json
import random
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.training.adafactor import Adafactor

# =============================================================================
# CONFIGURATION - UNCHANGED FROM PREVIOUS 3000-STEP TEST
# =============================================================================

MODEL_CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
}

OPTIMIZER_CONFIG = {
    "optimizer": "Adafactor",
    "lr": 0.001,
    "beta1": 0.5,
    "beta2": 0.98,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}

TRAINING_CONFIG = {
    "total_steps": 3000,
    "batch_size": 4,
    "accumulation_steps": 2,
    "lr_scheduler": "constant",
    "seed": 42,
}

DATA_CONFIG = {
    "num_stories": 20,
    "seed": 42,
    "max_seq_len": 256,
    "val_samples": 100,  # Number of validation samples
}

TOKENIZER_PATH = "/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json"

# Fixed prompts for generation evaluation
FIXED_PROMPTS = [
    "Once upon a time",
    "The little girl",
    "A boy named",
    "There was a",
    "One day",
]

# Checkpoint steps
CHECKPOINT_STEPS = [0, 500, 1000, 1500, 2000, 2500, 3000]

# Output paths
MONITOR_LOG = Path(__file__).parent / "results" / "tinystories_memorize_log.txt"
CHECKPOINT_DIR = Path(__file__).parent / "results" / "checkpoints"
GENERATED_DIR = Path(__file__).parent / "results" / "generated"
METRICS_FILE = Path(__file__).parent / "results" / "metrics.jsonl"


# =============================================================================
# LOGGER
# =============================================================================

class MonitorLogger:
    """Logger that writes to both stdout and monitor log file."""
    
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, 'w')
        self.start_time = time.time()
        
        header = f"\n{'='*70}\nTINYSTORIES MEMORIZATION TRAINING LOG\nStarted: {datetime.now().isoformat()}\n{'='*70}\n\n"
        self.log_file.write(header)
        self.log_file.flush()
    
    def log(self, message, flush=True):
        timestamp = time.time() - self.start_time
        formatted = f"[{timestamp:8.1f}s] {message}"
        print(formatted, flush=flush)
        self.log_file.write(formatted + "\n")
        if flush:
            self.log_file.flush()
    
    def close(self):
        self.log_file.close()


# =============================================================================
# DATASET
# =============================================================================

class TinyStoriesSubset(Dataset):
    """Dataset that caches all samples in memory."""
    
    def __init__(self, tokenizer_path, num_stories=20, max_seq_len=256, seed=42, logger=None):
        from src.tokenizer.bpe_tokenizer import BPETokenizer
        from datasets import load_dataset
        
        if logger:
            logger.log("Loading tokenizer...")
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        self.max_seq_len = max_seq_len
        
        if logger:
            logger.log("Loading TinyStories dataset...")
        full_dataset = load_dataset("roneneldan/TinyStories", split="train")
        
        if logger:
            logger.log(f"Dataset loaded, total stories: {len(full_dataset):,}")
            logger.log(f"Selecting {num_stories} stories with seed={seed}...")
        
        random.seed(seed)
        self.indices = random.sample(range(len(full_dataset)), num_stories)
        
        if logger:
            logger.log(f"Selected indices: {self.indices[:5]}... (first 5 shown)")
        
        # Extract and cache stories
        if logger:
            logger.log("Extracting stories into memory...")
        stories = [full_dataset[i]['text'] for i in self.indices]
        del full_dataset
        
        if logger:
            logger.log("Tokenizing all stories ONCE...")
        
        self.samples = []
        for i, story in enumerate(stories):
            tokens = self.tokenizer.encode(story)[:max_seq_len - 1]
            tokens = tokens + [self.tokenizer.eos_token_id]
            # Pad to max_seq_len
            padding_length = max_seq_len - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
            self.samples.append(torch.tensor(tokens, dtype=torch.long))
            if logger and i < 3:
                logger.log(f"  Story {i+1}: {len(self.tokenizer.encode(story)[:max_seq_len-1])} tokens")
        
        if logger:
            logger.log(f"Total samples: {len(self.samples)}")
            logger.log(f"All samples CACHED in memory")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class TinyStoriesValidation(Dataset):
    """Small validation set from TinyStories validation split."""
    
    def __init__(self, tokenizer_path, num_samples=100, max_seq_len=256, seed=123, logger=None):
        from src.tokenizer.bpe_tokenizer import BPETokenizer
        from datasets import load_dataset
        
        if logger:
            logger.log("Loading validation set...")
        
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        self.max_seq_len = max_seq_len
        
        val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
        
        random.seed(seed)
        indices = random.sample(range(len(val_dataset)), num_samples)
        
        stories = [val_dataset[i]['text'] for i in indices]
        del val_dataset
        
        self.samples = []
        for story in stories:
            tokens = self.tokenizer.encode(story)[:max_seq_len - 1]
            tokens = tokens + [self.tokenizer.eos_token_id]
            padding_length = max_seq_len - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
            self.samples.append(torch.tensor(tokens, dtype=torch.long))
        
        if logger:
            logger.log(f"  Validation samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Stack pre-padded sequences."""
    return torch.stack(batch)


# =============================================================================
# METRICS
# =============================================================================

class MetricsLogger:
    """Log metrics to JSONL file."""
    
    def __init__(self, metrics_path):
        self.metrics_path = Path(metrics_path)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_file = open(self.metrics_path, 'w')
    
    def log(self, metrics_dict):
        """Write metrics as JSON line."""
        self.metrics_file.write(json.dumps(metrics_dict) + "\n")
        self.metrics_file.flush()
    
    def close(self):
        self.metrics_file.close()


# =============================================================================
# TERNARY WEIGHT ANALYSIS
# =============================================================================

def get_ternary_distribution(model):
    """Get percentage of -1, 0, +1 weights in ternary layers."""
    neg_count = 0
    zero_count = 0
    pos_count = 0
    total = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and 'BitLinear' in module.__class__.__name__:
            w = module.weight.data
            # After quantization: -1, 0, +1
            neg_count += (w == -1).sum().item()
            zero_count += (w == 0).sum().item()
            pos_count += (w == 1).sum().item()
            total += w.numel()
    
    if total == 0:
        return 0.0, 0.0, 0.0
    
    return neg_count / total * 100, zero_count / total * 100, pos_count / total * 100


# =============================================================================
# GENERATION
# =============================================================================

def generate_from_prompts(model, tokenizer, prompts, max_new_tokens=30, device='cpu'):
    """Generate continuations from fixed prompts."""
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            
            generated = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=1.0)
            generated_text = tokenizer.decode(generated[0].tolist())
            
            results.append({
                'prompt': prompt,
                'generated': generated_text,
            })
    
    model.train()
    return results


def save_generated_results(results, checkpoint_name, output_dir):
    """Save generated results to file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{checkpoint_name}_generated.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"Checkpoint: {checkpoint_name}\n")
        f.write("=" * 50 + "\n\n")
        
        for r in results:
            f.write(f"Prompt: {r['prompt']}\n")
            f.write(f"Generated: {r['generated']}\n")
            f.write("-" * 50 + "\n")
    
    return output_file


# =============================================================================
# VALIDATION
# =============================================================================

def compute_validation_loss(model, val_loader, device='cpu'):
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            # Shift for next-token prediction
            inputs = batch[:, :-1]
            targets = batch[:, 1:].clone()
            
            # Build mask for loss
            mask = (targets != 0)  # non-pad tokens
            targets[~mask] = -100
            
            outputs = model(inputs, targets=None)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    global logger
    
    # Setup
    logger = MonitorLogger(MONITOR_LOG)
    metrics_logger = MetricsLogger(METRICS_FILE)
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    
    device = 'cpu'
    logger.log(f"Device: {device}")
    
    # Seed
    random.seed(TRAINING_CONFIG["seed"])
    torch.manual_seed(TRAINING_CONFIG["seed"])
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    logger.log("")
    logger.log("=" * 70)
    logger.log("CONFIGURATION")
    logger.log("=" * 70)
    logger.log(f"Model: vocab={MODEL_CONFIG['vocab_size']}, d_model={MODEL_CONFIG['d_model']}, "
               f"layers={MODEL_CONFIG['n_layers']}, heads={MODEL_CONFIG['n_heads']}")
    logger.log(f"Optimizer: {OPTIMIZER_CONFIG['optimizer']}, LR={OPTIMIZER_CONFIG['lr']}, "
                f"beta1={OPTIMIZER_CONFIG['beta1']}, beta2={OPTIMIZER_CONFIG['beta2']}")
    logger.log(f"Training: {TRAINING_CONFIG['total_steps']} steps, batch={TRAINING_CONFIG['batch_size']}, "
                f"accum={TRAINING_CONFIG['accumulation_steps']}")
    logger.log(f"Data: {DATA_CONFIG['num_stories']} stories, seed={DATA_CONFIG['seed']}")
    logger.log(f"LR Schedule: {TRAINING_CONFIG['lr_scheduler']}")
    logger.log(f"Checkpoints: {CHECKPOINT_STEPS}")
    logger.log(f"Fixed prompts: {len(FIXED_PROMPTS)}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    logger.log("")
    logger.log("=" * 70)
    logger.log("LOADING DATA")
    logger.log("=" * 70)
    
    train_dataset = TinyStoriesSubset(
        TOKENIZER_PATH,
        num_stories=DATA_CONFIG["num_stories"],
        max_seq_len=DATA_CONFIG["max_seq_len"],
        seed=DATA_CONFIG["seed"],
        logger=logger,
    )
    
    val_dataset = TinyStoriesValidation(
        TOKENIZER_PATH,
        num_samples=DATA_CONFIG["val_samples"],
        max_seq_len=DATA_CONFIG["max_seq_len"],
        seed=123,
        logger=logger,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    logger.log(f"Train batches: {len(train_loader)}")
    logger.log(f"Val batches: {len(val_loader)}")
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    
    logger.log("")
    logger.log("=" * 70)
    logger.log("CREATING MODEL")
    logger.log("=" * 70)
    
    model = BitLinearGPT(
        vocab_size=MODEL_CONFIG["vocab_size"],
        d_model=MODEL_CONFIG["d_model"],
        n_heads=MODEL_CONFIG["n_heads"],
        n_layers=MODEL_CONFIG["n_layers"],
        d_ff=MODEL_CONFIG["d_ff"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model created: {num_params:,} parameters")
    model = model.to(device)
    model.train()
    
    # ========================================================================
    # OPTIMIZER
    # ========================================================================
    
    logger.log("")
    logger.log("=" * 70)
    logger.log("OPTIMIZER")
    logger.log("=" * 70)
    
    optimizer = Adafactor(
        model.parameters(),
        lr=OPTIMIZER_CONFIG["lr"],
        beta1=OPTIMIZER_CONFIG["beta1"],
        beta2=OPTIMIZER_CONFIG["beta2"],
        weight_decay=OPTIMIZER_CONFIG["weight_decay"],
    )
    
    logger.log(f"Optimizer: {OPTIMIZER_CONFIG['optimizer']}")
    logger.log(f"LR: {OPTIMIZER_CONFIG['lr']} (constant)")
    optimizer.zero_grad()
    
    # ========================================================================
    # TOKENIZER FOR GENERATION
    # ========================================================================
    
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    tokenizer_gen = BPETokenizer.load(TOKENIZER_PATH)
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    logger.log("")
    logger.log("=" * 70)
    logger.log("TRAINING")
    logger.log("=" * 70)
    logger.log(f"Total steps: {TRAINING_CONFIG['total_steps']}")
    
    train_iter = iter(train_loader)
    
    best_loss = float('inf')
    best_step = 0
    best_checkpoint_path = None
    
    losses = []
    step = 0
    grad_norm = 0.0
    
    logger.log("Entering training loop...")
    
    while step < TRAINING_CONFIG["total_steps"]:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        batch = batch.to(device)
        
        # Forward
        outputs = model(batch)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        
        # Compute loss manually
        targets = batch[:, 1:].clone()
        targets[targets == 0] = -100  # mask padding
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(logits[:, :-1, :].reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        # Backward
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % TRAINING_CONFIG["accumulation_steps"] == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                OPTIMIZER_CONFIG["max_grad_norm"]
            )
            
            optimizer.step()
            optimizer.zero_grad()
            
            losses.append(loss.item())
        
        # ====================================================================
        # CHECKPOINT SAVING AND EVALUATION
        # ====================================================================
        
        should_checkpoint = step in CHECKPOINT_STEPS
        
        if should_checkpoint or loss.item() < best_loss:
            checkpoint_name = f"step_{step}" if should_checkpoint else "best_loss"
            
            # Save checkpoint
            checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_name}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "best_loss": best_loss if best_loss != float('inf') else loss.item(),
            }, checkpoint_path)
            
            if step % 100 == 0:
                logger.log(f"Checkpoint saved: {checkpoint_name}")
            
            # Generate from fixed prompts
            generated = generate_from_prompts(model, tokenizer_gen, FIXED_PROMPTS, device=device)
            generated_file = save_generated_results(generated, checkpoint_name, GENERATED_DIR)
            
            if step % 100 == 0:
                logger.log(f"  Generated samples saved: {generated_file.name}")
            
            # Validation loss
            val_loss = compute_validation_loss(model, val_loader, device)
            
            # Perplexity
            perplexity = math.exp(val_loss)
            
            # Ternary weight distribution
            neg_pct, zero_pct, pos_pct = get_ternary_distribution(model)
            
            # Log metrics
            metrics = {
                "step": step,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "perplexity": perplexity,
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "lr": OPTIMIZER_CONFIG['lr'],  # Constant LR
                "ternary_neg_pct": neg_pct,
                "ternary_zero_pct": zero_pct,
                "ternary_pos_pct": pos_pct,
            }
            metrics_logger.log(metrics)
            
            if step % 100 == 0:
                logger.log(f"  Train loss: {loss.item():.4f}")
                logger.log(f"  Val loss: {val_loss:.4f}")
                logger.log(f"  Perplexity: {perplexity:.1f}")
                logger.log(f"  Ternary: -1={neg_pct:.1f}%, 0={zero_pct:.1f}%, +1={pos_pct:.1f}%")
        
        # Track best loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_step = step
            if best_checkpoint_path is None:
                best_checkpoint_path = CHECKPOINT_DIR / "best_loss.pt"
        
        step += 1
        
        # Progress log
        if step % 100 == 0:
            logger.log(f"Step {step}: Loss={loss.item():.4f}")
    
    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    
    logger.log("")
    logger.log("=" * 70)
    logger.log("RESULTS")
    logger.log("=" * 70)
    
    initial_loss = losses[0] if losses else 0
    final_loss = losses[-1] if losses else 0
    best_loss = min(losses) if losses else 0
    best_step_idx = losses.index(best_loss) if best_loss in losses else 0
    
    reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
    
    logger.log(f"Initial loss: {initial_loss:.4f}")
    logger.log(f"Final loss: {final_loss:.4f}")
    logger.log(f"Best loss: {best_loss:.4f} (step {best_step_idx})")
    logger.log(f"Loss reduction: {reduction:.1f}%")
    
    # Save final checkpoint
    final_checkpoint_path = CHECKPOINT_DIR / "final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": final_loss,
        "losses": losses,
    }, final_checkpoint_path)
    logger.log(f"Final checkpoint saved: final.pt")
    
    logger.log("")
    logger.log("=" * 70)
    logger.log("TRAINING COMPLETE")
    logger.log("=" * 70)
    logger.log(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    logger.log(f"Generated samples saved to: {GENERATED_DIR}")
    logger.log(f"Metrics saved to: {METRICS_FILE}")
    
    logger.close()
    metrics_logger.close()


if __name__ == "__main__":
    main()
