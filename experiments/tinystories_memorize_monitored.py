#!/usr/bin/env python3
"""
TinyStories Memorization Test with Monitor Logging.

Monitor log: experiments/results/tinystories_memorize_log.txt

Run with:
  cd /root/agent-sandbox/ARTN
  PYTHONUNBUFFERED=1 /root/agent-sandbox/.venv/bin/python experiments/tinystories_memorize_monitored.py
  
Monitor progress:
  tail -f experiments/results/tinystories_memorize_log.txt
"""

import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT

# ============================================================
# CONFIGURATION - EXACT MATCH TO 50K TRAINING
# ============================================================

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
}

TOKENIZER_PATH = "/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json"

# Monitor log path
MONITOR_LOG = Path(__file__).parent / "results" / "tinystories_memorize_log.txt"


class MonitorLogger:
    """Logger that writes to both stdout and monitor log file."""
    
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, 'w')
        self.start_time = time.time()
        
        # Write header
        header = f"\n{'='*70}\nTINYSTORIES MEMORIZATION TRAINING LOG\nStarted: {datetime.now().isoformat()}\n{'='*70}\n\n"
        self.log_file.write(header)
        self.log_file.flush()
    
    def log(self, message, flush=True):
        """Log message to both stdout and file."""
        timestamp = time.time() - self.start_time
        formatted = f"[{timestamp:8.1f}s] {message}"
        print(formatted, flush=flush)
        self.log_file.write(formatted + "\n")
        if flush:
            self.log_file.flush()
    
    def progress(self, step, loss=None, grad_norm=None, extra=""):
        """Log training progress."""
        if loss is not None and grad_norm is not None:
            msg = f"Step {step:4d}: Loss={loss:.4f}, Grad={grad_norm:.4f} {extra}"
        else:
            msg = f"Step {step:4d}: {extra}"
        self.log(msg)
    
    def close(self):
        self.log_file.close()


# Global logger
logger = None


class TinyStoriesSubset(Dataset):
    """Small deterministic subset of TinyStories for memorization test.
    
    Pre-loads and caches all data in memory to avoid repeated TinyStories reloading.
    """
    
    def __init__(
        self,
        tokenizer_path: str,
        num_stories: int = 20,
        max_seq_len: int = 256,
        seed: int = 42,
        monitor_log=None,
    ):
        from src.tokenizer.bpe_tokenizer import BPETokenizer
        
        logger.log("Loading tokenizer...")
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        logger.log(f"Tokenizer loaded, vocab size: {len(self.tokenizer.vocab)}")
        
        self.max_seq_len = max_seq_len
        
        logger.log("Loading TinyStories dataset...")
        from datasets import load_dataset
        full_dataset = load_dataset("roneneldan/TinyStories", split="train")
        logger.log(f"Dataset loaded, total stories: {len(full_dataset):,}")
        
        logger.log(f"Selecting {num_stories} stories with seed={seed}...")
        random.seed(seed)
        self.indices = random.sample(range(len(full_dataset)), num_stories)
        logger.log(f"Selected indices: {self.indices[:5]}... (first 5 shown)")
        
        # EXTRACT stories into memory BEFORE creating dataset
        logger.log("Extracting stories into memory...")
        stories = [full_dataset[i]['text'] for i in self.indices]
        
        # Delete reference to TinyStories dataset (free memory)
        del full_dataset
        logger.log("TinyStories dataset reference released")
        
        logger.log("Tokenizing all stories ONCE...")
        # Pre-tokenize ALL stories and cache in memory
        self.samples = []
        for i, story in enumerate(stories):
            tokens = self.tokenizer.encode(story)[:self.max_seq_len - 1]
            tokens = tokens + [self.tokenizer.eos_token_id]
            self.samples.append(torch.tensor(tokens, dtype=torch.long))
            if i < 3:
                logger.log(f"  Story {i+1}: {len(tokens)} tokens")
        
        logger.log(f"Total samples: {len(self.samples)}")
        logger.log(f"Total tokens: {sum(len(s) for s in self.samples):,}")
        logger.log(f"All samples CACHED in memory - TinyStories reference released")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Return cached tensor (no disk I/O)
        return self.samples[idx]


def collate_fn(batch):
    """Pad sequences to same length."""
    max_len = max(len(seq) for seq in batch)
    padded = []
    for seq in batch:
        if len(seq) < max_len:
            padding = torch.zeros(max_len - len(seq), dtype=torch.long)
            seq = torch.cat([seq, padding])
        padded.append(seq)
    return torch.stack(padded)


def train_memorize():
    """Run memorization test with EXACT config as 50k training."""
    
    global logger
    logger = MonitorLogger(MONITOR_LOG)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    
    logger.log(f"Device: {device}")
    logger.log("")
    logger.log("=" * 70)
    logger.log("CONFIGURATION")
    logger.log("=" * 70)
    logger.log(f"Model: vocab={MODEL_CONFIG['vocab_size']}, d_model={MODEL_CONFIG['d_model']}, "
               f"layers={MODEL_CONFIG['n_layers']}, heads={MODEL_CONFIG['n_heads']}")
    logger.log(f"Optimizer: {OPTIMIZER_CONFIG['optimizer']}, LR={OPTIMIZER_CONFIG['lr']}, "
               f"beta1={OPTIMIZER_CONFIG['beta1']}, beta2={OPTIMIZER_CONFIG['beta2']}")
    logger.log(f"Training: {TRAINING_CONFIG['total_steps']} steps, "
               f"batch={TRAINING_CONFIG['batch_size']}, accum={TRAINING_CONFIG['accumulation_steps']}")
    logger.log(f"Data: {DATA_CONFIG['num_stories']} stories, seed={DATA_CONFIG['seed']}")
    logger.log(f"Token frequency: Constant (no cosine decay)")
    logger.log("")
    
    # Create dataset
    logger.log("=" * 70)
    logger.log("LOADING DATA")
    logger.log("=" * 70)
    
    dataset = TinyStoriesSubset(
        tokenizer_path=TOKENIZER_PATH,
        num_stories=DATA_CONFIG["num_stories"],
        max_seq_len=DATA_CONFIG["max_seq_len"],
        seed=DATA_CONFIG["seed"],
    )
    
    logger.log("Creating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Single process to avoid data transfer overhead
    )
    logger.log(f"Dataloader created, batches per epoch: {len(dataloader)}")
    logger.log("NOTE: Using PyTorch DataLoader same as train_optimized.py")
    
    # Create model
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
    logger.log(f"Model moved to {device}")
    model = model.to(device)
    model.train()
    
    # Create optimizer
    logger.log("")
    logger.log("=" * 70)
    logger.log("OPTIMIZER")
    logger.log("=" * 70)
    
    from src.training.adafactor import Adafactor
    
    optimizer = Adafactor(
        model.parameters(),
        lr=OPTIMIZER_CONFIG["lr"],
        beta1=OPTIMIZER_CONFIG["beta1"],
        beta2=OPTIMIZER_CONFIG["beta2"],
        weight_decay=OPTIMIZER_CONFIG["weight_decay"],
    )
    
    logger.log(f"Optimizer: Adafactor")
    logger.log(f"LR: {OPTIMIZER_CONFIG['lr']} (constant)")
    logger.log(f"Reset optimizer gradients")
    optimizer.zero_grad()
    
    # Training loop
    logger.log("")
    logger.log("=" * 70)
    logger.log("TRAINING")
    logger.log("=" * 70)
    logger.log(f"Total steps: {TRAINING_CONFIG['total_steps']}")
    logger.log(f"Estimated time: ~{TRAINING_CONFIG['total_steps'] * 0.89 / 60:.0f} minutes")
    logger.log("=" * 70)
    logger.log("")
    
    losses = []
    grad_norms = []
    
    step = 0
    data_iter = iter(dataloader)
    epoch_start_time = time.time()
    best_loss = float('inf')
    best_step = 0
    best_checkpoint_path = None
    checkpoint_dir = Path("experiments/results/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    logger.log(f"Entering training loop...")
    logger.log(f"Checkpoints will be saved to: {checkpoint_dir}")
    logger.log(f"  - Every 500 steps")
    logger.log(f"  - Best loss checkpoint (auto-saved)")
    logger.log(f"  - Final checkpoint at step 3000")
    
    while step < TRAINING_CONFIG["total_steps"]:
        # Get batch
        if step % 100 == 0:
            logger.log(f"Step {step}: Fetching batch...")
        
        try:
            batch = next(data_iter)
        except StopIteration:
            # Epoch complete - report actual duration
            epoch_elapsed = time.time() - epoch_start_time
            logger.log(f"Epoch complete in {epoch_elapsed:.1f}s, restarting...")
            data_iter = iter(dataloader)
            epoch_start_time = time.time()
            batch = next(data_iter)
        
        if step % 100 == 0:
            logger.log(f"Step {step}: Batch fetched, shape={batch.shape}")
        
        batch = batch.to(device)
        
        # Targets = shifted input
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        if step % 100 == 0:
            logger.log(f"Step {step}: Running forward pass...")
        
        # Forward
        start_time = time.time()
        outputs = model(inputs, targets)
        forward_time = time.time() - start_time
        loss = outputs["loss"]
        
        if step % 100 == 0:
            logger.log(f"Step {step}: Forward completed in {forward_time:.2f}s")
        
        # Check for NaN
        if torch.isnan(loss):
            logger.log(f"ERROR: NaN loss at step {step}")
            logger.log("TRAINING FAILED")
            logger.close()
            return {"status": "failed", "reason": "NaN loss", "step": step}
        
        # Backward
        if step % 100 == 0:
            logger.log(f"Step {step}: Running backward pass...")
        
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        if step % 100 == 0:
            logger.log(f"Step {step}: Backward completed in {backward_time:.2f}s")
        
        # Gradient accumulation
        if (step + 1) % TRAINING_CONFIG["accumulation_steps"] == 0:
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                OPTIMIZER_CONFIG["max_grad_norm"]
            )
            
            optimizer.step()
            optimizer.zero_grad()
            
            losses.append(loss.item())
            grad_norms.append(grad_norm.item())
            
            # Log every 100 steps
            if step % 100 == 0:
                logger.progress(step, loss.item(), grad_norm.item())
            
            # Track best loss and save checkpoint
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_step = step
                # Save best checkpoint (overwrite)
                best_checkpoint_path = checkpoint_dir / "best_loss.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "best_loss": best_loss,
                }, best_checkpoint_path)
                if step % 100 == 0:
                    logger.log(f"  → New best loss: {best_loss:.4f} (checkpoint saved)")
        
        step += 1
        
        # Save checkpoint every 500 steps
        if step % 500 == 0 and step > 0:
            checkpoint_path = checkpoint_dir / f"step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item() if 'loss' in dir() else losses[-1],
            }, checkpoint_path)
            logger.log(f"Checkpoint saved: step_{step}.pt")
        
        # Early termination check
        if step % 500 == 0 and len(losses) > 0:
            recent_loss = losses[-1]
            logger.log(f"Step {step}: Current loss = {recent_loss:.4f}")
    
    # Results
    logger.log("")
    logger.log("=" * 70)
    logger.log("RESULTS")
    logger.log("=" * 70)
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": final_loss,
        "losses": losses,
    }, final_checkpoint_path)
    logger.log(f"Final checkpoint saved: final.pt")
    
    # Find best loss from tracking
    best_loss = min(losses)
    best_step = losses.index(best_loss)
    
    # Generate samples from best checkpoint
    if best_checkpoint_path and best_checkpoint_path.exists():
        logger.log("")
        logger.log("=" * 70)
        logger.log("GENERATING SAMPLES FROM BEST CHECKPOINT")
        logger.log("=" * 70)
        
        # Load best checkpoint
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.log(f"Loaded best checkpoint (step {checkpoint['step']}, loss {checkpoint['loss']:.4f})")
        
        # Load tokenizer and 5 sample stories
        from src.tokenizer.bpe_tokenizer import BPETokenizer
        tokenizer_analysis = BPETokenizer.load(TOKENIZER_PATH)
        
        # Generate samples from first 3 stories
        logger.log("")
        logger.log("Generating samples from first 3 stories:")
        
        # We need to get the original stories - reload dataset
        from datasets import load_dataset
        import random
        random.seed(42)
        full_ds = load_dataset("roneneldan/TinyStories", split="train")
        indices = random.sample(range(len(full_ds)), 20)
        
        for i in range(3):
            story = full_ds[indices[i]]["text"]
            tokens = tokenizer_analysis.encode(story)[:50]
            prompt_text = tokenizer_analysis.decode(tokens[:20])
            
            # Generate continuation
            input_ids = torch.tensor([tokens[:20]], dtype=torch.long)
            with torch.no_grad():
                generated = model.generate(input_ids, max_new_tokens=30, temperature=1.0)
            
            generated_text = tokenizer_analysis.decode(generated[0].tolist())
            original_text = tokenizer_analysis.decode(tokens)
            
            logger.log(f"")
            logger.log(f"Story {i+1}:")
            logger.log(f"  Prompt: {prompt_text[:80]}...")
            logger.log(f"  Original: {original_text[:120]}...")
            logger.log(f"  Generated: {generated_text[:120]}...")
        
        logger.log("")
    
    if initial_loss > 0:
        reduction = (initial_loss - final_loss) / initial_loss * 100
    else:
        reduction = 0
    
    logger.log(f"Initial loss: {initial_loss:.4f}")
    logger.log(f"Final loss: {final_loss:.4f}")
    logger.log(f"Best loss: {best_loss:.4f} (step {best_step})")
    logger.log(f"Loss reduction: {reduction:.1f}%")
    
    # Determine status
    logger.log("")
    logger.log("=" * 70)
    logger.log("INTERPRETATION")
    logger.log("=" * 70)
    
    if final_loss < 1.0 and reduction > 80:
        logger.log("STATUS: MEMORIZATION SUCCESS")
        status = "success"
    elif reduction > 50 and best_loss < 3.0:
        logger.log("STATUS: PARTIAL MEMORIZATION")
        status = "partial"
    else:
        logger.log("STATUS: MEMORIZATION FAILED")
        status = "failed"
    
    # Save results
    results = {
        "status": status,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_step": best_step,
        "loss_reduction_pct": reduction,
        "losses": losses[-100:],  # Last 100
        "config": {
            "model": MODEL_CONFIG,
            "optimizer": OPTIMIZER_CONFIG,
            "training": TRAINING_CONFIG,
            "data": DATA_CONFIG,
        }
    }
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "tinystories_memorize_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.log(f"Results saved to: {output_file}")
    logger.close()
    
    return results


if __name__ == "__main__":
    results = train_memorize()
