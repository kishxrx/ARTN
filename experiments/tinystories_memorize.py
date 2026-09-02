#!/usr/bin/env python3
"""
TinyStories Memorization Test for Full Ternary Model.

PURPOSE: Determine if the 6.8M ternary model can memorize actual TinyStories data.

IMPORTANT: This uses EXACT configuration as 50k training to isolate the problem.

If SUCCESS:
  - Investigate full-training strategy and timeline
  
If FAILURE:
  - Systematically investigate: tokenizer, preprocessing, targets, masking, 
    loss, gradients, optimizer, ternary quantization, STE, architecture
"""

import sys
import json
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
    "lr": 0.001,  # SAME as 50k training
    "beta1": 0.5,  # SAME as 50k training
    "beta2": 0.98,  # SAME as 50k training
    "weight_decay": 0.01,  # SAME as 50k training
    "max_grad_norm": 1.0,  # SAME as 50k training
}

TRAINING_CONFIG = {
    "total_steps": 3000,  # Enough to determine if loss decreases
    "batch_size": 4,  # SAME as 50k training
    "accumulation_steps": 2,  # SAME as 50k training
    "lr_scheduler": "constant",  # NO decay (different from 50k)
    "seed": 42,
}

DATA_CONFIG = {
    "num_stories": 20,  # Small deterministic subset
    "seed": 42,  # Deterministic selection
    "max_seq_len": 256,  # SAME as 50k training
}

# Tokenizer path - SAME as 50k training
TOKENIZER_PATH = "/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json"


# ============================================================
# DATASET
# ============================================================

class TinyStoriesSubset(Dataset):
    """Small deterministic subset of TinyStories for memorization test."""
    
    def __init__(
        self,
        tokenizer_path: str,
        num_stories: int = 20,
        max_seq_len: int = 256,
        seed: int = 42,
    ):
        from src.tokenizer.bpe_tokenizer import BPETokenizer
        
        # Load tokenizer - SAME as 50k training
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        
        self.max_seq_len = max_seq_len
        
        # Load TinyStories
        from datasets import load_dataset
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        
        # Deterministic random selection
        random.seed(seed)
        indices = random.sample(range(len(dataset)), num_stories)
        
        # Extract stories
        stories = [dataset[i]['text'] for i in indices]
        
        # Print story selection for reproducibility
        print(f"\nSelected story indices (seed={seed}):")
        print(f"  {indices}")
        
        # Tokenize - SAME method as 50k training
        self.samples = []
        for i, story in enumerate(stories):
            tokens = self.tokenizer.encode(story)[:self.max_seq_len - 1]
            
            # Add EOS token
            tokens = tokens + [self.tokenizer.eos_token_id]
            
            # Print first tokens for verification
            print(f"\nStory {i+1}: {len(tokens)} tokens")
            print(f"  First 20 tokens: {tokens[:20]}")
            decoded_preview = self.tokenizer.decode(tokens[:50])
            print(f"  Preview: {decoded_preview[:80]}...")
            
            self.samples.append(tokens)
        
        print(f"\nTotal samples: {len(self.samples)}")
        print(f"Total tokens: {sum(len(s) for s in self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        return torch.tensor(tokens, dtype=torch.long)


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


# ============================================================
# TRAINING
# ============================================================

def train_memorize():
    """Run memorization test with EXACT config as 50k training."""
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    
    print("=" * 60)
    print("TINYSTORIES MEMORIZATION TEST")
    print("=" * 60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Print configuration
    print("\nMODEL CONFIG (SAME as 50k training):")
    for k, v in MODEL_CONFIG.items():
        print(f"  {k}: {v}")
    
    print("\nOPTIMIZER CONFIG (SAME as 50k training):")
    for k, v in OPTIMIZER_CONFIG.items():
        print(f"  {k}: {v}")
    
    print("\nTRAINING CONFIG:")
    for k, v in TRAINING_CONFIG.items():
        print(f"  {k}: {v}")
    
    print("\nDATA CONFIG:")
    for k, v in DATA_CONFIG.items():
        print(f"  {k}: {v}")
    
    print(f"\nTOKENIZER: {TOKENIZER_PATH}")
    
    # Create dataset
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    dataset = TinyStoriesSubset(
        tokenizer_path=TOKENIZER_PATH,
        num_stories=DATA_CONFIG["num_stories"],
        max_seq_len=DATA_CONFIG["max_seq_len"],
        seed=DATA_CONFIG["seed"],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Create model - SAME architecture as 50k training
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    
    model = BitLinearGPT(
        vocab_size=MODEL_CONFIG["vocab_size"],
        d_model=MODEL_CONFIG["d_model"],
        n_heads=MODEL_CONFIG["n_heads"],
        n_layers=MODEL_CONFIG["n_layers"],
        d_ff=MODEL_CONFIG["d_ff"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
    )
    
    print(f"Model: BitLinearGPT (Ternary)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model = model.to(device)
    model.train()
    
    # Create optimizer - SAME as 50k training
    print("\n" + "=" * 60)
    print("OPTIMIZER")
    print("=" * 60)
    
    from src.training.adafactor import Adafactor
    
    optimizer = Adafactor(
        model.parameters(),
        lr=OPTIMIZER_CONFIG["lr"],
        beta1=OPTIMIZER_CONFIG["beta1"],
        beta2=OPTIMIZER_CONFIG["beta2"],
        weight_decay=OPTIMIZER_CONFIG["weight_decay"],
    )
    
    print(f"Optimizer: Adafactor")
    print(f"LR: {OPTIMIZER_CONFIG['lr']} (constant, no decay)")
    print(f"Beta1: {OPTIMIZER_CONFIG['beta1']}")
    print(f"Beta2: {OPTIMIZER_CONFIG['beta2']}")
    print(f"Weight decay: {OPTIMIZER_CONFIG['weight_decay']}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Total steps: {TRAINING_CONFIG['total_steps']}")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"Gradient accumulation: {TRAINING_CONFIG['accumulation_steps']}")
    print("Learning rate schedule: CONSTANT (no cosine decay)")
    print("Estimated time: ~89 minutes (0.89s per step)")
    print("=" * 60)
    print("Starting training loop...", flush=True)
    
    losses = []
    grad_norms = []
    
    step = 0
    data_iter = iter(dataloader)
    optimizer.zero_grad()
    
    print(f"Step {step:4d}: Starting first forward pass...", flush=True)
    
    while step < TRAINING_CONFIG["total_steps"]:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        batch = batch.to(device)
        
        # Targets = shifted input
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Forward
        outputs = model(inputs, targets)
        loss = outputs["loss"]
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\nERROR: NaN loss at step {step}")
            print("TRAINING FAILED")
            return {"status": "failed", "reason": "NaN loss", "step": step}
        
        # Backward
        loss.backward()
        
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
            if step % 100 == 0 or step == TRAINING_CONFIG["total_steps"] - 1:
                print(f"Step {step:4d}: Loss = {loss.item():.4f}, Grad norm = {grad_norm:.4f}", flush=True)
        
        step += 1
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    best_loss = min(losses)
    best_step = losses.index(best_loss)
    
    if initial_loss > 0:
        reduction = (initial_loss - final_loss) / initial_loss * 100
    else:
        reduction = 0
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Best loss: {best_loss:.4f} (step {best_step})")
    print(f"Loss reduction: {reduction:.1f}%")
    
    # Determine status
    # Success: loss降到很低（例如<1.0）并且保持下降趋势
    # Failure: loss几乎不变或者下降后停滞
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if final_loss < 1.0 and reduction > 80:
        print("STATUS: MEMORIZATION SUCCESS")
        print("\nFINDING:")
        print("  Model successfully memorized TinyStories subset")
        print("  Ternary architecture CAN learn structured language data")
        print("\nNEXT STEP:")
        print("  Investigate why 50k training failed:")
        print("  - Learning rate schedule (cosine decay)")
        print("  - Training duration")
        print("  - Batch size")
        print("  - Data shuffling")
        status = "success"
    elif reduction > 50 and best_loss < 3.0:
        print("STATUS: PARTIAL MEMORIZATION")
        print("\nFINDING:")
        print("  Model shows significant learning")
        print("  May need more steps or different approach")
        print("\nNEXT STEP:")
        print("  Continue training or adjust config")
        status = "partial"
    else:
        print("STATUS: MEMORIZATION FAILED")
        print("\nFINDING:")
        print("  Model could NOT memorize TinyStories subset")
        print("  Problem is NOT specific to full training run")
        print("\nNEXT STEP:")
        print("  Systematically investigate:")
        print("  1. Tokenizer and preprocessing")
        print("  2. Target shifting")
        print("  3. Causal masking")
        print("  4. Loss calculation")
        print("  5. Gradient flow")
        print("  6. Optimizer behavior")
        print("  7. Ternary quantization")
        print("  8. STE implementation")
        print("  9. Architecture details")
        status = "failed"
    
    # Save results
    results = {
        "status": status,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_step": best_step,
        "loss_reduction_pct": reduction,
        "losses": losses,
        "grad_norms": grad_norms,
        "config": {
            "model": MODEL_CONFIG,
            "optimizer": OPTIMIZER_CONFIG,
            "training": TRAINING_CONFIG,
            "data": DATA_CONFIG,
        }
    }
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "memorize_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = train_memorize()
    
    # Print loss trajectory summary
    if results["status"] != "error":
        print("\n" + "=" * 60)
        print("LOSS TRAJECTORY")
        print("=" * 60)
        losses = results["losses"]
        print("Every 100 steps:")
        for i in range(0, len(losses), 100):
            print(f"  Step {i:4d}: {losses[i]:.4f}")
