#!/usr/bin/env python3
"""
ARTN Optimized Training Script - CORRECT IMPLEMENTATION.

EXACTLY as originally planned:
- LR 0.003 (max), 0.0001 (min)
- Warmup 1000 steps
- Effective batch 32 via gradient accumulation
- Gradient clipping 1.0
- Cosine decay schedule

CORRECT gradient accumulation:
- Accumulate gradients over MULTIPLE batches (not split one batch)

Usage:
    python scripts/train_optimized.py --total-steps 50000
    python scripts/train_optimized.py --total-steps 100000 --resume checkpoint.pt
"""

import sys
import time
import math
import torch
import argparse
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.adafactor import Adafactor


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

class CosineWarmupScheduler:
    """Cosine learning rate schedule with warmup and decay."""
    
    def __init__(self, warmup_steps, total_steps, max_lr, min_lr=1e-5):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
    
    def get_lr(self, step):
        """Get learning rate for current step."""
        if step < self.warmup_steps:
            # Warmup: linear increase
            return self.max_lr * (step + 1) / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# Configuration - ORIGINAL PLAN (with two options)
# =============================================================================

CONFIG = {
    # Model architecture
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    
    # Optimized training parameters - STABLE FOR LONG TRAINING
    "batch_size": 4,  # Physical batch size
    "gradient_accumulation_steps": 8,  # Accumulate over 8 batches -> Effective batch = 32
    "max_lr": 0.0003,  # SAFE LR for ternary (proven to work)
    "min_lr": 0.00003,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "beta1": 0.9,
    "beta2": 0.98,
    
    # Training schedule - ORIGINAL PLAN
    "total_steps": 20000,
    "warmup_steps": 1000,
    "checkpoint_every": 1000,
    "eval_every": 500,
    
    # Data
    "max_samples": 50000,
    "dataset": "roneneldan/TinyStories",
    
    # Output
    "output_dir": "outputs/artn-optimized",
}


# =============================================================================
# Dataset
# =============================================================================

class TinyStoriesDataset(Dataset):
    """Dataset for TinyStories."""
    
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)[:self.max_length - 1]
        
        # Add EOS token
        tokens = tokens + [self.tokenizer.eos_token_id]
        
        # Pad
        padding_length = self.max_length - len(tokens)
        tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
        
        # Create tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        
        # Mask padding (but NOT EOS)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": input_ids, "labels": labels}


# =============================================================================
# Training Functions - CORRECT IMPLEMENTATION
# =============================================================================

def get_batch(data_iter, train_loader):
    """Get next batch, resetting iterator if needed."""
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)
    return batch, data_iter


def train_step(model, data_iter, train_loader, optimizer, scheduler, config, device, step):
    """
    Training step with CORRECT gradient accumulation.
    
    Accumulate gradients over MULTIPLE batches, not split one batch.
    """
    
    model.train()
    accum_steps = config["gradient_accumulation_steps"]
    
    # Get current learning rate
    lr = scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Clear gradients BEFORE accumulation
    optimizer.zero_grad()
    
    accumulated_loss = 0.0
    nan_detected = False
    
    # Accumulate gradients over MULTIPLE batches
    for i in range(accum_steps):
        # Get NEW batch each time (correct way)
        batch, data_iter = get_batch(data_iter, train_loader)
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(input_ids, targets=labels)
        loss = outputs["loss"]
        
        # Check for NaN before accumulating
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[WARNING] NaN/Inf loss at step {step}, batch {i+1}/{accum_steps}")
            nan_detected = True
            continue
        
        # Scale loss for gradient accumulation (BEFORE backward)
        scaled_loss = loss / accum_steps
        
        # Backward pass (accumulates gradients)
        scaled_loss.backward()
        
        accumulated_loss += loss.item()
    
    if nan_detected:
        optimizer.zero_grad()
        return None, lr, data_iter
    
    # Gradient clipping (AFTER all accumulation)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
    
    # Check for NaN gradients
    has_nan_grad = False
    for p in model.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                has_nan_grad = True
                break
    
    if has_nan_grad:
        print(f"\n[WARNING] NaN gradients at step {step}, skipping")
        optimizer.zero_grad()
        return None, lr, data_iter
    
    # Optimizer step
    optimizer.step()
    
    # Return average loss and updated data_iter
    avg_loss = accumulated_loss / accum_steps
    return avg_loss, lr, data_iter


# =============================================================================
# Progress Display
# =============================================================================

def display_progress(step, total_steps, loss, lr, start_time):
    """Display training progress."""
    
    elapsed = time.time() - start_time
    steps_per_sec = step / elapsed if elapsed > 0 else 0
    remaining_steps = total_steps - step
    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
    
    eta_h = int(eta_seconds // 3600)
    eta_m = int((eta_seconds % 3600) // 60)
    eta_s = int(eta_seconds % 60)
    
    progress = step / total_steps
    bar_length = 30
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"\r[{bar}] Step {step}/{total_steps} | Loss: {loss:.4f} | LR: {lr:.6f} | ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}", end="", flush=True)


# =============================================================================
# Main Training
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimized ARTN Training - ORIGINAL PLAN")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total steps")
    parser.add_argument("--max-lr", type=float, default=None, help="Override max learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--accum-steps", type=int, default=None, help="Override gradient accumulation steps")
    args = parser.parse_args()
    
    # Override config if specified
    if args.total_steps:
        CONFIG["total_steps"] = args.total_steps
    if args.max_lr:
        CONFIG["max_lr"] = args.max_lr
    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
    if args.accum_steps:
        CONFIG["gradient_accumulation_steps"] = args.accum_steps
    
    config = CONFIG
    
    # Calculate effective batch
    effective_batch = config["batch_size"] * config["gradient_accumulation_steps"]
    
    print("\n" + "="*60)
    print("ARTN OPTIMIZED TRAINING - ORIGINAL PLAN")
    print("="*60)
    print(f"Total steps: {config['total_steps']}")
    print(f"Max LR: {config['max_lr']}")
    print(f"Min LR: {config['min_lr']}")
    print(f"Warmup: {config['warmup_steps']} steps")
    print(f"Batch size: {config['batch_size']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"EFFECTIVE BATCH: {effective_batch}")
    print(f"Gradient clipping: {config['max_grad_norm']}")
    print("="*60 + "\n")
    
    # Device
    device = torch.device("cpu")
    print(f"[Device] {device}")
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    
    # Load/create tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    
    if tokenizer_path.exists():
        print(f"\n[Tokenizer] Loading from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print("\n[Tokenizer] Training new BPE tokenizer...")
        
        # Load data for tokenizer
        dataset = load_dataset(config["dataset"], split="train")
        texts = [item["text"] for item in dataset.select(range(min(10000, len(dataset))))]
        corpus = "\n".join(texts)
        
        tokenizer = BPETokenizer(vocab_size=4000)
        tokenizer.train(corpus)
        tokenizer.save(tokenizer_path)
        print(f"[Tokenizer] Saved to {tokenizer_path}")
    
    print(f"[Tokenizer] Vocab size: {len(tokenizer.vocab)}")
    
    # Create model
    print("\n[Model] Creating BitLinearGPT...")
    model = BitLinearGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {num_params:,} parameters")
    
    # Optimizer - ADAFACTOR (not AdamW, which failed)
    optimizer = Adafactor(
        model.parameters(),
        lr=config["max_lr"],
        weight_decay=config["weight_decay"],
        beta1=0.9,  # MUST KEEP - momentum is critical for stability
        beta2=0.999,
    )
    
    # Scheduler - ORIGINAL PLAN
    scheduler = CosineWarmupScheduler(
        warmup_steps=config["warmup_steps"],
        total_steps=config["total_steps"],
        max_lr=config["max_lr"],
        min_lr=config["min_lr"],
    )
    
    print(f"[Optimizer] Adafactor (lr={config['max_lr']}, weight_decay={config['weight_decay']})")
    
    # Load dataset
    print("\n[Dataset] Loading TinyStories...")
    dataset = load_dataset(config["dataset"], split="train")
    
    if config["max_samples"]:
        dataset = dataset.select(range(min(config["max_samples"], len(dataset))))
    
    print(f"[Dataset] {len(dataset):,} samples")
    
    # Create dataloader
    train_dataset = TinyStoriesDataset(dataset, tokenizer, config["max_seq_len"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    
    print(f"[DataLoader] Batches: {len(train_loader)}, Batch size: {config['batch_size']}")
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60 + "\n")
    
    start_time = time.time()
    data_iter = iter(train_loader)
    
    losses = []
    start_step = 1
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n[Resuming] Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        print(f"[Resuming] Starting from step {start_step}")
        print(f"[Resuming] Previous loss: {checkpoint.get('loss', 'N/A')}\n")
    
    for step in range(start_step, config["total_steps"] + 1):
        # Train step (passes data_iter to update it)
        result = train_step(model, data_iter, train_loader, optimizer, scheduler, config, device, step)
        
        # Unpack result
        loss_or_none, lr, data_iter = result
        
        if loss_or_none is None:
            # NaN detected, skip
            continue
        
        loss = loss_or_none
        losses.append(loss)
        
        # Display progress
        display_progress(step, config["total_steps"], loss, lr, start_time)
        
        # Checkpoint
        if step % config["checkpoint_every"] == 0:
            checkpoint_path = output_dir / "checkpoints" / f"checkpoint_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "config": config,
            }, checkpoint_path)
            
            avg_loss = sum(losses[-100:]) / min(100, len(losses))
            print(f"\n[Checkpoint] Step {step}, Avg Loss: {avg_loss:.4f}, LR: {lr:.6f}")
        
        # Evaluation (sample generation + MONITORING)
        if step % config["eval_every"] == 0 and len(losses) > 0:
            model.eval()
            with torch.no_grad():
                prompt = "Once upon a time"
                input_ids = torch.tensor([tokenizer.encode(prompt)[:50]])
                
                # MONITORING: Check entropy
                outputs = model(input_ids)
                logits = outputs["logits"][0, -1, :]
                probs = torch.softmax(logits, dim=0)
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                max_entropy = math.log(4000)
                entropy_pct = (entropy / max_entropy) * 100
                
                top_prob, top_idx = torch.max(probs, dim=0)
                top_token = tokenizer.decode([top_idx.item()])
                
                # Status determination
                status = "HEALTHY"
                if entropy_pct < 20 or top_prob.item() > 0.7:
                    status = "COLLAPSED"
                elif entropy_pct < 30 or top_prob.item() > 0.5:
                    status = "WARNING"
                
                print(f"\n{'='*70}")
                print(f"MONITOR CHECK @ Step {step}")
                print(f"{'='*70}")
                print(f"  Loss: {loss:.4f}")
                print(f"  LR: {lr:.6f}")
                print(f"  Entropy: {entropy:.4f} ({entropy_pct:.1f}% of max)")
                print(f"  Top token: '{top_token}' ({top_prob.item()*100:.1f}%)")
                
                # Show top 5 tokens
                top5 = torch.topk(probs, k=5)
                top5_list = [(tokenizer.decode([idx.item()]), f"{prob.item()*100:.1f}%") 
                             for prob, idx in zip(top5.values, top5.indices)]
                print(f"  Top 5: {top5_list}")
                
                print(f"  STATUS: {status}")
                
                # Generate sample with temperature + top-k
                generated = input_ids.clone()
                # DON'T set manual_seed - let it be random each time
                for _ in range(30):
                    outputs = model(generated)
                    logits = outputs["logits"][0, -1, :] / 0.9  # Temperature
                    
                    # Top-k filtering
                    top_k = 50
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                    
                    probs = torch.softmax(logits, dim=0)
                    next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
                    generated = torch.cat([generated, next_token], dim=1)
                
                text = tokenizer.decode(generated[0].tolist())
                print(f"\n  Sample: {text[:80]}...")
                print(f"{'='*70}\n")
                
                # Save to log
                import datetime
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(output_dir / "monitor_log.txt", "a") as f:
                    f.write(f"{timestamp} | Step {step} | Loss {loss:.4f} | Entropy {entropy_pct:.1f}% | TopProb {top_prob.item()*100:.1f}% | STATUS: {status}\n")
                
                # STOP if collapsed
                if status == "COLLAPSED":
                    print(f"\n{'!'*70}")
                    print(f"{'!'*70}")
                    print(f"!!! MODEL COLLAPSED - TRAINING STOPPED !!!")
                    print(f"{'!'*70}")
                    print(f"{'!'*70}\n")
                    # Save emergency checkpoint
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "config": config,
                    }, output_dir / "collapsed_checkpoint.pt")
                    break
    
    # Save final model
    final_path = output_dir / "final_model"
    final_path.mkdir(exist_ok=True)
    
    torch.save(model.state_dict(), final_path / "model.pt")
    
    import json
    with open(final_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Summary
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print(f"\n\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    if losses:
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Avg loss (last 100): {sum(losses[-100:])/min(100, len(losses)):.4f}")
    print(f"Model saved to: {final_path}")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
