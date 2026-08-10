#!/usr/bin/env python3
"""
ARTN Training Script - CORRECT VERSION
Based on original train.py that worked (loss 8.45 healthy)

CRITICAL MONITORING:
- Entropy (must stay >30% of max)
- Top token probability (must stay <50%)
- Sample quality with temperature
"""

import os
import sys
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.training.adafactor import Adafactor
from src.training.checkpoint import CheckpointManager, save_final_model
from src.tokenizer.bpe_tokenizer import BPETokenizer


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Model
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    "dropout": 0.1,
    
    # Training - CORRECT SETTINGS FROM ORIGINAL
    "batch_size": 4,
    "gradient_accumulation_steps": 2,  # Your request: effective batch 8
    "learning_rate": 0.0003,  # ORIGINAL: 3e-4 - PROVEN TO WORK
    "weight_decay": 0.01,
    "beta1": 0.9,  # ORIGINAL: 0.9 - MUST KEEP
    "beta2": 0.999,
    "relative_step": False,  # ORIGINAL: False - MUST KEEP
    "scale_parameter": True,  # ORIGINAL: True - MUST KEEP
    "warmup_init": True,  # ORIGINAL: True - MUST KEEP
    "warmup_steps": 500,
    "min_lr": 1e-5,
    "max_steps": 50000,
    "max_grad_norm": 1.0,
    
    # Checkpoint & Logging
    "checkpoint_every": 1000,
    "log_every": 500,
    "monitor_every": 1000,  # MONITOR entropy/probability
    
    # Output
    "output_dir": "outputs/artn-final",
    "experiment_name": "artn-correct",
    
    # Hardware
    "device": "cpu",
    "num_workers": 2,
}


# =============================================================================
# MONITORING - CRITICAL FOR DETECTING COLLAPSE
# =============================================================================

def monitor_model_health(model, tokenizer, step, device):
    """
    Check for collapse indicators.
    
    THRESHOLDS:
    - Entropy: Must be >30% of max (max = ln(4000) = 8.29)
    - Top token prob: Must be <50%
    - Random input loss: Must be >70% of training loss
    
    Returns: (status, metrics_dict)
    """
    model.eval()
    
    metrics = {}
    warnings = []
    
    # Test prompt
    prompt = "Once upon a time"
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits'][0, -1, :]
        probs = torch.softmax(logits, dim=0)
        
        # Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = torch.log(torch.tensor(4000.0)).item()
        entropy_pct = (entropy / max_entropy) * 100
        metrics['entropy'] = entropy
        metrics['entropy_pct'] = entropy_pct
        
        # Top token probability
        top_prob, top_idx = torch.max(probs, dim=0)
        top_token = tokenizer.decode([top_idx.item()])
        metrics['top_token'] = top_token
        metrics['top_prob'] = top_prob.item() * 100
        
        # Top 5 tokens
        top5 = torch.topk(probs, k=5)
        top5_list = []
        for prob, idx in zip(top5.values, top5.indices):
            token = tokenizer.decode([idx.item()])
            top5_list.append((token, prob.item() * 100))
        metrics['top5'] = top5_list
    
    # Check thresholds
    if entropy_pct < 20:
        warnings.append(f"CRITICAL: Entropy {entropy_pct:.1f}% < 20% - COLLAPSE DETECTED")
    elif entropy_pct < 30:
        warnings.append(f"WARNING: Entropy {entropy_pct:.1f}% < 30% - MONITOR CLOSELY")
    
    if metrics['top_prob'] > 70:
        warnings.append(f"CRITICAL: Top token '{top_token}' has {metrics['top_prob']:.1f}% - COLLAPSE DETECTED")
    elif metrics['top_prob'] > 50:
        warnings.append(f"WARNING: Top token '{top_token}' has {metrics['top_prob']:.1f}% - MONITOR CLOSELY")
    
    # Status
    status = "HEALTHY"
    if len(warnings) > 0:
        status = "WARNING"
        if any("CRITICAL" in w for w in warnings):
            status = "COLLAPSED"
    
    model.train()
    return status, metrics, warnings


def generate_sample(model, tokenizer, prompt, max_tokens=40, temperature=0.9, top_k=50, device='cpu'):
    """Generate sample with temperature + top-k sampling"""
    model.eval()
    
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(generated)
            logits = outputs['logits'][0, -1, :] / temperature
            
            # Top-k filtering
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    model.train()
    return tokenizer.decode(generated[0].tolist())


# =============================================================================
# Dataset
# =============================================================================

def prepare_dataset(config):
    print(f"\n[Dataset] Loading TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    if config.get("train_samples"):
        dataset = dataset.select(range(min(config["train_samples"], len(dataset))))
    print(f"[Dataset] {len(dataset)} samples")
    return dataset


class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        tokens = self.tokenizer.encode(text)[:self.max_length]
        padding_length = self.max_length - len(tokens)
        tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": input_ids, "labels": labels}


# =============================================================================
# Training
# =============================================================================

def train_step(model, batch, optimizer, scheduler, config, device, accumulation_step):
    model.train()
    
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    
    outputs = model(input_ids, targets=labels)
    loss = outputs["loss"]
    
    # Scale loss for gradient accumulation
    scaled_loss = loss / config["gradient_accumulation_steps"]
    scaled_loss.backward()
    
    # Only step optimizer after accumulation
    if accumulation_step == config["gradient_accumulation_steps"] - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
    
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    # Update config
    CONFIG["max_steps"] = args.total_steps
    CONFIG["batch_size"] = args.batch_size
    CONFIG["gradient_accumulation_steps"] = args.accum_steps
    CONFIG["learning_rate"] = args.lr
    
    print("\n" + "="*70)
    print("ARTN TRAINING - CORRECT VERSION")
    print("="*70)
    print(f"Total steps: {CONFIG['max_steps']}")
    print(f"LR: {CONFIG['learning_rate']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Accumulation: {CONFIG['gradient_accumulation_steps']}")
    print(f"Effective batch: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"Warmup: {CONFIG['warmup_steps']}")
    print(f"Optimizer: Adafactor (beta1={CONFIG['beta1']}, beta2={CONFIG['beta2']})")
    print(f"CRITICAL: Monitoring entropy every {CONFIG['monitor_every']} steps")
    print("="*70)
    
    device = torch.device(CONFIG["device"])
    print(f"\n[Device] {device}")
    
    # Output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Tokenizer
    print("\n[Tokenizer] Training new BPE tokenizer...")
    dataset_for_tok = load_dataset("roneneldan/TinyStories", split="train")
    texts = [item["text"] for item in dataset_for_tok.select(range(min(100000, len(dataset_for_tok))))]
    corpus = "\n".join(texts)
    
    tokenizer = BPETokenizer(vocab_size=CONFIG["vocab_size"], min_frequency=2)
    tokenizer.train(corpus)
    tokenizer.save(output_dir / "tokenizer.json")
    print(f"[Tokenizer] Vocab size: {CONFIG['vocab_size']}")
    
    # Model
    print("\n[Model] Creating BitLinearGPT...")
    model = BitLinearGPT(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        max_seq_len=CONFIG["max_seq_len"],
        dropout=CONFIG["dropout"],
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {num_params:,} parameters")
    
    # Optimizer - CORRECT SETTINGS FROM ORIGINAL
    optimizer = Adafactor(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        beta1=CONFIG["beta1"],  # 0.9 - MUST KEEP
        beta2=CONFIG["beta2"],
    )
    print(f"[Optimizer] Adafactor (lr={CONFIG['learning_rate']}, beta1={CONFIG['beta1']})")
    
    # Scheduler
    def lr_lambda(step):
        if step < CONFIG["warmup_steps"]:
            return step / CONFIG["warmup_steps"]
        else:
            progress = (step - CONFIG["warmup_steps"]) / (CONFIG["max_steps"] - CONFIG["warmup_steps"])
            return max(CONFIG["min_lr"] / CONFIG["learning_rate"], 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume
    start_step = 0
    if args.resume:
        print(f"\n[Resume] Loading from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        print(f"[Resume] Starting from step {start_step}")
    
    # Dataset
    raw_dataset = prepare_dataset(CONFIG)
    train_dataset = TinyStoriesDataset(raw_dataset, tokenizer, CONFIG["max_seq_len"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    
    print(f"\n[DataLoader] {len(train_dataset)} samples, batch_size={CONFIG['batch_size']}")
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)
    
    start_time = time.time()
    data_iter = iter(train_loader)
    accumulated_loss = 0.0
    
    for step in range(start_step + 1, CONFIG["max_steps"] + 1):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # Train step with accumulation
        accum_step = (step - 1) % CONFIG["gradient_accumulation_steps"]
        loss = train_step(model, batch, optimizer, scheduler, CONFIG, device, accum_step)
        accumulated_loss += loss
        
        # Progress
        if step % 500 == 0:
            avg_loss = accumulated_loss / 500
            elapsed = time.time() - start_time
            eta = (elapsed / step) * (CONFIG["max_steps"] - step)
            
            progress = step / CONFIG["max_steps"]
            bar_len = 30
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[{bar}] Step {step}/{CONFIG['max_steps']} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | ETA: {int(eta//3600):02d}:{int((eta%3600)//60):02d}:{int(eta%60):02d}")
            
            accumulated_loss = 0.0
        
        # MONITOR - CRITICAL (runs automatically when checkpoint is saved)
        if step % CONFIG["monitor_every"] == 0:
            status, metrics, warnings = monitor_model_health(model, tokenizer, step, device)
            
            print(f"\n{'='*70}")
            print(f"AUTOMATIC MONITOR CHECK @ Step {step}")
            print(f"{'='*70}")
            print(f"  Loss: {loss:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Entropy: {metrics['entropy']:.4f} ({metrics['entropy_pct']:.1f}% of max)")
            print(f"  Top token: '{metrics['top_token']}' ({metrics['top_prob']:.1f}%)")
            print(f"  Top 5: {[(t, f'{p:.1f}%') for t, p in metrics['top5']]}")
            print(f"  STATUS: {status}")
            
            if warnings:
                print(f"\n  WARNINGS:")
                for w in warnings:
                    print(f"    - {w}")
            
            # Generate sample
            sample = generate_sample(model, tokenizer, "Once upon a time", max_tokens=40, temperature=0.9, top_k=50, device=device)
            print(f"\n  Sample: {sample}")
            
            # Save to log file
            with open(output_dir / "monitor_log.txt", "a") as f:
                import datetime
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp} | Step {step} | Loss {loss:.4f} | Entropy {metrics['entropy_pct']:.1f}% | TopProb {metrics['top_prob']:.1f}% | STATUS: {status}\n")
            
            print(f"{'='*70}\n")
            
            # STOP if collapsed
            if status == "COLLAPSED":
                print(f"\n{'!'*70}")
                print(f"{'!'*70}")
                print(f"!!! MODEL COLLAPSED - TRAINING STOPPED AUTOMATICALLY !!!")
                print(f"{'!'*70}")
                print(f"{'!'*70}\n")
                break
        
        # Checkpoint
        if step % CONFIG["checkpoint_every"] == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "config": CONFIG,
            }, ckpt_path)
            print(f"[Checkpoint] Saved to {ckpt_path}")
    
    # Final
    if step >= CONFIG["max_steps"]:
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        total_time = time.time() - start_time
        print(f"Total time: {int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}")
        print(f"Final loss: {loss:.4f}")
        print(f"Model saved to: {output_dir}")
        print("="*70)
        
        # Save final model
        torch.save(model.state_dict(), output_dir / "final_model.pt")


if __name__ == "__main__":
    main()
