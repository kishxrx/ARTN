#!/usr/bin/env python3
"""
STAGE 2: Canonical Adafactor Convergence Test
==============================================

Compare Canonical Adafactor vs AdamW on FP32 TinyStories.

ONLY RUN AFTER STAGE 1 PASSES.

Tests:
- Short training run (2000 steps)
- Identical data, initialization, seed
- Report loss, perplexity, update/param ratio, RAM usage
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import json
import tracemalloc

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.model.gpt import GPT
from src.training.adafactor_canonical import CanonicalAdafactor

print("=" * 80)
print("STAGE 2: CANONICAL ADAFACTOR vs ADAMW CONVERGENCE TEST")
print("=" * 80)
print(f"\nExpected runtime: ~45-90 minutes on CPU")
print(f"                   ~10-20 minutes on GPU (if available)")
print()

# Configuration
config = {
    'model': {
        'vocab_size': 4000,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'max_seq_len': 256,
        'dropout': 0.1,
    },
    'training': {
        'max_steps': 2000,
        'batch_size': 4,
        'gradient_accumulation_steps': 2,
        'warmup_steps': 100,
        'max_lr': 1e-3,
        'seed': 42,
    },
    'adafactor': {
        'lr': 1e-3,
        'beta2_decay': -0.8,
        'eps': (1e-30, 1e-3),
        'd': 1.0,
        'weight_decay': 0.0,
    },
    'adamw': {
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.01,
    },
}

# Set seed
torch.manual_seed(config['training']['seed'])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Load data
print("Loading TinyStories data...")
data_dir = Path('/root/agent-sandbox/ARTN/outputs/tokenized_cache')
train_data = torch.load(data_dir / 'train_tokenized.pt', map_location='cpu', weights_only=False)
val_data = torch.load(data_dir / 'val_tokenized.pt', map_location='cpu', weights_only=False)
print(f"  Train: {len(train_data)} samples")
print(f"  Val: {len(val_data)} samples")

# Data loader
def get_batch(data, batch_size, block_size, device):
    """Get random batch."""
    indices = torch.randint(0, len(data), (batch_size,))
    batch_x, batch_y = [], []
    for i in indices:
        seq = data[i.item()]
        if len(seq) < block_size + 1:
            seq = seq + [0] * (block_size + 1 - len(seq))
        start = torch.randint(0, max(1, len(seq) - block_size), (1,)).item()
        batch_x.append(seq[start:start + block_size])
        batch_y.append(seq[start + 1:start + block_size + 1])
    return (
        torch.tensor(batch_x, dtype=torch.long, device=device),
        torch.tensor(batch_y, dtype=torch.long, device=device),
    )

# Learning rate schedule
def get_lr(step, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (2000 - warmup_steps)
        return max_lr * (0.1 + 0.9 * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2)

# Training function with RAM tracking
def train_optimizer(optimizer_name, optimizer_class, optimizer_kwargs):
    print(f"\n{'=' * 80}")
    print(f"TRAINING: {optimizer_name}")
    print(f"{'=' * 80}\n")
    
    # Reset seed for identical initialization
    torch.manual_seed(config['training']['seed'])
    
    # Create model
    model = GPT(**config['model']).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters\n")
    
    # Create optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    
    # Track RAM
    tracemalloc.start()
    
    results = {
        'train_loss': [], 'val_loss': [], 'lr': [], 'steps': [],
        'update_param_ratios': [], 'ram_mb': [],
    }
    
    block_size = config['model']['max_seq_len']
    batch_size = config['training']['batch_size']
    grad_accum = config['training']['gradient_accumulation_steps']
    
    model.train()
    
    for step in range(config['training']['max_steps']):
        # LR schedule
        lr = get_lr(step, config['training']['warmup_steps'], config['training']['max_lr'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr if optimizer_name == "AdamW" else lr  # AdamW uses absolute lr
        
        # Gradient accumulation
        optimizer.zero_grad()
        total_loss = 0.0
        total_update_norm = 0.0
        total_param_norm = 0.0
        
        for _ in range(grad_accum):
            x, y = get_batch(train_data, batch_size, block_size, device)
            outputs = model(x, y)
            loss = outputs['loss'] / grad_accum
            loss.backward()
            total_loss += loss.item()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Track update/param ratio before step
        param_norm = sum(p.data.norm() ** 2 for p in model.parameters()).sqrt().item()
        grad_norm = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None).sqrt().item()
        update_param_ratio = grad_norm / (param_norm + 1e-10)
        
        # Optimizer step
        optimizer.step()
        
        # Track RAM
        current, peak = tracemalloc.get_traced_memory()
        ram_mb = peak / 1024 / 1024
        
        # Log every 100 steps
        if step % 100 == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(10):
                    x, y = get_batch(val_data, batch_size, block_size, device)
                    outputs = model(x, y)
                    val_losses.append(outputs['loss'].item())
                val_loss = sum(val_losses) / len(val_losses)
            model.train()
            
            results['train_loss'].append(total_loss)
            results['val_loss'].append(val_loss)
            results['lr'].append(lr)
            results['steps'].append(step)
            results['update_param_ratios'].append(update_param_ratio)
            results['ram_mb'].append(ram_mb)
            
            print(f"Step {step:5d} | Train: {total_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Upd/Param: {update_param_ratio:.4f} | RAM: {ram_mb:.1f}MB")
    
    tracemalloc.stop()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_losses = []
        for _ in range(50):
            x, y = get_batch(val_data, batch_size, block_size, device)
            outputs = model(x, y)
            val_losses.append(outputs['loss'].item())
        final_val_loss = sum(val_losses) / len(val_losses)
    
    results['final_val_loss'] = final_val_loss
    results['final_val_ppl'] = torch.exp(torch.tensor(final_val_loss)).item()
    results['peak_ram_mb'] = max(results['ram_mb'])
    
    print(f"\nFinal val loss: {final_val_loss:.4f}")
    print(f"Final val PPL: {results['final_val_ppl']:.2f}")
    print(f"Peak RAM: {results['peak_ram_mb']:.1f}MB")
    
    return results

# Train with Canonical Adafactor
print("\n" + "=" * 80)
print("OPTIMIZER 1: CANONICAL ADAFACTOR")
print("=" * 80)

adafactor_results = train_optimizer(
    "Canonical Adafactor",
    CanonicalAdafactor,
    config['adafactor']
)

# Train with AdamW
print("\n" + "=" * 80)
print("OPTIMIZER 2: ADAMW")
print("=" * 80)

adamw_results = train_optimizer(
    "AdamW",
    torch.optim.AdamW,
    config['adamw']
)

# Comparison
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"\n{'Optimizer':<25} {'Final Val Loss':<18} {'Final Val PPL':<15} {'Peak RAM':<12}")
print("-" * 70)
print(f"{'Canonical Adafactor':<25} {adafactor_results['final_val_loss']:<18.4f} "
      f"{adafactor_results['final_val_ppl']:<15.2f} {adafactor_results['peak_ram_mb']:<12.1f}MB")
print(f"{'AdamW':<25} {adamw_results['final_val_loss']:<18.4f} "
      f"{adamw_results['final_val_ppl']:<15.2f} {adamw_results['peak_ram_mb']:<12.1f}MB")

ratio = adafactor_results['final_val_loss'] / adamw_results['final_val_loss']
print(f"\nVal Loss Ratio (Adafactor / AdamW): {ratio:.2f}x")

# Verdict
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if ratio < 1.2:
    print(f"\n✓ Canonical Adafactor performs COMPARABLY to AdamW")
    print(f"  Ratio: {ratio:.2f}x (within 20% of AdamW)")
    verdict = "PASS"
elif ratio < 1.5:
    print(f"\n⚠ Canonical Adafactor performs MODERATELY WORSE than AdamW")
    print(f"  Ratio: {ratio:.2f}x")
    verdict = "PARTIAL"
else:
    print(f"\n✗ Canonical Adafactor performs SIGNIFICANTLY WORSE than AdamW")
    print(f"  Ratio: {ratio:.2f}x")
    verdict = "FAIL"

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = Path(f'/root/agent-sandbox/ARTN/outputs/stage2_canonical_adafactor_{timestamp}')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'adafactor_results.json', 'w') as f:
    json.dump(adafactor_results, f, indent=2)

with open(output_dir / 'adamw_results.json', 'w') as f:
    json.dump(adamw_results, f, indent=2)

comparison = {
    'adafactor_final_val_loss': adafactor_results['final_val_loss'],
    'adamw_final_val_loss': adamw_results['final_val_loss'],
    'ratio': ratio,
    'verdict': verdict,
    'adafactor_peak_ram_mb': adafactor_results['peak_ram_mb'],
    'adamw_peak_ram_mb': adamw_results['peak_ram_mb'],
}

with open(output_dir / 'comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"\nResults saved to: {output_dir}")

print("\n" + "=" * 80)
print("STAGE 2 COMPLETE")
print("=" * 80)
