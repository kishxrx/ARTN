#!/usr/bin/env python3
"""
Adafactor Learning Rate Sensitivity Test
========================================

Test whether different learning rates improve Adafactor performance.

Tests:
- lr = 1e-4 (10x lower)
- lr = 5e-4 (2x lower)
- lr = 1e-3 (current)
- lr = 2e-3 (2x higher)

Short run (500 steps each) to quickly identify best LR.

Expected runtime: ~20-40 minutes
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.model.gpt import GPT
from src.training.adafactor_canonical import CanonicalAdafactor

print("=" * 80)
print("ADAFACTOR LEARNING RATE SENSITIVITY TEST")
print("=" * 80)
print(f"\nExpected runtime: ~20-40 minutes")
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
        'max_steps': 500,  # Short run
        'batch_size': 4,
        'gradient_accumulation_steps': 2,
        'seed': 42,
    },
    'learning_rates': [1e-4, 5e-4, 1e-3, 2e-3],
}

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Load data
data_dir = Path('/root/agent-sandbox/ARTN/outputs/tokenized_cache')
train_data = torch.load(data_dir / 'train_tokenized.pt', map_location='cpu', weights_only=False)
val_data = torch.load(data_dir / 'val_tokenized.pt', map_location='cpu', weights_only=False)

# Data loader
def get_batch(data, batch_size, block_size, device):
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

# Training function
def train_with_lr(lr):
    torch.manual_seed(config['training']['seed'])
    model = GPT(**config['model']).to(device)
    optimizer = CanonicalAdafactor(
        model.parameters(),
        lr=lr,
        beta2_decay=-0.8,
        eps=(1e-30, 1e-3),
        d=1.0,
        weight_decay=0.0,
    )
    
    block_size = config['model']['max_seq_len']
    batch_size = config['training']['batch_size']
    grad_accum = config['training']['gradient_accumulation_steps']
    
    model.train()
    
    for step in range(config['training']['max_steps']):
        optimizer.zero_grad()
        total_loss = 0.0
        
        for _ in range(grad_accum):
            x, y = get_batch(train_data, batch_size, block_size, device)
            outputs = model(x, y)
            loss = outputs['loss'] / grad_accum
            loss.backward()
            total_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_losses = []
        for _ in range(20):
            x, y = get_batch(val_data, batch_size, block_size, device)
            outputs = model(x, y)
            val_losses.append(outputs['loss'].item())
        final_val_loss = sum(val_losses) / len(val_losses)
    
    return final_val_loss

# Test each LR
results = {}

print(f"{'=' * 80}")
print("TESTING LEARNING RATES")
print(f"{'=' * 80}\n")

for lr in config['learning_rates']:
    print(f"\nTesting lr = {lr:.1e}...")
    val_loss = train_with_lr(lr)
    results[lr] = val_loss
    print(f"  Final val loss: {val_loss:.4f}")

# Find best LR
best_lr = min(results.keys(), key=lambda x: results[x])
best_loss = results[best_lr]

print(f"\n{'=' * 80}")
print("RESULTS")
print(f"{'=' * 80}")

print(f"\n{'Learning Rate':<20} {'Final Val Loss':<18} {'Relative to Best':<20}")
print("-" * 60)

for lr, loss in sorted(results.items()):
    rel = loss / best_loss
    marker = " ← BEST" if lr == best_lr else ""
    print(f"{lr:<20.1e} {loss:<18.4f} {rel:<20.2f}x{marker}")

print(f"\n{'=' * 80}")
print("COMPARISON WITH ADAMW")
print(f"{'=' * 80}")

# Reference AdamW result from Stage 2
adamw_val_loss = 2.4058

print(f"\nAdamW (lr=1e-3):  {adamw_val_loss:.4f}")
print(f"Best Adafactor (lr={best_lr:.1e}): {best_loss:.4f}")
print(f"Ratio: {best_loss / adamw_val_loss:.2f}x")

if best_loss < 3.5:
    print(f"\n⚠️ Adafactor with tuned LR improves to {best_loss / adamw_val_loss:.2f}x vs AdamW")
    print("   Gap reduced but still underperforms")
else:
    print(f"\n✗ Adafactor still significantly underperforms even with LR tuning")

print(f"\n{'=' * 80}")
print("SENSITIVITY TEST COMPLETE")
print(f"{'=' * 80}")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = Path(f'/root/agent-sandbox/ARTN/outputs/adafactor_lr_sensitivity_{timestamp}')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'results.json', 'w') as f:
    json.dump({
        'learning_rates': {str(k): v for k, v in results.items()},
        'best_lr': best_lr,
        'best_loss': best_loss,
        'adamw_reference': adamw_val_loss,
        'ratio_to_adamw': best_loss / adamw_val_loss,
    }, f, indent=2)

print(f"\nResults saved to: {output_dir}")
