#!/usr/bin/env python3
"""
CORRECTED ADAFACTOR vs ADAMW - SHORT CONVERGENCE TEST
======================================================

Tests whether the corrected Adafactor implementation converges on FP32 TinyStories.

Comparison:
- Corrected Adafactor (variance applied TO momentum)
- AdamW (proven to work)

Setup:
- FP32 GPT model (6.8M params)
- TinyStories dataset (tokenized)
- Identical hyperparameters
- Short run (2000 steps)

Expected: Both should converge. If corrected Adafactor fails, there may be
additional issues beyond the momentum/variance bug.
"""

import sys
import os
import json
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.model.gpt import GPT
from src.training.adafactor_corrected import Adafactor as AdafactorCorrected

print("=" * 80)
print("CORRECTED ADAFACTOR vs ADAMW - SHORT CONVERGENCE TEST")
print("=" * 80)

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
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'warmup_steps': 100,
        'max_lr': 1e-3,
        'seed': 42,
    },
    'adafactor': {
        'beta1': 0.9,
        'beta2': 0.999,
        'eps1': 1e-30,
        'eps2': 1e-3,
        'weight_decay': 0.01,
        'clip_threshold': 1.0,
        'lr': 1e-3,
    },
    'adamw': {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.01,
        'lr': 1e-3,
    },
}

# Set seed
torch.manual_seed(config['training']['seed'])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load tokenized data
print("\nLoading tokenized TinyStories data...")
data_dir = Path('/root/agent-sandbox/ARTN/outputs/tokenized_cache')
train_data = torch.load(data_dir / 'train_tokenized.pt', map_location='cpu', weights_only=False)
val_data = torch.load(data_dir / 'val_tokenized.pt', map_location='cpu', weights_only=False)

print(f"  Train samples: {len(train_data)}")
print(f"  Val samples: {len(val_data)}")

# Create data loader
def get_batch(data, batch_size, block_size, device):
    """Get a random batch of data."""
    # data is a list of token lists
    indices = torch.randint(0, len(data), (batch_size,))
    batch_x = []
    batch_y = []
    for i in indices:
        seq = data[i.item()]
        if len(seq) < block_size + 1:
            # Pad if too short
            seq = seq + [0] * (block_size + 1 - len(seq))
        # Random starting position
        start = torch.randint(0, max(1, len(seq) - block_size), (1,)).item()
        x = seq[start:start + block_size]
        y = seq[start + 1:start + block_size + 1]
        batch_x.append(x)
        batch_y.append(y)
    x = torch.tensor(batch_x, dtype=torch.long, device=device)
    y = torch.tensor(batch_y, dtype=torch.long, device=device)
    return x, y

# Learning rate schedule
def get_lr(step, warmup_steps, max_lr):
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        # Cosine decay to 10% of max_lr
        progress = (step - warmup_steps) / (config['training']['max_steps'] - warmup_steps)
        return max_lr * (0.1 + 0.9 * (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2)

# Training function
def train_optimizer(optimizer_name, optimizer_class, optimizer_kwargs):
    """Train with specified optimizer."""
    print(f"\n{'=' * 80}")
    print(f"TRAINING: {optimizer_name}")
    print(f"{'=' * 80}")
    
    # Create model
    torch.manual_seed(config['training']['seed'])
    model = GPT(**config['model']).to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Create optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    
    # Training loop
    results = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'steps': [],
    }
    
    block_size = config['model']['max_seq_len']
    batch_size = config['training']['batch_size']
    grad_accum = config['training']['gradient_accumulation_steps']
    
    model.train()
    
    for step in range(config['training']['max_steps']):
        # Get learning rate
        lr = get_lr(step, config['training']['warmup_steps'], config['training']['max_lr'])
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Gradient accumulation
        optimizer.zero_grad()
        total_loss = 0.0
        
        for _ in range(grad_accum):
            x, y = get_batch(train_data, batch_size, block_size, device)
            outputs = model(x, y)
            loss = outputs['loss'] / grad_accum
            loss.backward()
            total_loss += loss.item()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
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
            
            train_loss = total_loss
            results['train_loss'].append(train_loss)
            results['val_loss'].append(val_loss)
            results['learning_rate'].append(lr)
            results['steps'].append(step)
            
            print(f"Step {step:5d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_losses = []
        for _ in range(50):
            x, y = get_batch(val_data, batch_size, block_size, device)
            outputs = model(x, y)
            val_losses.append(outputs['loss'].item())
        final_val_loss = sum(val_losses) / len(val_losses)
    
    print(f"\nFinal validation loss: {final_val_loss:.4f}")
    
    results['final_val_loss'] = final_val_loss
    results['final_val_ppl'] = torch.exp(torch.tensor(final_val_loss)).item()
    
    return results

# Train with corrected Adafactor
print("\n" + "=" * 80)
print("OPTIMIZER 1: CORRECTED ADAFACTOR")
print("=" * 80)

adafactor_results = train_optimizer(
    "Corrected Adafactor",
    AdafactorCorrected,
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

print(f"\n{'Optimizer':<25} {'Final Val Loss':<18} {'Final Val PPL':<15}")
print("-" * 60)
print(f"{'Corrected Adafactor':<25} {adafactor_results['final_val_loss']:<18.4f} {adafactor_results['final_val_ppl']:<15.2f}")
print(f"{'AdamW':<25} {adamw_results['final_val_loss']:<18.4f} {adamw_results['final_val_ppl']:<15.2f}")

ratio = adafactor_results['final_val_loss'] / adamw_results['final_val_loss']
print(f"\nVal Loss Ratio (Adafactor / AdamW): {ratio:.2f}x")

# Verdict
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if ratio < 1.5:
    print(f"\n✓ CORRECTED ADAFACTOR CONVERGES")
    print(f"\nThe corrected implementation successfully trains the model.")
    print(f"Ratio of {ratio:.2f}x is within acceptable range for different optimizers.")
    verdict = "PASS"
elif ratio < 2.0:
    print(f"\n⚠ CORRECTED ADAFACTOR CONVERGES BUT SLOWER")
    print(f"\nRatio of {ratio:.2f}x suggests Adafactor may need tuning.")
    verdict = "PARTIAL"
else:
    print(f"\n✗ CORRECTED ADAFACTOR DOES NOT CONVERGE WELL")
    print(f"\nRatio of {ratio:.2f}x indicates significant convergence issues.")
    print(f"May need further investigation beyond the momentum/variance bug.")
    verdict = "FAIL"

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = Path(f'/root/agent-sandbox/ARTN/outputs/corrected_adafactor_test_{timestamp}')
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
}

with open(output_dir / 'comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"\nResults saved to: {output_dir}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
