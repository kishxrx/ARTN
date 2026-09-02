#!/usr/bin/env python3
"""Check actual weight values in BitLinear"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.model.bitlinear import BitLinear

# Create a BitLinear layer
layer = BitLinear(256, 256)

# Check actual weight values
print("Raw weight sample (first 10):")
print(layer.weight.data.flatten()[:10])

# Forward pass triggers quantization
x = torch.randn(1, 256)
out = layer(x)

# Check weight again after forward
print("\nWeight after forward (first 10):")
print(layer.weight.data.flatten()[:10])

# Check if weights are actually ternary
w = layer.weight.data
neg_count = (w == -1).sum().item()
zero_count = (w == 0).sum().item()
pos_count = (w == 1).sum().item()

print(f"\nTernary check on stored weights:")
print(f"  -1 count: {neg_count}")
print(f"   0 count: {zero_count}")
print(f"  +1 count: {pos_count}")

# Check what values are in weights
print(f"\nActual weight range:")
print(f"  Min: {w.min().item():.4f}")
print(f"  Max: {w.max().item():.4f}")
print(f"  Mean: {w.mean().item():.4f}")
print(f"  Std: {w.std().item():.4f}")

# Check sparsity tracking
print(f"\nRunning sparsity: {layer.running_sparsity.item():.4f}")

# Check if dequantization happens
print("\n" + "=" * 60)
print("ANALYZING QUANTIZATION BEHAVIOR")
print("=" * 60)

# Manual quantization check
scale = w.abs().mean()
w_scaled = w / scale
threshold = 0.5

# How many weights should be ternary?
should_be_neg = (w_scaled < -threshold).sum().item()
should_be_zero = ((w_scaled >= -threshold) & (w_scaled <= threshold)).sum().item()
should_be_pos = (w_scaled > threshold).sum().item()

total = w.numel()

print(f"\nIf weights were quantized:")
print(f"  -1: {should_be_neg}/{total} ({should_be_neg/total*100:.1f}%)")
print(f"   0: {should_be_zero}/{total} ({should_be_zero/total*100:.1f}%)")
print(f"  +1: {should_be_pos}/{total} ({should_be_pos/total*100:.1f}%)")
