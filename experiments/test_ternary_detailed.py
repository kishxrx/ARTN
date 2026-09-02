#!/usr/bin/env python3
"""
TernarySTE Line-by-Line Investigation
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("TERNARYSTE FORWARD LINE-BY-LINE ANALYSIS")
print("=" * 70)

# Create weight tensor matching typical BitLinear layer
weight = torch.randn(1024, 256)  # d_ff=1024, d_model=256
scale = weight.abs().mean().clamp(min=1e-8)
threshold = 0.5

print(f"\n[INPUT TENSOR STATE]")
print(f"  weight.shape: {weight.shape}")
print(f"  weight.dtype: {weight.dtype}")
print(f"  weight.device: {weight.device}")
print(f"  weight.numel(): {weight.numel():,}")
print(f"  weight.memory: {weight.element_size() * weight.numel():,} bytes ({weight.element_size() * weight.numel() / 1024:.2f} KB)")
print(f"  weight.mean(): {weight.mean():.6f}")
print(f"  weight.std(): {weight.std():.6f}")
print(f"  weight.min(): {weight.min():.6f}")
print(f"  weight.max(): {weight.max():.6f}")

print(f"\n[SCALE TENSOR STATE]")
print(f"  scale: {scale:.6f}")
print(f"  scale.dtype: {scale.dtype}")
print(f"  scale.device: {scale.device}")

print(f"\n[PARAMETERS]")
print(f"  threshold: {threshold}")

print(f"\n[LINE 49] weight_scaled = weight / scale.clamp(min=1e-8)")
start = time.time()
weight_scaled = weight / scale.clamp(min=1e-8)
elapsed_line49 = time.time() - start
print(f"  ✓ COMPLETED in {elapsed_line49:.6f}s")
print(f"  weight_scaled.shape: {weight_scaled.shape}")
print(f"  weight_scaled.dtype: {weight_scaled.dtype}")
print(f"  weight_scaled.numel(): {weight_scaled.numel():,}")
print(f"  weight_scaled.memory: {weight_scaled.element_size() * weight_scaled.numel():,} bytes")
print(f"  weight_scaled.mean(): {weight_scaled.mean():.6f}")
print(f"  weight_scaled.std(): {weight_scaled.std():.6f}")

print(f"\n[LINE 55] weight_quantized = torch.zeros_like(weight)")
print(f"  About to create zero tensor with shape {weight.shape}...")
start = time.time()
weight_quantized = torch.zeros_like(weight)
elapsed_line55 = time.time() - start
print(f"  ✓ COMPLETED in {elapsed_line55:.6f}s")
print(f"  weight_quantized.shape: {weight_quantized.shape}")
print(f"  weight_quantized.dtype: {weight_quantized.dtype}")
print(f"  weight_quantized.numel(): {weight_quantized.numel():,}")
print(f"  weight_quantized.memory: {weight_quantized.element_size() * weight_quantized.numel():,} bytes")

print(f"\n[LINE 56] weight_quantized[weight_scaled > threshold] = 1.0")
print(f"  Creating boolean mask: weight_scaled > {threshold}")
start = time.time()
mask_pos = weight_scaled > threshold
elapsed_mask1 = time.time() - start
print(f"  Mask created in {elapsed_mask1:.6f}s")
print(f"  mask_pos.sum(): {mask_pos.sum().item():,} elements")

print(f"  Assigning 1.0 to masked positions...")
start = time.time()
weight_quantized[mask_pos] = 1.0
elapsed_line56 = time.time() - start
print(f"  ✓ COMPLETED in {elapsed_line56:.6f}s")

print(f"\n[LINE 57] weight_quantized[weight_scaled < -threshold] = -1.0")
print(f"  Creating boolean mask: weight_scaled < {-threshold}")
start = time.time()
mask_neg = weight_scaled < -threshold
elapsed_mask2 = time.time() - start
print(f"  Mask created in {elapsed_mask2:.6f}s")
print(f"  mask_neg.sum(): {mask_neg.sum().item():,} elements")

print(f"  Assigning -1.0 to masked positions...")
start = time.time()
weight_quantized[mask_neg] = -1.0
elapsed_line57 = time.time() - start
print(f"  ✓ COMPLETED in {elapsed_line57:.6f}s")

print(f"\n[RESULT]")
print(f"  weight_quantized.shape: {weight_quantized.shape}")
print(f"  weight_quantized.unique(): {weight_quantized.unique().tolist()}")
print(f"  Zeros: {(weight_quantized == 0).sum().item():,}")
print(f"  Ones: {(weight_quantized == 1).sum().item():,}")
print(f"  Neg ones: {(weight_quantized == -1).sum().item():,}")
print(f"  Total: {weight_quantized.numel():,}")

print(f"\n[PERFORMANCE SUMMARY]")
total_time = elapsed_line49 + elapsed_line55 + elapsed_line56 + elapsed_line57
print(f"  Line 49 (scale):     {elapsed_line49:.6f}s")
print(f"  Line 55 (zeros):     {elapsed_line55:.6f}s")
print(f"  Line 56 (pos mask):  {elapsed_line56:.6f}s")
print(f"  Line 57 (neg mask):  {elapsed_line57:.6f}s")
print(f"  TOTAL:               {total_time:.6f}s")

print("\n" + "=" * 70)
print("NOW TEST: Full TernarySTE.apply() call")
print("=" * 70)

from src.model.bitlinear import TernarySTE

weight2 = torch.randn(1024, 256)
print(f"  Input shape: {weight2.shape}")
print(f"  Input numel: {weight2.numel():,}")

start = time.time()
result_quant, result_scale = TernarySTE.apply(weight2)
elapsed = time.time() - start

print(f"  ✓ TernarySTE.apply() completed in {elapsed:.6f}s")
print(f"  Output shape: {result_quant.shape}")
print(f"  Scale: {result_scale:.6f}")

print("\n" + "=" * 70)
print("ALL TESTS PASSED - NO HANG DETECTED")
print("=" * 70)
