#!/usr/bin/env python3
"""
CANONICAL ADAFACTOR VERIFICATION
=================================

Verify against:
1. Original Shazeer & Stern 2018 paper Algorithm 4
2. PyTorch's torch.optim.Adafactor

Key questions:
1. Does canonical Adafactor use momentum?
2. What order: gradient→variance vs momentum→variance?
3. Relative vs absolute step size?
4. Parameter scaling?
"""

import torch
import torch.nn as nn

print("=" * 80)
print("CANONICAL ADAFACTOR VERIFICATION")
print("=" * 80)

# Test with PyTorch's Adafactor
from torch.optim import Adafactor as PyTorchAdafactor

# Create simple model
model = nn.Linear(4, 4)
x = torch.randn(2, 4)
y = model(x)
loss = y.sum()
loss.backward()

# Get gradient
grad = model.weight.grad.clone()
print(f"\nGradient norm: {grad.norm().item():.6f}")
print(f"Gradient shape: {grad.shape}")

# Save initial weight
weight_before = model.weight.data.clone()

# Test PyTorch Adafactor with NO momentum (beta1=None)
print("\n" + "=" * 80)
print("PYTORCH ADAFACTOR (NO MOMENTUM, beta1=None)")
print("=" * 80)

optimizer_no_momentum = PyTorchAdafactor(
    model.weight,
    lr=1e-3,
    beta1=None,  # NO momentum
    beta2=0.999,
    eps1=1e-30,
    eps2=1e-3,
    weight_decay=0.0,
    clip_threshold=1.0,
)

optimizer_no_momentum.step()
weight_after_no_momentum = model.weight.data.clone()
delta_no_momentum = weight_after_no_momentum - weight_before

print(f"Parameter delta norm: {delta_no_momentum.norm().item():.6f}")
print(f"Delta sample: {delta_no_momentum[0, :4].tolist()}")

# Reset gradient
model.weight.grad = grad.clone()
model.weight.data = weight_before.clone()

# Test PyTorch Adafactor WITH momentum (beta1=0.9)
print("\n" + "=" * 80)
print("PYTORCH ADAFACTOR (WITH MOMENTUM, beta1=0.9)")
print("=" * 80)

optimizer_with_momentum = PyTorchAdafactor(
    model.weight,
    lr=1e-3,
    beta1=0.9,  # WITH momentum
    beta2=0.999,
    eps1=1e-30,
    eps2=1e-3,
    weight_decay=0.0,
    clip_threshold=1.0,
)

optimizer_with_momentum.step()
weight_after_with_momentum = model.weight.data.clone()
delta_with_momentum = weight_after_with_momentum - weight_before

print(f"Parameter delta norm: {delta_with_momentum.norm().item():.6f}")
print(f"Delta sample: {delta_with_momentum[0, :4].tolist()}")

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

diff = (delta_no_momentum - delta_with_momentum).norm().item()
print(f"\nDifference between NO momentum vs WITH momentum: {diff:.6e}")

if diff < 1e-6:
    print("→ PyTorch Adafactor produces SAME result with/without momentum")
    print("→ This suggests momentum is NOT applied in the same way as Adam!")
else:
    print("→ PyTorch Adafactor produces DIFFERENT results with/without momentum")

print("\n" + "=" * 80)
print("PAPER ALGORITHM ANALYSIS")
print("=" * 80)

print("""
From Shazeer & Stern 2018, Algorithm 4:

  Line 4:  Gt = ∇ft(Xt-1)  ← GRADIENT
  Line 5-6: Rt, Ct = row/col sums of G²t  ← SECOND MOMENT
  Line 7:   V̂t = Rt × Ct / (1n × Rt)  ← FACTORED VARIANCE
  Line 8:   Ut = Gt / √V̂t  ← GRADIENT / VARIANCE
  Line 9:   Ût = clip(Ut)  ← CLIP
  Line 10:  Xt = Xt-1 - αt × Ût  ← UPDATE

KEY: Line 8 uses GRADIENT (Gt), NOT momentum!

PAPER QUOTE:
  "We present the algorithm with β1 fixed at 0 so as to focus our 
   attention on the second moments. First moments can be included 
   as in Adam without modification if desired."

INTERPRETATION:
  - Canonical Adafactor: Ut = Gt / √V̂t  (GRADIENT / variance)
  - Adafactor with momentum: Ut = mt / √V̂t  (MOMENTUM / variance)
  
The "bug" was NOT in the formula Ut = Gt / √V̂t!
The bug was OVERWRITING Ut with mt/bias_correction AFTER clipping!
""")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
