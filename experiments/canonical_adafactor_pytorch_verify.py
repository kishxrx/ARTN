#!/usr/bin/env python3
"""
CANONICAL ADAFACTOR VERIFICATION - AGAINST PYTORCH
===================================================

Verify that our implementation matches PyTorch's canonical Adafactor.

PyTorch's Adafactor:
- NO momentum (no beta1 parameter!)
- Uses GRADIENT / variance (NOT momentum / variance)
- Relative step size
"""

import torch
import torch.nn as nn
import sys

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from torch.optim import Adafactor as PyTorchAdafactor

print("=" * 80)
print("CANONICAL ADAFACTOR VERIFICATION - PYTORCH COMPARISON")
print("=" * 80)

# Create simple model
torch.manual_seed(42)
model = nn.Linear(4, 4)

# Create gradient
x = torch.randn(2, 4)
y = model(x)
loss = y.sum()
loss.backward()

grad = model.weight.grad.clone()
print(f"\nGradient norm: {grad.norm().item():.6f}")
print(f"Gradient shape: {grad.shape}")

# Save initial weight
weight_before = model.weight.data.clone()

# Test PyTorch's canonical Adafactor
print("\n" + "=" * 80)
print("PYTORCH CANONICAL ADAFACTOR")
print("=" * 80)

optimizer_pytorch = PyTorchAdafactor(
    [model.weight],
    lr=0.001,
    beta2_decay=0.0,  # No decay (fixed beta2 ~ 1.0)
    eps=(1e-30, 1e-3),
    d=1.0,
    weight_decay=0.0,
)

optimizer_pytorch.step()
weight_after_pytorch = model.weight.data.clone()
delta_pytorch = weight_after_pytorch - weight_before

print(f"Parameter delta norm: {delta_pytorch.norm().item():.6f}")
print(f"Delta sample: {delta_pytorch[0, :4].tolist()}")

# Now test our implementation (canonical version)
print("\n" + "=" * 80)
print("OUR CANONICAL ADAFACTOR (Gt / variance, NO momentum)")
print("=" * 80)

# Reset
model.weight.grad = grad.clone()
model.weight.data = weight_before.clone()

# Manual canonical Adafactor step
def canonical_adafactor_step(weight, grad, beta2, eps1, eps2, lr, clip_threshold):
    """Canonical Adafactor from paper Algorithm 4."""
    # Initialize state
    exp_avg_sq_row = torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)
    exp_avg_sq_col = torch.zeros(weight.shape[1], dtype=weight.dtype, device=weight.device)
    
    # Step 1: Compute second moment (row/col sums)
    grad_sq = grad * grad
    
    # Row sums: Rt = β2 × Rt-1 + (1 - β2) × mean(G², dim=1)
    exp_avg_sq_row = beta2 * exp_avg_sq_row + (1 - beta2) * grad_sq.mean(dim=1)
    
    # Col sums: Ct = β2 × Ct-1 + (1 - β2) × mean(G², dim=0)
    exp_avg_sq_col = beta2 * exp_avg_sq_col + (1 - beta2) * grad_sq.mean(dim=0)
    
    # Step 2: Factored variance: V̂t = Rt × Ct / (1n × Rt)
    global_mean = exp_avg_sq_row.mean()  # 1n × Rt
    factored_var = exp_avg_sq_row.unsqueeze(1) * exp_avg_sq_col.unsqueeze(0) / global_mean.add(eps1)
    
    # Step 3: Update = GRADIENT / variance (NOT momentum!)
    update = grad / (factored_var.sqrt() + eps2)
    
    # Step 4: Clip
    update_norm = update.norm()
    if update_norm > clip_threshold:
        update.mul_(clip_threshold / update_norm)
    
    # Step 5: Apply (absolute step size for simplicity)
    delta = -lr * update
    
    return delta, update, factored_var

# Run our canonical version
beta2 = 0.999
eps1 = 1e-30
eps2 = 1e-3
lr = 0.001
clip_threshold = 1.0

delta_ours, update_ours, var_ours = canonical_adafactor_step(
    weight_before, grad, beta2, eps1, eps2, lr, clip_threshold
)

print(f"Parameter delta norm: {delta_ours.norm().item():.6f}")
print(f"Delta sample: {delta_ours[0, :4].tolist()}")

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

diff = (delta_pytorch - delta_ours).norm().item()
relative_diff = diff / (delta_pytorch.norm().item() + 1e-10)

print(f"\nPyTorch delta norm: {delta_pytorch.norm().item():.6f}")
print(f"Our delta norm: {delta_ours.norm().item():.6f}")
print(f"Absolute difference: {diff:.6e}")
print(f"Relative difference: {relative_diff * 100:.2f}%")

if relative_diff < 0.01:
    print("\n✓ MATCHES PyTorch canonical Adafactor!")
    status = "PASS"
elif relative_diff < 0.05:
    print("\n⚠ CLOSE to PyTorch (within 5%)")
    status = "CLOSE"
else:
    print("\n✗ DOES NOT MATCH PyTorch")
    status = "FAIL"

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"""
CANONICAL ADAFACTOR (Paper + PyTorch):
  update = GRADIENT / sqrt(variance)
  NO momentum (β1 = 0)
  Clip the variance-scaled GRADIENT
  Apply update

Our original "buggy" implementation:
  update = GRADIENT / sqrt(variance)  ← CORRECT for canonical!
  clip update  ← CORRECT
  update = momentum / bias_correction  ← OVERWRITES! THIS IS THE BUG!
  apply update  ← Momentum only, variance lost

Our "corrected" implementation:
  m = momentum(gradient)
  update = m / sqrt(variance)  ← NOT canonical! This is Adam-style!

VERDICT:
  - Canonical Adafactor: Ut = Gt / √V̂t  (gradient / variance)
  - With momentum extension: Ut = mt / √V̂t  (momentum / variance)
  - Our "buggy" version overwrote the correct canonical formula
  - Our "corrected" version is Adam-style, not canonical
""")

print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
