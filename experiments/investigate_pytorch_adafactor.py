#!/usr/bin/env python3
"""
PYTORCH ADAFACTOR INVESTIGATION
================================

Investigate what PyTorch's Adafactor actually does by tracing through the steps.

We need to understand:
1. Relative step size (αt = max(ε2, RMS(X)) × ρt)
2. Exact variance formula
3. Clipping mechanism
4. Any other differences
"""

import torch
import torch.nn as nn

print("=" * 80)
print("PYTORCH ADAFACTOR STEP-BY-STEP INVESTIGATION")
print("=" * 80)

# Create simple test case
torch.manual_seed(42)
weight = torch.randn(4, 4)
grad = torch.randn(4, 4)

print(f"\nWeight norm: {weight.norm().item():.6f}")
print(f"Weight RMS: {(weight ** 2).mean().sqrt().item():.6f}")
print(f"Grad norm: {grad.norm().item():.6f}")

# Manual step with PyTorch
from torch.optim import Adafactor

# Create optimizer
param = torch.nn.Parameter(weight.clone())
optimizer = Adafactor(
    [param],
    lr=0.001,  # This is ρt (relative step size)
    beta2_decay=0.0,  # No decay
    eps=(1e-30, 1e-3),
    d=1.0,
    weight_decay=0.0,
)

# Set gradient
param.grad = grad.clone()

# Step through manually
print("\n" + "=" * 80)
print("MANUAL STEP-THROUGH")
print("=" * 80)

# Get state
state = optimizer.state[param]
print(f"\nInitial state keys: {list(state.keys())}")

# Do one step
weight_before = param.data.clone()
optimizer.step()
weight_after = param.data.clone()

delta_pytorch = weight_after - weight_before
print(f"\nPyTorch delta norm: {delta_pytorch.norm().item():.6f}")
print(f"PyTorch delta sample: {delta_pytorch[0, :4].tolist()}")

# Try to understand what happened
print("\n" + "=" * 80)
print("REVERSE ENGINEERING")
print("=" * 80)

# From the paper, the relative step size is:
# αt = max(ε2, RMS(Xt-1)) × ρt
rms_weight = (weight_before ** 2).mean().sqrt()
eps2 = 1e-3
rho_t = 0.001  # lr parameter

alpha_t = max(eps2, rms_weight.item()) * rho_t
print(f"\nRMS(weight): {rms_weight.item():.6f}")
print(f"Relative step size αt = max({eps2}, {rms_weight.item():.6f}) × {rho_t} = {alpha_t:.6f}")

# The update should be:
# delta = -αt × clipped_update
# So: clipped_update = -delta / αt
clipped_update = -delta_pytorch / alpha_t
print(f"\nClipped update norm: {clipped_update.norm().item():.6f}")
print(f"Clipped update sample: {clipped_update[0, :4].tolist()}")

# Now let's compute variance ourselves
grad_sq = grad ** 2

# Row and column sums
row_mean = grad_sq.mean(dim=1)
col_mean = grad_sq.mean(dim=0)

print(f"\nRow mean: {row_mean.tolist()}")
print(f"Col mean: {col_mean.tolist()}")

# Factored variance
# From paper: V̂t = Rt × Ct / (1n × Rt)
# But first step, Rt and Ct are just the current values
exp_avg_sq_row = row_mean  # First step
exp_avg_sq_col = col_mean  # First step

# Global mean
global_mean = grad_sq.mean()
print(f"\nGlobal mean: {global_mean.item():.6f}")

# Factored variance
# Paper says: V̂t = Rt × Ct / (1n × Rt)
# 1n × Rt is the mean of Rt
# So: V̂t = outer(Rt, Ct) / mean(Rt)
factored_var = exp_avg_sq_row.unsqueeze(1) * exp_avg_sq_col.unsqueeze(0) / exp_avg_sq_row.mean()
print(f"\nFactored variance shape: {factored_var.shape}")
print(f"Factored variance sample: {factored_var[0, :4].tolist()}")

# Update = grad / sqrt(variance + eps2)
eps1 = 1e-30
eps2 = 1e-3
update = grad / (factored_var.sqrt() + eps2)
print(f"\nUpdate norm: {update.norm().item():.6f}")
print(f"Update sample: {update[0, :4].tolist()}")

# Clip
d = 1.0
update_rms = (update ** 2).mean().sqrt()
print(f"\nUpdate RMS: {update_rms.item():.6f}")

if update_rms > d:
    clipped = update * d / update_rms
    print(f"Clipped update norm: {clipped.norm().item():.6f}")
else:
    clipped = update
    print(f"No clipping needed")

# Final delta
delta_manual = -alpha_t * clipped
print(f"\nManual delta norm: {delta_manual.norm().item():.6f}")
print(f"Manual delta sample: {delta_manual[0, :4].tolist()}")

# Compare
diff = (delta_pytorch - delta_manual).norm().item()
relative_diff = diff / (delta_pytorch.norm().item() + 1e-10)
print(f"\nDifference: {diff:.6e}")
print(f"Relative difference: {relative_diff * 100:.2f}%")

if relative_diff < 0.01:
    print("✓ MATCHES PyTorch!")
else:
    print("✗ Still does not match")

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
