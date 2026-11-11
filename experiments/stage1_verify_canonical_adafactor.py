#!/usr/bin/env python3
"""
STAGE 1: Canonical Adafactor Verification
==========================================

Verify that our CanonicalAdafactor matches PyTorch's torch.optim.Adafactor.

Tests:
- Multiple steps (not just one)
- Both 2D (matrix) and 1D (vector) parameters
- Identical initial parameters and gradients
- Report max and relative numerical differences

Expected: < 0.1% relative difference across all tests.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.training.adafactor_canonical import CanonicalAdafactor
from torch.optim import Adafactor as PyTorchAdafactor

print("=" * 80)
print("STAGE 1: CANONICAL ADAFACTOR VERIFICATION")
print("=" * 80)
print(f"\nExpected runtime: ~30-60 seconds")
print()

# Test configuration
NUM_STEPS = 100
TOLERANCE = 0.001  # 0.1% relative difference

# Set seed for reproducibility
torch.manual_seed(42)

# Create a model with both 2D and 1D parameters
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 32)  # 2D weight + 1D bias
        self.linear2 = nn.Linear(32, 16)
        self.bn = nn.BatchNorm1d(16)  # More 1D params
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.bn(x)
        return x

# Create two identical models
model_canonical = TestModel()
model_pytorch = TestModel()
model_pytorch.load_state_dict(model_canonical.state_dict())

# Create optimizers
optimizer_canonical = CanonicalAdafactor(
    model_canonical.parameters(),
    lr=1e-3,
    beta2_decay=-0.8,
    eps=(1e-30, 1e-3),
    d=1.0,
    weight_decay=0.0,
)

optimizer_pytorch = PyTorchAdafactor(
    model_pytorch.parameters(),
    lr=1e-3,
    beta2_decay=-0.8,
    eps=(1e-30, 1e-3),
    d=1.0,
    weight_decay=0.0,
)

print(f"Model parameters:")
total_params = sum(p.numel() for p in model_canonical.parameters())
print(f"  Total: {total_params:,} parameters")
print(f"  2D params: {sum(p.numel() for p in model_canonical.parameters() if p.dim() == 2):,}")
print(f"  1D params: {sum(p.numel() for p in model_canonical.parameters() if p.dim() == 1):,}")

# Store initial parameters
params_initial = {
    name: p.data.clone() 
    for name, p in model_canonical.named_parameters()
}

# Track differences
all_diffs = []
max_diff = 0.0
max_diff_step = 0
max_diff_param = ""

print(f"\n{'=' * 80}")
print(f"RUNNING {NUM_STEPS} OPTIMIZATION STEPS")
print(f"{'=' * 80}")

# Run multiple steps
for step in range(NUM_STEPS):
    # Create random input
    x = torch.randn(4, 16)
    
    # Forward pass
    y_canonical = model_canonical(x)
    y_pytorch = model_pytorch(x.clone())
    
    # Compute loss
    loss_canonical = y_canonical.sum()
    loss_pytorch = y_pytorch.sum()
    
    # Backward pass
    loss_canonical.backward()
    loss_pytorch.backward()
    
    # Optimizer step
    optimizer_canonical.step()
    optimizer_pytorch.step()
    
    # Zero gradients
    optimizer_canonical.zero_grad()
    optimizer_pytorch.zero_grad()
    
    # Compare parameters after this step
    for (name_c, p_c), (name_p, p_p) in zip(
        model_canonical.named_parameters(),
        model_pytorch.named_parameters()
    ):
        diff = (p_c.data - p_p.data).abs().max().item()
        rel_diff = diff / (p_p.data.abs().max().item() + 1e-10)
        
        all_diffs.append(rel_diff)
        
        if rel_diff > max_diff:
            max_diff = rel_diff
            max_diff_step = step
            max_diff_param = name_c
    
    # Progress
    if step % 10 == 0 or step == NUM_STEPS - 1:
        print(f"Step {step:4d}: max_rel_diff = {max_diff:.6e}")

print(f"\n{'=' * 80}")
print("VERIFICATION RESULTS")
print(f"{'=' * 80}")

# Summary statistics
import numpy as np
all_diffs = np.array(all_diffs)

print(f"\nNumerical Differences:")
print(f"  Mean relative difference: {all_diffs.mean():.6e}")
print(f"  Max relative difference:  {all_diffs.max():.6e}")
print(f"  Min relative difference:  {all_diffs.min():.6e}")
print(f"  Std relative difference:  {all_diffs.std():.6e}")

print(f"\nWorst Case:")
print(f"  Step: {max_diff_step}")
print(f"  Parameter: {max_diff_param}")
print(f"  Relative difference: {max_diff:.6e}")

# Compare optimizer states
print(f"\n{'=' * 80}")
print("OPTIMIZER STATE COMPARISON")
print(f"{'=' * 80}")

state_mismatches = 0

for (name_c, p_c), (name_p, p_p) in zip(
    model_canonical.named_parameters(),
    model_pytorch.named_parameters()
):
    state_c = optimizer_canonical.state[p_c]
    state_p = optimizer_pytorch.state[p_p]
    
    # Compare state keys
    if set(state_c.keys()) != set(state_p.keys()):
        print(f"\n✗ {name_c}: State key mismatch")
        print(f"  Canonical: {set(state_c.keys())}")
        print(f"  PyTorch:   {set(state_p.keys())}")
        state_mismatches += 1
        continue
    
    # Compare state values
    for key in state_c.keys():
        if isinstance(state_c[key], torch.Tensor):
            diff = (state_c[key] - state_p[key]).abs().max().item()
            rel_diff = diff / (state_p[key].abs().max().item() + 1e-10)
            
            if rel_diff > TOLERANCE:
                print(f"\n✗ {name_c}.{key}: rel_diff = {rel_diff:.6e}")
                state_mismatches += 1

if state_mismatches == 0:
    print("\n✓ All optimizer states match")

# Final verdict
print(f"\n{'=' * 80}")
print("VERDICT")
print(f"{'=' * 80}")

if all_diffs.max() < TOLERANCE:
    print(f"\n✓ PASS: Canonical Adafactor matches PyTorch")
    print(f"  Max relative difference: {all_diffs.max():.6e} < {TOLERANCE}")
    print(f"  Ready for Stage 2 convergence test")
    exit_code = 0
else:
    print(f"\n✗ FAIL: Canonical Adafactor does NOT match PyTorch")
    print(f"  Max relative difference: {all_diffs.max():.6e} > {TOLERANCE}")
    print(f"  Fix implementation before Stage 2")
    exit_code = 1

print(f"\n{'=' * 80}")
print("STAGE 1 COMPLETE")
print(f"{'=' * 80}")

exit(exit_code)
