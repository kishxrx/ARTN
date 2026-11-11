#!/usr/bin/env python3
"""
ADAFACTOR SINGLE-STEP VERIFICATION
===================================

Verifies the corrected Adafactor implementation against reference algorithm.

Tests:
1. Second-moment update
2. Factored row/column reconstruction
3. Variance normalization
4. Update clipping/scaling
5. Optional momentum handling
6. Exact tensor ultimately applied to parameters

Compares against:
- PyTorch reference implementation (if available)
- Direct mathematical calculation
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.training.adafactor import Adafactor

print("=" * 80)
print("ADAFACTOR SINGLE-STEP VERIFICATION")
print("=" * 80)

# Set seed for reproducibility
torch.manual_seed(42)

# Create a simple model with 2D and 1D parameters
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 32, bias=True)  # 2D weight + 1D bias
        self.linear2 = nn.Linear(32, 16, bias=True)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

model = SimpleModel()

print(f"\nModel parameters:")
for name, p in model.named_parameters():
    print(f"  {name}: {p.shape}")

# Create identical copies for comparison
model_ref = SimpleModel()
model_ref.load_state_dict(model.state_dict())

model_fixed = SimpleModel()
model_fixed.load_state_dict(model.state_dict())

# Create a batch of data
batch_size = 4
x = torch.randn(batch_size, 16)
y = torch.randn(batch_size, 16)

# Compute gradients on both models
def compute_loss(m, x, y):
    pred = m(x)
    loss = ((pred - y) ** 2).mean()
    return loss

# Original (buggy) Adafactor
optimizer_original = Adafactor(
    model.parameters(),
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps1=1e-30,
    eps2=1e-3,
    weight_decay=0.0,
    clip_threshold=1.0,
)

# PyTorch reference Adafactor (if available)
try:
    from torch.optim import Adafactor as PyTorchAdafactor
    optimizer_pytorch = PyTorchAdafactor(
        model_ref.parameters(),
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps1=1e-30,
        eps2=1e-3,
        weight_decay=0.0,
        clip_threshold=1.0,
    )
    has_pytorch = True
    print("\n✓ PyTorch Adafactor available for reference")
except (ImportError, TypeError) as e:
    has_pytorch = False
    print(f"\n✗ PyTorch Adafactor not available ({e}), using manual reference")

# Compute initial gradients
loss = compute_loss(model, x, y)
loss.backward()

# Save gradients
grads_original = {name: p.grad.clone() for name, p in model.named_parameters()}

# Also compute for reference model
loss_ref = compute_loss(model_ref, x.clone(), y.clone())
loss_ref.backward()

grads_ref = {name: p.grad.clone() for name, p in model_ref.named_parameters()}

print(f"\nInitial loss: {loss.item():.6f}")
print(f"\nGradients computed")

# Manual reference calculation for a 2D weight
def manual_adafactor_step(weight, grad, beta1, beta2, eps1, eps2, lr, step, clip_threshold):
    """
    Manual calculation of correct Adafactor step for 2D parameter.
    
    CORRECT algorithm:
    1. Compute momentum (first moment)
    2. Compute variance (second moment)
    3. Apply variance normalization TO momentum
    4. Clip the variance-normalized momentum
    5. Apply update
    """
    # Initialize state
    exp_avg = torch.zeros_like(weight)
    exp_avg_sq_row = torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)
    exp_avg_sq_col = torch.zeros(weight.shape[1], dtype=weight.dtype, device=weight.device)
    
    # Step 1: Compute momentum (first moment)
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    bias_correction = 1 - beta1 ** step
    m_hat = exp_avg / bias_correction
    
    # Step 2: Compute variance (second moment)
    grad_sq = grad * grad
    exp_avg_sq_row.mul_(beta2).add_(grad_sq.mean(dim=1), alpha=1 - beta2)
    exp_avg_sq_col.mul_(beta2).add_(grad_sq.mean(dim=0), alpha=1 - beta2)
    
    # Factorized variance
    global_mean = grad_sq.mean()
    numerator = exp_avg_sq_row.unsqueeze(1) * exp_avg_sq_col.unsqueeze(0)
    factored_var = numerator / global_mean.add(eps1)
    
    # Step 3: Apply variance normalization TO momentum (CRITICAL!)
    update = m_hat / (factored_var.sqrt() + eps2)
    
    # Step 4: Clip update
    update_norm = update.norm()
    if update_norm > clip_threshold:
        update.mul_(clip_threshold / update_norm)
    
    # Step 5: Compute final parameter delta
    delta = -lr * update
    
    return delta, m_hat, factored_var, update

print("\n" + "=" * 80)
print("SINGLE-STEP VERIFICATION")
print("=" * 80)

# Get the first 2D weight for detailed analysis
weight_name = 'linear1.weight'
weight = dict(model.named_parameters())[weight_name]
grad = grads_original[weight_name]

print(f"\nAnalyzing parameter: {weight_name} (shape: {weight.shape})")
print(f"  Gradient norm: {grad.norm().item():.6f}")
print(f"  Gradient mean: {grad.mean().item():.6f}")
print(f"  Gradient std: {grad.std().item():.6f}")

# Manual reference calculation
print("\n" + "-" * 80)
print("MANUAL REFERENCE CALCULATION")
print("-" * 80)

beta1, beta2, eps1, eps2 = 0.9, 0.999, 1e-30, 1e-3
lr = 1e-3
step = 1
clip_threshold = 1.0

delta_ref, m_hat_ref, var_ref, update_ref = manual_adafactor_step(
    weight.data.clone(), grad.clone(), beta1, beta2, eps1, eps2, lr, step, clip_threshold
)

print(f"\nMomentum (m_hat):")
print(f"  Norm: {m_hat_ref.norm().item():.6f}")
print(f"  Mean: {m_hat_ref.mean().item():.6f}")
print(f"  Std: {m_hat_ref.std().item():.6f}")

print(f"\nFactored variance:")
print(f"  Norm: {var_ref.norm().item():.6f}")
print(f"  Mean: {var_ref.mean().item():.6f}")
print(f"  Min: {var_ref.min().item():.6f}")
print(f"  Max: {var_ref.max().item():.6f}")

print(f"\nVariance-normalized momentum update:")
print(f"  Norm: {update_ref.norm().item():.6f}")
print(f"  Mean: {update_ref.mean().item():.6f}")
print(f"  Std: {update_ref.std().item():.6f}")

print(f"\nFinal parameter delta (should be applied):")
print(f"  Norm: {delta_ref.norm().item():.6f}")
print(f"  Mean: {delta_ref.mean().item():.6f}")
print(f"  Std: {delta_ref.std().item():.6f}")
print(f"  Sample values: {delta_ref[0, :5].tolist()}")

# Now test the original (buggy) implementation
print("\n" + "-" * 80)
print("ORIGINAL (BUGGY) ADAFACTOR IMPLEMENTATION")
print("-" * 80)

# Save original parameters
params_before = {name: p.data.clone() for name, p in model.named_parameters()}

# Step the original optimizer
optimizer_original.step()

# Check what was actually applied
params_after = {name: p.data.clone() for name, p in model.named_parameters()}
delta_actual_original = params_after[weight_name] - params_before[weight_name]

print(f"\nActual parameter delta from buggy Adafactor:")
print(f"  Norm: {delta_actual_original.norm().item():.6f}")
print(f"  Mean: {delta_actual_original.mean().item():.6f}")
print(f"  Std: {delta_actual_original.std().item():.6f}")
print(f"  Sample values: {delta_actual_original[0, :5].tolist()}")

# Compare against reference
diff_original = (delta_actual_original - delta_ref).norm().item()
relative_diff_original = diff_original / (delta_ref.norm().item() + 1e-10)

print(f"\nComparison with CORRECT reference:")
print(f"  Absolute difference norm: {diff_original:.6e}")
print(f"  Relative difference: {relative_diff_original * 100:.2f}%")

if relative_diff_original < 0.01:
    print(f"  Status: ✓ MATCHES reference (within 1%)")
else:
    print(f"  Status: ✗ DOES NOT MATCH reference")

# Test PyTorch reference if available
if has_pytorch:
    print("\n" + "-" * 80)
    print("PYTORCH REFERENCE ADAFACTOR")
    print("-" * 80)
    
    params_before_ref = {name: p.data.clone() for name, p in model_ref.named_parameters()}
    optimizer_pytorch.step()
    params_after_ref = {name: p.data.clone() for name, p in model_ref.named_parameters()}
    
    delta_pytorch = params_after_ref[weight_name] - params_before_ref[weight_name]
    
    print(f"\nParameter delta from PyTorch Adafactor:")
    print(f"  Norm: {delta_pytorch.norm().item():.6f}")
    print(f"  Mean: {delta_pytorch.mean().item():.6f}")
    print(f"  Sample values: {delta_pytorch[0, :5].tolist()}")
    
    # Compare manual reference with PyTorch
    diff_pytorch = (delta_pytorch - delta_ref).norm().item()
    relative_diff_pytorch = diff_pytorch / (delta_ref.norm().item() + 1e-10)
    
    print(f"\nComparison with manual reference:")
    print(f"  Absolute difference norm: {diff_pytorch:.6e}")
    print(f"  Relative difference: {relative_diff_pytorch * 100:.2f}%")

print("\n" + "=" * 80)
print("VERIFICATION RESULTS")
print("=" * 80)

print("\n" + "-" * 80)
print("BUG ANALYSIS")
print("-" * 80)

print("""
The original implementation has this bug:

  LINE 152: update = grad / (factored_var.sqrt() + eps2)  # Variance-scaled
  LINE 161-163: Clip this variance-scaled update
  LINE 170: update = exp_avg / bias_correction  # OVERWRITES with momentum!
  LINE 173: p.add_(update, alpha=-lr)  # Applies momentum, NOT variance

The CORRECT algorithm should be:

  STEP 1: Compute momentum
    m_hat = exp_avg / bias_correction

  STEP 2: Compute variance
    factored_var = row × col / global_mean

  STEP 3: Apply variance TO momentum (CRITICAL!)
    update = m_hat / (sqrt(factored_var) + eps2)

  STEP 4: Clip and apply
    clip update
    p.add_(update, alpha=-lr)
""")

print("-" * 80)
print("EXPECTED BEHAVIOR")
print("-" * 80)

print(f"""
If the original implementation is buggy:
  - It should apply ONLY momentum (NO variance normalization)
  - The delta should be: -lr × (exp_avg / bias_correction)
  - This will NOT match the correct reference delta

If the implementation is correct:
  - It should apply variance-normalized momentum
  - The delta should match the manual reference delta

Current status: {"BUGGY - does not match reference" if relative_diff_original > 0.01 else "CORRECT - matches reference"}
""")

print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
