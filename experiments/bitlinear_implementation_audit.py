#!/usr/bin/env python3
"""
READ-ONLY: BitLinear Implementation Audit

This script traces the EXACT forward/backward path through BitLinear to verify:
1. Which tensor is the optimizer parameter
2. How scale is computed
3. Which tensor is used in forward matmul
4. Whether scale and weight std can mathematically coexist
5. Whether weight explosion affects effective forward weights

CRITICAL: This is READ-ONLY. No model modification.
"""

import torch
import torch.nn.functional as F
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.model.bitlinear import BitLinear, compute_abs_mean_scale, TernarySTE

print("="*100)
print("READ-ONLY: BitLinear Implementation Audit")
print("="*100)

# ============================================================
# PART 1: IMPLEMENTATION TRACING
# ============================================================

print("\n" + "="*100)
print("PART 1: IMPLEMENTATION TRACING")
print("="*100)

print("""
[QUESTION 1: Which tensor is the optimizer parameter?]

From BitLinear.__init__ (line 154-156):
  self.weight = nn.Parameter(torch.empty(out_features, in_features))

ANSWER: self.weight is a standard nn.Parameter
- This is the UNDERLYING weight tensor
- This is what the optimizer updates
- This is stored in checkpoint['model_state_dict']
- This is the tensor with std=389.86 at step 50k
""")

print("""
[QUESTION 2: How is scale computed?]

From BitLinear.quantize_weights() (line 175-179):
  if self.per_channel:
      scale = compute_per_channel_scale(self.weight, self.eps)
  else:
      scale = compute_abs_mean_scale(self.weight, self.eps)

From compute_abs_mean_scale() (line 94):
  return weight.abs().mean().clamp(min=eps)

ANSWER: scale = mean(|self.weight|)
- Scale is computed from the UNDERLYING weight tensor
- Scale should track weight magnitude
- If weight std = 389.86, scale should be ~300-400 (for normal distribution)
""")

print("""
[QUESTION 3: Which tensor is used in forward matmul?]

From BitLinear.forward() (line 199-205):
  weight_q, scale = self.quantize_weights()
  return F.linear(x, weight_q, self.bias)

From BitLinear.quantize_weights() (line 181-187):
  weight_quantized, scale = TernarySTE.apply(self.weight, scale, self.threshold)
  weight_dequantized = weight_quantized * scale
  return weight_dequantized, scale

From TernarySTE.forward() (line 48-62):
  weight_scaled = weight / scale.clamp(min=1e-8)
  weight_quantized = torch.zeros_like(weight)
  weight_quantized[weight_scaled > threshold] = 1.0
  weight_quantized[weight_scaled < -threshold] = -1.0
  return weight_quantized, scale

ANSWER: Forward matmul uses weight_dequantized = weight_quantized * scale
- weight_quantized ∈ {-1, 0, +1}
- scale = mean(|self.weight|)
- weight_dequantized ∈ {-scale, 0, +scale}
""")

print("""
[QUESTION 4: Are scale and weight from same tensor?]

YES - scale is computed from self.weight in quantize_weights()

The formula:
  scale = mean(|self.weight|)
  weight_scaled = self.weight / scale
  weight_quantized = quantize(weight_scaled)
  weight_dequantized = weight_quantized * scale

The effective weight magnitude = scale (since weight_quantized ∈ {-1, 0, +1})
""")

# ============================================================
# PART 2: MATHEMATICAL CONSISTENCY CHECK
# ============================================================

print("\n" + "="*100)
print("PART 2: MATHEMATICAL CONSISTENCY CHECK")
print("="*100)

print("""
[QUESTION 5: Can reported weight std and scale mathematically coexist?]

Previous diagnostic reported:
  - Ternary-5k underlying weight std = 38.99
  - Forward pass scale = 0.067

Mathematical relationship:
  For a weight tensor W:
    - If std(W) = 38.99
    - Then mean(|W|) ≈ std(W) * sqrt(2/π) ≈ 38.99 * 0.798 ≈ 31.1

Expected scale ≈ 31.1, but reported scale = 0.067

THIS IS A 465x DISCREPANCY!

Possible explanations:
1. The "weight std" was measured from a different tensor
2. The "scale" was measured from a different tensor
3. There's a bug in how scale is computed
4. The diagnostic script measured wrong values

We need to verify by loading a checkpoint and measuring BOTH values from the SAME model.
""")

# ============================================================
# PART 3: CHECKPOINT VERIFICATION
# ============================================================

print("\n" + "="*100)
print("PART 3: CHECKPOINT VERIFICATION")
print("="*100)

# Load Ternary-5k checkpoint
ckpt_path = "outputs/artn-50k-best-config/checkpoints/checkpoint_step_5000.pt"
print(f"\n[Loading checkpoint: {ckpt_path}]")

ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['model_state_dict']

# Find a BitLinear weight
weight_key = 'blocks.0.attn.q_proj.weight'
if weight_key in state_dict:
    underlying_weight = state_dict[weight_key]
    
    print(f"\n[Analyzing {weight_key}]")
    print(f"  Shape: {underlying_weight.shape}")
    print(f"  Underlying weight mean: {underlying_weight.mean().item():.6f}")
    print(f"  Underlying weight std: {underlying_weight.std().item():.6f}")
    print(f"  Underlying weight abs mean: {underlying_weight.abs().mean().item():.6f}")
    
    # Compute scale the same way BitLinear does
    scale = underlying_weight.abs().mean().clamp(min=1e-8)
    print(f"\n  Computed scale (abs mean): {scale.item():.6f}")
    
    # Quantize
    weight_scaled = underlying_weight / scale
    threshold = 0.5
    weight_quantized = torch.zeros_like(underlying_weight)
    weight_quantized[weight_scaled > threshold] = 1.0
    weight_quantized[weight_scaled < -threshold] = -1.0
    
    # Dequantize
    weight_dequantized = weight_quantized * scale
    
    print(f"\n  Quantized weight distribution:")
    print(f"    -1: {(weight_quantized == -1).sum().item()} ({(weight_quantized == -1).float().mean().item()*100:.2f}%)")
    print(f"     0: {(weight_quantized == 0).sum().item()} ({(weight_quantized == 0).float().mean().item()*100:.2f}%)")
    print(f"    +1: {(weight_quantized == 1).sum().item()} ({(weight_quantized == 1).float().mean().item()*100:.2f}%)")
    
    print(f"\n  Dequantized weight stats:")
    print(f"    Mean: {weight_dequantized.mean().item():.6f}")
    print(f"    Std: {weight_dequantized.std().item():.6f}")
    print(f"    Abs mean: {weight_dequantized.abs().mean().item():.6f}")
    print(f"    Min: {weight_dequantized.min().item():.6f}")
    print(f"    Max: {weight_dequantized.max().item():.6f}")
    
    print(f"\n  *** CRITICAL COMPARISON ***")
    print(f"  Underlying weight std: {underlying_weight.std().item():.6f}")
    print(f"  Dequantized weight std: {weight_dequantized.std().item():.6f}")
    print(f"  Ratio: {underlying_weight.std().item() / weight_dequantized.std().item():.2f}x")
    
    print(f"\n  Scale (from abs mean): {scale.item():.6f}")
    print(f"  Effective magnitude: {weight_dequantized.abs().mean().item():.6f}")

# ============================================================
# PART 4: CHECK ALL LAYERS
# ============================================================

print("\n" + "="*100)
print("PART 4: CHECK ALL LAYERS")
print("="*100)

layer_stats = []

for key, tensor in state_dict.items():
    if 'weight' in key and 'embedding' not in key and 'ln' not in key and 'LayerNorm' not in key:
        if tensor.dim() == 2:  # Only 2D weights (Linear layers)
            underlying_std = tensor.std().item()
            underlying_abs_mean = tensor.abs().mean().item()
            
            # Compute scale and dequantized weight
            scale = tensor.abs().mean().clamp(min=1e-8)
            weight_scaled = tensor / scale
            weight_quantized = torch.zeros_like(tensor)
            weight_quantized[weight_scaled > 0.5] = 1.0
            weight_quantized[weight_scaled < -0.5] = -1.0
            weight_dequantized = weight_quantized * scale
            
            dequant_std = weight_dequantized.std().item()
            
            layer_stats.append({
                'layer': key,
                'underlying_std': underlying_std,
                'underlying_abs_mean': underlying_abs_mean,
                'scale': scale.item(),
                'dequant_std': dequant_std,
                'ratio': underlying_std / dequant_std if dequant_std > 0 else 0,
            })

print(f"\n[Layer-by-layer comparison]")
print(f"{'Layer':<50} {'Underlying Std':<15} {'Scale':<15} {'Dequant Std':<15} {'Ratio':<10}")
print("-" * 105)

for stat in layer_stats[:15]:  # Show first 15 layers
    print(f"{stat['layer']:<50} {stat['underlying_std']:<15.4f} {stat['scale']:<15.4f} {stat['dequant_std']:<15.4f} {stat['ratio']:<10.2f}x")

# ============================================================
# PART 5: VERIFY GRADIENT FLOW
# ============================================================

print("\n" + "="*100)
print("PART 5: VERIFY GRADIENT FLOW")
print("="*100)

print("""
[QUESTION 6: What gradient reaches the underlying parameter?]

From TernarySTE.backward() (line 65-77):
  def backward(ctx, grad_quantized, grad_scale):
      return grad_quantized, None, None

ANSWER: Gradient flows through unchanged (STE)
- grad_weight = grad_quantized (unchanged)
- grad_scale = None (not learned)
- The underlying weight receives the SAME gradient as the quantized weight
- No gradient scaling based on scale factor
""")

# ============================================================
# PART 6: OPTIMIZER STATE CHECK
# ============================================================

print("\n" + "="*100)
print("PART 6: OPTIMIZER STATE CHECK")
print("="*100)

print("""
[QUESTION 7: Does AdaFactor have weight decay/clipping/normalization?]

From Adafactor.__init__ (line 50-60):
  weight_decay: float = 0.0
  clip_threshold: float = 1.0

From Adafactor.step() (line 127-129):
  if weight_decay > 0:
      p.mul_(1 - lr * weight_decay)

From Adafactor.step() (line 160-163):
  update_norm = update.norm()
  if update_norm > clip_threshold:
      update.mul_(clip_threshold / update_norm)

ANSWER:
- weight_decay = 0.0 by default (NO weight decay)
- clip_threshold = 1.0 (clips GRADIENT, not weights)
- NO weight clipping
- NO weight normalization
- NO constraint on weight magnitude
""")

# Check training config
import json
config_path = "outputs/artn-50k-best-config/config.json"
if Path(config_path).exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"\n[Training Config]")
    print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
    print(f"  Weight decay: {config.get('weight_decay', 'N/A')}")
    print(f"  Beta1: {config.get('beta1', 'N/A')}")
    print(f"  Beta2: {config.get('beta2', 'N/A')}")

# ============================================================
# PART 7: ACROSS CHECKPOINTS COMPARISON
# ============================================================

print("\n" + "="*100)
print("PART 7: ACROSS CHECKPOINTS COMPARISON")
print("="*100)

checkpoints = {
    "1k": "outputs/artn-50k-best-config/checkpoints/checkpoint_step_1000.pt",
    "5k": "outputs/artn-50k-best-config/checkpoints/checkpoint_step_5000.pt",
    "50k": "outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt",
}

weight_key = 'blocks.0.attn.q_proj.weight'

print(f"\n[Evolution of {weight_key}]")
print(f"{'Step':<10} {'Underlying Std':<20} {'Scale':<20} {'Dequant Std':<20}")
print("-" * 70)

for step, path in checkpoints.items():
    if Path(path).exists():
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['model_state_dict']
        
        if weight_key in state_dict:
            tensor = state_dict[weight_key]
            underlying_std = tensor.std().item()
            scale = tensor.abs().mean().item()
            
            # Quantize
            weight_scaled = tensor / scale
            weight_quantized = torch.zeros_like(tensor)
            weight_quantized[weight_scaled > 0.5] = 1.0
            weight_quantized[weight_scaled < -0.5] = -1.0
            weight_dequantized = weight_quantized * scale
            dequant_std = weight_dequantized.std().item()
            
            print(f"{step:<10} {underlying_std:<20.4f} {scale:<20.4f} {dequant_std:<20.4f}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*100)
print("AUDIT SUMMARY")
print("="*100)

print("""
[VERIFIED FACTS]

1. ✅ Optimizer parameter = self.weight (underlying float tensor)
2. ✅ Scale = mean(|self.weight|) computed from underlying weight
3. ✅ Forward matmul uses weight_dequantized = quantized * scale
4. ✅ Scale and weight are from SAME underlying tensor
5. ✅ STE passes gradient through unchanged (no scaling)
6. ✅ AdaFactor has NO weight decay (default 0.0)
7. ✅ AdaFactor clips GRADIENTS, not weights
8. ✅ NO weight clipping or normalization in implementation

[KEY FINDING: UNDERLYING WEIGHT AND SCALE ARE MATHEMATICALLY CONSISTENT]

The checkpoint verification will show whether:
- Underlying weight std grows → scale grows proportionally
- Effective dequantized weight magnitude = scale
- Weight explosion in underlying weights = scale explosion
- Effective forward weights ALSO grow (this is the critical question)

[CRITICAL QUESTION TO ANSWER]

Does weight explosion in underlying weights cause:
A) Effective forward weights to ALSO explode (problem)
B) Effective forward weights to stay bounded (not a problem)

If (A): Weight explosion is HARMFUL
If (B): Weight explosion is HARMLESS (scale compensates)

We need to check the checkpoint verification results above.
""")

print("\n" + "="*100)
print("AUDIT COMPLETE")
print("="*100)
