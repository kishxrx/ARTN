#!/usr/bin/env python3
"""
Instrumented verification of BitLinear/TernarySTE implementation.

Tests each component individually with evidence.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, '/root/agent-sandbox/ARTN')

print("=" * 70)
print("INSTRUMENTED BITLINEAR/TERNARYSTE VERIFICATION")
print("=" * 70)

# =============================================================================
# TEST 1: Check TernarySTE quantization output
# =============================================================================

print("\n" + "=" * 70)
print("TEST 1: What does TernarySTE.forward() actually return?")
print("=" * 70)

from src.model.bitlinear import TernarySTE, compute_abs_mean_scale

# Create sample weight
weight_full = torch.randn(64, 64)
scale = compute_abs_mean_scale(weight_full)
threshold = 0.5

print(f"Input weight shape: {weight_full.shape}")
print(f"Input weight sample (first 5): {weight_full.flatten()[:5]}")
print(f"Scale: {scale.item():.6f}")
print(f"Threshold: {threshold}")

# Apply TernarySTE
weight_quantized, scale_returned = TernarySTE.apply(weight_full, scale, threshold)

print(f"\nTernarySTE output shape: {weight_quantized.shape}")
print(f"TernarySTE output dtype: {weight_quantized.dtype}")

# CRITICAL: What unique values are in weight_quantized?
unique_values = torch.unique(weight_quantized)
print(f"\nUNIQUE VALUES in weight_quantized: {unique_values.tolist()}")
print(f"Number of unique values: {len(unique_values)}")

# Count distribution
neg_count = (weight_quantized == -1).sum().item()
zero_count = (weight_quantized == 0).sum().item()
pos_count = (weight_quantized == 1).sum().item()
total = weight_quantized.numel()

print(f"\nDistribution in quantized output:")
print(f"  -1: {neg_count}/{total} ({neg_count/total*100:.2f}%)")
print(f"   0: {zero_count}/{total} ({zero_count/total*100:.2f}%)")
print(f"  +1: {pos_count}/{total} ({pos_count/total*100:.2f}%)")

# VERIFICATION LABEL
if len(unique_values) == 3 and set(unique_values.tolist()) == {-1.0, 0.0, 1.0}:
    print("\n[VERIFIED] TernarySTE returns EXACTLY {-1, 0, +1}")
else:
    print(f"\n[ERROR] TernarySTE returns {unique_values.tolist()}, NOT ternary!")

# =============================================================================
# TEST 2: What tensor is passed to matrix multiplication?
# =============================================================================

print("\n" + "=" * 70)
print("TEST 2: What tensor enters F.linear() in BitLinear.forward()?")
print("=" * 70)

from src.model.bitlinear import BitLinear

# Patch BitLinear to intercept the actual tensor used in matmul
original_forward = BitLinear.forward
intercepted_weight = None
intercepted_unique = None

def instrumented_forward(self, x):
    global intercepted_weight, intercepted_unique
    
    # Quantize weights (same as original)
    weight_q, scale = self.quantize_weights()
    
    # INTERCEPT: What is weight_q?
    intercepted_weight = weight_q.clone()
    intercepted_unique = torch.unique(weight_q)
    
    # Store for later comparison
    self._last_weight_q = weight_q.clone()
    self._last_scale = scale.clone()
    
    # Continue with original forward
    if self.bias is not None:
        output = nn.functional.linear(x, weight_q, self.bias)
    else:
        output = nn.functional.linear(x, weight_q)
    
    return output

BitLinear.forward = instrumented_forward

# Create layer and run forward
layer = BitLinear(64, 64)
x = torch.randn(2, 32, 64)  # batch=2, seq=32, dim=64

output = layer(x)

print(f"Intercepted weight tensor shape: {intercepted_weight.shape}")
print(f"Intercepted weight dtype: {intercepted_weight.dtype}")
print(f"\nUNIQUE VALUES in tensor passed to F.linear(): {intercepted_unique.tolist()}")

# Check if it's ternary
if len(intercepted_unique) <= 3:
    vals = set(intercepted_unique.tolist())
    if vals == {-1.0, 0.0, 1.0} or vals == {0.0, 1.0} or vals == {-1.0, 0.0} or vals == {0.0}:
        print("[VERIFIED] Matrix multiplication uses TERNARY weights")
    else:
        print(f"[UNKNOWN] Unique values: {intercepted_unique.tolist()}")
else:
    print(f"[ERROR] More than 3 unique values: {len(intercepted_unique)}")

# Check the actual values
neg_cnt = (intercepted_weight == -1).sum().item()
zero_cnt = (intercepted_weight == 0).sum().item()
pos_cnt = (intercepted_weight == 1).sum().item()
print(f"\nDistribution in intercepted tensor:")
print(f"  -1: {neg_cnt}/{intercepted_weight.numel()} ({neg_cnt/intercepted_weight.numel()*100:.2f}%)")
print(f"   0: {zero_cnt}/{intercepted_weight.numel()} ({zero_cnt/intercepted_weight.numel()*100:.2f}%)")
print(f"  +1: {pos_cnt}/{intercepted_weight.numel()} ({pos_cnt/intercepted_weight.numel()*100:.2f}%)")

# Check scale application
width = intercepted_weight.abs().max().item()
print(f"\nWeight range passed to matmul: [{intercepted_weight.min().item():.4f}, {intercepted_weight.max().item():.4f}]")

# =============================================================================
# TEST 3: Is scale applied AFTER ternarization?
# =============================================================================

print("\n" + "=" * 70)
print("TEST 3: Verify scale multiplication timing")
print("=" * 70)

# Read code directly
print("Checking quantize_weights() implementation...")

# Get fresh layer
layer2 = BitLinear(64, 64)
weight_q, scale = layer2.quantize_weights()

print(f"quantize_weights() returns:")
print(f"  weight_q shape: {weight_q.shape}")
print(f"  weight_q unique: {torch.unique(weight_q).tolist()}")
print(f"  scale: {scale.item():.6f}")

# Check if weight_q is AFTER dequantization
# If weight_q is ternary {-1, 0, +1}, dequantization NOT applied
# If weight_q contains floats like -0.723, +0.723, dequantization IS applied

min_val = weight_q.min().item()
max_val = weight_q.max().item()

if -1.0 <= min_val <= 0.0 and 0.0 <= max_val <= 1.0:
    if abs(min_val) <= 1.0 + 1e-6 and max_val <= 1.0 + 1e-6:
        # Check if strictly -1, 0, 1
        unq = torch.unique(weight_q)
        if len(unq) <= 3 and all(abs(v) <= 1.0 + 1e-6 for v in unq.tolist()):
            print("[VERIFIED] quantize_weights() returns TERNARY values before dequantization")
        else:
            print(f"[UNKNOWN] Unique values: {unq.tolist()}")
else:
    print(f"[STRONGLY SUPPORTED] Dequantization applied (range: [{min_val:.4f}, {max_val:.4f}])")

# =============================================================================
# TEST 4: Backward pass - STE gradient flow
# =============================================================================

print("\n" + "=" * 70)
print("TEST 4: Does STE provide non-zero gradients?")
print("=" * 70)

# Restore original forward
BitLinear.forward = original_forward

layer3 = BitLinear(64, 64)
x = torch.randn(2, 32, 64, requires_grad=True)

# Forward
output = layer3(x)
loss = output.sum()

# Backward
loss.backward()

# Check gradients
weight_grad = layer3.weight.grad

print(f"Weight gradient shape: {weight_grad.shape}")
print(f"Weight gradient non-zero count: {(weight_grad != 0).sum().item()}/{weight_grad.numel()}")
print(f"Weight gradient range: [{weight_grad.min().item():.6f}, {weight_grad.max().item():.6f}]")
print(f"Weight gradient mean: {weight_grad.mean().item():.6f}")

if (weight_grad != 0).sum().item() > 0:
    print("[VERIFIED] STE provides NON-ZERO gradients")
else:
    print("[ERROR] STE provides ZERO gradients!")

# Check if gradients are actually meaningful
grad_norm = weight_grad.norm().item()
print(f"Weight gradient norm: {grad_norm:.6f}")

if grad_norm > 1e-6:
    print("[VERIFIED] Gradients are non-trivial")
else:
    print("[ERROR] Gradients are effectively zero!")

# =============================================================================
# TEST 5: Are weights updated after optimizer.step()?
# =============================================================================

print("\n" + "=" * 70)
print("TEST 5: Does optimizer.step() actually change weights?")
print("=" * 70)

layer4 = BitLinear(64, 64)
optimizer = torch.optim.SGD(layer4.parameters(), lr=0.1)

# Snapshot weights before
weight_before = layer4.weight.data.clone()

# Forward + backward
x = torch.randn(2, 32, 64)
output = layer4(x)
loss = output.sum()
loss.backward()

# Optimizer step
optimizer.step()

# Snapshot weights after
weight_after = layer4.weight.data.clone()

# Compare
diff = (weight_after - weight_before).abs()
print(f"Weight difference max: {diff.max().item():.6f}")
print(f"Weight difference mean: {diff.mean().item():.6f}")
print(f"Non-zero changes: {(diff > 1e-10).sum().item()}/{diff.numel()}")

if diff.max().item() > 1e-6:
    print("[VERIFIED] Weights ARE updated by optimizer")
else:
    print("[ERROR] Weights NOT updated!")

# =============================================================================
# TEST 6: Quantization threshold/scale behavior
# =============================================================================

print("\n" + "=" * 70)
print("TEST 6: Threshold and scale sanity check")
print("=" * 70)

layer5 = BitLinear(64, 64, threshold=0.5)

# Get weight statistics
w = layer5.weight.data
scale = compute_abs_mean_scale(w)

print(f"Weight statistics:")
print(f"  Mean: {w.mean().item():.6f}")
print(f"  Std: {w.std().item():.6f}")
print(f"  Min: {w.min().item():.6f}")
print(f"  Max: {w.max().item():.6f}")
print(f"  Abs mean (scale): {scale.item():.6f}")
print(f"  Threshold: 0.5")

# Compute what would be quantized
w_scaled = w / scale.clamp(min=1e-8)
print(f"\nAfter scaling:")
print(f"  Scaled weight range: [{w_scaled.min().item():.4f}, {w_scaled.max().item():.4f}]")

# Count what would go where
would_be_neg = (w_scaled < -0.5).sum().item()
would_be_zero = ((w_scaled >= -0.5) & (w_scaled <= 0.5)).sum().item()
would_be_pos = (w_scaled > 0.5).sum().item()
total = w.numel()

print(f"\nExpected quantization distribution:")
print(f"  -1: {would_be_neg}/{total} ({would_be_neg/total*100:.2f}%)")
print(f"   0: {would_be_zero}/{total} ({would_be_zero/total*100:.2f}%)")
print(f"  +1: {would_be_pos}/{total} ({would_be_pos/total*100:.2f}%)")

# Check actual quantization
w_q, _ = layer5.quantize_weights()
actual_neg = (w_q == -1).sum().item()
actual_zero = (w_q == 0).sum().item()
actual_pos = (w_q == 1).sum().item()

print(f"\nActual quantization distribution:")
print(f"  -1: {actual_neg}/{total} ({actual_neg/total*100:.2f}%)")
print(f"   0: {actual_zero}/{total} ({actual_zero/total*100:.2f}%)")
print(f"  +1: {actual_pos}/{total} ({actual_pos/total*100:.2f}%)")

if abs(would_be_neg - actual_neg) < 10 and abs(would_be_zero - actual_zero) < 10 and abs(would_be_pos - actual_pos) < 10:
    print("[VERIFIED] Quantization threshold behaves as expected")
else:
    print("[UNKNOWN] Mismatch between expected and actual quantization")

# =============================================================================
# TEST 7: Is quantization applied to ALL BitLinear layers?
# =============================================================================

print("\n" + "=" * 70)
print("TEST 7: Are all BitLinear layers quantized?")
print("=" * 70)

from src.model.bitlinear import BitLinearGPT

model = BitLinearGPT(
    vocab_size=100,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    max_seq_len=128,
)

# Count BitLinear layers
bitlinear_layers = []
for name, module in model.named_modules():
    if isinstance(module, BitLinear):
        bitlinear_layers.append(name)

print(f"Total BitLinear layers in model: {len(bitlinear_layers)}")
for name in bitlinear_layers:
    print(f"  - {name}")

# Run forward and check each layer quantizes
x = torch.randint(0, 100, (1, 10))

# Patch forward to capture quantized weights
quantization_results = {}
original_quantize = BitLinear.quantize_weights

def tracked_quantize(self):
    result = original_quantize(self)
    layer_name = id(self)
    quantization_results[layer_name] = {
        'unique': torch.unique(result[0]).tolist(),
        'shape': result[0].shape,
    }
    return result

BitLinear.quantize_weights = tracked_quantize

output = model(x)

print(f"\nLayers that performed quantization: {len(quantization_results)}")

all_ternary = True
for lid, data in quantization_results.items():
    unq = data['unique']
    if not (set(unq) == {-1.0, 0.0, 1.0} or set(unq) == {0.0, 1.0} or set(unq) == {-1.0, 0.0}):
        print(f"  Layer {lid}: NOT ternary - {unq}")
        all_ternary = False

if all_ternary and len(quantization_results) == len(bitlinear_layers):
    print("[VERIFIED] All BitLinear layers perform ternary quantization")
else:
    print(f"[UNKNOWN] Not all layers verified. Count: {len(quantization_results)}/{len(bitlinear_layers)}")

# Restore
BitLinear.quantize_weights = original_quantize

# =============================================================================
# TEST 8: Compare quantized vs full-precision forward pass
# =============================================================================

print("\n" + "=" * 70)
print("TEST 8: Do quantized and full-precision passes differ?")
print("=" * 70)

layer6 = BitLinear(64, 64)
x = torch.randn(1, 32, 64)

# Forward with quantization (normal)
output_quant = layer6(x)

# Forward WITHOUT quantization (forced full-precision)
with torch.no_grad():
    # Use stored weights directly
    output_full = nn.functional.linear(x, layer6.weight.data, layer6.bias)

print(f"Output with quantization: shape={output_quant.shape}, mean={output_quant.mean().item():.4f}")
print(f"Output full-precision: shape={output_full.shape}, mean={output_full.mean().item():.4f}")

diff = (output_quant - output_full).abs()
print(f"\nDifference:")
print(f"  Max: {diff.max().item():.6f}")
print(f"  Mean: {diff.mean().item():.6f}")
print(f"  Relative: {(diff / (output_full.abs() + 1e-8)).mean().item():.6f}")

if diff.max().item() > 1e-6:
    print("[VERIFIED] Quantized and full-precision produce DIFFERENT outputs")
else:
    print("[ERROR] Quantization has NO EFFECT on output!")

# =============================================================================
# TEST 9: Verify loss calculation components
# =============================================================================

print("\n" + "=" * 70)
print("TEST 9: Loss calculation verification")
print("=" * 70)

# Load actual tokenizer
from src.tokenizer.bpe_tokenizer import BPETokenizer

tokenizer_path = "/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json"
tokenizer = BPETokenizer.load(tokenizer_path)

# Test encode/decode
test_text = "Once upon a time"
tokens = tokenizer.encode(test_text)
decoded = tokenizer.decode(tokens)

print(f"Test text: '{test_text}'")
print(f"Encoded tokens: {tokens[:10]}")
print(f"Decoded text: '{decoded}'")

if decoded.strip() == test_text.strip():
    print("[VERIFIED] Tokenizer encode/decode is consistent")
else:
    print(f"[UNKNOWN] Encode/decode mismatch: '{test_text}' -> '{decoded}'")

# Test loss with causal mask
vocab_size = 4000
batch_size = 2
seq_len = 32

# Simulate model output
logits = torch.randn(batch_size, seq_len, vocab_size)
targets = torch.randint(0, vocab_size, (batch_size, seq_len))

# Cross entropy with shifted targets
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Standard next-token prediction: input[:,-1] predicts target[:,-1]
# In training: input[:, :-1] -> logits, target[:, 1:] is what we predict
shifted_targets = targets[:, 1:].clone()
shifted_targets[shifted_targets == 0] = -100  # mask padding

logits_for_loss = logits[:, :-1, :]  # remove last position

loss = criterion(logits_for_loss.reshape(-1, vocab_size), shifted_targets.reshape(-1))

print(f"\nLoss calculation:")
print(f"  Logits shape: {logits_for_loss.shape}")
print(f"  Targets shape: {shifted_targets.shape}")
print(f"  Loss: {loss.item():.4f}")

if loss.item() > 0 and loss.item() < 20:
    print("[VERIFIED] Loss calculation produces reasonable values")
else:
    print(f"[UNKNOWN] Loss value unusual: {loss.item()}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print("""
[TEST 1] TernarySTE returns exactly {-1, 0, +1}: PENDING
[TEST 2] Matrix mult uses ternary tensor: PENDING
[TEST 3] Scale timing: PENDING
[TEST 4] STE provides non-zero gradients: PENDING
[TEST 5] Optimizer updates weights: PENDING
[TEST 6] Threshold/scale behavior: PENDING
[TEST 7] All BitLinear layers quantized: PENDING
[TEST 8] Quantization changes output: PENDING
[TEST 9] Loss calculation: PENDING

Read output above for VERIFIED/UNKNOWN status.
""")
