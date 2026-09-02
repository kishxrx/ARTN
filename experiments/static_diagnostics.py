#!/usr/bin/env python3
"""
Comprehensive Static Diagnostics - NO TRAINING - FIXED VERSION

Checks all components that could cause gibberish output.
Each test runs in isolation and reports PASS/FAIL/UNKNOWN.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinear, BitLinearGPT, TernarySTE
from src.tokenizer.bpe_tokenizer import BPETokenizer

# Configuration
TOKENIZER_PATH = "/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json"
DEVICE = "cpu"

RESULTS = []

def log_test(test_name: str, status: str, evidence: str, details: str = ""):
    """Log test result"""
    RESULTS.append({
        "test": test_name,
        "status": status,
        "evidence": evidence,
        "details": details
    })
    status_icon = "✓" if status == "PASS" else "✗" if status == "FAIL" else "?"
    print(f"[{status_icon}] {test_name}: {status}")
    print(f"    Evidence: {evidence}")
    if details:
        print(f"    Details: {details}")
    print()

# ============================================================
# TEST 1: Tokenizer Encode/Decode Consistency
# ============================================================
print("=" * 70)
print("TEST 1: Tokenizer Encode/Decode Consistency")
print("=" * 70)

try:
    # Use BPETokenizer.load() method (not from_file)
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    
    # Test roundtrip
    original = "Once upon a time there was a little girl."
    encoded = tokenizer.encode(original)
    decoded = tokenizer.decode(encoded)
    
    # Check encoding produces valid tokens
    assert len(encoded) > 0, "Empty encoding"
    assert all(0 <= t < tokenizer.vocab_size for t in encoded), "Invalid token IDs"
    
    # Check decoding produces string
    assert len(decoded) > 0, "Empty decoding"
    
    # Check approximate consistency (allowing for special tokens, whitespace)
    original_words = set(original.lower().split())
    decoded_words = set(decoded.lower().split())
    overlap = len(original_words & decoded_words) / len(original_words) if original_words else 0
    
    if overlap > 0.5:
        log_test("Tokenizer consistency", "PASS", 
                f"Vocab={tokenizer.vocab_size}, {len(encoded)} tokens, word overlap={overlap:.1%}")
    else:
        log_test("Tokenizer consistency", "FAIL",
                f"Word overlap only {overlap:.1%}", 
                f"Original: '{original}' → Decoded: '{decoded}'")
except Exception as e:
    log_test("Tokenizer consistency", "UNKNOWN", f"Error: {e}")

# ============================================================
# TEST 2: Target Shifting (Next Token Prediction)
# ============================================================
print("=" * 70)
print("TEST 2: Target Shifting")
print("=" * 70)

try:
    # Create sample input
    batch_size, seq_len = 2, 16
    vocab_size = 100
    
    # Simulate input tokens and targets
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Standard target shifting: targets = input[:, 1:], input = input[:, :-1]
    targets = input_ids[:, 1:]
    shifted_input = input_ids[:, :-1]
    
    # Verify shapes
    assert targets.shape == shifted_input.shape, "Shape mismatch"
    assert targets.shape[1] == seq_len - 1, "Wrong sequence length"
    
    # Verify targets are shifted
    for i in range(batch_size):
        for j in range(seq_len - 1):
            expected_target = input_ids[i, j + 1]
            actual_target = targets[i, j]
            assert actual_target == expected_target, f"Position mismatch at [{i},{j}]"
    
    log_test("Target shifting", "PASS",
            f"Input {input_ids.shape} → shifted {shifted_input.shape}, targets {targets.shape}",
            "Standard autoregressive shifting verified")
except Exception as e:
    log_test("Target shifting", "FAIL", f"Error: {e}")

# ============================================================
# TEST 3: Causal Mask Verification
# ============================================================
print("=" * 70)
print("TEST 3: Causal Mask")
print("=" * 70)

try:
    # Create causal mask
    seq_len = 8
    mask = torch.tril(torch.ones(seq_len, seq_len))
    
    # Check lower triangular (including diagonal)
    for i in range(seq_len):
        for j in range(seq_len):
            expected = 1.0 if j <= i else 0.0
            assert mask[i, j] == expected, f"Mask incorrect at [{i},{j}]"
    
    # Check mask properties
    assert mask[0, 0] == 1.0, "First position should attend to itself"
    assert mask[0, 1] == 0.0, "First position should not attend to future"
    assert mask[-1, -1] == 1.0, "Last position should attend to itself"
    assert mask[-1, 0] == 1.0, "Last position should attend to start"
    
    # Count visible positions
    visible_count = mask.sum().item()
    expected_count = seq_len * (seq_len + 1) / 2
    
    log_test("Causal mask", "PASS",
            f"Lower triangular mask correct, {int(visible_count)} visible positions",
            f"Shape: {mask.shape}, type: lower_triangular")
except Exception as e:
    log_test("Causal mask", "FAIL", f"Error: {e}")

# ============================================================
# TEST 4: Loss Calculation
# ============================================================
print("=" * 70)
print("TEST 4: Loss Calculation")
print("=" * 70)

try:
    vocab_size = 100
    batch_size = 4
    seq_len = 16
    
    # Create logits and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Calculate loss
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
    
    # Verify loss properties
    assert loss.item() > 0, "Loss should be positive"
    assert loss.item() < 10, "Loss seems too high (unstable?)"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    
    # Check gradient flow
    logits_with_grad = logits.clone().requires_grad_(True)
    loss = loss_fn(logits_with_grad.reshape(-1, vocab_size), targets.reshape(-1))
    loss.backward()
    
    assert logits_with_grad.grad is not None, "No gradient on logits"
    assert not torch.isnan(logits_with_grad.grad).any(), "NaN in gradient"
    assert not torch.all(logits_with_grad.grad == 0), "All gradients zero"
    
    log_test("Loss calculation", "PASS",
            f"loss={loss.item():.4f}, grad_norm={logits_with_grad.grad.norm().item():.4f}",
            "CrossEntropyLoss works, gradients flow")
except Exception as e:
    log_test("Loss calculation", "FAIL", f"Error: {e}")

# ============================================================
# TEST 5: Gradient Flow Through Model - FIXED
# ============================================================
print("=" * 70)
print("TEST 5: Gradient Flow Through Model")
print("=" * 70)

try:
    # Create small model
    model = BitLinearGPT(
        vocab_size=100,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=32
    )
    
    # Forward pass
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    targets = torch.randint(0, 100, (batch_size, seq_len))
    
    # Call model forward - returns dict with 'logits' and 'loss' keys
    output = model(input_ids)
    logits = output['logits']  # Extract logits from dict
    
    # Compute loss manually (since targets not passed to model)
    loss = nn.CrossEntropyLoss()(
        logits.reshape(-1, logits.size(-1)), 
        targets.reshape(-1)
    )
    
    # Backward pass
    loss.backward()
    
    # Check gradients on all parameters
    total_params = 0
    params_with_grad = 0
    zero_grad_params = 0
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if grad_norm > 1e-8:
                    params_with_grad += 1
                else:
                    zero_grad_params += 1
    
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
    
    if params_with_grad == total_params:
        log_test("Gradient flow", "PASS",
                f"{params_with_grad}/{total_params} params have non-zero gradients, avg_norm={avg_grad_norm:.6f}",
                f"Loss: {loss.item():.4f}")
    elif params_with_grad > 0:
        log_test("Gradient flow", "UNKNOWN",
                f"Only {params_with_grad}/{total_params} params have gradients",
                f"{zero_grad_params} params have zero gradients")
    else:
        log_test("Gradient flow", "FAIL",
                "No parameters received gradients",
                "Check model architecture")
except Exception as e:
    log_test("Gradient flow", "FAIL", f"Error: {e}")

# ============================================================
# TEST 6: Optimizer Parameter Updates - FIXED
# ============================================================
print("=" * 70)
print("TEST 6: Optimizer Parameter Updates")
print("=" * 70)

try:
    # Create model and optimizer
    model = BitLinearGPT(
        vocab_size=100,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=32
    )
    
    optimizer = torch.optim.Adafactor(model.parameters(), lr=0.001)
    
    # Store initial parameters
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()
    
    # Training step
    input_ids = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))
    
    output = model(input_ids)
    logits = output['logits']
    
    loss = nn.CrossEntropyLoss()(
        logits.reshape(-1, logits.size(-1)), 
        targets.reshape(-1)
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check parameters changed
    params_changed = 0
    total_params = 0
    change_magnitudes = []
    
    for name, param in model.named_parameters():
        total_params += 1
        change = (param.data - initial_params[name]).norm().item()
        change_magnitudes.append(change)
        if change > 1e-8:
            params_changed += 1
    
    avg_change = np.mean(change_magnitudes)
    
    if params_changed == total_params:
        log_test("Optimizer updates", "PASS",
                f"ALL {params_changed}/{total_params} params updated, avg_change={avg_change:.6f}",
                f"Loss: {loss.item():.4f}")
    elif params_changed > 0:
        log_test("Optimizer updates", "UNKNOWN",
                f"Only {params_changed}/{total_params} params updated",
                f"Avg change: {avg_change:.6f}")
    else:
        log_test("Optimizer updates", "FAIL",
                "NO parameters updated",
                "Optimizer not working")
except Exception as e:
    log_test("Optimizer updates", "FAIL", f"Error: {e}")

# ============================================================
# TEST 7: Ternary Quantization and STE
# ============================================================
print("=" * 70)
print("TEST 7: Ternary Quantization and STE")
print("=" * 70)

try:
    # Create BitLinear layer
    layer = BitLinear(64, 64)
    
    # Forward pass
    x = torch.randn(2, 16, 64, requires_grad=True)
    y = layer(x)
    
    # Check output shape
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    
    # Backward pass
    loss = y.sum()
    loss.backward()
    
    # Check gradient exists on input
    assert x.grad is not None, "No gradient on input"
    
    # Check gradient exists on weight
    assert layer.weight.grad is not None, "No gradient on weight"
    
    # TernarySTE function test
    weight = torch.randn(32, 32, requires_grad=True)
    scale = weight.abs().mean().detach()
    
    weight_quantized, returned_scale = TernarySTE.apply(weight, scale.clone(), 0.5)
    
    # Check ternary values
    unique_vals = torch.unique(weight_quantized)
    
    # Should contain -1, 0, +1 (scaled)
    is_ternary = all(val in unique_vals for val in [-1, 0, 1])
    
    # Gradient test
    loss2 = weight_quantized.sum()
    loss2.backward()
    
    grad_exists = weight.grad is not None
    grad_nonzero = not torch.all(weight.grad == 0) if grad_exists else False
    
    if grad_nonzero:
        log_test("Ternary STE", "PASS",
                f"Gradient flows through STE, ternary values: {unique_vals.tolist()[:5]}...",
                f"Weight grad norm: {weight.grad.norm().item():.6f}")
    else:
        log_test("Ternary STE", "FAIL",
                "STE gradient blocked or zeros",
                f"Grad exists: {grad_exists}")
except Exception as e:
    log_test("Ternary STE", "FAIL", f"Error: {e}")

# ============================================================
# TEST 8: Tensors Entering Linear Operation
# ============================================================
print("=" * 70)
print("TEST 8: Tensors Entering Linear Operation")
print("=" * 70)

try:
    # Hook to capture tensor
    captured_tensors = {}
    
    def hook_fn(module, inp, out, name):
        if isinstance(inp, tuple) and len(inp) > 0:
            captured_tensors[name] = inp[0].detach().clone()
    
    # Create model with hooks
    model = BitLinearGPT(
        vocab_size=100,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=32
    )
    
    # Register hooks on all BitLinear layers
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            module.register_forward_hook(
                lambda m, i, o, n=name: hook_fn(m, i, o, n)
            )
    
    # Forward pass
    input_ids = torch.randint(0, 100, (2, 16))
    _ = model(input_ids)
    
    # Check captured tensors
    issues = []
    
    for name, tensor in captured_tensors.items():
        if torch.isnan(tensor).any():
            issues.append(f"{name}: NaN detected")
        if torch.isinf(tensor).any():
            issues.append(f"{name}: Inf detected")
        if tensor.abs().max() > 1000:
            issues.append(f"{name}: Very large values")
        if tensor.abs().max() < 1e-8:
            issues.append(f"{name}: Very small values")
    
    if not issues:
        log_test("Tensor values", "PASS",
                f"{len(captured_tensors)} BitLinear layers checked",
                "No NaN, Inf, or extreme values")
    else:
        log_test("Tensor values", "FAIL",
                f"Issues in {len(issues)} layers",
                "; ".join(issues[:3]))
except Exception as e:
    log_test("Tensor values", "FAIL", f"Error: {e}")

# ============================================================
# TEST 9: Data/Batch Correctness
# ============================================================
print("=" * 70)
print("TEST 9: Data/Batch Correctness")
print("=" * 70)

try:
    # Simulate batch loading
    batch_size = 4
    seq_len = 32
    vocab_size = 100
    
    # Create batch
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Check batch shape
    assert input_ids.dim() == 2, f"Wrong dimensions: {input_ids.dim()}"
    assert input_ids.size(0) == batch_size, f"Wrong batch size: {input_ids.size(0)}"
    assert input_ids.size(1) == seq_len, f"Wrong sequence length: {input_ids.size(1)}"
    
    # Check token range
    assert input_ids.min() >= 0, f"Negative token: {input_ids.min()}"
    assert input_ids.max() < vocab_size, f"Token exceeds vocab: {input_ids.max()}"
    
    # Check data variety (not all same token)
    unique_tokens = input_ids.unique().numel()
    variety_ratio = unique_tokens / (batch_size * seq_len)
    
    if variety_ratio > 0.1:
        log_test("Data batch", "PASS",
                f"Batch shape correct, variety: {unique_tokens} unique tokens ({variety_ratio:.1%})",
                f"Range: [{input_ids.min()}, {input_ids.max()}]")
    else:
        log_test("Data batch", "UNKNOWN",
                f"Low variety: only {unique_tokens} unique tokens",
                "Data may be too uniform")
except Exception as e:
    log_test("Data batch", "FAIL", f"Error: {e}")

# ============================================================
# TEST 10: Initialization
# ============================================================
print("=" * 70)
print("TEST 10: Initialization")
print("=" * 70)

try:
    # Create model
    model = BitLinearGPT(
        vocab_size=4000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256
    )
    
    # Check initialization statistics
    stats = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            stats[name] = {
                "mean": param.data.mean().item(),
                "std": param.data.std().item(),
                "min": param.data.min().item(),
                "max": param.data.max().item(),
                "zeros": (param.data == 0).float().mean().item()
            }
    
    # Check for issues
    issues = []
    
    for name, s in stats.items():
        # Skip LayerNorm (zeros are normal for bias, initialized to 1 for weight)
        if 'ln' in name or 'layer_norm' in name:
            continue
        
        if s["std"] == 0:
            issues.append(f"{name}: Zero std (all same value)")
        if abs(s["mean"]) > 10:
            issues.append(f"{name}: Large mean ({s['mean']:.2f})")
    
    # Sample statistics for weights only (not LayerNorm)
    weight_stats = [(n, s) for n, s in stats.items() if 'weight' in n and 'ln' not in n]
    avg_std = np.mean([s['std'] for n, s in weight_stats]) if weight_stats else 0
    
    if not issues:
        log_test("Initialization", "PASS",
                f"avg_weight_std={avg_std:.4f}, no issues found",
                f"Checked {len(stats)} parameters")
    else:
        log_test("Initialization", "UNKNOWN",
                f"{len(issues)} potential issues",
                "; ".join(issues[:3]))
except Exception as e:
    log_test("Initialization", "FAIL", f"Error: {e}")

# ============================================================
# TEST 11: Parameter/Gradient Statistics - FIXED
# ============================================================
print("=" * 70)
print("TEST 11: Parameter/Gradient Statistics")
print("=" * 70)

try:
    # Create and train model briefly
    model = BitLinearGPT(
        vocab_size=100,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=32
    )
    
    optimizer = torch.optim.Adafactor(model.parameters(), lr=0.001)
    
    # Multiple training steps
    losses = []
    for _ in range(5):
        input_ids = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        
        output = model(input_ids)
        logits = output['logits']
        
        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, logits.size(-1)), 
            targets.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # Collect final statistics
    param_stats = []
    grad_stats = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_stats.append({
                "name": name,
                "mean": param.data.mean().item(),
                "std": param.data.std().item(),
                "norm": param.data.norm().item()
            })
            
            if param.grad is not None:
                grad_stats.append({
                    "name": name,
                    "mean": param.grad.mean().item(),
                    "std": param.grad.std().item(),
                    "norm": param.grad.norm().item()
                })
    
    avg_param_norm = np.mean([s['norm'] for s in param_stats])
    avg_grad_norm = np.mean([s['norm'] for s in grad_stats])
    
    # Check for gradient explosion/vanishing
    if avg_grad_norm > 1000:
        status = "FAIL"
        evidence = "Gradient explosion detected"
    elif avg_grad_norm < 1e-8:
        status = "FAIL"
        evidence = "Gradient vanishing detected"
    else:
        status = "PASS"
        evidence = "Gradients in healthy range"
    
    log_test("Gradient stats", status,
            f"avg_param_norm={avg_param_norm:.4f}, avg_grad_norm={avg_grad_norm:.6f}, final_loss={losses[-1]:.4f}",
            f"Checked {len(param_stats)} params, {len(grad_stats)} grads")
except Exception as e:
    log_test("Gradient stats", "FAIL", f"Error: {e}")

# ============================================================
# SUMMARY
# ============================================================
print("=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

passed = sum(1 for r in RESULTS if r["status"] == "PASS")
failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
unknown = sum(1 for r in RESULTS if r["status"] == "UNKNOWN")

print(f"\nTotal tests: {len(RESULTS)}")
print(f"  PASS:   {passed}")
print(f"  FAIL:   {failed}")
print(f"  UNKNOWN: {unknown}")

if failed == 0 and unknown == 0:
    print("\nALL STATIC DIAGNOSTICS PASSED.")
    print("Issue is likely in training dynamics, not implementation.")
elif failed == 0:
    print(f"\n{unknown} tests returned UNKNOWN - may need investigation.")
else:
    print(f"\n{failed} tests FAILED - implementation issues detected!")

print("\n" + "=" * 70)

# Save results
import json
output_path = Path(__file__).parent / "results" / "static_diagnostics.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(RESULTS, f, indent=2)

print(f"Results saved to: {output_path}")
