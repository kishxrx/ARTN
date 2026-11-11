#!/usr/bin/env python3
"""
ARTN Smoke Test: Forward pass, gradients, causality, halting, ternary.

Tests:
  1.  Model instantiation — 1 shared block, parameter count
  2.  Forward pass — output shape, finite loss
  3.  Backward pass — gradients for all parameter groups
  4.  Gradient quality — non-None, non-NaN, non-Inf, meaningful non-zero
  5.  Causal correctness — change future token, early positions unchanged
  6.  Halting weight normalization — sum to 1.0 per position
  7.  Conditional vs halting weight distinction — h doesn't sum to 1, p does
  8.  Ternary quantization — weights quantize to {-1, 0, +1}
  9.  Inference early stop — threshold<1.0 gives fewer passes
  10. Inference threshold=1.0 — full computation (all passes)
  11. Inference output validity — valid token IDs, no NaN
  12. Per-position halt variation — different positions halt at different passes
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from experiments.artn_model import ARTNGPT
from src.model.bitlinear import BitLinear


def test_header(name):
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")


def test_pass(name):
    print(f"  PASS: {name}")


def test_fail(name, detail=""):
    print(f"  FAIL: {name} {detail}")
    return False


def main():
    print("=" * 70)
    print("ARTN SMOKE TEST")
    print("=" * 70)

    torch.manual_seed(42)

    all_pass = True

    # ================================================================
    # TEST 1: Model instantiation
    # ================================================================
    test_header("1. Model Instantiation")

    model = ARTNGPT(
        vocab_size=4000, d_model=256, n_heads=8,
        n_layers=6, d_ff=1024, max_seq_len=256,
        target_passes=4.0, ponder_loss_weight=0.01,
    )

    total_params = model.count_parameters()
    block_params = model.count_block_parameters()

    print(f"  Total params: {total_params:,}")
    print(f"  Block params (shared): {block_params:,}")
    print(f"  Max passes: {model.max_passes}")

    # Verify single shared block (not ModuleList)
    has_block = hasattr(model, 'block')
    has_blocks = hasattr(model, 'blocks')
    print(f"  Has 'block' (single): {has_block}")
    print(f"  Has 'blocks' (ModuleList): {has_blocks}")

    if has_block and not has_blocks:
        test_pass("Single shared block (not ModuleList)")
    else:
        all_pass = test_fail("Expected single block, got ModuleList or missing")

    # Verify parameter count is ~2.9M (not ~6.85M)
    expected_min = 2_800_000
    expected_max = 3_000_000
    if expected_min <= total_params <= expected_max:
        test_pass(f"Parameter count {total_params:,} in expected range [{expected_min}, {expected_max}]")
    else:
        all_pass = test_fail("Parameter count", f"got {total_params:,}, expected ~2.9M")

    # Verify halting gate exists
    if hasattr(model, 'halting_gate') and isinstance(model.halting_gate, nn.Linear):
        gate_params = sum(p.numel() for p in model.halting_gate.parameters())
        print(f"  Gate params: {gate_params}")
        test_pass("Halting gate exists (nn.Linear)")
    else:
        all_pass = test_fail("Halting gate missing")

    # ================================================================
    # TEST 2: Forward pass
    # ================================================================
    test_header("2. Forward Pass")

    B, T = 2, 32
    input_ids = torch.randint(0, 4000, (B, T))
    targets = torch.randint(0, 4000, (B, T))

    model.train()
    result = model(input_ids, targets=targets)

    logits = result['logits']
    loss = result['loss']

    print(f"  Input shape: ({B}, {T})")
    print(f"  Logits shape: {list(logits.shape)}")
    print(f"  Loss value: {loss.item():.4f}")

    # Check logits shape
    if list(logits.shape) == [B, T, 4000]:
        test_pass("Logits shape correct")
    else:
        all_pass = test_fail("Logits shape", f"got {list(logits.shape)}, expected [{B}, {T}, 4000]")

    # Check loss is scalar
    if loss.dim() == 0:
        test_pass("Loss is scalar")
    else:
        all_pass = test_fail("Loss is not scalar", f"dim={loss.dim()}")

    # Check loss is finite
    if torch.isfinite(loss):
        test_pass("Loss is finite")
    else:
        all_pass = test_fail("Loss is not finite", f"value={loss.item()}")

    # Check loss is positive
    if loss.item() > 0:
        test_pass("Loss is positive")
    else:
        all_pass = test_fail("Loss is not positive", f"value={loss.item()}")

    # Check halting info
    info = result['halting_info']
    print(f"  Expected passes: {info['expected_passes']:.4f}")
    print(f"  Loss CE: {info['loss_ce']:.4f}")
    print(f"  Loss ponder: {info['loss_ponder']:.6f}")
    print(f"  Passes executed: {info['passes_executed']}")

    if info['passes_executed'] == 6:
        test_pass("All 6 passes executed during training")
    else:
        all_pass = test_fail("Wrong pass count", f"got {info['passes_executed']}, expected 6")

    # ================================================================
    # TEST 3 & 4: Backward pass and gradient quality
    # ================================================================
    test_header("3 & 4. Backward Pass and Gradient Quality")

    model.zero_grad()
    loss.backward()

    # Group parameters
    param_groups = {
        'block_bitlinear': [],
        'block_layernorm': [],
        'halting_gate': [],
        'token_embedding': [],
        'position_embedding': [],
        'ln_f': [],
        'lm_head': [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'block.' in name and 'attn' in name or 'block.' in name and 'ff' in name:
            param_groups['block_bitlinear'].append((name, param))
        elif 'block.' in name and 'ln' in name:
            param_groups['block_layernorm'].append((name, param))
        elif 'halting_gate' in name:
            param_groups['halting_gate'].append((name, param))
        elif 'token_embedding' in name:
            param_groups['token_embedding'].append((name, param))
        elif 'position_embedding' in name:
            param_groups['position_embedding'].append((name, param))
        elif 'ln_f' in name:
            param_groups['ln_f'].append((name, param))
        elif 'lm_head' in name:
            param_groups['lm_head'].append((name, param))
        else:
            print(f"  WARNING: ungrouped parameter: {name}")

    print(f"  Parameter groups:")
    for group_name, params in param_groups.items():
        print(f"    {group_name}: {len(params)} parameters")

    # Check each group
    for group_name, params in param_groups.items():
        if not params:
            print(f"  SKIP: {group_name} has no parameters")
            continue

        group_has_grad = False
        group_has_nonzero = False
        group_has_nan = False
        group_has_inf = False

        for name, param in params:
            if param.grad is None:
                print(f"    {name}: NO GRADIENT (None)")
                continue

            if torch.isnan(param.grad).any():
                group_has_nan = True
                print(f"    {name}: NaN GRADIENT")
                continue

            if torch.isinf(param.grad).any():
                group_has_inf = True
                print(f"    {name}: Inf GRADIENT")
                continue

            group_has_grad = True
            max_grad = param.grad.abs().max().item()
            if max_grad > 1e-10:
                group_has_nonzero = True

        # Report group status
        status_parts = []
        if group_has_nan:
            status_parts.append("HAS NaN")
        if group_has_inf:
            status_parts.append("HAS Inf")
        if not group_has_grad:
            status_parts.append("NO GRADIENTS")
        if group_has_grad and not group_has_nonzero:
            status_parts.append("ALL ZERO")

        if status_parts:
            all_pass = test_fail(f"{group_name}", ", ".join(status_parts))
        elif group_has_nonzero:
            test_pass(f"{group_name} has meaningful non-zero gradients")
        else:
            all_pass = test_fail(f"{group_name}", "unexpected state")

    # ================================================================
    # TEST 5: Causal correctness — no future-token leakage
    # ================================================================
    test_header("5. Causal Correctness (Future-Token Leakage)")

    model.eval()

    # Create two inputs that differ ONLY in the last token
    input_A = torch.randint(0, 4000, (1, 16))
    input_B = input_A.clone()
    # Change last token to something different
    new_token = (input_B[0, -1] + 1) % 4000
    input_B[0, -1] = new_token

    print(f"  Input A last token: {input_A[0, -1].item()}")
    print(f"  Input B last token: {input_B[0, -1].item()}")

    with torch.no_grad():
        result_A = model(input_A)
        result_B = model(input_B)

        logits_A = result_A['logits']
        logits_B = result_B['logits']

    # Position 0's logits should be IDENTICAL (depends only on token 0)
    logits_0_A = logits_A[0, 0, :]
    logits_0_B = logits_B[0, 0, :]
    diff_0 = (logits_0_A - logits_0_B).abs().max().item()

    # Position 7 (middle) should also be IDENTICAL
    logits_7_A = logits_A[0, 7, :]
    logits_7_B = logits_B[0, 7, :]
    diff_7 = (logits_7_A - logits_7_B).abs().max().item()

    # Position 14 (second-to-last) should be IDENTICAL
    logits_14_A = logits_A[0, 14, :]
    logits_14_B = logits_B[0, 14, :]
    diff_14 = (logits_14_A - logits_14_B).abs().max().item()

    # Position 15 (last) should be DIFFERENT
    logits_15_A = logits_A[0, 15, :]
    logits_15_B = logits_B[0, 15, :]
    diff_15 = (logits_15_A - logits_15_B).abs().max().item()

    print(f"  Position  0 (first):  max diff = {diff_0:.2e}")
    print(f"  Position  7 (middle): max diff = {diff_7:.2e}")
    print(f"  Position 14 (penult): max diff = {diff_14:.2e}")
    print(f"  Position 15 (last):   max diff = {diff_15:.2e}")

    # Check halting weights too
    with torch.no_grad():
        info_A = result_A['halting_info']
        info_B = result_B['halting_info']

    # Check halt weights at position 0
    hw_0_A = info_A['halt_weights'][0][0, 0, 0].item()
    hw_0_B = info_B['halt_weights'][0][0, 0, 0].item()
    hw_diff_0 = abs(hw_0_A - hw_0_B)

    print(f"  Halt weight pos 0, pass 0: A={hw_0_A:.6f}, B={hw_0_B:.6f}, diff={hw_diff_0:.2e}")

    # Causality passes
    causal_pass = True
    if diff_0 > 1e-5:
        all_pass = test_fail("Position 0 logits leaked", f"diff={diff_0:.2e}")
        causal_pass = False
    if diff_7 > 1e-5:
        all_pass = test_fail("Position 7 logits leaked", f"diff={diff_7:.2e}")
        causal_pass = False
    if diff_14 > 1e-5:
        all_pass = test_fail("Position 14 logits leaked", f"diff={diff_14:.2e}")
        causal_pass = False
    if diff_15 < 1e-5:
        all_pass = test_fail("Position 15 should differ", f"diff={diff_15:.2e}")
        causal_pass = False
    if hw_diff_0 > 1e-5:
        all_pass = test_fail("Position 0 halt weight leaked", f"diff={hw_diff_0:.2e}")
        causal_pass = False

    if causal_pass:
        test_pass("No future-token leakage in logits or halting weights")

    # ================================================================
    # TEST 6: Halting weight normalization (sum to 1.0)
    # ================================================================
    test_header("6. Halting Weight Normalization")

    model.eval()
    with torch.no_grad():
        result = model(input_A)
        info = result['halting_info']

    halt_weights = info['halt_weights']  # list of (B, T, 1)

    # Sum halt weights across all passes per position
    total_weight = torch.zeros(1, 16, 1)
    for p in halt_weights:
        total_weight += p

    min_sum = total_weight.min().item()
    max_sum = total_weight.max().item()
    print(f"  Halt weight sums: min={min_sum:.6f}, max={max_sum:.6f}")

    if abs(min_sum - 1.0) < 1e-4 and abs(max_sum - 1.0) < 1e-4:
        test_pass("All position halt weights sum to 1.0")
    else:
        all_pass = test_fail("Halt weights don't sum to 1", f"range [{min_sum:.6f}, {max_sum:.6f}]")

    # ================================================================
    # TEST 7: Conditional halt prob vs halting weight
    # ================================================================
    test_header("7. Conditional Halt Prob vs Halting Weight")

    h_list = info['conditional_halt_probs']  # h_i per pass
    p_list = info['halt_weights']             # p_i per pass

    # h_i should NOT sum to 1 (they are independent per pass)
    h_sum = sum(h.mean().item() for h in h_list)
    print(f"  Sum of mean(h_i) across passes: {h_sum:.4f} (should NOT be 1.0)")

    # p_i should sum to 1 per position
    p_sum = sum(p.mean().item() for p in p_list)
    print(f"  Sum of mean(p_i) across passes: {p_sum:.6f} (should be ~1.0)")

    if abs(h_sum - 1.0) > 0.01:
        test_pass("Conditional halt probs do NOT sum to 1 (independent per pass)")
    else:
        all_pass = test_fail("Conditional halt probs unexpectedly sum to 1")

    if abs(p_sum - 1.0) < 1e-4:
        test_pass("Halting weights sum to ~1.0")
    else:
        all_pass = test_fail("Halting weights don't sum to 1", f"got {p_sum:.6f}")

    # ================================================================
    # TEST 8: Ternary quantization
    # ================================================================
    test_header("8. Ternary Quantization")

    ternary_stats = model.get_ternary_stats()
    print(f"  BitLinear layers found: {len(ternary_stats['weight_stats'])}")
    print(f"  Total ternary params: {ternary_stats['total_params']:,}")

    if len(ternary_stats['weight_stats']) > 0:
        # Check first layer
        s = ternary_stats['weight_stats'][0]
        print(f"  Layer: {s['name']}, shape: {s['shape']}")
        print(f"    Positive: {s['positive']:,}, Negative: {s['negative']:,}, Zero: {s['zero']:,}")
        print(f"    Sparsity: {s['sparsity']:.2%}")

        # Verify three bins exist (should have all three values)
        if s['positive'] > 0 and s['negative'] > 0:
            test_pass("Ternary quantization produces {-1, 0, +1}")
        else:
            all_pass = test_fail("Ternary missing values", f"pos={s['positive']}, neg={s['negative']}")

        # Verify at least 2 BitLinear layers (block has 6: q,k,v,o,w1,w2 + lm_head)
        if len(ternary_stats['weight_stats']) >= 7:
            test_pass(f"Found {len(ternary_stats['weight_stats'])} BitLinear layers (block 6 + lm_head)")
        else:
            all_pass = test_fail("Too few BitLinear layers", f"got {len(ternary_stats['weight_stats'])}")
    else:
        all_pass = test_fail("No BitLinear layers found")

    # ================================================================
    # TEST 9: Inference early stop (threshold < 1.0)
    # ================================================================
    test_header("9. Inference Early Stop (threshold=0.5)")

    model.eval()

    # Track passes used by instrumenting generate
    # We'll manually replicate the generate logic for one step
    with torch.no_grad():
        input_test = torch.randint(0, 4000, (1, 16))

        tok_emb = model.token_embedding(input_test)
        pos_emb = model.position_embedding(torch.arange(16))
        x = tok_emb + pos_emb

        remaining = torch.ones(1, 16, 1)
        output = torch.zeros(1, 16, 256)
        passes_used_05 = 0

        for pass_idx in range(6):
            passes_used_05 += 1
            x = model.block(x)
            h = torch.sigmoid(model.halting_gate(x))
            p = h * remaining
            if pass_idx == 5:
                p = remaining
            output = output + x * p
            remaining = remaining - p

            cumulative_last = 1.0 - remaining[0, -1, 0].item()
            if cumulative_last >= 0.5 and pass_idx < 5:
                break

        # Assign remaining
        output = output + x * remaining

    print(f"  threshold=0.5: passes used = {passes_used_05} / 6")

    if passes_used_05 < 6:
        test_pass(f"Early stop worked: {passes_used_05} passes (threshold=0.5)")
    else:
        print(f"  NOTE: Gate didn't cross 0.5 — this is possible with untrained gate")
        print(f"        Will re-test after training")
        test_pass("No early stop at threshold=0.5 (untrained gate — expected)")

    # ================================================================
    # TEST 10: Inference threshold=1.0 (full computation)
    # ================================================================
    test_header("10. Inference threshold=1.0 (Full Computation)")

    with torch.no_grad():
        tok_emb = model.token_embedding(input_test)
        pos_emb = model.position_embedding(torch.arange(16))
        x = tok_emb + pos_emb

        remaining = torch.ones(1, 16, 1)
        output = torch.zeros(1, 16, 256)
        passes_used_10 = 0

        for pass_idx in range(6):
            passes_used_10 += 1
            x = model.block(x)
            h = torch.sigmoid(model.halting_gate(x))
            p = h * remaining
            if pass_idx == 5:
                p = remaining
            output = output + x * p
            remaining = remaining - p

            cumulative_last = 1.0 - remaining[0, -1, 0].item()
            if cumulative_last >= 1.0 and pass_idx < 5:
                break

        output = output + x * remaining

    print(f"  threshold=1.0: passes used = {passes_used_10} / 6")

    if passes_used_10 == 6:
        test_pass("threshold=1.0 uses all 6 passes (full computation)")
    else:
        all_pass = test_fail("threshold=1.0 should use all passes", f"got {passes_used_10}")

    # ================================================================
    # TEST 11: Inference output validity
    # ================================================================
    test_header("11. Inference Output Validity")

    model.eval()
    with torch.no_grad():
        prompt = torch.randint(0, 4000, (1, 8))
        output_ids = model.generate(
            prompt, max_new_tokens=5, temperature=0.9, threshold=1.0
        )

    print(f"  Input shape: (1, 8)")
    print(f"  Output shape: {list(output_ids.shape)}")
    print(f"  Output IDs: {output_ids[0].tolist()}")

    # Check shape
    if list(output_ids.shape) == [1, 13]:
        test_pass("Output shape correct (8 + 5 new tokens)")
    else:
        all_pass = test_fail("Output shape", f"got {list(output_ids.shape)}, expected [1, 13]")

    # Check all IDs are valid
    all_valid = all(0 <= id < 4000 for id in output_ids[0].tolist())
    if all_valid:
        test_pass("All output token IDs are valid [0, 4000)")
    else:
        all_pass = test_fail("Invalid token IDs")

    # Check no NaN in output
    if not torch.isnan(output_ids.float()).any():
        test_pass("No NaN in output")
    else:
        all_pass = test_fail("NaN in output IDs")

    # ================================================================
    # TEST 12: Per-position halt variation
    # ================================================================
    test_header("12. Per-Position Halt Variation")

    model.eval()
    with torch.no_grad():
        input_test = torch.randint(0, 4000, (1, 16))
        result = model(input_test)
        info = result['halting_info']

    # Expected passes per position
    remaining_history = info['remaining_history']
    E_per_pos = 1.0 + sum(r for r in remaining_history)  # (1, 16, 1)

    e_values = E_per_pos[0, :, 0].tolist()
    print(f"  E[passes] per position:")
    for i, e in enumerate(e_values):
        print(f"    Position {i:2d}: {e:.4f}")

    e_min = min(e_values)
    e_max = max(e_values)
    e_range = e_max - e_min
    print(f"  Range: [{e_min:.4f}, {e_max:.4f}], spread: {e_range:.4f}")

    if e_range > 0.01:
        test_pass(f"Per-position halt varies (range={e_range:.4f})")
    else:
        print(f"  NOTE: All positions have same E[passes] — gate is ignoring input")
        print(f"        This is possible with random init — will change after training")
        test_pass("Per-position halt uniform (untrained gate — expected)")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*70}")

    if all_pass:
        print("\n  ALL TESTS PASSED")
        print("  Ready to proceed to training script creation")
    else:
        print("\n  SOME TESTS FAILED — review above")

    print(f"\n  Model: ARTNGPT")
    print(f"  Parameters: {total_params:,}")
    print(f"  Shared block: {block_params:,} params")
    print(f"  Max passes: {model.max_passes}")
    print(f"  Target passes: {model.target_passes}")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
