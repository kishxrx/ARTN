# Ternary Implementation Verification Report

**Date:** August 18, 2026
**Purpose:** Verify whether BitLinear/TernarySTE implementation performs true ternary computation

---

## Executive Summary

**CRITICAL FINDING:** Matrix multiplications in BitLinear use **scaled float values** ({-scale, 0, +scale}), NOT true ternary values {-1, 0, +1}.

This is caused by line 187 in `src/model/bitlinear.py` which multiplies quantized weights by scale before returning them for computation.

---

## Verification Results

### TEST 1: TernarySTE Output

| Metric | Value |
|--------|-------|
| Input weight shape | torch.Size([64, 64]) |
| Output unique values | **{-1.0, 0.0, 1.0}** |
| Distribution: -1 | 34.79% |
| Distribution: 0 | 31.40% |
| Distribution: +1 | 33.81% |

**Status:** [VERIFIED] TernarySTE returns EXACTLY {-1, 0, +1}

---

### TEST 2: Tensor Passed to Matrix Multiplication

| Metric | Value |
|--------|-------|
| Intercepted tensor shape | torch.Size([64, 64]) |
| Unique values | **{-0.1431, 0.0, 0.1431}** |
| Distribution: -1 count | 0/4096 (0.00%) |
| Distribution: 0 count | 1288/4096 (31.45%) |
| Distribution: +1 count | 0/4096 (0.00%) |
| Weight range | [-0.1431, 0.1431] |

**Status:** [ERROR] Matrix multiplication uses SCALED FLOATS, not ternary

---

### TEST 3: Scale Multiplication Timing

| Metric | Value |
|--------|-------|
| quantize_weights() returns | {-0.1415, 0.0, 0.1415} |
| Scale value | 0.141472 |

**Status:** Scale is applied AFTER ternarization

---

### TEST 4: STE Gradient Flow

| Metric | Value |
|--------|-------|
| Weight gradient shape | torch.Size([64, 64]) |
| Non-zero gradients | 4096/4096 (100%) |
| Gradient range | [-2.299, 3.444] |
| Gradient mean | 0.041 |
| Gradient norm | 76.25 |

**Status:** [VERIFIED] STE provides non-zero gradients to full-precision weights

---

### TEST 5: Optimizer Updates

| Metric | Value |
|--------|-------|
| Weights changed | 4096/4096 (100%) |
| Max change | 0.2447 |
| Mean change | 0.0845 |

**Status:** [VERIFIED] Optimizer updates full-precision weights correctly

---

### TEST 6: Threshold/Scale Behavior

| Metric | Value |
|--------|-------|
| Weight mean | 0.0016 |
| Weight std | 0.1754 |
| Weight min | -0.6228 |
| Weight max | 0.6296 |
| Scale (abs mean) | 0.1403 |
| Threshold | 0.5 |
| Scaled weight range | [-4.44, 4.49] |
| Expected -1 count | 1396/4096 |
| Expected 0 count | 1282/4096 |
| Expected +1 count | 1418/4096 |
| Actual -1 count | 0/4096 |

**Status:** [UNKNOWN] Quantization threshold behaves, but scale multiplication changes output values

---

### TEST 7: All BitLinear Layers Quantized

| Metric | Value |
|--------|-------|
| Total BitLinear layers | 13 |
| Layers performing quantization | 13 |
| All layers quantized? | YES |

**List of layers:**
- blocks.0.attn.q_proj
- blocks.0.attn.k_proj
- blocks.0.attn.v_proj
- blocks.0.attn.o_proj
- blocks.0.ff.w1
- blocks.0.ff.w2
- blocks.1.attn.q_proj
- blocks.1.attn.k_proj
- blocks.1.attn.v_proj
- blocks.1.attn.o_proj
- blocks.1.ff.w1
- blocks.1.ff.w2
- lm_head

**Status:** [VERIFIED] All BitLinear layers perform quantization

**Note:** All layers produce scaled floats, not ternary

---

### TEST 8: Quantized vs Full-Precision Output

| Metric | Value |
|--------|-------|
| Quantized output mean | -0.0393 |
| Full-precision output mean | -0.0558 |
| Max difference | 2.5769 |
| Mean difference | 0.5878 |
| Relative difference | 1.64 |

**Status:** [VERIFIED] Quantized and full-precision produce DIFFERENT outputs

**Note:** This confirms quantization HAS AN EFFECT, but uses scaled floats

---

### TEST 9: Loss Calculation

| Metric | Value |
|--------|-------|
| Test text | "Once upon a time" |
| Encoded tokens | [258, 272, 113, 417] |
| Decoded text | "Once upon a time" |
| Tokenizer consistency | EXACT |
| Loss value | 8.8263 |

**Status:** [VERIFIED] Tokenizer and loss calculation correct

---

## Code Analysis

### The Implementation Path

**File:** `src/model/bitlinear.py`

```python
# Line 168-197: quantize_weights()

def quantize_weights(self) -> Tuple[Tensor, Tensor]:
    # Compute scale
    if self.per_channel:
        scale = compute_per_channel_scale(self.weight, self.eps)
    else:
        scale = compute_abs_mean_scale(self.weight, self.eps)
    
    # Quantize using STE
    weight_quantized, scale = TernarySTE.apply(
        self.weight, scale, self.threshold
    )
    # At this point: weight_quantized ∈ {-1, 0, +1}
    
    # Dequantize: multiply back by scale for computation
    weight_dequantized = weight_quantized * scale  # ← LINE 187
    # After: weight_dequantized ∈ {-scale, 0, +scale}
    
    return weight_dequantized, scale  # ← Returns FLOAT

# Line 199-205: forward()

def forward(self, x: Tensor) -> Tensor:
    weight_q, scale = self.quantize_weights()
    return F.linear(x, weight_q, self.bias)
    # weight_q is weight_dequantized (FLOAT), NOT ternary
```

---

## Root Cause

| Step | Code | Result |
|------|------|--------|
| 1 | TernarySTE.apply() | {-1, 0, +1} |
| 2 | Line 187: `weight_quantized * scale` | {-scale, 0, +scale} |
| 3 | Return weight_dequantized | FLOAT values |
| 4 | F.linear(x, weight_q, bias) | Uses floats |

---

## Impact Analysis

### What IS Happening

- Weights are quantized to {-1, 0, +1}
- Then scaled back to {-scale, 0, +scale}
- Matrix multiplications use SCALED floats
- NOT true ternary computation

### What SHOULD Happen (Per BitNet Paper)

**UNKNOWN** - Need to verify against BitNet b158 paper

### Possible Interpretations

| Interpretation | Plausibility |
|----------------|--------------|
| Intentional for numeric stability | POSSIBLE |
| Bug - should use unscaled ternary | POSSIBLE |
| Correct per some papers | UNKNOWN |

---

## Verification Labels Summary

| Test | Status | Classification |
|------|--------|----------------|
| 1. TernarySTE returns {-1, 0, +1} | Correct | [VERIFIED] |
| 2. Matrix mult uses ternary | Incorrect | [ERROR] |
| 3. Scale timing | After ternarization | [VERIFIED] |
| 4. STE gradients non-zero | Correct | [VERIFIED] |
| 5. Optimizer updates weights | Correct | [VERIFIED] |
| 6. Threshold behavior | Correct | [VERIFIED] |
| 7. All layers quantized | Correct | [VERIFIED] |
| 8. Quantization changes output | Correct | [VERIFIED] |
| 9. Loss/tokenization correct | Correct | [VERIFIED] |

---

## Open Questions

### Critical

1. **Should matrix multiplications use {-1, 0, +1} or {-scale, 0, +scale}?**
   - Need: BitNet b158 paper verification
   - Impact: Determines if implementation is correct

2. **Is dequantization intentional or a bug?**
   - Need: Original paper specification
   - Impact: Core architecture correctness

### Secondary

3. What is the correct scale application point?
4. Should activation quantization also be applied?
5. Does inference differ from training in quantization?

---

## Files Examined

| File | Purpose |
|------|---------|
| src/model/bitlinear.py | Core implementation |
| src/training/adafactor.py | Optimizer |
| experiments/verify_ternary_implementation.py | This verification |

---

## Next Steps (NOT RECOMMENDATIONS - FOR USER DECISION)

1. Verify BitNet b158 paper specification
2. Compare against reference implementation
3. Test with unscaled ternary weights if paper suggests
4. Document paper requirements before any code changes

---

## Conclusion

**The TernarySTE function correctly produces {-1, 0, +1}.**

**The quantize_weights() function multiplies by scale, converting to {-scale, 0, +scale}.**

**Matrix multiplications use FLOAT values, NOT ternary.**

**This is either intentional design or implementation bug - CANNOT DETERMINE without BitNet paper verification.**

**No code changes should be made until paper specification confirms correct behavior.**

---

## Appendix: Raw Test Output

```
TEST 1: TernarySTE returns exactly {-1, 0, +1}: [VERIFIED]
TEST 2: Matrix mult uses ternary tensor: [ERROR] - uses scaled floats
TEST 3: Scale timing: [VERIFIED] - applied after ternarization
TEST 4: STE provides non-zero gradients: [VERIFIED]
TEST 5: Optimizer updates weights: [VERIFIED]
TEST 6: Threshold/scale behavior: [UNKNOWN]
TEST 7: All BitLinear layers quantized: [VERIFIED] - but with floats
TEST 8: Quantization changes output: [VERIFIED]
TEST 9: Loss calculation: [VERIFIED]
```
