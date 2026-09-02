# BitNet b1.58 AbsMean Quantization Verification

**Date:** August 19, 2026
**Status:** VERIFICATION COMPLETE

---

## Executive Summary

**CLASSIFICATION: VERIFIED INCORRECT**

Our implementation has a **FUNDAMENTAL ERROR in scale computation**.

---

## BitNet b1.58 Official Formula

From BitNet paper and official implementations:

### Weight Quantization (AbsMean)

```python
# Step 1: Compute scale
scale = 1.0 / mean(|W|)

# Step 2: Scale and round
W_scaled = W * scale

# Step 3: Round and clamp
W_quantized = round(W_scaled).clamp(-1, 1)
```

**Key insight:** Weights are MULTIPLIED by scale (1/mean), making small weights LARGER.

---

## Our Implementation

From `src/model/bitlinear.py`:

```python
# Line 80-94: compute_abs_mean_scale
def compute_abs_mean_scale(weight: Tensor, eps: float = 1e-8) -> Tensor:
    """scale = mean(|weight|)"""
    return weight.abs().mean().clamp(min=eps)

# Line 49: In TernarySTE.forward
weight_scaled = weight / scale.clamp(min=1e-8)

# Line 56-57: Threshold comparison
weight_quantized[weight_scaled > threshold] = 1.0
weight_quantized[weight_scaled < -threshold] = -1.0
```

---

## THE BUG

### Our Implementation:

1. `scale = mean(|W|)` ← We compute mean, not inverse
2. `weight_scaled = W / scale` ← We DIVIDE by scale
3. Result: Small weights get SMALLER

### BitNet Official:

1. `scale = 1.0 / mean(|W|)` ← BitNet uses inverse
2. `weight_scaled = W * scale` ← BitNet MULTIPLIES by scale
3. Result: Small weights get LARGER

---

## Mathematical Proof

### Example with Real Checkpoint Values:

```
Checkpoint statistics (from analysis):
  mean(|W|) = 0.065 (typical)
  std(W) = 0.082
  range: [-0.42, 0.44]
```

### Our Implementation:

```python
scale = mean(|W|) = 0.065
weight_scaled = W / 0.065

For W = 0.04 (typical):
  weight_scaled = 0.04 / 0.065 = 0.615
  Since 0.615 > 0.5, this quantizes to +1 ✓

For W = 0.02 (small):
  weight_scaled = 0.02 / 0.065 = 0.308
  Since 0.308 < 0.5, this quantizes to 0 ✓

For W = -0.03:
  weight_scaled = -0.03 / 0.065 = -0.462
  Since -0.462 > -0.5, this quantizes to 0 ✓
```

**Wait... this should work!**

### Re-analyzing...

Actually, our formula:
- `weight_scaled = W / mean(|W|)`
- Threshold at 0.5

Is equivalent to checking if `|W| > 0.5 * mean(|W|)`

### BitNet's formula:
- `weight_scaled = W * (1/mean(|W|))`
- Round to nearest integer

Is equivalent to checking if `|W| > 0.5 * mean(|W|)`

**They're THE SAME!**

---

## Verification: Are they actually the same?

### Mathematical Equivalence:

**Our approach:**
- `weight / scale` where `scale = mean(|W|)`
- Check if `weight/mean(|W|) > 0.5`
- Equivalent to: `weight > 0.5 * mean(|W|)`

**BitNet approach:**
- `weight * (1/mean(|W|))`
- Round to nearest integer
- Round rule: in (-0.5, 0.5) → 0, >= 0.5 → 1, <= -0.5 → -1
- Equivalent to: `weight/mean(|W|) > 0.5`

**CONCLUSION: THE FORMULAS ARE MATHEMATICALLY EQUIVALENT!**

---

## Then WHY are all weights zero?

### Re-examining the evidence:

Checkpoint analysis showed:
- Weight range: [-0.42, 0.44]
- Weight std: 0.082
- mean(|W|): ~0.065

### Our threshold of 0.5 is applied to SCALED weights:

```
W_scaled = W / 0.065

For this to quantize to +1:
  W / 0.065 > 0.5
  W > 0.5 * 0.065
  W > 0.0325

For this to quantize to -1:
  W / 0.065 < -0.5
  W < -0.0325
```

### So weights should quantize as:
- `W > 0.0325` → +1
- `W < -0.0325` → -1
- `-0.0325 <= W <= 0.0325` → 0

### But checkpoint shows:
- Weight range: [-0.42, 0.44]
- Most weights ARE outside [-0.0325, 0.0325]!

**Why are they all zero then?**

---

## CRITICAL REALIZATION

The checkpoint analysis showed:

```python
blocks.0.attn.q_proj.weight:
  Min: -0.371766
  Max: 0.381790
  ...
  Ternary: -1=0.00%, 0=100.00%, +1=0.00%
```

**But this was computed with threshold 0.5 on UNSCALED weights!**

The correct analysis should apply the actual quantization:

```python
# What we did:
if weight < -0.5:  # WRONG
    quantized = -1
elif weight > 0.5:  # WRONG
    quantized = 1
else:
    quantized = 0

# What should happen:
scale = mean(|weight|) = 0.065
weight_scaled = weight / 0.065
if weight_scaled < -0.5:  # CORRECT
    quantized = -1
elif weight_scaled > 0.5:  # CORRECT
    quantized = 1
else:
    quantized = 0
```

---

## Re-analyzing with CORRECT threshold

```
scale = mean(|W|) = 0.065

Weight_scaled = Weight / 0.065

For Weight = 0.38 (max from checkpoint):
  Weight_scaled = 0.38 / 0.065 = 5.85
  Since 5.85 > 0.5, quantizes to +1 ✓

For Weight = -0.37 (min from checkpoint):
  Weight_scaled = -0.37 / 0.065 = -5.69
  Since -5.69 < -0.5, quantizes to -1 ✓

For Weight = 0.01 (small):
  Weight_scaled = 0.01 / 0.065 = 0.15
  Since -0.5 < 0.15 < 0.5, quantizes to 0 ✓
```

**Expected distribution with scale=0.065:**

For weights with std=0.082 and mean(|W|)=0.065:
- ~33% should be +1
- ~33% should be -1
- ~33% should be 0

**But we see 100% zeros!**

---

## THE REAL BUG

The checkpoint analysis used **threshold=0.5 on UNSCALED weights**!

From my earlier analysis script:
```python
neg = (w < -0.5).sum().item()  # WRONG: Should use scaled weights
zero = ((w >= -0.5) & (w <= 0.5)).sum().item()
pos = (w > 0.5).sum().item()
```

**This was MY analysis bug, not the implementation bug!**

---

## Final Verification

### What the implementation does:

1. `scale = mean(|W|)` ✓
2. `W_scaled = W / scale` ✓
3. Compare to threshold=0.5 ✓

### What my analysis did:

1. Take raw weights W
2. Directly compare to threshold=0.5 ✗ (should divide by scale first)

---

## CLASSIFICATION: VERIFIED CORRECT (Implementation), VERIFIED INCORRECT (My Analysis)

**The BitLinear implementation is CORRECT.**

**My checkpoint analysis was WRONG.**

---

## Correct Analysis (Needs Verification)

To correctly analyze ternary distribution:

```python
# For each layer:
W = layer.weight.data
scale = W.abs().mean()
W_scaled = W / scale

neg = (W_scaled < -0.5).sum().item()
zero = ((W_scaled >= -0.5) & (W_scaled <= 0.5)).sum().item()
pos = (W_scaled > 0.5).sum().item()
total = W.numel()
```

**This will show the TRUE ternary distribution.**

---

## Recommendation

Re-run checkpoint analysis with CORRECT scale application:

1. For each checkpoint
2. For each BitLinear layer
3. Compute `scale = mean(|W|)`
4. Compute `W_scaled = W / scale`
5. Apply threshold on SCALED weights
6. Record distribution

This will reveal if the implementation is actually working correctly.

---

## Current Status

| Item | Status |
|------|--------|
| BitNet formula | Documented correctly |
| Implementation math | MATHEMATICALLY SOUND |
| My analysis | INCORRECT (used unscaled weights) |
| Actual ternary distribution | UNKNOWN (needs re-analysis) |

---

## Next Step

Re-run checkpoint analysis with corrected threshold application.
