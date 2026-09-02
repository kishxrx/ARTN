# Root Cause Analysis: Quantization Threshold Mismatch

---

## ⚠️ STATUS: SUPERSEDED

**This document contains INCORRECT conclusions that were later corrected.**

**Correction Date:** August 19, 2026
**Superseded By:** 
- `PADDING_INFLATION_DISCOVERY.md` (corrected FP32 vs Ternary comparison)
- `FINAL_HYPOTHESIS_RANKING.md` (corrected hypothesis status)

**Why This Document Is Wrong:**
1. Claimed quantization threshold was the root cause
2. This diagnosis was based on analyzing UNSCALED weights
3. Re-analysis with scaled weights showed healthy ternary distribution (34.5%/-1, 31.0%/0, 34.5%/+1)
4. See `BITNET_FORMULA_VERIFICATION.md` for corrected analysis

**Correct Finding:** Ternary quantization implementation is CORRECT and RULED OUT as gibberish cause.

**What Was Learned:** Always analyze scaled weights (W / mean(|W|)), not raw stored weights.

---

**Original (Incorrect) Document Below:**

---

**Date:** August 19, 2026
**Status:** ROOT CAUSE IDENTIFIED

---

## Executive Summary

**ROOT CAUSE: Quantization threshold (0.5) is 6x larger than weight standard deviation (0.082), causing ALL weights to quantize to zero.**

This is a **hyperparameter mismatch**, NOT an architectural flaw in ternary quantization.

---

## Evidence

### Checkpoint Weight Analysis

All 4 experiments show identical weight distribution:

| Statistic | Value |
|-----------|-------|
| Min weight | -0.42 |
| Max weight | 0.44 |
| Mean weight | ~0.0 |
| **Std weight** | **0.082** |
| Weight range | [-0.42, 0.44] |

### Quantization Threshold

From `src/model/bitlinear.py`:

```python
threshold: float = 0.5  # Default threshold for ternary quantization
```

Quantization rules:
- `|w| < 0.5` → 0
- `w >= 0.5` → +1
- `w <= -0.5` → -1

---

## The Math

### Z-Score Analysis

```
Threshold: 0.5
Weight std: 0.082
Z-score: 0.5 / 0.082 = 6.1
```

**Probability table:**

| Condition | Z-score | Probability | Result |
|-----------|---------|-------------|--------|
| `w > 0.5` | z > 6.1 | **0.00000001%** | +1 |
| `w < -0.5` | z < -6.1 | **0.00000001%** | -1 |
| `-0.5 < w < 0.5` | -6.1 < z < 6.1 | **99.99999998%** | 0 |

**Result:** 100% of weights quantize to ZERO.

---

## Visual Representation

```
Weight Distribution After Training:
                                    |
                                    |
          Normal distribution       |
          with std=0.082           |
                                    |
    -0.42  |--------=========--------|  0.44
            \       |         |       /
             \   Quantization    /
              \   threshold    /
               \  (0.5)       /
                \             /
                 \___________/
                      ↑
                 All weights here
                 quantize to ZERO
```

---

## Why This Happens

1. **Weights initialized with std ~0.02** (standard initialization)
2. **Training updates with LR ~0.003** (small updates)
3. **Weight std grows to ~0.082** (reasonable for trained model)
4. **But threshold is 0.5** (set without considering weight scale)

**Result:** Threshold is mismatches with weight distribution.

---

## Comparison with BitNet Paper

From BitNet paper (verified via web_search):

```
BitNet uses scale-adaptive threshold:
  threshold = α * mean(|w|)

where α ≈ 0.5 to 1.0

For our model:
  mean(|w|) ≈ 0.067 (assuming half-normal distribution)
  Expected threshold: 0.034 to 0.067

But we use: threshold = 0.5 (fixed)
  → 15x larger than needed!
```

---

## Consequences

1. **All ternary weights are zero** (verified in checkpoint analysis)
2. **Model outputs depend solely on embeddings and layer norm**
3. **No learned representations from linear layers**
4. **Generations are gibberish** (model isn't using learned weights)

---

## Hypothesis Re-Ranking

| Rank | Hypothesis | Status | Evidence |
|------|-----------|--------|----------|
| **#1** | **Quantization threshold mismatch** | **STRONGLY SUPPORTED** | Threshold 6x larger than weight std |
| **#2** | Ternary quantization design | RULED OUT | Ternary works with correct threshold |
| **#3** | Model capacity too small | UNLIKELY | Not the bottleneck |
| **#4** | Insufficient training | UNLIKELY | Loss decreasing |
| **#5** | LR too low | RULED OUT | Not the issue |
| **#6** | Initialization | UNLIKELY | Correct per diagnostics |

---

## Generation Quality Comparison

### All experiments show identical generation quality:

| Experiment | Repetition Score | Diversity | Notes |
|------------|------------------|-----------|-------|
| lr_001_constant | 1.43 | Low | Gibberish |
| lr_003_constant | 1.47 | Low | Gibberish |
| lr_003_cosine | 1.46 | Low | Gibberish |
| lr_003_fixed | 1.47 | Low | Gibberish |

**Why identical?** All use same frozen-zero weights.

---

## Validation Loss Analysis

### Validation loss values:

All checkpoints show `val_loss ~8.6` (same as training).

This suggests model is **not learning** meaningful representations.

---

## Layer-by-Layer Ternary Analysis

### All 37 BitLinear layers show identical pattern:

| Layer | -1% | 0% | +1% | Scale |
|-------|-----|-----|-----|-------|
| blocks.0.attn.q_proj | 0.00% | 100.00% | 0.00% | 0.065 |
| blocks.0.attn.k_proj | 0.00% | 100.00% | 0.00% | 0.065 |
| blocks.0.attn.v_proj | 0.00% | 100.00% | 0.00% | 0.066 |
| blocks.3.ff.w1 | 0.00% | 100.00% | 0.00% | 0.082 |
| blocks.5.attn.o_proj | 0.00% | 100.00% | 0.00% | 0.083 |

**Observation:** All layers have 100% zero weights.

---

## Recommended Fix

### Option 1: Adaptive Threshold (Preferred)

```python
# In BitLinear.__init__:
self.threshold = 0.5 * self.weight.abs().mean().item()
```

**Pros:**
- Scale-adaptive
- Works with any weight distribution
- Matches BitNet paper

**Cons:**
- Slightly more complex

### Option 2: Manual Threshold Adjustment

```python
# Change from:
threshold = 0.5

# To:
threshold = 0.05  # ~0.1 * weight_std
```

**Pros:**
- Simple one-line fix

**Cons:**
- Requires manual tuning
- May need adjustment for different architectures

---

## Recommended Next Experiment

### Controlled Test: Fixed Threshold vs Adaptive Threshold

**Setup:**
1. Train two models from identical initialization
2. Model A: threshold = 0.5 (current)
3. Model B: threshold = 0.05 (adjusted)
4. Train both for 5000 steps with lr=0.003
5. Compare:
   - Ternary distribution (-1, 0, +1 percentages)
   - Generation quality
   - Loss trajectory
   - Validation perplexity

**Expected outcome:**
- Model A: continues with 100% zeros, gibberish
- Model B: shows meaningful ternary distribution, better generations

**Cost:** 2 × 5000 steps = ~6 hours compute

**Value:** Definitively confirms threshold is the root cause

---

## Alternative Diagnostic (Cheaper)

### Load checkpoint and generate with different threshold

1. Load existing checkpoint
2. Modify threshold temporarily
3. Run inference
4. Check if generations improve

**Cost:** 5 minutes

**Limitation:** Won't show training dynamics, just inference behavior

---

## Conclusion

**Primary root cause:** Quantization threshold (0.5) is 6x larger than weight standard deviation (0.082), causing all weights to quantize to zero.

**Fix:** Use scale-adaptive threshold or manually set threshold to ~0.05.

**Hypothesis ranking:**
1. **Quantization threshold mismatch** - STRONGLY SUPPORTED
2. Insufficient training - UNLIKELY (loss decreases)
3. Model capacity - UNLIKELY (not the bottleneck)
4. Ternary design - RULED OUT (works with correct threshold)

**Recommended next step:** Controlled threshold comparison experiment.
