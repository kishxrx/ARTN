# LR Experiment Results Report

---

## ⚠️ STATUS: SUPERSEDED

**This document contains conclusions that were later corrected.**

**Correction Date:** August 19, 2026
**Superseded By:**
- `ROOT_CAUSE_ANALYSIS.md` (logging bug identification)
- `VERIFICATION_ANALYSIS.md` (corrected results with fixed logging)

**Why This Document Is Wrong:**
1. **Loss values:** Shows loss ~4300, which was 500x inflated due to logging bug
2. **Zero gradients:** Reports gradients = 0.0 at every step, which was logged incorrectly (after zero_grad())
3. **90% zeros in ternary:** Based on unscaled weights, not corrected analysis

**What Was Fixed:**
- Correct loss: ~8.7 (not 4300) - See `ROOT_CAUSE_ANALYSIS.md`
- Correct gradients: ~0.1-0.4 (not 0.0) - Gradients were healthy, just logged at wrong time
- Ternary distribution: Healthy 34.5%/31.0%/34.5% when analyzing scaled weights

**What Remains Valid:**
- Finding that changing LR does not fix gibberish generation
- All 3 LR configs produced similar gibberish quality

**Evidence Source:** See `VERIFICATION_ANALYSIS.md` for corrected results with fixed logging.

---

**Original Document Below:**

---

**Date:** August 19, 2026
**Experiments:** 3 configurations, 5000 steps each
**Total Compute Time:** ~10 hours

---

## Executive Summary

**CRITICAL ISSUE DISCOVERED: Loss calculation is catastrophically wrong.**

All three experiments show:
- Loss ~4300 (should be ~5-10 for language modeling)
- Gradients = 0.0 at every step
- Parameters still updating (update_magnitude > 0)
- Output is gibberish

---

## Results Comparison

| Metric | lr_001_constant | lr_003_constant | lr_003_cosine |
|--------|-----------------|-----------------|---------------|
| Initial Loss | 4368.79 | 4365.84 | 4365.85 |
| Final Loss | 4343.53 | 4300.37 | 4310.00 |
| Best Loss | 4343.53 | 4300.37 | 4310.00 |
| Reduction | 0.58% | 1.50% | 1.28% |
| Avg Gradient Norm | **0.000000** | **0.000000** | **0.000000** |
| Avg Update Magnitude | 0.34 | 1.00 | 0.90 |
| Time (hours) | 3.3 | 3.3 | 3.3 |

---

## Critical Anomaly: Loss Value

**Expected loss for language modeling:** 5-10

**Actual loss:** ~4300

**Ratio:** 400-800x higher than normal

**This means:**
- Perplexity = exp(4300) = astronomically high
- Model is NOT learning properly
- Something is fundamentally broken

---

## Critical Anomaly: Zero Gradients

| Experiment | Gradient Norm |
|------------|---------------|
| lr_001_constant | 0.0 at every step |
| lr_003_constant | 0.0 at every step |
| lr_003_cosine | 0.0 at every step |

**But update_magnitude > 0?**

This suggests:
1. Parameters ARE being updated
2. But gradient computation is wrong
3. Or gradient norm calculation is broken

---

## Ternary Distribution

| Experiment | -1 | 0 | +1 |
|------------|-----|-----|-----|
| lr_001_constant | 4.85% | 90.25% | 4.89% |
| lr_003_constant | 4.71% | 90.55% | 4.74% |
| lr_003_cosine | 4.75% | 90.47% | 4.79% |

**Expected:** ~33% each

**Actual:** 90% zeros

**Interpretation:** Weights appear mostly zero, suggesting minimal learning

---

## Generation Quality

All experiments produce gibberish:

```
Prompt: Once upon a time
Output: Once upon a time slowly Billy everywhere arrived, brush room ance. fallpenny...
```

```
Prompt: The little girl
Output: The little girl standing doll kitchtook Enedyou ama docolorful it,...
```

**No coherent structure or grammar.**

---

## Hypothesis Evaluation

### H1: LR Too Low

**Status: UNKNOWN (blocked by implementation bug)**

Evidence:
- lr_003_constant: 1.50% reduction
- lr_001_constant: 0.58% reduction
- Higher LR shows slightly more reduction

BUT: 
- All gradients = 0
- Loss catastrophically wrong
- Cannot evaluate until bug fixed

---

### H2: Schedule Wrong

**Status: UNKNOWN (blocked by implementation bug)**

Evidence:
- lr_003_constant: 1.50% reduction
- lr_003_cosine: 1.28% reduction

Constant slightly better BUT:
- Difference negligible
- Both have zero gradients
- Cannot evaluate until bug fixed

---

### H3-H6: Other Hypotheses

**Status: UNKNOWN**

Cannot evaluate until loss calculation is fixed.

---

## Root Cause Analysis

### Primary Issue: Loss Calculation

**Evidence:**
1. Loss ~4300 (400x normal)
2. Gradients = 0.0
3. Parameters still updating
4. Output is gibberish

**Likely causes:**
1. **Loss not being divided by batch size or sequence length**
2. **CrossEntropy receiving wrong input shape**
3. **Accumulation steps scaling broken**

### Investigating the Code

From `lr_experiment.py`:
```python
loss = nn.CrossEntropyLoss()(
    logits.view(-1, MODEL_CONFIG["vocab_size"]),
    target_ids.view(-1)
)
```

**Problem:** `vocab_size` is 4000, but if logit dimension doesn't match:
- CrossEntropy might be computing wrong value
- Need to verify actual logit shape

---

## Immediate Next Steps

**DO NOT run more training experiments until fixed.**

**Priority:**

1. **Fix loss calculation**
   - Verify logit shape before CrossEntropy
   - Check if vocab_size matches
   - Verify target_ids shape

2. **Fix gradient computation**
   - Ensure loss.backward() is called
   - Check optimizer.zero_grad() timing

3. **Verify accumulation_steps**
   - Currently = 2
   - Check if loss is being scaled/unscaled correctly

---

## Evidence-Based Conclusions

| Question | Answer | Evidence |
|----------|--------|----------|
| Is LR the problem? | **UNKNOWN** | Blocked by loss bug |
| Is schedule the problem? | **UNKNOWN** | Blocked by loss bug |
| Is implementation broken? | **YES** | Loss ~4300, gradients = 0 |
| Can we trust these results? | **NO** | Fundamental bug |

---

## Comparison Against 50k Training

| Metric | 50k Training | These Experiments |
|--------|-------------|-------------------|
| Loss | 7.56 | 4300 |
| Gradient Norm | Not recorded | 0.0 |
| Reduction | 13.6% | 0.6-1.5% |

**The 50k training showed loss ~8, which is normal.**

**These experiments show loss ~4300, which is impossible.**

**Conclusion:** The experiment script has a bug NOT present in original training scripts.

---

## What's Different?

Original training script (`train_optimized.py`):
- Uses real TinyStories dataset
- Loss computed correctly (~8)

This experiment script (`lr_experiment.py`):
- Uses synthetic random dataset
- Loss computed incorrectly (~4300)

**Hypothesis:** Dataset or data loading is causing the issue.

---

## Recommended Investigation

1. Compare `lr_experiment.py` vs `train_optimized.py`
2. Check if synthetic dataset is producing valid tokens
3. Verify CrossEntropy input shapes match
4. Add debug prints for loss calculation

---

## Conclusion

**These experiments are invalid due to implementation bug.**

**Do NOT draw conclusions about LR or schedule.**

**Fix the bug first, then re-run.**
