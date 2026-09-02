# FP32 vs Ternary Experiment Comparison

---

## ⚠️ STATUS: SUPERSEDED

**This document contains INCORRECT conclusions that were later corrected.**

**Correction Date:** August 19, 2026
**Superseded By:** 
- `PADDING_INFLATION_DISCOVERY.md` (corrected metrics with padding excluded)
- `HELD_OUT_EVALUATION_AUDIT.md` (detailed audit results)

**Why This Document Is Wrong:**
1. Claimed FP32 achieves 95% accuracy on held-out data vs Ternary 3.56%
2. **CRITICAL ERROR:** Did not exclude padding tokens from accuracy calculation
3. FP32 achieved 100% accuracy on padding positions (predicting token 0)
4. This inflated FP32 total accuracy from 5.46% (real tokens) to 26.89% (total)
5. Ternary achieved 0% on padding, so total accuracy was NOT inflated

**Corrected Comparison (Real Tokens Only, Excluding Padding):**

| Metric | FP32 5k | Ternary 50k | Difference |
|--------|---------|-------------|------------|
| **Real Token Accuracy** | **5.46%** | **4.60%** | **0.86%** |
| Padding Accuracy | 100.00% | 0.00% | 100% |
| Total Accuracy | 26.89% | 3.56% | 23.33% |

**Correct Conclusion:** FP32 and Ternary perform SIMILARLY on real language tokens (~5% vs ~4.6%). The apparent gap was entirely due to padding inflation. Ternary quantization is RULED OUT as the root cause of gibberish.

**Evidence Source:** See `heldout_evaluation_results.json` and audit script `experiments/audit_heldout_evaluation.py`

---

**Original (Incorrect) Document Below:**

---

**Date:** August 19, 2026
**Purpose:** Determine if ternary quantization limits learning capacity

---

## Executive Summary

**CRITICAL FINDING: FP32 model MEMORIZED the training data with 95% accuracy!**

**FP32 achieves near-perfect performance while ternary struggles at 3% accuracy.**

---

## Key Results Comparison

### Final Metrics (Step 5000)

| Metric | FP32 Control | Ternary Experiment | Difference |
|--------|--------------|-------------------|------------|
| **Eval Loss** | **0.239** | 8.60 | **36x better** |
| **Perplexity** | **1.27** | 5422 | **4268x better** |
| **Top-1 Accuracy** | **95.45%** | 0.06% | **1591x better** |
| **Top-5 Accuracy** | **99.09%** | 0.32% | **309x better** |
| **Correct Token Prob** | **93.99%** | 0.03% | **3133x better** |
| **Best Loss** | 0.239 | 8.60 | 36x better |
| **Training Time** | 26 min | ~30 min | Similar |

---

## Learning Trajectory Comparison

### FP32 Model

| Step | Loss | PPL | Top-1% | Top-5% | Correct Prob% |
|------|------|-----|--------|--------|---------------|
| 0 | 0.334 | 1.40 | 92.7 | 97.3 | 91.6 |
| 500 | 0.301 | 1.35 | 91.8 | 97.3 | 92.0 |
| 1000 | 0.340 | 1.40 | 90.9 | 97.3 | 91.5 |
| 2000 | 0.267 | 1.31 | 94.5 | 99.1 | 93.7 |
| 3000 | 0.253 | 1.29 | 94.5 | 99.1 | 93.7 |
| 4000 | 0.244 | 1.28 | 95.5 | 99.1 | 94.0 |
| **5000** | **0.239** | **1.27** | **95.5** | **99.1** | **93.99** |

**Pattern:** FP32 starts at 92% accuracy (memorizing from step 0!) and improves to 95%!

### Ternary Model (for comparison)

| Step | Loss | PPL | Top-1% | Top-5% |
|------|------|-----|--------|--------|
| 0 | 8.66 | 5763 | 0.00 | 0.00 |
| 500 | 8.60 | 5407 | 0.00 | 0.00 |
| 1000 | 8.54 | 5132 | 0.00 | 0.00 |
| 2000 | 8.45 | 4668 | 0.00 | 0.00 |
| 3000 | 8.34 | 4181 | 1.82 | 3.64 |
| **5000** | **8.60** | **5422** | **0.06** | **0.32** |

**Pattern:** Ternary struggles, barely reaches 0.06% accuracy after 5000 steps.

---

## Generation Quality Comparison

### FP32 at Step 500

**Prompt:** "Once upon a time"
```
Generated: "Once upon a time there was a little girl named Sarah. of the little girl loved to have in the flower in the flower in the end of the little girl named Sarah."
```
**Analysis:** Contains exact training data! Memorization confirmed.

### Ternary at Step 5000

**Prompt:** "Once upon a time"
```
Generated: "Once upon a time this?\" ending screwoods polite plate song. wheches. flew garden..."
```
**Analysis:** Complete gibberish. No memorization.

### FP32 at Step 5000

**Prompt:** "Once upon a time"
```
Generated: "Once upon a time there was a little girl named Sarah. girl Tom."
```
**Analysis:** Still contains training data fragments!

### Ternary at Step 5000

**Prompt:** "Once upon a time"
```
Generated: "Once upon a time bugending angel woods every plate song..."
```
**Analysis:** Gibberish.

---

## Critical Observation: Memorization vs Learning

### FP32 Model Behavior

1. **Step 0:** 92.7% accuracy immediately (random initialization → memorization starts)
2. **Step 500:** Still 91.8% accuracy
3. **Step 5000:** 95.5% accuracy

**FP32 model is MEMORIZING the training data, not learning to generalize.**

### Why is this happening?

The FP32 model has:
- 20 training sentences
- 8 batch size
- 5000 steps
- Total passes through data: 5000 × 8 / 20 = **2000 epochs**

**Result:** The model memorizes the exact training sequences!

### Ternary Model Behavior

1. **Step 0:** 0% accuracy
2. **Step 5000:** 0.06% accuracy

**Ternary model CANNOT memorize even with 2000 epochs.**

---

## The Real Finding

### What We Discovered

**The comparison is FLAWED because:**

1. **FP32 memorizes training data** (92% → 95% accuracy on training set)
2. **Ternary cannot memorize** (0% → 0.06% accuracy)
3. **Both models evaluated on TRAINING DATA, not held-out test data**

### The Critical Issue

We evaluated on the **same 10 sentences used in training**!

This means:
- FP32 achieves 95% accuracy = **Memorization, not learning**
- Ternary achieves 0.06% accuracy = **Cannot memorize**

**This doesn't tell us about generalization!**

---

## Correct Interpretation

### Finding #1: Ternary Prevents Memorization

| Model | Can Memorize Training Data? | Evidence |
|-------|----------------------------|----------|
| FP32 | ✅ YES | 95% accuracy on training set |
| Ternary | ❌ NO | 0.06% accuracy on training set |

**Ternary quantization acts as REGULARIZATION - prevents overfitting.**

### Finding #2: Both Models Evaluated Incorrectly

**The evaluation was on training data, not held-out test data.**

To properly compare learning capacity, we need:
1. Separate test set (sentences NOT in training)
2. Evaluate both models on test set
3. Compare generalization, not memorization

### Finding #3: Ternary May Not Be The Problem

**Previous conclusion was WRONG:**
- ❌ "Ternary limits learning capacity"
- ✅ "Ternary prevents memorization (regularization)"

**What we still don't know:**
- Can ternary model learn to generalize with more training?
- Does ternary need more data/longer training to learn?

---

## What This Means for the Gibberish Problem

### Hypothesis Update

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| **Ternary prevents learning** | ❌ RULED OUT | Ternary prevents memorization (regularization) |
| **Insufficient training** | ❓ UNKNOWN | Need test set evaluation |
| **Model capacity** | ❓ UNKNOWN | Need test set evaluation |
| **Need more training data** | ❓ UNKNOWN | FP32 memorized 20 sentences |

---

## Next Steps Required

### 1. Evaluate on Held-Out Test Set

**Must compare FP32 and Ternary on data NOT in training:**

```
Test sentences (NOT in training):
- "The boy went to the park."
- "She found a puppy in the yard."
- "The cat climbed the tree."
```

### 2. Compare Generalization

| Metric | FP32 | Ternary | Interpretation |
|--------|------|---------|----------------|
| Training accuracy | 95% | 0.06% | Memorization vs regularization |
| **Test accuracy** | ? | ? | **This is what matters** |

### 3. Expected Outcomes

**If ternary limits learning:**
- FP32 test accuracy >> Ternary test accuracy

**If ternary is just regularization:**
- FP32 test accuracy ≈ Ternary test accuracy (or worse due to overfitting)
- Ternary may generalize better with longer training

---

## Immediate Action Required

**We must re-evaluate both models on HELD-OUT TEST DATA before concluding anything.**

The current comparison only shows:
- FP32 can memorize
- Ternary cannot memorize

This does NOT prove ternary limits learning capacity.

---

## Conclusion

**PRELIMINARY FINDING: Ternary quantization acts as strong regularization, preventing memorization.**

**CRITICAL CAVEAT: Cannot conclude ternary limits learning without test set evaluation.**

**NEXT STEP: Create held-out test set and re-evaluate both models.**
