# Comprehensive Checkpoint Evaluation Analysis

---

## ⚠️ STATUS: SUPERSEDED

**This document contains conclusions that were later corrected.**

**Correction Date:** August 19, 2026
**Superseded By:**
- `FORWARD_PASS_VERIFICATION_WITH_EVIDENCE.md` (corrected evaluation methodology)
- `PADDING_INFLATION_DISCOVERY.md` (corrected accuracy metrics)

**Why This Document Is Wrong:**
1. Evaluated checkpoints using RANDOM token sequences, not real TinyStories text
2. Random tokens are nonsensical - the model never trained on such sequences
3. Reported ~0% accuracy across all checkpoints
4. Re-evaluation with REAL text showed the model CAN achieve meaningful accuracy

**Corrected Finding:**
- On REAL TinyStories text: 50k model achieves ~3.64% Top-1 accuracy (not 0.06%)
- Model IS learning and improving over time
- See `FORWARD_PASS_VERIFICATION_WITH_EVIDENCE.md` for corrected evaluation

**What Was Learned:** Always evaluate language models on real text from the training distribution, not random tokens.

---

**Original Document Below:**

---

**Date:** August 19, 2026
**Evaluation:** All available checkpoints from 500 steps to 50,000 steps

---

## Executive Summary

**KEY FINDING: CASE B - Poor next-token prediction throughout all training**

The 50,000-step model shows **NO meaningful improvement** in next-token prediction accuracy compared to early checkpoints.

---

## Results Table

| Checkpoint | Steps | Top-1 Acc | Top-5 Acc | Loss | Perplexity | Diversity |
|------------|-------|-----------|-----------|------|------------|-----------|
| step_500 | 500 | 0.00% | 0.26% | 8.73 | 6173 | 1.000 |
| step_1000 | 1000 | 0.00% | 0.26% | 8.72 | 6121 | 1.000 |
| step_1500 | 1500 | 0.00% | 0.26% | 8.71 | 6073 | 1.000 |
| step_2000 | 2000 | 0.00% | 0.26% | 8.70 | 6027 | 1.000 |
| step_2500 | 2500 | 0.00% | 0.26% | 8.70 | 5990 | 1.000 |
| step_3000 | 3000 | 0.00% | 0.26% | 8.69 | 5962 | 1.000 |
| lr_001_constant_5k | 5000 | 0.06% | 0.26% | 8.68 | 5908 | 1.000 |
| lr_003_constant_5k | 5000 | 0.06% | 0.32% | 8.60 | 5422 | 0.989 |
| **step_50k** | **50000** | **0.06%** | **0.26%** | **8.50** | **4898** | **0.990** |

---

## Critical Observations

### 1. Top-1 Accuracy: Essentially Zero

| Training Duration | Top-1 Accuracy | Expected (Random Model) |
|-------------------|----------------|--------------------------|
| 500 steps | 0.00% | 0.025% (1/4000) |
| 5000 steps | 0.06% | 0.025% |
| **50000 steps** | **0.06%** | 0.025% |

**Result:** 50k-step model performs at **RANDOM BASELINE**.

### 2. Perplexity Improvement

| Training | Perplexity | Improvement |
|----------|------------|-------------|
| 500 steps | 6173 | - |
| 5000 steps | 5908 | ↓ 4.3% |
| 50000 steps | 4898 | ↓ 20.7% |

**Result:** Loss decreases significantly, but **accuracy doesn't improve**.

### 3. Generation Quality: All Gibberish

All checkpoints from 500 to 50,000 steps produce **identical gibberish**:

```
Step 500:   "Once upon a time this?\" ending 't rain. pieces..."
Step 5000:  "Once upon a time this?\" ending screwoods polite..."
Step 50000: "Once upon a time bugending angel woods every..."
```

**All generations:**
- No coherent structure
- Random word sequences
- Diversity = 1.0 (all unique words, no repetition)
- No meaningful content

---

## Loss vs Accuracy Paradox

### What's Happening?

| Metric | Step 500 | Step 50k | Change |
|--------|----------|----------|--------|
| Training Loss | 8.62 | ~8.0 (estimated) | ↓ Decreasing |
| Eval Loss | 8.73 | 8.50 | ↓ Decreasing |
| Perplexity | 6173 | 4898 | ↓ 21% |
| Top-1 Accuracy | 0.00% | 0.06% | → Flat |

**Paradox:** Loss decreases 21%, but accuracy stays at random baseline.

---

## Diagnosis

### Case B: Poor Next-Token Prediction Throughout

**Evidence:**
1. Top-1 accuracy at random baseline (0.06% vs 0.025%)
2. Top-5 accuracy near random baseline (0.26% vs 0.125%)
3. 50,000 steps of training produces no accuracy improvement
4. All generations are gibberish

**Conclusion:** Fundamental training/architecture issue.

---

## Root Cause Analysis

### What is NOT the Problem

| Hypothesis | Evidence | Status |
|------------|----------|--------|
| Ternary quantization | Verified correct (34.31/34 ternary distribution) | ✅ RULED OUT |
| Quantization threshold | Verified correct (scales properly) | ✅ RULED OUT |
| Insufficient training | 50k steps shows no improvement | ✅ RULED OUT |
| LR too low | Tested multiple LRs, same result | ✅ RULED OUT |
| Generator/decoding bug | Same evaluation procedure used | UNLIKELY |
| Data/preprocessing | Need investigation | UNKNOWN |

### What Could Be the Problem

| Hypothesis | Evidence | Rank |
|------------|----------|------|
| **Model architecture** | Loss decreases but accuracy stays flat | #1 POSSIBLE |
| **Training objective mismatch** | Optimizing wrong thing | #2 POSSIBLE |
| **Token distribution issue** | Vocab imbalance or wrong targets | #3 POSSIBLE |
| **Forward pass bug** | Model not learning representations | #4 POSSIBLE |
| **Training data quality** | Real vs synthetic data | #5 UNKNOWN |

---

## Detailed Analysis

### 1. Model Architecture Investigation Needed

**Observation:** Loss decreases but accuracy doesn't improve.

**Possible causes:**
- Model learning wrong representations
- Attention not attending properly
- Residual connections not working
- Layer architecture issue

**Test needed:** Verify forward pass produces correct attention patterns.

### 2. Training Objective Mismatch

**Observation:** The model is trained to minimize loss, but not learning to predict correctly.

**Possible causes:**
- Loss computed on wrong targets
- Target shifting incorrect (should predict next token, not current)
- Padding tokens causing issues

**Test needed:** Verify target tensor construction.

### 3. Token Distribution Issue

**Observation:** Accuracy at random baseline.

**Possible causes:**
- Vocab imbalance (some tokens dominate)
- Special tokens treated incorrectly
- Tokenizer encoding/decoding mismatch

**Test needed:** Analyze token distribution in training data.

### 4. Forward Pass Bug

**Observation:** Consistently poor predictions despite training.

**Possible causes:**
- Embeddings not being used
- Attention heads all zero
- Output projection incorrect

**Test needed:** Trace forward pass and inspect intermediate activations.

---

## Hypothesis Ranking

| Rank | Hypothesis | Status | Evidence |
|------|-----------|--------|----------|
| **#1** | **Forward pass/attention bug** | STRONGLY SUSPECTED | Loss ↓ but accuracy = random |
| **#2** | **Training target construction** | POSSIBLE | Predictions at baseline |
| **#3** | **Model architecture issue** | POSSIBLE | 50k steps no improvement |
| **#4** | Token distribution imbalance | UNKNOWN | Need analysis |
| ~~Quantization~~ | RULED OUT | Verified correct |
| ~~Insufficient training~~ | RULED OUT | 50k steps tested |
| ~~LR/optimizer~~ | RULED OUT | Multiple configs tested |

---

## Highest-Value Next Diagnostic

### Trace Forward Pass and Attention

**Goal:** Verify model is actually learning representations.

**Procedure:**
```python
1. Load 50k checkpoint
2. Run single forward pass on known input
3. Inspect:
   - Embedding outputs (are they non-zero?)
   - Attention weights (patterns meaningful?)
   - Attention outputs (non-zero?)
   - FFN outputs (non-zero?)
   - Logits distribution (uniform or peaked?)
   - Predicted tokens (consistent with loss?)
```

**Cost:** 5 minutes
**Value:** Will definitively show if model is broken or just poorly trained

**Expected outcome:**
- If attention/FFN all zeros → Forward pass bug
- If logits uniform → Model not learning
- If attention patterns random → Initialization issue

---

## Alternative Diagnostic (Lower Priority)

### Verify Target Construction

**Goal:** Check if targets are correct.

**Procedure:**
```python
1. Load training data batch
2. Print input tokens
3. Print target tokens
4. Verify: target[i] = input[i+1]
5. Check for padding tokens
```

**Cost:** 2 minutes
**Value:** Rules out target construction bug

---

## Conclusion

**KEY FINDING:** The 50,000-step model performs at **random baseline** for next-token prediction.

**Case diagnosis:** **CASE B - Fundamental training/architecture issue**

**The model is minimizing loss but NOT learning to predict correctly.**

**Top hypothesis:** Forward pass bug (attention not working, embeddings not used, or output projection broken).

**Highest-value next step:** Trace forward pass and inspect attention patterns.
