# Verification Run Analysis

**Date:** August 19, 2026
**Config:** lr_003_constant (LR=0.003, constant schedule)
**Steps:** 5000
**Time:** 3.3 hours

---

## Key Results

| Metric | Value | Status |
|--------|-------|--------|
| **Initial Loss** | 8.73 | ✅ NORMAL |
| **Final Loss** | 8.60 | ✅ NORMAL |
| **Best Loss** | 8.60 | ✅ NORMAL |
| **Loss Reduction** | 1.50% | ✅ LEARNING |
| **Initial Gradient Norm** | 0.1145 | ✅ HEALTHY |
| **Final Gradient Norm** | 0.0981 | ✅ HEALTHY |
| **Initial Perplexity** | 6196 | ✅ REASONABLE |
| **Final Perplexity** | 5436 | ✅ IMPROVING |

---

## Comparison: Fixed vs Buggy Runs

| Metric | Buggy Run | Fixed Run | Correct? |
|--------|-----------|-----------|----------|
| Absolute Loss | 4368 | 8.73 | ✅ Fixed (was 500x inflated) |
| Gradient Norm | 0.000000 | 0.1145 | ✅ Fixed (was logged after zero_grad) |
| Perplexity | 9999.0 | 6196 | ✅ Fixed (now reasonable) |
| Loss Reduction % | 0.58% | 1.50% | ✅ Valid (percentages were correct) |

---

## Loss Trajectory

```
Step   500: 8.7317 (PPL: 6196)
Step  1000: 8.7143 (PPL: 6089)
Step  1500: 8.6978 (PPL: 5990)
Step  2000: 8.6820 (PPL: 5896)
Step  2500: 8.6669 (PPL: 5807)
Step  3000: 8.6525 (PPL: 5724)
Step  3500: 8.6387 (PPL: 5646)
Step  4000: 8.6255 (PPL: 5572)
Step  4500: 8.6129 (PPL: 5502)
Step  5000: 8.6007 (PPL: 5436)
```

**Observation:** Steady decrease throughout training (no plateau).

---

## Gradient Analysis

| Step | Gradient Norm | Status |
|------|---------------|--------|
| 500 | 0.1145 | ✅ Healthy |
| 2500 | 0.1036 | ✅ Healthy |
| 5000 | 0.0981 | ✅ Healthy |

**Observation:** Gradients stable, not vanishing or exploding.

---

## Ternary Weight Distribution

| Step | -1% | 0% | +1% |
|------|-----|-----|-----|
| 500 | 4.90% | 90.16% | 4.94% |
| 2500 | 4.82% | 90.33% | 4.86% |
| 5000 | 4.71% | 90.55% | 4.74% |

**Observation:** Slight reduction in ±1 weights, slight increase in zeros.

---

## Generation Samples

All generations are **gibberish** (same as before):

```
Prompt: Once upon a time
Output: Once upon a time slowly youneverywhere arrived, brush room ance...
```

**Observation:** No improvement in generation quality despite loss decrease.

---

## Hypothesis Evaluation

### H1: LR Too Low

**Status: STRONGLY SUPPORTED**

Evidence:
- Initial loss: 8.73 (normal for language modeling)
- Loss decreased 1.50% over 5000 steps (learning)
- Gradient norms healthy (0.10 average)
- Perplexity improving (6196 → 5436)

Comparison with lr_001_constant:
- lr_001 showed 0.58% reduction
- lr_003 showed 1.50% reduction
- lr_003 performs **2.6x better** in loss reduction

**Conclusion:** LR=0.003 is more effective than LR=0.001 for this model/data.

---

### H2: Schedule Wrong

**Status: UNKNOWN**

Need to compare lr_003_constant vs lr_003_cosine with fixed logging.

From buggy runs (reduction percentages are valid):
- lr_003_constant: 1.50%
- lr_003_cosine: 1.28%

Constant appears slightly better, but need verification with fixed metrics.

---

### H3: Model Too Small

**Status: POSSIBLE**

Evidence:
- Loss stabilizes around 8.6 after 5000 steps
- Generations remain gibberish
- Ternary distribution shows 90% zeros

These could indicate model capacity limitation, but:
- Need validation loss (not available)
- Need longer training to see if loss continues decreasing
- Gibberish could be due to short training, not model size

---

### H4: Not Enough Training

**Status: POSSIBLE**

Evidence:
- Loss still decreasing at step 5000
- No plateau in perplexity
- Gradient norms stable

Need longer training to determine if:
- Loss continues to decrease
- Generations improve
- Or if model reaches capacity limit

---

## What Was Fixed

1. ✅ **Loss calculation**: Now shows correct scale (~8.x instead of ~4300)
2. ✅ **Gradient logging**: Captured before zero_grad(), now shows healthy values
3. ✅ **Accumulation tracking**: Properly averages loss over logging window

---

## What's Still Wrong

1. ❌ **Generations are gibberish** - No improvement despite loss decrease
2. ❌ **No validation loss** - Can't measure generalization
3. ❌ **Ternary weights mostly zero** - 90% zeros (may be architectural limit)
4. ❌ **No coherence metrics** - Need structured evaluation of generations

---

## Comparison: Against Previous Runs

**Can we use previous 3 runs?**

| Metric | Valid? | Why |
|--------|--------|-----|
| Loss reduction % | ✅ YES | Percentages unaffected by scaling |
| Gradient norms | ❌ NO | Wrongly computed (after zero_grad) |
| Absolute loss | ❌ NO | 500x inflated |
| Generations | ✅ YES | Real outputs |
| Update magnitude | ✅ YES | Correctly tracked |

**Valid conclusions from previous runs:**
1. lr_003_constant > lr_001_constant (1.50% vs 0.58% reduction)
2. lr_003_constant > lr_003_cosine (1.50% vs 1.28% reduction)

---

## Recommendations

1. ✅ **LR=0.003 is supported** - Better than 0.001
2. ⚠️ **Run longer training** - Loss still decreasing, haven't reached limit
3. ⚠️ **Add validation loss** - Need to measure generalization
4. ⚠️ **Investigate gibberish** - Loss decreasing but generations not improving

---

## Final Conclusions

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: LR too low | **STRONGLY SUPPORTED** | lr_003 > lr_001 (2.6x improvement) |
| H2: Schedule wrong | **UNKNOWN** | Need fixed cosine run |
| H3: Model too small | **POSSIBLE** | Gibberish persists |
| H4: Not enough training | **POSSIBLE** | Loss still decreasing |
