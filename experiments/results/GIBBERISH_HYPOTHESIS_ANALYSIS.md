# Gibberish Hypothesis Analysis

---

## ⚠️ STATUS: SUPERSEDED

**This document contains an outdated hypothesis ranking that was later corrected.**

**Correction Date:** August 19, 2026
**Superseded By:** `FINAL_HYPOTHESIS_RANKING.md`

**What Changed:**
1. **"Learning Rate Too Low"** was ranked as STRONGLY SUPPORTED
   - Later ruled out: 3x LR produced same gibberish, FP32 vs Ternary showed same issue
   - See `VERIFICATION_ANALYSIS.md`, `PADDING_INFLATION_DISCOVERY.md`

2. **"Weight Quantization Never Activates"** showed 0% for all values
   - This was based on analyzing UNSCALED weights
   - Re-analysis with scaled weights showed healthy 34.5%/31.0%/34.5% distribution
   - See `BITNET_FORMULA_VERIFICATION.md`

3. **Missing hypotheses:**
   - Ternary quantization limiting learning - RULED OUT by FP32 control experiment
   - Model capacity - Now being investigated
   - Training data size - Now being investigated

**What Remains Valid:**
- Training/Val loss tracking together (no overfitting)
- Entropy remained healthy (no collapse)
- Observation that output was gibberish throughout

---

**Original Document Below:**

---

**Date:** August 18, 2026
**Purpose:** Rank remaining hypotheses by EVIDENCE, not intuition

---

## Evidence Summary

### 50k Training Run (Attempt 4)

| Metric | Start | End | Observation |
|--------|-------|-----|-------------|
| Loss | 8.75 | 7.56 | Decreased 13.6% |
| Entropy | 95.1% | 96.1% | STABLE (healthy) |
| TopProb | 0.5% | 0.6% | STABLE (healthy) |
| Train/Val | N/A | N/A | Not measured |

**Key Evidence:**
- Loss decreased throughout
- Entropy REMAINED HIGH (96%+)
- No collapse (unlike Attempt 2)
- Output: GIBBERISH throughout

### 3k Instrumented Run

| Metric | Start | End | Observation |
|--------|-------|-----|-------------|
| Train Loss | 8.75 | 8.15 | Decreased 6.9% |
| Val Loss | 8.73 | 8.33 | Decreased 4.6% |
| Perplexity | 6161 | 4066 | Decreased 34% |
| Entropy | N/A | N/A | Not measured (3k run) |
| Ternary Distribution | 0%/0%/0% | 0%/0%/0% | NOT QUANTIZED |

**Key Evidence:**
- Train/Val losses TRACK TOGETHER
- No overfitting (no divergence)
- Ternary distribution shows 0% for all values
- Output: GIBBERISH at all checkpoints

---

## Hypotheses Ranked by Evidence

### Hypothesis 1: Learning Rate Too Low / Progress Too Slow

**Evidence FOR:**
- Loss decreased only 6.9% in 3k steps (3k run)
- Loss decreased only 13.6% in 50k steps (50k run)
- Loss rate slowing: 5k-10k=0.28 loss drop, 40k-50k=0.06 loss drop
- Loss still at 7.56 after 50k steps (high)
- BitNet papers use LR 0.0004-0.001 for models up to 2B

**Evidence AGAINST:**
- LR was 0.001 (within BitNet range)
- Loss IS decreasing (just slowly)
- No divergence or stability issues

**Classification:** [STRONGLY SUPPORTED]

**Evidence Strength:** Direct measurements from training logs

---

### Hypothesis 2: Model Too Small for Vocabulary

**Evidence FOR:**
- Model: 6.8M params
- Vocabulary: 4000 tokens
- Ratio: 1,714 params per vocab token
- BitNet b1.58 2B uses vocab 128256 (ratio: 15.6 params per vocab)
- Our ratio 110x larger per token but model tiny

**Evidence AGAINST:**
- Model DOES learn (loss decreases)
- Entropy healthy (no collapse)
- Cannot prove vocabulary size is the blocker

**Classification:** [POSSIBLE]

**Evidence Strength:** Circumstantial (architecture comparison)

---

### Hypothesis 3: Weight Quantization Never Activates (Ternary Distribution 0%)

**Evidence FOR:**
- Ternary distribution shows 0% for all values at ALL checkpoints
- This means weights are NOT being quantized to {-1, 0, +1}
- Weights remain full-precision floats
- Monitoring function measures stored weights, not quantized

**Evidence AGAINST:**
- BitNet implementation VERIFIED CORRECT on Aug 18
- Scaled weights ARE intended behavior
- kyegomez reference also shows scaled floats
- Quantization happens in forward pass, weights stored as dequantized

**Classification:** [VERIFIED - NOT THE CAUSE]

**Evidence Strength:** This is correct BitNet behavior per verification

---

### Hypothesis 4: Dataset Too Small (20 Stories)

**Evidence FOR:**
- 3k run used only 20 stories (3,825 tokens)
- Model may overfit to specific phrases
- Limited vocabulary exposure

**Evidence AGAINST:**
- 50k run used full TinyStories dataset
- Loss decreased on full dataset too
- Full dataset still produced gibberish

**Classification:** [UNKNOWN - NEED TO CHECK]

**Evidence Strength:** Contradicted by 50k run (full dataset)

---

### Hypothesis 5: Ternary Quantization Prevents Learning Fine Details

**Evidence FOR:**
- Ternary weights have only 3 values
- Cannot represent fine-grained patterns
- May require more training to compensate

**Evidence AGAINST:**
- Loss IS decreasing (learning happens)
- BitNet papers show ternary works at scale
- Random test memorized perfectly (different config)

**Classification:** [POSSIBLE]

**Evidence Strength:** Theoretical (no direct evidence)

---

### Hypothesis 6: Learning Rate Schedule Wrong

**Evidence FOR:**
- 50k run used cosine decay (LR decreased over time)
- Loss plateaued after 20k steps
- Cosine decay stopped learning too early

**Evidence AGAINST:**
- 3k run used constant LR
- Still slow progress (6.9% in 3k steps)
- 50k run loss kept decreasing (just slowly)

**Classification:** [POSSIBLE]

**Evidence Strength:** Supported by historical findings (LR decay issue)

---

### Hypothesis 7: Not Enough Training Steps

**Evidence FOR:**
- Loss still decreasing at 50k
- Loss still decreasing at 3k
- No plateau reached

**Evidence AGAINST:**
- 50k steps is substantial
- Other ternary models learn faster
- Loss should be lower by now if working

**Classification:** [POSSIBLE]

**Evidence Strength:** Cannot be ruled out, but weak evidence

---

### Hypothesis 8: Initialization Problem

**Evidence FOR:**
- Initial loss high (8.75)
- Initial perplexity high (6161)
- Slow start may indicate poor initialization

**Evidence AGAINST:**
- Initial loss similar to other transformers
- Loss does decrease from start

**Classification:** [UNKNOWN]

**Evidence Strength:** No comparative baseline

---

## Ranked Hypotheses by Evidence Strength

| Rank | Hypothesis | Classification | Evidence Strength |
|------|------------|----------------|-------------------|
| 1 | Learning rate too low / progress too slow | [STRONGLY SUPPORTED] | Direct measurement |
| 2 | Learning rate schedule wrong (cosine decay) | [POSSIBLE] | Historical finding |
| 3 | Model too small for vocabulary | [POSSIBLE] | Architecture comparison |
| 4 | Ternary prevents fine learning | [POSSIBLE] | Theoretical |
| 5 | Not enough training steps | [POSSIBLE] | Cannot rule out |
| 6 | Dataset too small | [UNKNOWN] | Contradicted by 50k |
| 7 | Initialization problem | [UNKNOWN] | No baseline |
| 8 | Weight quantization issue | [VERIFIED - NOT CAUSE] | Correct per BitNet |

---

## Top Two Hypotheses

### H1: Learning Rate Too Low / Progress Too Slow

**Evidence:**
- Loss 8.75 → 7.56 in 50k steps (13.6%)
- Loss 8.75 → 8.15 in 3k steps (6.9%)
- Rate slowing over time (plateau effect)

### H2: Learning Rate Schedule Wrong (Cosine Decay Stopped Learning)

**Evidence:**
- Historical finding: "Cosine decay stops learning after 20k"
- Loss plateau at 20k-25k in 50k run
- Loss rate: 5k-10k=0.28, 20k-25k=0.09, 40k-50k=0.06

---

## Distinguishing Experiment

### Question

Is the problem:
A) Absolute LR too low (need higher LR)?
B) LR schedule causing plateau (need constant LR)?

### Cheapest Experiment

**Compare: Same model, same steps, different LR/schedule**

| Config | Steps | Purpose |
|--------|-------|---------|
| LR=0.001 constant | 5000 | Match current config |
| LR=0.003 constant | 5000 | Test higher LR |
| LR=0.001 cosine | 5000 | Test schedule effect |

**Cost:** ~3 hours total (1 hour each config, 5000 steps)

**What it distinguishes:**
- If LR=0.003 faster than 0.001 → H1 supported (LR too low)
- If constant faster than cosine → H2 supported (schedule issue)
- If no difference → Neither hypothesis explains gibberish

**Why cheapest:**
- Uses existing model architecture
- Uses existing dataset
- Only changes LR/schedule
- Short runs (5000 steps)
- Can use 3k script with modified config

---

## Proposed Experiment

```bash
# Experiment A: LR=0.003 constant (higher LR)
python experiments/tinystories_memorize_observability.py --lr 0.003 --scheduler constant

# Experiment B: LR=0.001 constant (current LR, no decay)  
python experiments/tinystories_memorize_observability.py --lr 0.001 --scheduler constant

# Compare loss trajectories at 5000 steps
```

**Decision Rule:**
- If 0.003 reaches lower loss → LR was too low
- If 0.001 constant matches cosine → Schedule not the issue
- If both plateau → Architecture problem

---

## Conclusion

**Top hypotheses to test:**
1. LR too low (strongly supported by loss trajectory)
2. LR schedule wrong (possible, historical precedent)

**Cheapest experiment:** Run 3 short comparisons with different LR/schedule configs.

**Remaining hypotheses:** Model size, ternary limitations, training length are all [POSSIBLE] but have less direct evidence.
