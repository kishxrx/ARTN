# Comprehensive Analysis: All LR Experiments

---

## ⚠️ STATUS: SUPERSEDED

**This document contains conclusions that were later corrected.**

**Correction Date:** August 19, 2026
**Superseded By:**
- `ROOT_CAUSE_ANALYSIS.md` (logging bug identification)
- `VERIFICATION_ANALYSIS.md` (corrected results with fixed logging)
- `PADDING_INFLATION_DISCOVERY.md` (corrected FP32 vs Ternary)

**What Was Corrected:**
1. **Loss values:** This document shows loss ~4368, which was a logging bug. Correct loss is ~8.7. See `ROOT_CAUSE_ANALYSIS.md` for bug details.
2. **Ternary distribution:** Shows ~90% zeros in ternary weights. This was based on analyzing UNSCALED weights. Scaled weight analysis shows healthy 34.5%/31.0%/34.5% distribution. See `BITNET_FORMULA_VERIFICATION.md`.
3. **LR hypothesis:** Correctly identified that changing LR does not fix gibberish. This finding remains valid.

**What Remains Valid:**
- Generation quality comparison across LR experiments
- Finding that LR is not the root cause

---

**Original Document Below:**

---

**Date:** August 19, 2026
**Experiments Analyzed:** 4 runs (3 buggy + 1 fixed)

---

## Executive Summary

**CRITICAL FINDING: Changing LR does NOT meaningfully change generation quality.**

All 4 experiments produce nearly identical gibberish, despite:
- 2.6x difference in loss reduction (1.50% vs 0.58%)
- 3x difference in absolute LR (0.003 vs 0.001)
- 3x difference in update magnitude (0.999 vs 0.340)

**This rules out LR as the primary cause of gibberish.**

---

## Evidence Tables

### Loss Reduction (VALID across all runs)

| Experiment | Initial Loss | Final Loss | Reduction | Valid? |
|------------|--------------|------------|-----------|--------|
| lr_001_constant | 4368.79 | 4343.53 | **0.58%** | ✅ (percentage valid) |
| lr_003_constant | 4365.84 | 4300.37 | **1.50%** | ✅ (percentage valid) |
| lr_003_cosine | 4365.85 | 4310.00 | **1.28%** | ✅ (percentage valid) |
| lr_003_constant_fixed | 8.73 | 8.60 | **1.50%** | ✅ (all valid) |

**Key insight:** lr_003 shows 2.6x better reduction than lr_001.

---

### Update Magnitude (VALID across all runs)

| Experiment | Avg Update Magnitude | Ratio to lr_001 |
|------------|---------------------|-----------------|
| lr_001_constant | 0.340 | 1.0x (baseline) |
| lr_003_constant | 0.999 | 2.9x |
| lr_003_cosine | 0.900 | 2.6x |
| lr_003_constant_fixed | 0.999 | 2.9x |

**Key insight:** Update magnitude scales with LR as expected.

---

### Perplexity (Only valid for fixed run)

| Experiment | Initial PPL | Final PPL | Reduction |
|------------|-------------|-----------|-----------|
| lr_003_constant_fixed | 6196 | 5436 | 12.3% |

**Key insight:** Perplexity improving, but starts very high (normal should be <1000).

---

### Ternary Weight Distribution (VALID across all runs)

| Experiment | -1% | 0% | +1% |
|------------|-----|-----|-----|
| lr_001_constant | 4.85% | 90.25% | 4.89% |
| lr_003_constant | 4.71% | 90.55% | 4.74% |
| lr_003_cosine | 4.75% | 90.47% | 4.79% |
| lr_003_constant_fixed | 4.71% | 90.55% | 4.74% |

**Key insight:** All experiments show ~90% zeros, regardless of LR.

---

## Generation Quality Comparison

### All Prompts - Side by Side

**Prompt: "Once upon a time"**

| Experiment | Generation (first 80 chars) | Repetition |
|------------|----------------------------|------------|
| lr_001_constant | "Once upon a time slowly Billy everywhere arrived, brush room..." | 1.19 |
| lr_003_constant | "Once upon a time slowly youneverywhere arrived, brush room..." | 1.25 |
| lr_003_cosine | "Once upon a time slowly Billy everywhere arrived, brush room..." | 1.19 |
| lr_003_constant_fixed | "Once upon a time slowly youneverywhere arrived, brush room..." | 1.25 |

**Prompt: "The little girl"**

| Experiment | Generation (first 80 chars) | Repetition |
|------------|----------------------------|------------|
| lr_001_constant | "The little girl standing doll kitchtook Enedyou ama..." | 1.61 |
| lr_003_constant | "The little girl standing cut kitchtook Enlooks you ama..." | 1.61 |
| lr_003_cosine | "The little girl standing cut kitchtook Enlooks you ama..." | 1.56 |
| lr_003_constant_fixed | "The little girl standing cut kitchtook Enlooks you ama..." | 1.61 |

**Prompt: "A boy named"**

| Experiment | Generation (first 80 chars) | Repetition |
|------------|----------------------------|------------|
| lr_001_constant | "A boy named bird at. y,\" closces's erslide, pond..." | 1.47 |
| lr_003_constant | "A boy named bird at. y,\" closces's erslide, pond..." | 1.47 |
| lr_003_cosine | "A boy named bird at. y,\" closces's erslide, pond..." | 1.47 |
| lr_003_constant_fixed | "A boy named bird at. y,\" closces's erslide, pond..." | 1.47 |

---

### Generation Similarity Analysis

**Prompt "Once upon a time":**
- lr_001 and lr_003_cosine: **IDENTICAL** (both use "Billy")
- lr_003_constant and lr_003_fixed: **IDENTICAL** (both use "youneverywhere")
- Only 2 unique generations across 4 runs

**Prompt "The little girl":**
- lr_003_constant, lr_003_cosine, lr_003_fixed: **IDENTICAL** ("standing cut")
- lr_001_constant: Slightly different ("standing doll")
- Only 2 unique generations across 4 runs

**Prompt "A boy named":**
- Nearly **IDENTICAL** across all 4 runs
- Only 1-2 character differences

---

## Repetition Score Comparison

| Experiment | Avg Repetition across prompts |
|------------|-------------------------------|
| lr_001_constant | 1.43 |
| lr_003_constant | 1.47 |
| lr_003_cosine | 1.46 |
| lr_003_constant_fixed | 1.47 |

**Key insight:** No meaningful difference in repetition. All equally bad.

---

## Evidence-Based Hypothesis Ranking

### H1: LR Too Low

**Status: RULED OUT for gibberish cause**

**Evidence:**
- 3x higher LR produces same gibberish
- Generations nearly identical across LR values
- Loss reduction differs (0.58% vs 1.50%), but generations don't

**Conclusion:** Higher LR improves loss reduction metric, but NOT generation quality. LR is not the bottleneck.

---

### H2: LR Schedule Wrong (constant vs cosine)

**Status: RULED OUT for gibberish cause**

**Evidence:**
- lr_003_constant: 1.50% reduction, repetition 1.47
- lr_003_cosine: 1.28% reduction, repetition 1.46
- Generations identical between constant and cosine

**Conclusion:** Schedule difference doesn't affect generation quality.

---

### H3: Insufficient Training (5000 steps too few)

**Status: POSSIBLE**

**Evidence FOR:**
- Loss still decreasing at step 5000 (no plateau)
- Perplexity improving (6196 → 5436)
- Gradient norms stable (not vanishing)

**Evidence AGAINST:**
- Loss reduction is small (1.50% over 5000 steps)
- Generations don't improve at all
- Verified pipeline showed loss ~8 after 3000 steps with real data

**Conclusion:** Need longer training to definitively rule out, but 5000 steps should show SOME generation improvement if training is the issue.

---

### H4: Model Capacity Too Small

**Status: POSSIBLE**

**Evidence FOR:**
- 6.8M parameters is very small for language modeling
- Vocab size 4000 is large relative to model size
- Ternary quantization reduces effective capacity further
- 90% of ternary weights are zero (only 10% active)

**Evidence AGAINST:**
- TinyStories is designed for small models
- Similar-sized models in literature can generate coherent text
- No validation loss to confirm overfitting/underfitting

**Conclusion:** Possible, but not definitively supported. Need to test with larger model or smaller vocab.

---

### H5: Ternary Quantization Behavior

**Status: STRONGLY SUPPORTED as contributing factor**

**Evidence:**
- 90% of weights are zero (should be ~33%)
- Only 10% of weights active (90% disabled)
- Ternary distribution stable across all runs (architecture limitation)
- Quantization may be too aggressive for small model

**Evidence AGAINST:**
- BitNet papers show ternary works for larger models
- Ternary verified correct per BitNet design

**Conclusion:** Ternary quantization with 90% zeros severely limits model capacity. Only 10% of parameters are actually being used.

---

### H6: Initialization

**Status: UNLIKELY**

**Evidence:**
- All models use same seed=42 initialization
- Static diagnostics passed (parameters initialized correctly)
- Generations are deterministic given same seed

**Conclusion:** Initialization is correct per diagnostics.

---

### H7: Data/Preprocessing

**Status: STRONGLY SUPPORTED as primary cause**

**Evidence:**
- lr_experiment uses synthetic random tokens (not real TinyStories)
- Verified training with real data showed loss ~8
- Experiment with synthetic data shows loss ~8.7 (similar)
- BUT: Generations are gibberish in both cases

**Critical observation:**
- One-batch diagnostic showed real data produces loss ~8.6
- One-batch diagnostic showed synthetic data produces loss ~8.7
- Both produce gibberish

**Conclusion:** Data source doesn't matter - both produce gibberish. Problem is elsewhere.

---

### H8: Vocab/Tokenizer Mismatch

**Status: POSSIBLE**

**Evidence:**
- Vocab size 4000 is small for full language modeling
- Tokenizer trained on specific dataset
- Model may not have capacity to learn full vocab

**Need to test:**
- Smaller vocab (e.g., 2000)
- Different tokenizer
- Check vocab coverage of generated tokens

---

## Hypothesis Ranking Summary

| Rank | Hypothesis | Status | Priority |
|------|-----------|--------|----------|
| 1 | **Ternary quantization** | STRONGLY SUPPORTED | HIGH - 90% zeros suggests capacity issue |
| 2 | **Insufficient training** | POSSIBLE | MEDIUM - Need longer run |
| 3 | **Model capacity too small** | POSSIBLE | MEDIUM - 6.8M small for vocab 4000 |
| 4 | **Vocab/tokenizer mismatch** | POSSIBLE | LOW - Need vocab analysis |
| 5 | **LR too low** | **RULED OUT** | - |
| 6 | **LR schedule wrong** | **RULED OUT** | - |
| 7 | **Initialization** | UNLIKELY | - |
| 8 | **Data/preprocessing** | UNLIKELY (rule out after one-batch test) | - |

---

## Key Findings

### Finding 1: LR Does NOT Fix Gibberish

**Evidence:**
- 3x higher LR → 2.6x better loss reduction
- BUT: Identical gibberish generations
- Conclusion: LR is not the bottleneck

### Finding 2: Ternary Weights 90% Zero

**Evidence:**
- All 4 runs show 90% quaternary zeros
- Only 10% of weights active
- This severely limits model capacity

**Implication:** Model operating at 10% capacity.

### Finding 3: Generations Nearly Deterministic

**Evidence:**
- lr_003_constant and lr_003_constant_fixed produce identical outputs
- lr_003_cosine produces outputs matching lr_001_constant (strange)
- Only 2 unique generations per prompt across 4 runs

**Implication:** Model converges to similar local minimum regardless of LR.

### Finding 4: Loss Decreases but Generations Don't Improve

**Evidence:**
- Loss reduction: 1.50% (lr_003_constant_fixed)
- Repetition scores: 1.19, 1.25, 1.61, 1.47, 1.35 across prompts
- No prompt shows coherent structure

**Implication:** Loss is not measuring what we care about.

---

## Recommended Next Diagnostic

**Highest-value: Deploy checkpoint and generate 100 samples per checkpoint**

Run inference on all 4 checkpoints:
1. Load best_checkpoint.pt from each experiment
2. Generate 100 samples from same prompt
3. Measure:
   - Word diversity (unique words / total words)
   - N-gram repetition
   - Coherence score (manual inspection)
   - Perplexity on held-out text

**Why this diagnostic:**
1. Cheap (inference only, no training)
2. Distinguishes between:
   - Ternary quantization issue (all checkpoints same)
   - Training length issue (later checkpoints better)
   - Model capacity issue (all fail similarly)
3. Uses existing artifacts (checkpoints already saved)

**Cost:** ~30 minutes inference time
**Value:** Rule out 2-3 hypotheses with single diagnostic

---

## Alternative Diagnostics (lower priority)

1. **Train without ternary**: Test if float weights produce coherent text
   - Cost: ~3 hours compute
   - Value: Definitively rule out ternary as cause

2. **Smaller vocab**: Train with vocab_size=1000
   - Cost: ~3 hours compute
   - Value: Test if vocab/model size ratio matters

3. **Longer training**: Run 50k steps
   - Cost: ~33 hours compute
   - Value: Rule out "insufficient training" hypothesis

---

## Conclusion

**Primary hypothesis:** Ternary quantization with 90% zeros severely limits model capacity, causing gibberish.

**Secondary hypothesis:** Model too small for vocab size 4000.

**Ruled out:** LR too low, LR schedule wrong, data/preprocessing.

**Recommended action:** Analyze existing checkpoints with diverse generation metrics before further training.
