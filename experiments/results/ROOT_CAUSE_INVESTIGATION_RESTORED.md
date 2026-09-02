# Root Cause Investigation Summary: Gibberish Generation in Ternary BitLinearGPT

**Date Range:** August 18 - September 1, 2026 (Updated with Phase 20: BitLinear Implementation Audit - Phase 19 Overturned)
**Status:** Ongoing Investigation - Attention provides little functional benefit
**Model:** BitLinearGPT (6.8M params, vocab=4000, 6 layers, 256 dim)

**LATEST FINDING (Phase 16-17, August 31, 2026):** All 48 attention heads show diffuse patterns. Learned attention provides little measurable benefit over uniform attention (max degradation 0.55%). This is CORRELATED with weak contextual learning, but causation is NOT established.

---

## Executive Summary

### [ORIGINAL - August 19, 2026] - [SUPERSEDED]

**After 2 days of systematic investigation, we have RULED OUT ternary quantization as the root cause of gibberish generation.**

**Critical Finding: Both FP32 (no quantization) and Ternary models perform SIMILARLY on real language tokens (~5% accuracy), indicating the problem affects both architectures equally.**

**The gibberish problem is caused by something OTHER than quantization - likely insufficient training, model capacity, or training data.**

**[CAVEAT - August 26, 2026]:** FP32 experiment used only 20 stories (NOT 50k dataset). Cannot establish FP32-vs-Ternary equivalence on 50k dataset without running FP32 on 50k TinyStories.

---

### [UPDATED - August 27, 2026] - [SUPERSEDED BY CORRECTED ANALYSIS]

**After 9 days of systematic investigation, we have strong evidence that the model's predictive behavior is dominated by unigram/frequency information.**

**Critical Findings (August 27, 2026):**
- Model Top-1: 3.69% vs Unigram Top-1: 3.70% (approximately equal)
- Model Cross-Entropy: 7.7176 vs Unigram CE: 6.7504 (model WORSE than unigram)
- Bigram baseline Top-1: 20.71% (model substantially WORSE than bigram)
- Full-context loss: 7.7176, 8-token context: 7.7395, 4-token context: 7.7865 (minimal context benefit)
- Context shuffling: 7.6617 (slightly LOWER loss than normal context 7.7176)
- Greedy decoding produces repetitive output: "and and and..." / "the the the..."
- Temperature changes do NOT restore coherent generation

**[CORRECTION - August 27, 2026]:** The context benefit calculation was incorrect due to invalid ctx1_loss. Corrected analysis shows POSITIVE context benefit (0.4). See Phase 11.

---

### [CORRECTED - August 27, 2026] - FINAL

**The model learned MINIMAL contextual information, but remains substantially weaker than a simple bigram baseline.**

**Critical Findings (Corrected Trajectory Analysis):**

**Context Benefit (RECOVERED from saved results):**
- Context Benefit (ctx4): 0.13 → 0.44 across training (POSITIVE)
- Context Benefit (ctx8): 0.10 → 0.23 across training (POSITIVE)
- Shuffle Effect: 0.18 → 0.82 (POSITIVE - shuffling HURTS)
- Interpretation: Model DOES use context, but benefit is SMALL (~0.4 vs loss ~8)

**Model vs Baselines (VERIFIED across all 50 checkpoints):**
- Model Top-1: Never exceeds 3.79% (step 39000)
- Unigram Baseline: 3.85%
- Bigram Baseline: 21.93%
- Model vs Unigram: -0.06% at best (NEVER exceeds)
- Model vs Bigram: -18.14% (consistently WORSE)

**Learning Trajectory:**
- Loss: 8.60 → 7.70 (improved 0.90 over 50k steps)
- Top-1: 0.04% → 3.77%
- Context benefit INCREASES during training (0.13 → 0.44)
- Shuffle effect INCREASES during training (0.18 → 0.82)

**REVISED INTERPRETATION:**
- Model shows MINIMAL (not zero) contextual learning
- Context benefit is positive but small (~5% relative improvement)
- Shuffle effect is positive (context order matters slightly)
- Model STILL never exceeds unigram baseline
- Model STILL ~18% worse than bigram baseline

**PRIMARY QUESTION:** Why does the model learn MINIMAL contextual information despite 50k steps and 9.2M tokens?

**Remaining Hypotheses:**
- Training/data pipeline issue
- Target/context alignment issue
- Tokenizer/sequence construction issue
- Model architecture/training dynamics
- Ternary capacity limitation
- Insufficient data
- Other implementation/design issue

**CRITICAL ANOMALY:** The neural model performs substantially WORSE than a simple bigram baseline. This MUST be investigated before spending another large training run.

---

## Problem Statement

**Initial Observation:**
- Trained BitLinearGPT produces gibberish output
- Loss decreases during training but generation remains incoherent
- Phenomenon observed across multiple training runs (500 to 50,000 steps)

**Questions to Answer:**
1. Is ternary quantization breaking the model?
2. Are there implementation bugs in forward pass?
3. Is the training objective wrong?
4. Is learning rate insufficient?
5. Is model capacity insufficient?

---

## Investigation Timeline

### Phase 1: Initial Hypotheses (August 18, 2026)

**Hypotheses considered:**
- Ternary quantization implementation broken
- Forward pass bugs
- Training objective bugs
- Learning rate too low
- Model capacity insufficient

**Actions:**
- Analyzed ternary weight distribution
- Checked forward pass stages
- Reviewed training objective

**Status at this point:** All hypotheses under investigation, none ruled out.

---

### Phase 2: Logging Bug Discovery (August 18-19, 2026)

**Discovery:** Two critical logging bugs

| Bug | Problem | Effect | Fix |
|-----|---------|--------|-----|
| Loss inflation | Summed losses × batch × seq | 500x inflated | Track sum AND count |
| Zero gradients | Computed after zero_grad | Always 0.0 | Capture before zero_grad |

**Evidence:** `ROOT_CAUSE_ANALYSIS.md`
**Impact:** Previous experiments showed loss ~4300, correct loss is ~8.7

**Status at this point:** Logging bugs FIXED. Previous loss measurements INVALID. Need to re-run experiments with correct logging.

---

### Phase 3: Ternary Quantization Verification (August 19, 2026)

**Initial suspicion:** Ternary weights showed "90% zeros"

**Investigation:**
- Analyzed UNSCALED weights → showed 90% zeros
- Analyzed SCALED weights (W / mean(|W|)) → showed healthy 34.5%/31.0%/34.5% distribution

**Conclusion at this point:** Ternary quantization implementation is CORRECT

**Evidence:** `BITNET_FORMULA_VERIFICATION.md`, `TERNARY_IMPLEMENTATION_VERIFICATION.md`

**[STATUS - August 27, 2026]:** STILL VALID - Ternary implementation verified correct. However, this does NOT rule out ternary quantization as a capacity limitation for learning contextual patterns.

---

### Phase 4: Forward Pass Verification (August 19, 2026)

**Method:** Trace every forward pass stage with measurements

**Results:**

| Stage | Status | Evidence |
|-------|--------|----------|
| Embeddings | ✅ VERIFIED CORRECT | Diff = 0.98 (different tokens embed differently) |
| Positional Encoding | ✅ VERIFIED CORRECT | Diff = 0.87 (different positions encode differently) |
| Q/K/V Projections | ✅ VERIFIED CORRECT | Q ≠ K ≠ V (diff = 0.65-0.66) |
| Attention Scores | ✅ VERIFIED CORRECT | Std = 0.32, 100% non-zero |
| Causal Mask | ✅ VERIFIED CORRECT | 55 masked positions, correct triangular pattern |
| Attention Probabilities | ✅ VERIFIED CORRECT | Sum = 1.000000 (exact) |
| Attention Output | ✅ VERIFIED CORRECT | Std = 0.26, non-zero |
| Residual Connections | ✅ VERIFIED CORRECT | Change = 0.20 magnitude |
| FFN | ✅ VERIFIED CORRECT | Std = 0.25, non-zero |
| Final Logits | ✅ VERIFIED CORRECT | Std = 1.17, varies by position |
| Training Objective | ✅ VERIFIED CORRECT | Target[i] = Input[i+1], manually verified |

**Conclusion at this point:** No forward pass bugs found

**Evidence:** `FORWARD_PASS_VERIFICATION_WITH_EVIDENCE.md`

**[STATUS - August 27, 2026]:** STILL VALID - Forward/backward implementation verified correct. This addresses IMPLEMENTATION correctness, NOT architecture CAPACITY (separate hypothesis).

---

### Phase 5: FP32 Control Experiment (August 19, 2026)

**Purpose:** Compare FP32 (no quantization) vs Ternary to determine if quantization is the problem

**Setup:**
- Identical architecture (6 layers, 256 dim, 8 heads)
- Identical training data (20 TinyStories sentences)
- Identical hyperparameters (LR=0.003, batch=8, 5000 steps)
- ONLY difference: FP32 Linear vs BitLinear

**Initial Results (MISLEADING):**

| Model | Held-Out Top-1 Accuracy | Initial Interpretation |
|-------|------------------------|------------------------|
| FP32 5k | **26.89%** | FP32 learned! |
| Ternary 50k | **3.56%** | Ternary failed |
| Difference | **23.33%** | FP32 7.5x better |

**Evidence:** `FP32_VS_TERNARY_COMPARISON.md` (now marked SUPERSEDED)

**[STATUS - August 27, 2026]:** ⚠️ **SUPERSEDED BY PADDING INFLATION DISCOVERY** - See Phase 6.

---

### Phase 6: Padding Inflation Discovery (August 19, 2026) - CRITICAL

**Audit Trigger:** 26.89% vs 3.56% seemed too large a gap

**Audit Findings:**

| Model | Total Accuracy | Real Token Accuracy | Padding Accuracy |
|-------|----------------|---------------------|------------------|
| FP32 5k | 26.89% | **5.46%** | **100.00%** |
| Ternary 50k | 3.56% | **4.60%** | **0.00%** |
| Difference | 23.33% | **0.86%** | 100% |

**What Happened:**
- FP32 learned to predict padding token (token 0) with 100% accuracy
- Padding inflated total accuracy from 5.46% to 26.89%
- Ternary did NOT learn to predict padding
- On REAL language tokens, both models perform SIMILARLY (~5%)

**Conclusion at this point:** Ternary quantization is RULED OUT as the root cause

**Evidence:** `PADDING_INFLATION_DISCOVERY.md`

**[STATUS - August 27, 2026]:** ⚠️ **IMPORTANT CAVEAT:** FP32 experiment used only 20 TinyStories sentences. This does NOT establish FP32-vs-Ternary equivalence on the 50k TinyStories dataset. Cannot rule out ternary quantization as a capacity limitation for contextual learning without FP32 50k comparison.

---

### Phase 7: Longitudinal Analysis (August 26, 2026)

**Purpose:** Analyze complete training trajectory from 1k to 50k steps

**Method:** Evaluate all 50 checkpoints (every 1000 steps)

**Key Findings:**

| Step | Train Loss | Held-Out Loss | Top-1 | Learning Rate |
|------|------------|---------------|-------|---------------|
| 1k | 8.45 | 8.61 | 0.03% | 0.001000 |
| 5k | 7.94 | 8.22 | 1.58% | 0.000984 |
| **10k** | **7.67** | **8.00** | **3.51%** | **0.000921** |
| 15k | 7.37 | 7.88 | 3.57% | 0.000817 |
| 20k | 7.22 | 7.82 | 3.51% | 0.000683 |
| 30k | 7.23 | 7.75 | 3.65% | 0.000377 |
| 40k | 7.06 | 7.72 | 3.65% | 0.000126 |
| 50k | 7.13 | 7.72 | 3.69% | 0.000030 |

**Held-out loss improvement rates:**
- Early (1k-10k): 0.000067 per step
- Middle (10k-20k): 0.000018 per step
- Late (40k-50k): 0.000001 per step
- Ratio (early/late): 120.5x slower

**Observations:**
- Model learns from 1k to ~10k-15k (loss drops, accuracy improves)
- After 15k, learning plateaus (held-out loss stabilizes at ~7.72)
- LR decays from 0.001 to 0.00003 (30x reduction)
- Generation remains gibberish at ALL stages

**Hypothesis at this point:** LR decay causes plateau

**Evidence:** `experiments/results/longitudinal_analysis.json`

**[STATUS - August 27, 2026]:** HYPOTHESIS TESTED - See Phase 8 (invalid) and Phase 9 (valid).

---

### Phase 8: Invalid AdamW Continuation Experiment (August 26, 2026) - ❌ INVALID EXPERIMENT

**Purpose:** Test if higher LR would restore learning

**Setup:**
- Start from 50k checkpoint
- Continue training with different LR values (3e-5, 1e-4, 3e-4)
- Measure held-out performance

**CRITICAL ERROR DISCOVERED:**

| Parameter | Original Training | Continuation Experiment |
|-----------|-------------------|-------------------------|
| Optimizer | **AdaFactor** (custom) | **AdamW** (PyTorch) |
| beta1 | 0.5 | 0.5 |
| beta2 | 0.999 | 0.98 |
| Optimizer State | 50k steps accumulated | **NOT RESTORED** |

**What Happened:**
- Used WRONG optimizer (AdamW instead of custom AdaFactor)
- Did NOT restore optimizer state (momentum, variance)
- Result: Catastrophic degradation (held-out loss 7.72 → 9-20)

**Status:** ❌ **INVALID EXPERIMENT**
- Causal conclusions from this experiment are INVALID
- Failure was due to optimizer mismatch, not LR
- This experiment DOES NOT prove anything about LR

**Lesson Learned:** Always verify optimizer type and state restoration before continuation experiments

**Evidence:** `experiments/results/lr_causal_experiment.json` (INVALID)

---

### Phase 9: Corrected AdaFactor LR Experiment (August 26, 2026)

**Purpose:** Valid test of LR hypothesis with correct optimizer

**Setup:**
- Correct optimizer: Custom AdaFactor (matching original training)
- Optimizer state: RESTORED from 50k checkpoint
- ONLY variable changed: Learning rate
- Duration: 1000 continuation steps

**Results:**

| Experiment | LR | Baseline Loss | Final Loss | Change |
|------------|-----|---------------|------------|--------|
| Control | 3e-5 | 7.6981 | 7.6980 | -0.0001 |
| Experiment A | 1e-4 | 7.6981 | 7.6979 | -0.0002 |
| Experiment B | 3e-4 | 7.6981 | 7.6969 | -0.0012 |

**Findings:**
- All LR values produced similar results (< 0.02 difference in held-out loss)
- Held-out loss essentially unchanged across tested LR range (3e-5, 1e-4, 3e-4)
- Model did not resume learning despite higher LR

**Conclusion:** STRONG evidence against LR being the primary cause of the plateau within the tested range (AdaFactor, optimizer state restored, 1000 continuation steps, held-out loss unchanged).

**Status:** ✅ **VALID EXPERIMENT**

**[CAVEAT - August 27, 2026]:** This is experimental evidence for the tested conditions, not an unlimited universal proof. Different LR schedules, ranges, or training durations were not tested.

**Evidence:** `experiments/results/lr_causal_corrected/lr_causal_corrected.json`

---

### Phase 10: Context Forensic Diagnostic (August 27, 2026) - NEW EVIDENCE

**Purpose:** Determine what the model has ACTUALLY learned

**Method:** Comprehensive evaluation with multiple baselines and ablations

**Results:**

**10.1 Top-K Accuracy:**

| Metric | Model | Unigram Baseline | Bigram Baseline |
|--------|-------|------------------|-----------------|
| Top-1 Accuracy | **3.69%** | **3.70%** | **20.71%** |
| Top-5 Accuracy | 15.76% | - | - |
| Top-10 Accuracy | 18.13% | - | - |
| Top-50 Accuracy | 23.06% | - | - |
| Top-100 Accuracy | 26.35% | - | - |

**Observation:** Model Top-1 is approximately equal to unigram Top-1. Model is substantially WORSE than unigram at larger K (unigram accumulates correct predictions across frequent tokens).

**10.2 Cross-Entropy and Perplexity:**

| Metric | Model | Unigram Baseline | Interpretation |
|--------|-------|------------------|----------------|
| Cross-Entropy | **7.7176** | **6.7504** | Model WORSE than unigram |
| Perplexity | **2247.5** | **854.4** | Model WORSE than unigram |

**10.3 Context Ablation:**

| Context Limit | Loss | Top-1 | Change from Full |
|---------------|------|-------|------------------|
| 256 tokens (full) | 7.7176 | 3.71% | Baseline |
| 8 tokens | 7.7395 | 3.81% | +0.02 loss |
| 4 tokens | 7.7865 | 3.35% | +0.07 loss |

**Observation:** Minimal context benefit. Removing most context has negligible effect on loss.

**10.4 Context Shuffling:**

| Condition | Loss | Interpretation |
|-----------|------|----------------|
| Normal context | 7.7176 | Baseline |
| Shuffled context | 7.6617 | **SLIGHTLY LOWER** |
| Change | -0.0559 | Shuffling slightly HELPS |

**Observation:** Shuffling context order produces slightly LOWER loss. This is unexpected if model uses context meaningfully.

**10.5 Greedy Decoding:**

Real held-out prefixes produce incoherent/repetitive continuations:

| Prefix | Greedy Generation |
|--------|-------------------|
| "Once upon a time there was a big room. In" | "and and and and and and..." |
| "Once upon a time, there were two friends called Sam" | "and and and and and and..." |
| "Once upon a time" | "the the the the the the..." |

**10.6 Temperature Variation:**

Temperature changes (0.5, 0.7, 1.0) do NOT restore coherent generation. All produce gibberish.

**10.7 Tokenizer Diagnostic:**

Token IDs and decoded strings are correct around predictions. Tokenization/decoding is working as expected.

**CRITICAL ANOMALY:**

**The neural model performs substantially WORSE than a simple bigram baseline (3.69% vs 20.71%).**

This is unexpected and MUST be investigated before spending another large training run.

**Evidence:** `experiments/results/context_forensic_diagnostic/context_forensic_diagnostic.json`

---

### Phase 11: Corrected Trajectory Analysis (August 27, 2026) - RECOVERED DATA

**Purpose:** Reco...[truncated]

### Phase 11: Corrected Trajectory Analysis (August 27, 2026) - RECOVERED DATA

**Purpose:** Reco...[truncated]

### Phase 11: Corrected Trajectory Analysis (August 27, 2026) - RECOVERED DATA

**Purpose:** Recover context benefit from existing saved results (NO rerun)

**Issues Fixed:**
1. Context benefit was -Infinity due to invalid ctx1_loss
2. Checkpoint ordering was lexicographic (1000, 10000, 11000...)
3. Previous conclusion "model learned ONLY unigram statistics" was incorrect

**Method:**
- Loaded existing trajectory results (all 50 checkpoints)
- Recomputed context benefit using ctx4_loss (valid)
- Sorted checkpoints numerically
- NO checkpoint inference rerun

**Results:**

**Context Benefit (RECOVERED):**
| Metric | Step 1k | Step 50k | Interpretation |
|--------|---------|----------|----------------|
| Context Benefit (ctx4) | 0.13 | 0.44 | POSITIVE - context helps |
| Context Benefit (ctx8) | 0.10 | 0.23 | POSITIVE - context helps |
| Shuffle Effect | 0.18 | 0.81 | POSITIVE - shuffling hurts |

**Model vs Baselines (VERIFIED):**
| Checkpoint | Model Top-1 | Unigram | Bigram | vs Unigram | vs Bigram |
|------------|-------------|---------|--------|------------|-----------|
| Step 1k | 0.04% | 3.85% | 21.93% | -3.81% | -21.89% |
| Step 39k (best) | 3.79% | 3.85% | 21.93% | -0.06% | -18.14% |
| Step 50k | 3.77% | 3.85% | 21.93% | -0.08% | -18.16% |

**Key Findings:**

1. **Context Benefit is POSITIVE (not zero):**
   - ctx4_loss > full_loss at ALL 50 checkpoints
   - Benefit: ~0.4 (compared to loss ~8)
   - Relative improvement: ~5%

2. **Shuffle Effect is POSITIVE:**
   - shuffled_loss > full_loss at ALL 50 checkpoints
   - Effect: ~0.7
   - Context ORDER matters slightly

3. **Model NEVER Exceeds Unigram Baseline:**
   - Best model: 3.79% (step 39000)
   - Unigram: 3.85%
   - Gap: -0.06%

4. **Model is ~18% WORSE Than Bigram:**
   - Consistent across all checkpoints
   - Simple bigram outperforms neural network by 5.8x

**CORRECTED CONCLUSION:**

The model learned MINIMAL contextual information (context benefit positive), but remains substantially weaker than a simple bigram baseline.

**PREVIOUS CONCLUSION SUPERSEDED:**

The claim "model learned ONLY unigram statistics" is INCORRECT. The model shows minimal contextual learning (context benefit = 0.4, shuffle effect = 0.7), but this is too weak to exceed unigram baseline or approach bigram baseline.

**ROOT CAUSE: STILL UNKNOWN**

**Evidence:** `experiments/results/checkpoint_trajectory_analysis/CORRECTED_TRAJECTORY_ANALYSIS.txt`
**Data:** `experiments/results/checkpoint_trajectory_analysis/corrected_trajectory_data.json`

---
...[truncated]

---

### Phase 12: Padding Hypothesis Verification (August 30, 2026) - RULED OUT

**Purpose:** Determine if padding tokens contaminate attention and cause weak contextual learning.

**Hypothesis:** Model attends to padding tokens during training, corrupting hidden states and preventing contextual learning.

**Method:**
- Analyzed 50,000 training sequences for padding percentage
- Computed attention events to padding vs real tokens
- Verified causal mask behavior

**Results:**

**Padding Statistics:**
| Metric | Value |
|--------|-------|
| Total sequences | 50,000 |
| Total real tokens | 9,219,277 |
| Total padding tokens | 3,580,723 |
| Overall padding % | 27.97% |
| Median padding % | 30.86% |

**High Padding Sequences:**
| Padding Level | Count | Percentage |
|---------------|-------|------------|
| >90% padding | 12 | 0.0% |
| >80% padding | 14 | 0.0% |
| >70% padding | 96 | 0.2% |
| >50% padding | 4,175 | 8.3% |

**Attention Analysis:**
| Metric | Value |
|--------|-------|
| Attention events to real tokens | 9,262,522 (100%) |
| Attention events to padding | 0 (0%) |

**CRITICAL FINDING:** Causal mask prevents attending to padding tokens.

Padding tokens are at END of sequences (after real tokens). Causal mask prevents attending to future positions. Therefore model NEVER attends to padding during training.

**VERDICT:**

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Padding contaminates attention | ❌ RULED OUT | 0% attention to padding |
| Weak contextual learning from padding | ❌ NOT SUPPORTED | Causal mask prevents contamination |

**Evidence:** `experiments/results/padding_hypothesis_verification/` (output from diagnostic)

---

### Phase 13: Training vs Evaluation Pipeline Audit (August 30, 2026) - VERIFIED DIFFERENCES

**Purpose:** Identify discrepancies between training and evaluation pipelines that could explain weak contextual learning.

**Method:**
- Compared training script (`scripts/train_optimized.py`) with evaluation script (`experiments/checkpoint_trajectory_analysis.py`)
- Analyzed tokenization, sequence construction, loss calculation, attention masking

**Results:**

| # | Discrepancy | Status | Impact |
|---|-------------|--------|--------|
| 1 | EOS token in training, NOT in evaluation | ❌ VERIFIED ISSUE | Model trained on EOS, evaluated without EOS |
| 2 | Padding in training, NOT in evaluation | ⚠️ DIFFERENCE | Not critical (causal mask handles) |
| 3 | Input/target shifting | ✅ NOT AN ISSUE | Both shift correctly |
| 4 | Loss masking (ignore_index) | ⚠️ DIFFERENCE | May affect loss comparability |
| 5 | Accuracy masking | ✅ NOT AN ISSUE | Correctly implemented |
| 6 | Sequence length (fixed vs variable) | ⚠️ DIFFERENCE | May affect generalization |
| 7 | Positional embeddings | ⚠️ UNKNOWN | May affect generalization |

**CRITICAL FINDING:**

Training adds EOS token to every sequence:
```python
tokens = tokenizer.encode(text)[:max_length - 1]
tokens = tokens + [tokenizer.eos_token_id]  # EOS added
```

Evaluation does NOT add EOS:
```python
tokens = tokenizer.encode(story)[:256]  # No EOS
```

**Objective Function Mismatch:**
- Training: Model learns to predict EOS at end of sequences
- Evaluation: Model NEVER tested on EOS prediction

**Evidence:** `experiments/training_eval_pipeline_audit.py`

---

### Phase 14: EOS Isolation Diagnostic (August 30, 2026) - RULED OUT

**Purpose:** Determine if EOS token mismatch materially affects next-token prediction.

**Hypothesis:** Model performance differs when evaluated with vs without EOS token.

**Method:**
- Evaluated same 50k checkpoint on same held-out stories
- Method A: Current evaluation (WITHOUT EOS)
- Method B: Training-style evaluation (WITH EOS)
- Fair comparison on identical prediction positions

**Results:**

**Performance Comparison:**
| Metric | WITHOUT EOS | WITH EOS | Difference |
|--------|-------------|----------|------------|
| Cross-Entropy | 7.6989 | 7.6981 | -0.0008 |
| Perplexity | 2205.84 | 2204.15 | -1.69 |
| Top-1 Accuracy | 3.77% | 3.75% | -0.02% |

**Fair Comparison (Shared Positions):**
| Method | Top-1 Accuracy | Difference |
|--------|----------------|------------|
| WITHOUT EOS | 3.7739% | - |
| WITH EOS | 3.7739% | 0.0000% |

**Context Benefit:**
| Method | Benefit | Difference |
|--------|---------|------------|
| WITHOUT EOS | 0.1328 | - |
| WITH EOS | 0.1328 | 0.0000 |

**EOS Prediction:**
| Metric | Value |
|--------|-------|
| EOS positions tested | 500 |
| EOS correctly predicted | 0 |
| EOS accuracy | 0.00% |

**VERDICT:**

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| EOS mismatch affects prediction | ❌ NOT SUPPORTED | 0.0000% difference on shared positions |

**ADDITIONAL FINDING:** Model has 0% accuracy on EOS prediction despite being trained to predict it.

**Evidence:** `experiments/eos_isolation_diagnostic.py`

---

### Phase 15: Training Dynamics Diagnostic (August 30, 2026) - CRITICAL DISCOVERY

**Purpose:** Determine if training dynamics/gradient updates are capable of learning contextual relationships.

**Questions:**
1. Are gradients flowing through all layers?
2. Are update-to-parameter ratios healthy?
3. Is attention learning contextual patterns?
4. Are embeddings dominating attention (unigram vs context)?

**Method:**
- Loaded checkpoints at steps 1k, 10k, 25k, 50k
- Computed gradients on held-out data (READ-ONLY, no training)
- Analyzed gradient distribution by layer type
- Compared gradient magnitudes across training
- Analyzed attention-specific gradients (Q, K, V, O projections)

**Results:**

**Part 1: Gradient Flow at Step 50k**
| Layer Type | Avg Gradient | Avg Ratio | Status |
|------------|--------------|-----------|--------|
| Embedding | 0.483 | 0.0014 | HEALTHY |
| Attention | 0.463 | 1.33 | TOO LARGE |
| FeedForward | 0.698 | 1.82 | TOO LARGE |
| LayerNorm | 0.614 | 1.51 | TOO LARGE |

**Part 2: Gradient Evolution Across Training**
| Step | Total Norm | Avg Loss |
|------|------------|----------|
| 1k | 14.21 | 8.61 |
| 10k | 10.15 | 8.01 |
| 25k | 9.23 | 7.78 |
| 50k | 9.12 | 7.72 |

**Part 3: Parameter Changes Across Training**
| Layer Type | 1k→10k | 10k→25k | 25k→50k |
|------------|--------|---------|---------|
| Embedding | -8.39% | -10.56% | -5.18% |
| Attention | -7.76% | -10.13% | -4.99% |
| FeedForward | -7.77% | -10.12% | -4.99% |
| LayerNorm | -7.25% | -9.60% | -4.70% |

**Part 4: CRITICAL - Attention Layer Gradients**

| Projection | Step 1k | Step 10k | Step 25k | Step 50k | Trend |
|------------|---------|----------|----------|----------|-------|
| Q_PROJ | 0.140 | 0.067 | 0.037 | **0.028** | Decreasing |
| K_PROJ | 0.045 | 0.020 | 0.009 | **0.006** | Near-zero |
| V_PROJ | 1.441 | 0.961 | 0.826 | **0.786** | Normal |
| O_PROJ | 1.556 | 1.116 | 1.037 | **1.031** | Normal |

**K_PROJ gradients are 10-100x SMALLER than V_PROJ and O_PROJ!**

**Part 4: K_PROJ Bias Gradients**

All 6 layers have K_PROJ bias gradients = EXACTLY ZERO:
```
blocks.0.attn.k_proj.bias: grad_norm=0.000000
blocks.1.attn.k_proj.bias: grad_norm=0.000000
blocks.2.attn.k_proj.bias: grad_norm=0.000000
blocks.3.attn.k_proj.bias: grad_norm=0.000000
blocks.4.attn.k_proj.bias: grad_norm=0.000000
blocks.5.attn.k_proj.bias: grad_norm=0.000000
```

**Part 5: Embedding vs Attention Balance**

| Metric | Value |
|--------|-------|
| Embedding avg gradient | 0.483 |
| Attention avg gradient | 0.463 |
| Ratio (emb/attn) | 1.04x |

**VERDICT:** Gradients are balanced - embeddings do NOT dominate.

**CRITICAL FINDING:**

| Finding | Status | Evidence |
|---------|--------|----------|
| Gradients flow through all layers | ✅ VERIFIED | Total norm > 0 |
| K_PROJ gradients near-zero | ✅ **VERIFIED** | 0.006 vs 0.786 (V_PROJ) |
| K_PROJ bias gradients = 0 | ✅ **VERIFIED** | All 6 layers = 0.0 |
| V_PROJ and O_PROJ normal | ✅ VERIFIED | Gradients ~1.0 |
| Embeddings do NOT dominate | ✅ VERIFIED | Ratio = 1.04x |

**IMPLICATION:**

In attention mechanism:
```
Q = Query (what I'm looking for)
K = Key (what each position represents)
V = Value (what information to retrieve)

Attention = softmax(Q @ K^T / sqrt(d)) @ V
```

If K_PROJ has no gradients:
- Model cannot learn what each position represents
- Attention cannot learn to focus on relevant positions
- Model cannot learn contextual relationships
- This explains weak contextual learning!

**Evidence:** `experiments/training_dynamics_diagnostic.py`

---

### Phase 16: Attention Weight Analysis (August 31, 2026) - DIFFUSE PATTERNS

**Purpose:** Determine whether attention heads are learning meaningful contextual/token-to-token relationships.

**Method:**
- Extracted attention weights from all 6 layers × 8 heads on 50 held-out stories
- Measured attention patterns: previous token, first token, self-attention
- Computed entropy (measure of focus vs diffusion)
- Compared attention to relevant tokens vs random baseline
- Tested attention on original vs shuffled context
- Analyzed head specialization across layers

**Results:**

**Attention Pattern Summary (All 48 Heads):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Previous token attention | ~0.04-0.05 | Low |
| First token attention | ~0.04-0.05 | Low |
| Normalized entropy | ~0.76 | Near maximum (diffuse) |
| Pattern classification | DIFFUSE | All 48 heads |

**Relevant vs Random Token Attention:**
| Metric | Value |
|--------|-------|
| Attention to relevant tokens | 0.031806 |
| Attention to random tokens | 0.029096 |
| Ratio (relevant/random) | 1.09x |

**Original vs Shuffled Context:**
| Metric | Value |
|--------|-------|
| Original context entropy | 3.3936 |
| Shuffled context entropy | 3.3934 |
| Difference | 0.0003 |

**Head Specialization:**
| Metric | Value |
|--------|-------|
| Specialized heads | 1.5/48 (3.1%) |
| Diffuse heads | 46.5/48 (96.9%) |

**VERDICT:**

| Finding | Status | Evidence |
|---------|--------|----------|
| All 48 heads show diffuse/uniform attention | ✅ VERIFIED | Entropy ~0.76, no specialization |
| Attention does NOT focus on relevant tokens | ✅ VERIFIED | Ratio = 1.09x (barely above random) |
| Attention cannot distinguish original from shuffled | ✅ VERIFIED | Diff = 0.0003 |
| Heads show no meaningful specialization | ✅ VERIFIED | Only 3.1% specialized |

**Evidence:** `experiments/attention_weight_analysis.py`

---

### Phase 17: Causal Attention Ablation (August 31, 2026) - FUNCTIONALLY UNIMPORTANT

**Purpose:** Determine whether learned attention patterns causally contribute to prediction.

**Method:**
- Evaluated step 50k checkpoint on 500 held-out stories
- Compared three attention modes:
  1. Normal (learned) attention
  2. Uniform attention weights (destroy learned patterns)
  3. Random attention weights (maximum disruption)
- Measured cross-entropy and Top-1 accuracy for each mode

**Results:**

**Performance Comparison:**

| Attention Mode | Cross-Entropy | Top-1 Accuracy |
|----------------|---------------|----------------|
| Normal (learned) | 7.6981 | 3.7522% |
| Uniform (destroyed) | 7.6980 | 3.7672% |
| Random (destroyed) | 7.6995 | 3.7729% |

**Degradation from Normal Attention:**

| Comparison | CE Difference | Accuracy Difference |
|------------|---------------|---------------------|
| Normal → Uniform | -0.0001 | +0.0150% |
| Normal → Random | +0.0014 | +0.0207% |

**Relative Degradation (% change from normal):**

| Comparison | CE % Change | Accuracy % Change |
|------------|-------------|-------------------|
| Normal → Uniform | -0.001% | +0.40% |
| Normal → Random | +0.018% | +0.55% |

**CRITICAL FINDING:**
- Max relative CE change: 0.018%
- Max relative accuracy change: 0.55%
- Uniform attention performs slightly BETTER than learned attention (+0.40% accuracy)
- Random attention performs slightly BETTER than learned attention (+0.55% accuracy)

**VERDICT:**

| Finding | Status | Evidence |
|---------|--------|----------|
| Destroying attention causes < 1% degradation | ✅ VERIFIED | Max = 0.55% |
| Learned attention contributes LITTLE to prediction | ✅ VERIFIED | Uniform/Random ≈ Normal |
| Learned attention patterns provide minimal functional benefit | ✅ VERIFIED | Max CE change = 0.018% |

**OBSERVATION:**
The learned attention weighting provides little measurable benefit over uniform or random attention weighting. This does NOT prove the entire attention mechanism is mathematically equivalent to averaging - we only tested specific attention weight replacements.

**Evidence:** `experiments/causal_attention_ablation.py`

---

## Current Investigation Status

### VERIFIED FINDINGS:

1. ✅ Ternary quantization implementation is CORRECT
2. ✅ Forward pass implementation is CORRECT
3. ✅ Training objective implementation is CORRECT
4. ✅ Causal masking is CORRECT
5. ✅ Padding does NOT contaminate attention (0% attention to padding)
6. ✅ EOS mismatch does NOT affect performance
7. ⚠️ K_PROJ/Q_PROJ gradients are small but this is NOT K_PROJ-specific (Phase 15.5)
8. ✅ **All 48 attention heads show diffuse/uniform patterns** (Phase 16)
9. ✅ **Learned attention provides little functional benefit over uniform/random** (Phase 17)

### RULED OUT:

1. ❌ Ternary quantization bug
2. ❌ Forward pass bug
3. ❌ Target alignment bug
4. ❌ Causal masking bug
5. ❌ Padding contamination
6. ❌ EOS mismatch
7. ❌ Learning rate (tested range)
8. ❌ Embeddings dominating attention
9. ❌ K_PROJ-specific gradient issue (affects Q equally)

### REMAINING QUESTIONS:

1. **WHY does attention provide little functional benefit?**
   - All 48 heads show diffuse patterns (Phase 16)
   - Learned attention ≈ uniform attention in performance (Phase 17)
   - This could explain why model ≈ unigram baseline
   - But causality is NOT established - this is a correlation

2. **Is this a fundamental issue with ternary attention?**
   - Need to compare with FP32 attention patterns
   - Need to test if FP32 attention shows more specialization
   - Need to test if FP32 attention provides more functional benefit

3. **Is this a training issue or architecture issue?**
   - Training dynamics: attention learned, but provides no benefit
   - Could be: ternary capacity limitation, optimization issue, or something else
   - Requires FP32 comparison experiment

### NEXT STEP:

The attention mechanism shows diffuse patterns and provides little functional benefit. This is CORRELATED with weak contextual learning, but causality is NOT established.

Possible next investigations:
1. Compare with FP32 attention patterns and functional contribution
2. Investigate WHY attention learned diffuse patterns
3. Test if different initialization/architecture would help

---

### Phase 12: Padding Hypothesis Verification (August 30, 2026) - RULED OUT

**Purpose:** Determine if padding tokens contaminate attention and cause weak contextual learning.

**Hypothesis:** Model attends to padding tokens during training, corrupting hidden states and preventing contextual learning.

**Method:**
- Analyzed 50,000 training sequences for padding percentage
- Computed attention events to padding vs real tokens
- Verified causal mask behavior

**Results:**

**Padding Statistics:**
| Metric | Value |
|--------|-------|
| Total sequences | 50,000 |
| Total real tokens | 9,219,277 |
| Total padding tokens | 3,580,723 |
| Overall padding % | 27.97% |
| Median padding % | 30.86% |

**High Padding Sequences:**
| Padding Level | Count | Percentage |
|---------------|-------|------------|
| >90% padding | 12 | 0.0% |
| >80% padding | 14 | 0.0% |
| >70% padding | 96 | 0.2% |
| >50% padding | 4,175 | 8.3% |

**Attention Analysis:**
| Metric | Value |
|--------|-------|
| Attention events to real tokens | 9,262,522 (100%) |
| Attention events to padding | 0 (0%) |

**CRITICAL FINDING:** Causal mask prevents attending to padding tokens.

Padding tokens are at END of sequences (after real tokens). Causal mask prevents attending to future positions. Therefore model NEVER attends to padding during training.

**VERDICT:**

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Padding contaminates attention | ❌ RULED OUT | 0% attention to padding |
| Weak contextual learning from padding | ❌ NOT SUPPORTED | Causal mask prevents contamination |

**Evidence:** `experiments/results/padding_hypothesis_verification/` (output from diagnostic)

---

### Phase 13: Training vs Evaluation Pipeline Audit (August 30, 2026) - VERIFIED DIFFERENCES

**Purpose:** Identify discrepancies between training and evaluation pipelines that could explain weak contextual learning.

**Method:**
- Compared training script (`scripts/train_optimized.py`) with evaluation script (`experiments/checkpoint_trajectory_analysis.py`)
- Analyzed tokenization, sequence construction, loss calculation, attention masking

**Results:**

| # | Discrepancy | Status | Impact |
|---|-------------|--------|--------|
| 1 | EOS token in training, NOT in evaluation | ❌ VERIFIED ISSUE | Model trained on EOS, evaluated without EOS |
| 2 | Padding in training, NOT in evaluation | ⚠️ DIFFERENCE | Not critical (causal mask handles) |
| 3 | Input/target shifting | ✅ NOT AN ISSUE | Both shift correctly |
| 4 | Loss masking (ignore_index) | ⚠️ DIFFERENCE | May affect loss comparability |
| 5 | Accuracy masking | ✅ NOT AN ISSUE | Correctly implemented |
| 6 | Sequence length (fixed vs variable) | ⚠️ DIFFERENCE | May affect generalization |
| 7 | Positional embeddings | ⚠️ UNKNOWN | May affect generalization |

**CRITICAL FINDING:**

Training adds EOS token to every sequence:
```python
tokens = tokenizer.encode(text)[:max_length - 1]
tokens = tokens + [tokenizer.eos_token_id]  # EOS added
```

Evaluation does NOT add EOS:
```python
tokens = tokenizer.encode(story)[:256]  # No EOS
```

**Objective Function Mismatch:**
- Training: Model learns to predict EOS at end of sequences
- Evaluation: Model NEVER tested on EOS prediction

**Evidence:** `experiments/training_eval_pipeline_audit.py`

---

### Phase 14: EOS Isolation Diagnostic (August 30, 2026) - RULED OUT

**Purpose:** Determine if EOS token mismatch materially affects next-token prediction.

**Hypothesis:** Model performance differs when evaluated with vs without EOS token.

**Method:**
- Evaluated same 50k checkpoint on same held-out stories
- Method A: Current evaluation (WITHOUT EOS)
- Method B: Training-style evaluation (WITH EOS)
- Fair comparison on identical prediction positions

**Results:**

**Performance Comparison:**
| Metric | WITHOUT EOS | WITH EOS | Difference |
|--------|-------------|----------|------------|
| Cross-Entropy | 7.6989 | 7.6981 | -0.0008 |
| Perplexity | 2205.84 | 2204.15 | -1.69 |
| Top-1 Accuracy | 3.77% | 3.75% | -0.02% |

**Fair Comparison (Shared Positions):**
| Method | Top-1 Accuracy | Difference |
|--------|----------------|------------|
| WITHOUT EOS | 3.7739% | - |
| WITH EOS | 3.7739% | 0.0000% |

**Context Benefit:**
| Method | Benefit | Difference |
|--------|---------|------------|
| WITHOUT EOS | 0.1328 | - |
| WITH EOS | 0.1328 | 0.0000 |

**EOS Prediction:**
| Metric | Value |
|--------|-------|
| EOS positions tested | 500 |
| EOS correctly predicted | 0 |
| EOS accuracy | 0.00% |

**VERDICT:**

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| EOS mismatch affects prediction | ❌ NOT SUPPORTED | 0.0000% difference on shared positions |

**ADDITIONAL FINDING:** Model has 0% accuracy on EOS prediction despite being trained to predict it.

**Evidence:** `experiments/eos_isolation_diagnostic.py`

---

### Phase 15: Training Dynamics Diagnostic (August 30, 2026) - CRITICAL DISCOVERY

**Purpose:** Determine if training dynamics/gradient updates are capable of learning contextual relationships.

**Questions:**
1. Are gradients flowing through all layers?
2. Are update-to-parameter ratios healthy?
3. Is attention learning contextual patterns?
4. Are embeddings dominating attention (unigram vs context)?

**Method:**
- Loaded checkpoints at steps 1k, 10k, 25k, 50k
- Computed gradients on held-out data (READ-ONLY, no training)
- Analyzed gradient distribution by layer type
- Compared gradient magnitudes across training
- Analyzed attention-specific gradients (Q, K, V, O projections)

**Results:**

**Part 1: Gradient Flow at Step 50k**
| Layer Type | Avg Gradient | Avg Ratio | Status |
|------------|--------------|-----------|--------|
| Embedding | 0.483 | 0.0014 | HEALTHY |
| Attention | 0.463 | 1.33 | TOO LARGE |
| FeedForward | 0.698 | 1.82 | TOO LARGE |
| LayerNorm | 0.614 | 1.51 | TOO LARGE |

**Part 2: Gradient Evolution Across Training**
| Step | Total Norm | Avg Loss |
|------|------------|----------|
| 1k | 14.21 | 8.61 |
| 10k | 10.15 | 8.01 |
| 25k | 9.23 | 7.78 |
| 50k | 9.12 | 7.72 |

**Part 3: Parameter Changes Across Training**
| Layer Type | 1k→10k | 10k→25k | 25k→50k |
|------------|--------|---------|---------|
| Embedding | -8.39% | -10.56% | -5.18% |
| Attention | -7.76% | -10.13% | -4.99% |
| FeedForward | -7.77% | -10.12% | -4.99% |
| LayerNorm | -7.25% | -9.60% | -4.70% |

**Part 4: CRITICAL - Attention Layer Gradients**

| Projection | Step 1k | Step 10k | Step 25k | Step 50k | Trend |
|------------|---------|----------|----------|----------|-------|
| Q_PROJ | 0.140 | 0.067 | 0.037 | **0.028** | Decreasing |
| K_PROJ | 0.045 | 0.020 | 0.009 | **0.006** | Near-zero |
| V_PROJ | 1.441 | 0.961 | 0.826 | **0.786** | Normal |
| O_PROJ | 1.556 | 1.116 | 1.037 | **1.031** | Normal |

**K_PROJ gradients are 10-100x SMALLER than V_PROJ and O_PROJ!**

**Part 4: K_PROJ Bias Gradients**

All 6 layers have K_PROJ bias gradients = EXACTLY ZERO:
```
blocks.0.attn.k_proj.bias: grad_norm=0.000000
blocks.1.attn.k_proj.bias: grad_norm=0.000000
blocks.2.attn.k_proj.bias: grad_norm=0.000000
blocks.3.attn.k_proj.bias: grad_norm=0.000000
blocks.4.attn.k_proj.bias: grad_norm=0.000000
blocks.5.attn.k_proj.bias: grad_norm=0.000000
```

**Part 5: Embedding vs Attention Balance**

| Metric | Value |
|--------|-------|
| Embedding avg gradient | 0.483 |
| Attention avg gradient | 0.463 |
| Ratio (emb/attn) | 1.04x |

**VERDICT:** Gradients are balanced - embeddings do NOT dominate.

**CRITICAL FINDING:**

| Finding | Status | Evidence |
|---------|--------|----------|
| Gradients flow through all layers | ✅ VERIFIED | Total norm > 0 |
| K_PROJ gradients near-zero | ✅ **VERIFIED** | 0.006 vs 0.786 (V_PROJ) |
| K_PROJ bias gradients = 0 | ✅ **VERIFIED** | All 6 layers = 0.0 |
| V_PROJ and O_PROJ normal | ✅ VERIFIED | Gradients ~1.0 |
| Embeddings do NOT dominate | ✅ VERIFIED | Ratio = 1.04x |

**IMPLICATION:**

In attention mechanism:
```
Q = Query (what I'm looking for)
K = Key (what each position represents)
V = Value (what information to retrieve)

Attention = softmax(Q @ K^T / sqrt(d)) @ V
```

If K_PROJ has no gradients:
- Model cannot learn what each position represents
- Attention cannot learn to focus on relevant positions
- Model cannot learn contextual relationships
- This explains weak contextual learning!

**Evidence:** `experiments/training_dynamics_diagnostic.py`

---

### Phase 16: Attention Weight Analysis (August 31, 2026) - DIFFUSE PATTERNS

**Purpose:** Determine whether attention heads are learning meaningful contextual/token-to-token relationships.

**Method:**
- Extracted attention weights from all 6 layers × 8 heads on 50 held-out stories
- Measured attention patterns: previous token, first token, self-attention
- Computed entropy (measure of focus vs diffusion)
- Compared attention to relevant tokens vs random baseline
- Tested attention on original vs shuffled context
- Analyzed head specialization across layers

**Results:**

**Attention Pattern Summary (All 48 Heads):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Previous token attention | ~0.04-0.05 | Low |
| First token attention | ~0.04-0.05 | Low |
| Normalized entropy | ~0.76 | Near maximum (diffuse) |
| Pattern classification | DIFFUSE | All 48 heads |

**Relevant vs Random Token Attention:**
| Metric | Value |
|--------|-------|
| Attention to relevant tokens | 0.031806 |
| Attention to random tokens | 0.029096 |
| Ratio (relevant/random) | 1.09x |

**Original vs Shuffled Context:**
| Metric | Value |
|--------|-------|
| Original context entropy | 3.3936 |
| Shuffled context entropy | 3.3934 |
| Difference | 0.0003 |

**Head Specialization:**
| Metric | Value |
|--------|-------|
| Specialized heads | 1.5/48 (3.1%) |
| Diffuse heads | 46.5/48 (96.9%) |

**VERDICT:**

| Finding | Status | Evidence |
|---------|--------|----------|
| All 48 heads show diffuse/uniform attention | ✅ VERIFIED | Entropy ~0.76, no specialization |
| Attention does NOT focus on relevant tokens | ✅ VERIFIED | Ratio = 1.09x (barely above random) |
| Attention cannot distinguish original from shuffled | ✅ VERIFIED | Diff = 0.0003 |
| Heads show no meaningful specialization | ✅ VERIFIED | Only 3.1% specialized |

**Evidence:** `experiments/attention_weight_analysis.py`

---

### Phase 17: Causal Attention Ablation (August 31, 2026) - FUNCTIONALLY UNIMPORTANT

**Purpose:** Determine whether learned attention patterns causally contribute to prediction.

**Method:**
- Evaluated step 50k checkpoint on 500 held-out stories
- Compared three attention modes:
  1. Normal (learned) attention
  2. Uniform attention weights (destroy learned patterns)
  3. Random attention weights (maximum disruption)
- Measured cross-entropy and Top-1 accuracy for each mode

**Results:**

**Performance Comparison:**

| Attention Mode | Cross-Entropy | Top-1 Accuracy |
|----------------|---------------|----------------|
| Normal (learned) | 7.6981 | 3.7522% |
| Uniform (destroyed) | 7.6980 | 3.7672% |
| Random (destroyed) | 7.6995 | 3.7729% |

**Degradation from Normal Attention:**

| Comparison | CE Difference | Accuracy Difference |
|------------|---------------|---------------------|
| Normal → Uniform | -0.0001 | -0.0149% |
| Normal → Random | +0.0014 | -0.0207% |

**Relative Degradation (% change from normal):**

| Comparison | CE % Change | Accuracy % Change |
|------------|-------------|-------------------|
| Normal → Uniform | -0.00% | -0.40% |
| Normal → Random | +0.02% | -0.55% |

**CRITICAL FINDING:**
- Max relative degradation: 0.55%
- Uniform attention performs slightly BETTER than learned attention
- Random attention performs nearly identically to learned attention

**VERDICT:**

| Finding | Status | Evidence |
|---------|--------|----------|
| Destroying attention causes < 1% degradation | ✅ VERIFIED | Max = 0.55% |
| Learned attention contributes LITTLE to prediction | ✅ VERIFIED | Uniform ≈ Normal |
| Learned attention patterns provide minimal functional benefit | ✅ VERIFIED | 0.02% CE difference |

**OBSERVATION:**
The learned attention weighting provides little measurable benefit over uniform or random attention weighting. This does NOT prove the entire attention mechanism is mathematically equivalent to averaging - we only tested specific attention weight replacements.

**Evidence:** `experiments/causal_attention_ablation.py`

---

## Current Investigation Status

### VERIFIED FINDINGS:

1. ✅ Ternary quantization implementation is CORRECT
2. ✅ Forward pass implementation is CORRECT
3. ✅ Training objective implementation is CORRECT
4. ✅ Causal masking is CORRECT
5. ✅ Padding does NOT contaminate attention (0% attention to padding)
6. ✅ EOS mismatch does NOT affect performance
7. ⚠️ K_PROJ/Q_PROJ gradients are small but this is NOT K_PROJ-specific (Phase 15.5)
8. ✅ **All 48 attention heads show diffuse/uniform patterns** (Phase 16)
9. ✅ **Learned attention provides little functional benefit over uniform/random** (Phase 17)

### RULED OUT:

1. ❌ Ternary quantization bug
2. ❌ Forward pass bug
3. ❌ Target alignment bug
4. ❌ Causal masking bug
5. ❌ Padding contamination
6. ❌ EOS mismatch
7. ❌ Learning rate (tested range)
8. ❌ Embeddings dominating attention
9. ❌ K_PROJ-specific gradient issue (affects Q equally)

### REMAINING QUESTIONS:

1. **WHY does attention provide little functional benefit?**
   - All 48 heads show diffuse patterns (Phase 16)
   - Learned attention ≈ uniform attention in performance (Phase 17)
   - This could explain why model ≈ unigram baseline
   - But causality is NOT established - this is a correlation

2. **Is this a fundamental issue with ternary attention?**
   - Need to compare with FP32 attention patterns
   - Need to test if FP32 attention shows more specialization
   - Need to test if FP32 attention provides more functional benefit

3. **Is this a training issue or architecture issue?**
   - Training dynamics: attention learned, but provides no benefit
   - Could be: ternary capacity limitation, optimization issue, or something else
   - Requires FP32 comparison experiment

### NEXT STEP:

The attention mechanism shows diffuse patterns and provides little functional benefit. This is CORRELATED with weak contextual learning, but causality is NOT established.

Possible next investigations:
1. Compare with FP32 attention patterns and functional contribution
2. Investigate WHY attention learned diffuse patterns
3. Test if different initialization/architecture would help

---

## Key Evidence Tables

### FP32 vs Ternary: Corrected Comparison

| Metric | FP32 5k | Ternary 50k | Difference | Interpretation |
|--------|---------|-------------|------------|----------------|
| **Real Token Accuracy** | **5.46%** | **4.60%** | **0.86%** | Similar performance |
| Padding Accuracy | 100.00% | 0.00% | 100% | FP32 predicts padding |
| Total Accuracy | 26.89% | 3.56% | 23.33% | Inflated by padding |
| Training Accuracy | 94.55% | 4.09% | 90.46% | FP32 memorizes, Ternary doesn't |

**Conclusion:** Both fail on real held-out tokens → problem is not quantization

---

### Learning Trajectory (Ternary Model)

| Steps | Loss | Perplexity | Top-1 Accuracy |
|-------|------|------------|----------------|
| 500 | 8.73 | 6173 | 0.00% |
| 1000 | 8.72 | 6121 | 0.00% |
| 2000 | 8.70 | 6027 | 0.00% |
| 3000 | 8.69 | 5962 | 0.00% |
| 5000 | 8.60 | 5422 | 0.06% |
| **50000** | **8.50** | **4898** | **4.09%** |

**Observation:** Loss decreases, accuracy improves slightly, but remains very low (~4%)

---

### Ternary Weight Distribution (Corrected)

| Checkpoint | -1% | 0% | +1% | Active% | Status |
|------------|-----|-----|-----|---------|--------|
| lr_001_constant (5k) | 34.5% | 31.0% | 34.5% | 69.0% | ✅ Healthy |
| lr_003_constant (5k) | 34.5% | 31.0% | 34.5% | 69.0% | ✅ Healthy |
| 50k checkpoint | ~33% | ~34% | ~33% | ~66% | ✅ Healthy |

**Note:** Earlier documents showed "90% zeros" - this was analyzing UNSCALED weights
**Correct method:** Analyze scaled weights (W / mean(|W|))
**Evidence:** `BITNET_FORMULA_VERIFICATION.md`

---

## Key Evidence Tables

### FP32 vs Ternary: Corrected Comparison

| Metric | FP32 5k | Ternary 50k | Difference | Interpretation |
|--------|---------|-------------|------------|----------------...[truncated]

| Hypothesis | First Raised | Evidence Then | Status Then | New Evidence | Current Status |
|------------|--------------|---------------|-------------|--------------|----------------|
| **Ternary quantization implementation broken** | Aug 18 | 90% zeros in weights | SUSPECTED | Scaled weights: 34.31/35% | ✅ IMPLEMENTATION CORRECT (but capacity may be limited) |
| **Forward pass bugs** | Aug 18 | Unverified | SUSPECTED | All 11 stages verified | ✅ IMPLEMENTATION CORRECT |
| **Training objective bugs** | Aug 18 | Unverified | SUSPECTED | Target shifting verified | ✅ IMPLEMENTATION CORRECT |
| **Learning rate insufficient** | Aug 18 | Unverified | POSSIBLE | Phase 9: all LR values identical | ❌ RULED OUT (tested range: 3e-5 to 3e-4) |
| **Logging bugs** | Aug 18 | Loss ~4300 | CRITICAL | Fixed, correct loss ~8.7 | ✅ FIXED |
| **Ternary quantization causes gibberish** | Aug 19 | FP32 26.89% vs Ternary 3.56% | SUSPECTED | Phase 6: padding inflation, real tokens 5.46% vs 4.60% | ⚠️ IMPLEMENTATION CORRECT, but capacity NOT ruled out (FP32 only tested on 20 stories) |
| **LR decay causes plateau** | Aug 26 | Longitudinal: loss stabilizes at 15k | SUSPECTED | Phase 9: higher LR doesn't help | ❌ RULED OUT (within tested range) |
| **Model uses contextual information** | Aug 27 | Assumed | ASSUMED | Phase 10: context benefit = 0, shuffling helps slightly | ⚠️ MINIMAL (Phase 11: benefit = 0.4, positive but small) |
| **Model learned bigram patterns** | Aug 27 | Assumed | ASSUMED | Phase 10: Model 3.69% vs Bigram 20.71% | ❌ RULED OUT - model 18% WORSE than bigram |
| **Padding contaminates attention** | Aug 27 | Phase 10: 95% padding in sample | SUSPECTED | Phase 12: 0% attention to padding (causal mask) | ❌ RULED OUT |
| **EOS mismatch affects performance** | Aug 27 | Phase 13: train/eval difference | SUSPECTED | Phase 14: 0.0000% difference with/without EOS | ❌ NOT SUPPORTED |
| **Training dynamics prevent contextual learning** | Aug 30 | Phase 15: investigate gradients | NEW | K/Q gradients small but equal | ⚠️ NOT K_PROJ-specific |
| **K_PROJ gradient issue** | Aug 30 | Phase 15: K_PROJ = 0.006 vs V_PROJ = 0.786 | NEW | K and Q equally small, weights changed 22% | ⚠️ Normal attention gradient flow? |
| **Attention shows diffuse/uniform patterns** | Aug 31 | Phase 16: all 48 heads diffuse | NEW | Entropy ~0.76, no specialization, relevant/random ratio 1.09x | ✅ VERIFIED |
| **Learned attention provides little benefit** | Aug 31 | Phase 17: causal ablation | NEW | Max degradation 0.55%, uniform ≈ normal | ✅ VERIFIED |
| **Attention failure causes weak contextual learning** | Aug 31 | Phase 16-17: diffuse + no benefit | NEW | Correlation found, causation NOT established | ⚠️ PLAUSIBLE BUT UNVERIFIED |
| **Padding contaminates attention** | Aug 27 | Phase 10: 95% padding in sample | SUSPECTED | Phase 12: 0% attention to padding (causal mask) | ❌ RULED OUT |
| **EOS mismatch affects performance** | Aug 27 | Phase 13: train/eval difference | SUSPECTED | Phase 14: 0.0000% difference with/without EOS | ❌ NOT SUPPORTED |
| **Training dynamics prevent contextual learning** | Aug 30 | Phase 15: investigate gradients | NEW | K/Q gradients small but equal | ⚠️ NOT K_PROJ-specific |
| **K_PROJ gradient issue** | Aug 30 | Phase 15: K_PROJ = 0.006 vs V_PROJ = 0.786 | NEW | K and Q equally small, weights changed 22% | ⚠️ Normal attention gradient flow? |
| **Attention shows diffuse/uniform patterns** | Aug 31 | Phase 16: all 48 heads diffuse | NEW | Entropy ~0.76, no specialization, relevant/random ratio 1.09x | ✅ VERIFIED |
| **Learned attention provides little benefit** | Aug 31 | Phase 17: causal ablation | NEW | Max degradation 0.55%, uniform ≈ normal | ✅ VERIFIED |
| **Attention failure causes weak contextual learning** | Aug 31 | Phase 16-17: diffuse + no benefit | NEW | Correlation found, causation NOT established | ⚠️ PLAUSIBLE BUT UNVERIFIED |
| **Training/data pipeline issue** | Aug 27 | Phase 10: model worse than bigram | NEW | Model = Unigram, < Bigram | ⚠️ NEEDS INVESTIGATION |
| **Target/context alignment issue** | Aug 27 | Phase 10: minimal context benefit | NEW | Model shows minimal context learning | ⚠️ NEEDS INVESTIGATION |
| **Tokenizer/sequence construction issue** | Aug 27 | Phase 10: tokenizer verified | NEW | Tokenizer correct, but sequence construction? | ⚠️ NEEDS INVESTIGATION |
| **Insufficient data** | Aug 27 | 1.61 tokens/param | POSSIBLE | Model learns minimal context | ⚠️ POSSIBLE (heuristic, not proven) |
| **Ternary capacity limitation** | Aug 27 | Phase 10: minimal context learning | POSSIBLE | Need FP32 50k comparison | ⚠️ UNKNOWN |
| **Architecture capacity insufficient** | Aug 27 | 6.8M params, 6 layers | POSSIBLE | Need larger model comparison | ⚠️ UNKNOWN |

---

## What We Have Verified

### 1. Ternary Quantization Implementation

**Status:** ✅ IMPLEMENTATION CORRECT

**Evidence:**
- BitNet formula verified correct (`BITNET_FORMULA_VERIFICATION.md`)
- Ternary distribution healthy (34.5%/31.0%/34.5%)
- Reference implementations match our approach (`BITNET_IMPLEMENTATION_COMPARISON.md`)

**[CAVEAT - August 27, 2026]:** Implementation correctness does NOT rule out ternary quantization as a capacity limitation for learning contextual patterns. This requires FP32 comparison on same 50k dataset (NOT yet run).

### 2. Forward Pass Implementation

**Status:** ✅ IMPLEMENTATION CORRECT

**Evidence:**
- All 11 stages verified with measurements (`FORWARD_PASS_VERIFICATION_WITH_EVIDENCE.md`)
- Q/K/V are non-identical
- Attention probabilities sum to 1.0 exactly
- Logits vary by position

**[CAVEAT - August 27, 2026]:** This addresses IMPLEMENTATION correctness, NOT architecture CAPACITY (separate hypothesis).

### 3. Training Objective Implementation

**Status:** ✅ IMPLEMENTATION CORRECT

**Evidence:**
- Target shifting manually verified (`FP32_AUDIT_ROOT_CAUSE.md`)
- Input[i] → Target[i] = Input[i+1] confirmed
- Example: "Once" → target "upon", "upon" → target "a"

### 4. Learning Rate (Tested Range)

**Status:** ❌ RULED OUT as primary cause (within tested range)

**Evidence:**
- Phase 9: Corrected AdaFactor experiment (August 26)
- All LR values (3e-5, 1e-4, 3e-4) produced similar results
- Held-out loss essentially unchanged
- Tested conditions: AdaFactor, optimizer state restored, 1000 continuation steps

**[CAVEAT - August 27, 2026]:** This is experimental evidence for the tested conditions, not universal proof. Different schedules/ranges not tested.

---

## What We Have Evidence Against

### 1. Model Uses Contextual Information

**Status:** ❌ NOT SUPPORTED by current evidence

**Evidence:**
- Context ablation: No meaningful difference between 4 tokens and 256 tokens context
- Context shuffling: Shuffling context order slightly IMPROVES loss (unexpected)
- Model performs IDENTICALLY to unigram baseline

### 2. Model Learned Bigram/Trigram Patterns

**Status:** ❌ NOT SUPPORTED by current evidence

**Evidence:**
- Model Top-1: 3.69%
- Bigram baseline Top-1: 20.71%
- Model is 17% WORSE than simple bigram predictor
- **CRITICAL ANOMALY:** Neural model should not be worse than bigram

### 3. Sampling/Decoding as Primary Cause

**Status:** ❌ STRONGLY DISFAVORED but not absolutely ruled out

**Evidence:**
- Greedy decoding produces repetitive output: "and and and..." or "the the the..."
- All temperature settings produce gibberish
- Problem appears to be in model learning, not decoding method

**[CAVEAT - August 27, 2026]:** Generation/tokenization implementation should remain appropriately qualified as requiring further investigation.

---

## What Remains Unknown

### [ORIGINAL - August 19, 2026] - [SUPERSEDED BY LATER EVIDENCE]

### 1. Is Training Duration Sufficient?

**Current:** 50,000 steps with 20 sentences = 2500 epochs
**Question:** Is there a threshold for coherent generation?
**Next:** Test 100k+ steps

### 2. Is Model Capacity Sufficient?

**Current:** 6.8M params, 4000 vocab
**Question:** Can a 6.8M param model learn language?
**Next:** Test larger models (25M, 100M)

### 3. Is Training Data Size Sufficient?

**Current:** 20 sentences (~200 tokens)
**Question:** Is 200 tokens enough to learn language?
**Next:** Test larger datasets (100, 1000, 10000 sentences)

### 4. What Accuracy Level is Needed?

**Current:** ~5% held-out accuracy
**Question:** Is 5% sufficient for coherent text? What's the threshold?
**Unknown:** No established baseline for this architecture/dataset combo

**[CAVEAT - August 27, 2026]:** These questions were based on 20-sentence training. Current training uses 50k TinyStories (~9.22M tokens). See updated section below.
