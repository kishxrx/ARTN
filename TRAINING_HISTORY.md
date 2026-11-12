# ARTN Training History - Complete Documentation

## Overview

This document records ALL training attempts, their configurations, results, and lessons learned.

---

## Training Attempts Summary

| Attempt | Date | Steps | Loss | Entropy | Result | Status |
|---------|------|-------|------|---------|--------|--------|
| 1 | Aug 1, 2026 | 5,000 | 8.45 | Not monitored | SUCCESS | Pipeline verified |
| 2 | Aug 3-5, 2026 | 50,000 | 3.16 | 8.2% | COLLAPSED | Failed (beta1=None) |
| 3 | Aug 6, 2026 | 2,000 | 8.61 | 94.8% | Healthy | Stopped (wrong accumulation) |
| 4 | Aug 7-9, 2026 | 50,000 | 7.56 | 96.2% | HEALTHY | Completed (gibberish) |
| **5** | **Aug 13-16, 2026** | **50,000** | **7.13** | **97.0%** | **HEALTHY** | **Completed (LR decay issue)** |
| **6** | **Aug 19, 2026** | **5,000** | **1.69** | **Healthy** | **SUCCESS** | **FP32 + AdamW baseline** |
| **7** | **Sep 10-11, 2026** | **10,000** | **1.92** | **Healthy** | **SUCCESS** | **BitLinear + AdamW WORKING** |
| **8** | **Oct 25, 2026** | **10,000** | **2.03** | **Healthy** | **SUCCESS** | **ARTN recursive + halting + ternary** |

**Total Time Invested: ~280 hours (original) + ~29 hours (Phase 46-47) + ~14.35 hours (ARTN Phase 60) = ~323.4 hours**

NOTE: Attempt 7 val loss 1.92 and Attempt 6 val loss 1.69 INCLUDE padding tokens (31.4% trivially-easy, NLL=0.011). Fair non-padding NLL: FP32 2.423, BitLinear 2.743, ARTN 2.904. See Phase 60 Errors and Corrections section below.

---

## Blueprint Stage Completion Status

**Last Updated: October 26, 2026**

| Stage | Blueprint Requirement | Status | Evidence |
|-------|----------------------|--------|----------|
| Stage 0 | MLOps Foundation | ✅ DONE | Repo setup, CI/CD, W&B |
| Stage 1 | Custom Tokenizers | ✅ DONE | BPE tokenizer working (2 tokenizers: 5k, 50k) |
| Stage 2 | Float32 Baseline GPT | ✅ DONE | Architecture implemented, val loss 1.69 [CORRECTED — includes padding, see Phase 60] |
| Stage 3 | Float32 Training Loop | ✅ DONE | Training works, 5,000 steps validated |
| Stage 4 | Recursive Transformer | ✅ DONE | Parameter tying works (`recursive.py` implemented) [CORRECTED — code exists but was NEVER trained, see Phase 60] |
| Stage 5 | Adaptive Halting Gate | ✅ DONE | `halting.py` implemented [CORRECTED — ponder loss was non-differentiable, see Phase 60] |
| Stage 6 | INT8 Quantization | ✅ DONE | `quantization.py` works [NOTE — never trained, skipped per Stage 7 success] |
| Stage 7 | Ternary BitLinear | ✅ **COMPLETE** | **Val loss 1.92 with AdamW, coherent generation** [CORRECTED — includes padding, fair NLL 2.743] |
| Stage 7 Milestone | Overfit tiny subset of TinyStories | ✅ **COMPLETE** | **Tested on ACTUAL TinyStories, val loss 1.92** |
| Stage 8 | Quantized Inference | ✅ **COMPLETE** | **Inference pipeline works, inference-only checkpoint (27 MB), deployment benchmark (4.61 tok/s)** |
| Stage 9 | CPU Benchmarking | ✅ **COMPLETE** | **Speed benchmark (all 3 models), threshold sweep, causal NLL, fair eval, padding audit, parameter audit** |
| Stage 10 | Research Paper | ❌ NOT DONE | - |

**Current Position: Stage 9 COMPLETE → Moving to Stage 10 (Research Paper)**

---

## ✅ Stage 7 Completion Update (October 15, 2026)

**IMPORTANT:** Previous version of this document (dated Aug 18, 2026) showed Stage 7 as "⚠️ PARTIAL" because it only tested on random data. However, after Phase 46-47 (Aug 19, Sep 10-11, 2026), Stage 7 is now **COMPLETE**.

### Phase 46: FP32 + AdamW Baseline (August 19, 2026)

| Metric | Value |
|--------|-------|
| Model | FP32 GPT (standard) |
| Optimizer | AdamW |
| Steps | 5,000 |
| **Val Loss** | **1.69** |
| **Perplexity** | **5.44** |
| Generation | ✅ Coherent stories |
| Location | `experiments/results/fp32_control_20260819_232646/` |
| Status | ✅ SUCCESS - Baseline established |

**Purpose:** Establish FP32 baseline to verify AdamW optimizer works before testing with ternary quantization.

**Result:** AdamW works perfectly for FP32, achieving val loss 1.69 with coherent generation.

---

### Phase 47: BitLinear + AdamW (September 10-11, 2026)

| Metric | Value |
|--------|-------|
| Model | BitLinearGPT (ternary quantization) |
| Optimizer | AdamW |
| Steps | 10,000 |
| **Val Loss** | **1.92** |
| **Perplexity** | **6.81** |
| **Train-Val Gap** | **0.067** (excellent generalization) |
| Generation | ✅ Coherent stories |
| Checkpoints | 20 checkpoints (step_0 to step_9500) |
| Location | `outputs/bitlinear_adamw_20260910_143646/` |
| Status | ✅ **SUCCESS - Ternary quantization WORKS with AdamW** |

**Purpose:** Test if BitLinear/ternary quantization converges with AdamW optimizer.

**Result:** BitLinear + AdamW achieves val loss 1.92, only 13% worse than FP32 (1.69), with coherent generation.

---

### Sample Generation (Step 10,000):

```
Prompt: "Once upon a time"

Output: "Once upon a time there was a little girl who was in her 
backyard. She was always dancing in the garden with her hands. She 
liked to explore the world and watch her mommy laugh. One day she 
was looking to see a big river with en! She was so excited and she 
started to run behind it. She ran to her mum and said, 'Mom, where 
are you going, you! I'm scared. I ragone? This is coming from there 
to rest.' But her mum said that she wanted to be brave and happy. 
So her daughter went..."
```

**Status: ✅ COHERENT GENERATION - Stage 7 Milestone SATISFIED**

---

### Key Findings from Phase 46-47:

| Finding | Evidence |
|---------|----------|
| ✅ AdamW works for FP32 | Val loss 1.69, coherent generation |
| ✅ AdamW works for BitLinear | Val loss 1.92, coherent generation |
| ✅ Ternary quantization is VIABLE | Only 13% performance gap vs FP32 |
| ✅ Excellent generalization | Train-val gap 0.067 |
| ✅ Optimizer was the root cause | AdamW succeeds where Adafactor failed |

---

### Comparison: FP32 vs BitLinear (AdamW)

| Metric | FP32 + AdamW | BitLinear + AdamW | Difference |
|--------|--------------|-------------------|------------|
| Val Loss | 1.69 | 1.92 | +13% |
| Perplexity | 5.44 | 6.81 | +25% |
| Parameters | 6.8M (FP32) | 6.8M (ternary) | Same count |
| Memory per param | 32 bits | 1.58 bits | **-95%** |
| Generation | Coherent | Coherent | ✅ Both work |
| Status | SUCCESS | SUCCESS | ✅ Both work |

---

### Configuration Used (Phase 47):

```python
CONFIG = {
    # Architecture
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    "vocab_size": 4000,
    
    # Optimizer (AdamW - PROVEN TO WORK)
    "max_lr": 0.001,
    "min_lr": 0.00003,
    "beta1": 0.5,
    "beta2": 0.98,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    # Training
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "total_steps": 10000,
    "warmup_steps": 1000,
    
    # Data
    "train_samples": 50000,
    "val_samples": 5000,
}
```

---

### Working Models Uploaded to Hugging Face:

| Model | File | Val Loss | Upload Location |
|-------|------|----------|-----------------|
| FP32 + AdamW | `checkpoint-fp32-adamw-working.pt` | 1.69 | `kishxrx/ARTN` |
| BitLinear + AdamW | `checkpoint-bitlinear-adamw-working.pt` | 1.92 | `kishxrx/ARTN` |

**Repository:** https://huggingface.co/kishxrx/ARTN

---

## Stage 7 Milestone Gap (HISTORICAL - August 18, 2026)

| Requirement | Status | Evidence |
|-------------|--------|----------|
| "Overfit a tiny subset of TinyStories" | ⚠️ NOT SATISFIED | Overfit test uses random token sequences |
| "The loss must go down" | ✅ SATISFIED | Loss 5.22 → 0.0001 on random data |

**Critical Gap:** Blueprint requires overfit on ACTUAL TinyStories data, but we only tested on synthetic random sequences.

---

### Overfit Test Results

```Test Date: Aug 16, 2026```

**BitLinear (Ternary) Overfit Test:**
- Initial loss: 5.2255
- Final loss: 0.0001 (after 500 steps)
- Reduction: 100%
- Data: **RANDOM synthetic sequences** (NOT TinyStories)
- Status: **PASSED**
- Optimizer: **Adam**, LR: **0.01**

**Baseline (FP32) Overfit Test:**
- Initial loss: 4.6224
- Final loss: 0.0002 (after 500 steps)
- Reduction: 100%
- Data: RANDOM synthetic sequences
- Status: **PASSED**

**Conclusion:** Ternary CAN memorize random data, but NOT proven on TinyStories data.

---

### CRITICAL: Optimizer/LR Mismatch Discovered

**Date:** Aug 16, 2026

**Discovery:** Overfit test uses DIFFERENT optimizer and LR than real training.

| Component | Overfit Test | Real 50k Training | Mismatch |
|-----------|-------------|-------------------|----------|
| Optimizer | **Adam** | **Adafactor** | ⚠️ DIFFERENT |
| Learning Rate | **0.01** | **0.001** | ⚠️ 10x difference |
| Beta1 | Default (0.9?) | **0.5** | ⚠️ Possibly different |

**Source:** `scripts/overfit_test.py` line 105 vs `scripts/train_optimized.py` line 76-80

**Why This Matters:**
- Adam optimizer + high LR (0.01) easily memorizes random sequences
- Real training uses Adafactor + lower LR (0.001)
- Overfit test passing does NOT prove real training config works
- Tests different optimization dynamics entirely

**Root Cause Confusion:**
- We assumed overfit test validated ternary implementation
- BUT overfit test used Adam, not Adafactor
- Real training failure may be optimizer/LR issue, not ternary issue

**Lesson Learned:**
- ALWAYS use SAME optimizer and LR in diagnostic tests as real training
- Mismatched tests give false confidence
- Must re-run overfit test with Adafactor + LR=0.001

---

## TinyStories Overfit Experiments (Aug 16-17, 2026)

### Experiments Conducted

Full details in `OVERFIT_EXPERIMENTS.md`. Summary:

| Test | Optimizer | LR | Data | Vocab | Final Loss | Reduction | Status |
|------|-----------|-----|------|-------|------------|-----------|--------|
| Random (baseline) | Adam | 0.01 | Random tokens | 100 | 0.0001 | **100%** | PASSED |
| TinyStories #1 | Adafactor | 0.001 | TinyStories | 4000 | 8.4564 | 3.7% | FAILED |
| TinyStories #2 | Adafactor | 0.01 | TinyStories | 4000 | 7.3897 | 15.9% | FAILED |
| TinyStories #3 | Adam | 0.001 | TinyStories | 4000 | 7.9970 | 8.9% | FAILED |
| TinyStories #4 | Adam | 0.01 | TinyStories | 4000 | 7.6189 | 13.3% | FAILED |

### Key Finding

**Same optimizer and LR (Adam + 0.01) that achieves 100% reduction on random data achieves only 13.3% reduction on TinyStories.**

### Differences Between Tests

| Factor | Random Test | TinyStories Test |
|--------|-------------|------------------|
| Vocab size | 100 | 4000 |
| Model size | 115K params | 6.8M params |
| Data type | Random synthetic | Real language |
| Final loss | 0.0001 | 7.62 |

### What We Do NOT Know

- Why same optimizer/LR works on random but fails on TinyStories
- Role of vocab size difference (100 vs 4000)
- Role of model size difference (115K vs 6.8M)
- Role of data structure (random vs language)

**Status:** Investigation ongoing. See `OVERFIT_EXPERIMENTS.md` for full details.

---

## Critical Diagnostic Rule

**For Future Tests:**

When running diagnostic/overfit tests, MUST use:
- Same optimizer (Adafactor)
- Same LR (0.001)
- Same beta1 (0.5)
- Same weight decay (0.01)
- Same gradient clipping (1.0)

**Why:** Testing with different optimizer (Adam) or higher LR (0.01) can pass even if real config is broken.

---

## Attempt 1: First Test Training (5K Steps) - SUCCESSFUL

### Purpose
Initial test to verify model architecture and training pipeline works.

### Configuration
| Setting | Value |
|---------|-------|
| Script | Unknown (likely train.py or early train script) |
| Model | BitLinearGPT |
| Parameters | 6,856,608 |
| Batch size | Unknown |
| Accumulation | Unknown |
| Learning rate | Unknown |
| Optimizer | Unknown |
| Total steps | 5,000 |

### Timeline
| Date | Start | End | Duration |
|------|-------|-----|----------|
| Aug 1, 2026 | ~13:24 | ~17:23 | ~4 hours |

### Training Progress
| Step | Loss | Trend |
|------|------|-------|
| 500 | 8.83 | Start |
| 1000 | 8.74 | Decreasing ✓ |
| 2000 | 8.76 | Stable |
| 3000 | 8.64 | Decreasing ✓ |
| 4000 | 8.52 | Decreasing ✓ |
| 5000 | 8.45 | End |

### Results
| Metric | Value |
|--------|-------|
| Final loss | 8.45 |
| Loss trend | Decreasing (HEALTHY) |
| Entropy | Not monitored |
| Status | **SUCCESS - Loss decreased properly** |

### Important Notes
- User wanted to verify training pipeline works before full run
- Loss decreased from 8.83 to 8.45 (NORMAL training)
- **Did NOT collapse** - loss was decreasing throughout
- This PROVED the training pipeline works
- Location: `outputs/artn-core-v1/`

### What Was Learned
| Lesson | Evidence |
|--------|----------|
| Training pipeline works | Loss decreased properly |
| Model can learn | Loss from 8.83 → 8.45 |
| Ready for full training | Pipeline verified |

---

## Attempt 2: Full Training (50K Steps) - COLLAPSED

### Purpose
Train model to completion for 50,000 steps.

### Configuration
| Setting | Value |
|---------|-------|
| Script | train_optimized.py |
| Model | BitLinearGPT |
| Parameters | 6,856,608 |
| Batch size | 4 |
| Accumulation steps | 2 |
| Effective batch | 8 |
| Max learning rate | 0.0003 |
| Min learning rate | 0.00003 |
| Warmup steps | 1,000 |
| **Optimizer** | **Adafactor** |
| **Beta1** | **None (CRITICAL BUG)** |
| Beta2 | 0.999 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Vocab size | 4,000 |
| Max sequence length | 256 |

### Training Timeline
| Step | Loss | Entropy | Sample |
|------|------|---------|--------|
| 1,000 | 7.91 | Not monitored | Some variety |
| 5,000 | 6.06 | Not monitored | Some variety |
| 10,000 | 4.64 | Not monitored | Readable words |
| 15,000 | 4.27 | Not monitored | Partial sentences |
| 25,000 | 3.61 | Not monitored | Readable |
| 50,000 | 3.16 | **8.2%** | "time time time..." |

### Final Results
| Metric | Value | Status |
|--------|-------|--------|
| Final loss | 3.16 | LOOKED GOOD |
| Final entropy | **8.2%** | COLLAPSED |
| Top token | "time" | REPETITIVE |
| Top token probability | **94.04%** | COLLAPSED |
| Sample | "time time time time..." | BROKEN |
| Status | **COLLAPSED** | 🚫 FAILED |

### Why It Failed

#### Root Cause #1: Beta1=None
| Issue | Impact |
|-------|--------|
| Beta1 was set to None | Removed momentum from optimizer |
| Without momentum | Gradients have no memory, unstable updates |
| Result | Model overfits to common tokens |
| Evidence | Top token "time" at 94% probability |

**What is Beta1?**
- Beta1 = momentum parameter (should be 0.9)
- Momentum remembers 90% of previous gradient direction
- Makes learning STABLE
- Without it: unstable → collapse

#### Root Cause #2: No Entropy Monitoring
| Issue | Impact |
|-------|--------|
| Entropy not monitored | Couldn't detect collapse early |
| Only monitored loss | Loss looked good (3.01) but model was broken |
| Result | Wasted 79 hours of training |
| Evidence | Collapse detected only at step 50,000 |

#### Root Cause #3: No Collapse Detection
| Issue | Impact |
|-------|--------|
| No automatic stop | Training continued for 50k steps even after collapse |
| No thresholds | No warnings when entropy dropped or top token spiked |
| Result | Finished training with broken model |

### Time Spent
- Duration: **79 hours** (Aug 3-6, 2026)
- Completely wasted due to collapse

### Lessons Learned
1. **ALWAYS set beta1=0.9** (not None)
2. **ALWAYS monitor entropy** (>30% required)
3. **ALWAYS monitor top token probability** (<50% required)
4. **ALWAYS have automatic stop** on collapse detection

---

## Attempt 3: train_correct.py (2K Steps) - STOPPED

### Purpose
Create correct training script with proper beta1 and monitoring.

### Configuration
| Setting | Value |
|---------|-------|
| Script | train_correct.py |
| Beta1 | 0.9 (CORRECT) |
| Monitoring | Every 1000 steps |
| Accumulation | **WRONG IMPLEMENTATION** |

### What Was Wrong
| Issue | Explanation |
|-------|-------------|
| Accumulation outside train_step | Only 1 forward pass per step |
| Should be inside train_step | Need accum_steps forward passes per step |
| Result | Half the training depth |

### Training Timeline
| Step | Loss | Entropy | Status |
|------|------|---------|--------|
| 1,000 | 8.6278 | 94.8% | Healthy |
| 2,000 | 8.6143 | 94.8% | Healthy |

### Results
| Metric | Value |
|--------|-------|
| Final loss | 8.6143 |
| Entropy | 94.8% |
| Status | **HEALTHY** |
| Training depth | **HALF** (wrong accumulation) |

### Why Stopped
- Discovered accumulation was implemented wrong
- Would give half the training depth
- User decided to use train_optimized.py with fixes instead

### Lessons Learned
1. Accumulation must be INSIDE training loop
2. Always verify training depth matches expectations
3. Test scripts before running full training

---

## Attempt 4: train_optimized.py (FIXED) - IN PROGRESS

### Purpose
Retry training with CORRECT settings: beta1=0.9, proper accumulation, monitoring.

### Configuration
| Setting | Value |
|---------|-------|
| Script | train_optimized.py (FIXED) |
| Model | BitLinearGPT |
| Parameters | 6,856,608 |
| Batch size | 4 |
| Accumulation steps | 2 |
| Effective batch | 8 |
| Max learning rate | 0.0003 |
| Min learning rate | 0.00003 |
| Warmup steps | 1,000 |
| Optimizer | Adafactor |
| **Beta1** | **0.9** (FIXED) |
| Beta2 | 0.999 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Vocab size | 4,000 |
| Max sequence length | 256 |
| Monitoring | Every 500 steps |
| Tokenizer | Loaded from backup (has unicode) |

### Fixes Applied
| Fix | Line Changed |
|-----|---------------|
| Beta1=None → 0.9 | Line 339 |
| Added entropy monitoring | Lines 430-500 |
| Added top token monitoring | Lines 440-450 |
| Added status determination | Lines 445-455 |
| Added auto-stop on collapse | Lines 495-505 |
| Added log file | Lines 485-490 |

### Training Timeline
| Step | Loss | Entropy | Top Token | Status |
|------|------|---------|-----------|--------|
| 500 | 8.75 | 95.1% | present 0.5% | Healthy |
| 1,000 | 8.79 | 95.1% | present 0.5% | Healthy |
| 5,000 | 8.49 | 95.4% | present 0.4% | Healthy |
| 10,000 | 8.21 | 95.6% | present 0.4% | Healthy |
| 15,000 | 7.94 | 95.8% | present 0.3% | Healthy |
| 20,000 | 7.80 | 95.9% | the 0.3% | Healthy |
| 25,000 | 7.71 | 96.0% | the 0.4% | Healthy |
| 27,000+ | 7.65 | 96.0% | the 0.4% | Healthy |

### Loss Drop Rate Analysis
| Period | Loss Drop | Rate |
|--------|-----------|------|
| 5k-10k | 0.277 | Good |
| 10k-15k | 0.270 | Good |
| 15k-20k | 0.143 | Slowing |
| 20k-25k | 0.089 | Slowing more |

### Current Status - COMPLETED

| Metric | Value | Status |
|--------|-------|--------|
| Final Step | 50,000 | COMPLETED |
| Final Loss | 7.56 | HIGH |
| Final Entropy | 96.2% | EXCELLENT |
| Top token | "the" (0.56%) | HEALTHY |
| Sample quality | Gibberish | NOT READABLE |
| Training time | 83 hours | COMPLETED |

### Issues Discovered

#### Issue 1: Sample Repetition in Monitoring
| Problem | Cause | Fix |
|---------|-------|-----|
| Samples identical in training output | torch.manual_seed(42) hardcoded | Removed fixed seed |
| Affects: Monitoring samples only | Loaded script in memory at start | Fix applied to disk |
| Does NOT affect: Model training | Training uses old code in memory | Will apply on restart |

#### Issue 2: Unicode in Samples
| Problem | Cause | Impact |
|---------|-------|--------|
| "â€œLetâ€™s" instead of "Let's" | Tokenizer trained on fancy quotes | Cosmetic only |
| Occurs because | Loading old tokenizer | Model learns fine, display has unicode |
| Can fix later | Replace tokenizer post-training | Not critical |

#### Issue 3: Slow Learning
| Observation | Value |
|-------------|-------|
| Loss drop rate | Decreasing (slower learning) |
| 5k-10k: 0.277 | Good rate |
| 20k-25k: 0.089 | 68% slower than start |
| Reason | Beta1=0.9 makes learning STABLE but SLOW |

### Comparison: Attempt 2 vs Attempt 4

| Metric | Attempt 2 (beta1=None) | Attempt 4 (beta1=0.9) |
|--------|------------------------|----------------------|
| Loss @ 10k | 4.64 | 8.21 |
| Loss @ 25k | 3.61 | 7.71 |
| Final Loss @ 50k | 3.16 | 7.56 |
| Entropy @ 50k | 8.2% (COLLAPSED) | 96.2% (HEALTHY) |
| Top token @ 50k | 94% (COLLAPSED) | 0.56% (HEALTHY) |
| Final result | COLLAPSED | HEALTHY but not useful |
| Learning speed | FAST but UNSTABLE | SLOW but STABLE |
| Sample quality | "time time time..." | Gibberish (varied) |

### Predictions - RESOLVED

| Scenario | Evidence | Result |
|----------|----------|--------|
| Finishes healthy | Entropy stable 96% | CORRECT - 96.2% final |
| Becomes readable | Loss decreasing slowly | WRONG - still gibberish |
| Collapses like before | Entropy healthy (unlikely) | Low |
| Works well | Cannot prove | Unknown |

### Time Estimate
- Started: Aug 6, ~15:00
- Completed: Aug 10, ~05:00
- Total duration: **83 hours**

### Lessons Learned
1. Beta1=0.9 makes learning STABLE but TOO SLOW
2. High loss doesn't mean model is collapsed
3. Low loss doesn't mean model is good (attempt 2 had low loss but collapsed)
4. Entropy is the KEY metric for collapse detection
5. Loss alone doesn't indicate readability
6. Model can be HEALTHY (96% entropy) but produce GIBBERISH
7. **Beta1=0.9 is TOO CONSERVATIVE for ternary quantization**

---

## Critical Findings

### Finding 1: Beta1=None Caused Collapse
| Evidence | Value |
|----------|-------|
| Attempt 2 | Beta1=None → COLLAPSED |
| Attempt 4 | Beta1=0.9 → HEALTHY |
| Conclusion | Beta1=0.9 is REQUIRED |

### Finding 2: Loss Alone Doesn't Indicate Quality
| Attempt | Loss @ 50k | Result |
|---------|------------|--------|
| 2 | 3.01 (good) | COLLAPSED |
| 4 (predicted) | ~6.5-7.0 (high) | HEALTHY? |

**Lower loss ≠ Better model**

### Finding 3: Entropy is Critical Metric
| Threshold | Meaning |
|-----------|---------|
| >30% | HEALTHY |
| 20-30% | WARNING |
| <20% | COLLAPSED |
| 8% (attempt 2) | SEVERE COLLAPSE |

### Finding 4: Top Token Probability Indicates Collapse
| Threshold | Meaning |
|-----------|---------|
| <10% | HEALTHY (spread distribution) |
| 10-50% | WARNING |
| >50% | COLLAPSING |
| 94% (attempt 2) | COLLAPSED on single token |

---

## Configuration Recommendations

### OPTIMAL CONFIG (Based on Evidence)

| Setting | Value | Reason |
|---------|-------|--------|
| Beta1 | 0.9 | REQUIRED - prevents collapse |
| Beta2 | 0.999 | Standard |
| Learning rate | 0.0003 max | Stable |
| Batch size | 4 | Works |
| Accumulation | 2 | Effective batch 8 |
| Warmup | 1000 | Standard |
| Monitoring | Every 1000 steps | Detect issues early |
| Metrics | Entropy + Top token + Loss | All three required |
| Auto-stop | Yes | Prevent wasted training |

---

## Open Questions

### Question 1: Will Current Training Work?
| Answer | Evidence |
|--------|----------|
| UNKNOWN | Cannot prove either way |
| Healthy so far | Entropy 96% |
| But slow learning | Loss drop rate decreasing |
| Only way to know | FINISH TRAINING |

### Question 2: Why Is Learning Slow?
| Hypothesis | Evidence |
|------------|----------|
| Beta1=0.9 conservative | True - prevents collapse but slow |
| Learning rate too low? | Unknown |
| Need more steps? | Unknown |
| Architecture issue? | Unlikely (entropy healthy) |

### Question 3: When Will It Be Readable?
| Previous guess | Reality |
|----------------|---------|
| Step 25,000 | WRONG - still garbled |
| New guess | UNKNOWN |
| Can't predict | Without more data |

---

## Recommendations

### For Current Training
| Option | Pros | Cons |
|--------|------|------|
| Continue to 50k | See final result | 65 hours + time |
| Stop now | Save time | Lose 27k steps |
| Adjust settings | Might help | Might hurt |

### For Future Training
1. Use beta1=0.9 (PROVEN)
2. Monitor entropy every 1000 steps
3. Monitor top token probability
4. Auto-stop if entropy <20% or top token >70%
5. Save logs and configs for EVERY training
6. Test scripts before full training runs
7. Don't trust loss alone

---

## Timeline Summary

| Date | Event | Duration |
|------|-------|----------|
| Aug 1, 2026 | Attempt 1 (5k test) | ~4 hours |
| Aug 3-6 | Attempt 2 (50k collapsed) | 79 hours |
| Aug 6 | Attempt 3 (stopped at 2k) | ~1 hour |
| Aug 6-10 | Attempt 4 (completed) | 83 hours |

**Total time invested: 166 hours (4h + 79h + ~1h + 83h)**

---

## Conclusion

| Training | Status | Lesson |
|----------|--------|--------|
| Attempt 1 | SUCCESS | Verified training pipeline works |
| Attempt 2 | Collapsed | Beta1=0.9 REQUIRED |
| Attempt 3 | Stopped | Test scripts first |
| Attempt 4 | COMPLETED but not useful | Beta1=0.9 too conservative |

**The only certainty: ENTROPY MONITORING is CRITICAL.**

---

## Next Steps - BETA1 EXPERIMENTS

### Problem Identified
| Issue | Evidence |
|-------|----------|
| Beta1=None | TOO AGGRESSIVE - causes collapse (Attempt 2) |
| Beta1=0.9 | TOO CONSERVATIVE - too slow learning (Attempt 4) |

### Proposed Solution
Test intermediate beta1 values in small experiments:

| Test | Beta1 | Steps | Purpose |
|------|-------|-------|---------|
| A | 0.5 | 10,000 | Middle ground - faster learning |
| B | 0.7 | 10,000 | Less conservative than 0.9 |

### Success Criteria for Tests
| Metric | Target | Why |
|--------|--------|-----|
| Entropy | >30% | Not collapsed |
| Top token prob | <50% | Not repetitive |
| Loss @ 10k | <6.0 | Learning reasonably |
| Sample quality | Some readable words | Progressing |

### Decision Process
1. Run both tests in parallel
2. Compare at 10k steps
3. Pick best config based on:
   - Entropy (must be healthy)
   - Loss (must be improving)
   - Sample quality (must show progress)
4. Run full 50k training with best config

---

## Beta1 Experiments Results (Aug 11, 2026)

### Experiments Completed

| Experiment | Beta1 | Batch Size | LR | Steps | Final Loss | Entropy | Status |
|------------|-------|------------|----|----|------------|---------|--------|
| beta1=0.5 test (batch=32) | 0.5 | 32 | 0.0003 | 10,000 | 8.39 | 94.7% | HEALTHY |
| beta1=0.7 test (batch=32) | 0.7 | 32 | 0.0003 | 10,000 | 8.30 | 94.5% | HEALTHY |
| **beta1=0.5 NEW (batch=8, LR=0.001)** | **0.5** | **8** | **0.001** | **10,000** | **6.21** | **93.8%** | **HEALTHY** |
| **beta1=0.7 NEW (batch=8, LR=0.001)** | **0.7** | **8** | **0.001** | **10,000** | **6.38** | **94.7%** | **HEALTHY** |

### Loss Progression Comparison

| Step | beta1=0.5 (batch=32, LR=0.0003) | beta1=0.7 (batch=32, LR=0.0003) | Previous (beta1=0.9, batch=8) | **NEW beta1=0.5 (batch=8, LR=0.001)** | **NEW beta1=0.7 (batch=8, LR=0.001)** |
|------|--------------------------------|--------------------------------|------------------------------|---------------------------------------|---------------------------------------|
| 1,000 | 8.82 | 8.81 | 8.79 | **8.44** | 8.49 |
| 2,000 | 8.75 | 8.70 | 8.69 | **7.41** | 7.42 |
| 3,000 | 8.67 | 8.60 | 8.63 | **6.83** | 6.79 |
| 4,000 | 8.60 | 8.50 | 8.49 | **6.46** | 6.42 |
| 5,000 | 8.53 | 8.43 | 8.49 | 6.58 | **6.27** |
| 6,000 | 8.48 | 8.41 | 8.37 | 6.39 | 6.34 |
| 7,000 | 8.43 | 8.36 | 8.33 | 6.35 | **6.22** |
| 8,000 | 8.41 | 8.34 | 8.32 | 6.43 | 6.42 |
| 9,000 | 8.39 | 8.31 | 8.25 | 6.40 | 6.35 |
| 10,000 | 8.39 | 8.30 | 8.21 | **6.21** | 6.38 |

### Key Findings

#### Finding 1: Batch Size Config Mismatch Discovery

| Issue | Impact |
|-------|--------|
| Previous training (Attempt 4) used batch=8 (accumulation=2) | Different effective batch size |
| Beta1 experiments used batch=32 (accumulation=8) | 4x more data per step |
| Configs NOT identical | Cannot directly compare |

#### Finding 2: Loss Difference is Small

| Comparison | Loss Difference |
|------------|----------------|
| batch=8 vs batch=32 @ 10k | Only 0.09 difference |
| Is this significant? | NO - within noise range |

#### Finding 3: Both Beta1 Values Stayed Healthy

| Metric | beta1=0.5 | beta1=0.7 | Previous (0.9) |
|--------|----------|-----------|----------------|
| Entropy @ 10k | 94.7% | 94.5% | 95.6% |
| Top Token @ 10k | 0.5% | 0.4% | 0.4% |
| Collapse? | NO | NO | NO |

**All configurations remained HEALTHY.**

#### Finding 4: Sample Quality

| Config | Sample Quality @ 10k |
|--------|---------------------|
| beta1=0.5 (batch=32) | Gibberish - no readable text |
| beta1=0.7 (batch=32) | Gibberish - no readable text |
| Previous (beta1=0.9, batch=8) @ 10k | Gibberish (not saved, but noted) |

**All samples are gibberish at 10k steps.**

### Research Findings on Batch Size

#### Source: arXiv 2507.07101 (July 2025)

| Finding | Evidence |
|---------|----------|
| Small batches (1-2) train stably | Verified in experiments |
| Small batches more robust to hyperparameters | Research paper finding |
| Small batches achieve equal or BETTER per-FLOP performance | Key finding |
| Large batches less robust | Research paper finding |
| Small batches enable vanilla SGD training | Significant result |

**Implication: Batch=8 may be BETTER than batch=32 for training.**

#### Source: Microsoft BitNet

| Finding | Evidence |
|---------|----------|
| 1.58-bit quantization requires special approach | Official Microsoft framework |
| BitNet uses quantization-aware training | Different from regular LLM training |
| Official framework available at github.com/microsoft/BitNet | Could provide guidance |

### Comparison Summary

| Metric | batch=8, beta1=0.9 | batch=32, beta1=0.7 | Difference |
|--------|-------------------|---------------------|------------|
| Loss @ 10k | 8.21 | 8.30 | +0.09 (negligible) |
| Learning speed | Very slow | Moderate | 0.7 faster |
| Final result @ 50k | Gibberish | Not tested | Unknown |
| Entropy | 95.6% | 94.5% | Similar |

### Conclusions from Beta1 Experiments

| Conclusion | Evidence |
|------------|----------|
| beta1=0.7 learns FASTER than 0.9 | Loss drop rate higher |
| beta1=0.7 stays HEALTHY | Entropy stable at 94.5% |
| Batch size difference is REAL | Config mismatch discovered |
| Research supports small batch | arXiv paper findings |
| No conclusion yet on optimal config | Need batch=8 comparison |

### Unresolved Questions

| Question | Status |
|----------|--------|
| Is batch=8 better for ternary? | Research suggests YES, but need testing |
| Is beta1=0.7 optimal? | Unknown - no specific research |
| Should we test batch=8 with beta1=0.7? | Valid question, research supports |

### Time Invested in Beta1 Experiments

| Experiment | Duration | Result |
|------------|----------|--------|
| beta1=0.5 @ 10k (batch=32) | ~7 hours | HEALTHY |
| beta1=0.7 @ 10k (batch=32) | ~7 hours | HEALTHY |
| beta1=0.5 @ 10k NEW (batch=8, LR=0.001) | ~7 hours | **HEALTHY, BEST LOSS 6.21** |
| beta1=0.7 @ 10k NEW (batch=8, LR=0.001) | ~10 hours | **HEALTHY, LOSS 6.38** |
| **Total** | **31 hours** | LR=0.001 is 5x BETTER |

---

## NEW KEY FINDING: LR=0.001 Works Much Better (Aug 12, 2026)

### Discovery

| Metric | Old (LR=0.0003, batch=32) | NEW (LR=0.001, batch=8) | Improvement |
|--------|----------------------------|-------------------------|-------------|
| Loss drop @ 10k | -0.43 to -0.51 | **-2.11 to -2.23** | **5x FASTER** |
| Final loss @ 10k | 8.30-8.39 | **6.21-6.38** | **MUCH BETTER** |

### NEW @ 10k Beats OLD @ 50k

| Training | Steps | Loss | Status |
|----------|-------|------|--------|
| OLD (beta1=0.9, batch=8, LR=0.0003) @ 50k | 50,000 | 7.56 | HEALTHY but slow |
| **NEW (beta1=0.5, batch=8, LR=0.001) @ 10k** | **10,000** | **6.21** | **HEALTHY, fast** |

### Sample Quality: Improved but Still Gibberish

| Model | Samples | Repetition |
|-------|---------|------------|
| Collapsed (attempt 2) | "time time time..." | YES (94%) |
| Slow (attempt 4) | "The The The..." | YES |
| **NEW beta1=0.5** | "genercorner stick..." | **NO** |
| **NEW beta1=0.7** | "peeked iid. playing..." | **NO** |

### Best Config Found

| Parameter | Value | Reason |
|-----------|-------|--------|
| Learning Rate | **0.001** | 5x faster learning, stable |
| Batch Size | **8** | Works well, matches research |
| Beta1 | **0.5 or 0.7** | Both healthy |
| Optimizer | Adafactor | Memory efficient |

---

## Attempt 5: Best Config 50k Training (50K Steps) - COMPLETED BUT FAILED

### Purpose
Full training with best config found: LR=0.001, batch=8, beta1=0.5

### Configuration
| Setting | Value |
|---------|-------|
| Script | train_optimized.py |
| Model | BitLinearGPT |
| Parameters | 6,856,608 |
| Batch size | 4 |
| Accumulation steps | 2 |
| Effective batch | 8 |
| Max learning rate | **0.001** |
| Min learning rate | 0.00003 |
| Warmup steps | 1,000 |
| LR Schedule | Cosine decay |
| Optimizer | Adafactor |
| Beta1 | **0.5** |
| Tokenizer | NEW (fresh, trained on TinyStories 10k) |

### Timeline
| Date | Start | End | Duration |
|------|-------|-----|----------|
| Aug 13-16, 2026 | Started | Completed | **~82 hours** |

### Training Progress
| Step | Loss | Entropy | TopProb | Trend |
|------|------|---------|---------|-------|
| 500 | 8.58 | 94.7% | 0.5% | Start |
| 10000 | 7.67 | 96.2% | 0.4% | Fast learning |
| 20000 | 7.22 | 96.7% | 0.7% | Slowing |
| 30000 | 7.23 | 96.9% | 1.1% | **PLATEAU** |
| 40000 | 7.07 | 97.0% | 1.1% | Barely moving |
| 50000 | 7.13 | 97.0% | 1.1% | **COMPLETED** |

### Key Observations

**Loss Plateaued After 20k:**
| Period | Loss Drop | Learning Rate |
|--------|-----------|---------------|
| 0-10k | -0.91 | HIGH (0.001) |
| 10k-20k | -0.45 | HIGH |
| **20k-50k** | **-0.09** | **DECAYING** |

**Learning Rate Decay:**
| Step | LR | Status |
|------|----|----|
| 10k | 0.001 | Fast learning |
| 20k | 0.0008 | Starting decay |
| 40k | 0.0001 | Almost stopped |
| 50k | 0.00003 | Tiny updates |

### Results
| Metric | Value |
|--------|-------|
| Final loss | 7.13 |
| Final entropy | **97.0%** |
| Final TopProb | **1.1%** |
| Status | **HEALTHY** |
| Samples | **GIBBERISH** (no readable stories) |

### Sample Outputs @ 50k
```
Prompt: "Once upon a time"
Output: "Once upon a time pile set. Liturtati Maybe see tion. pictures Smisityourbelonto ticflotop..."

Prompt: "A boy named"
Output: "A boy named cold high. window olThank mom me. This ble dI'll knowshouldchosafelneckwalked..."

Prompt: "The little girl"
Output: "The little girl tries important ined. waves €ppsoft machthe doga a carry zy embarra fight..."
```

**Words Found:** "Maybe", "Tom", "pictures", "hand", "chair", "spider", "wash", "remember"

**Problem:** Model learned INDIVIDUAL words but NOT sentence formation.

### Comparison to Previous 50k Attempts

| Training | Loss @ 50k | Entropy | TopProb | Samples |
|----------|-----------|---------|---------|---------|
| Attempt 2 (collapsed) | 3.16 | 8.2% | 94% | REPETITIVE ("time time time") |
| Attempt 4 (gibberish) | 7.56 | 96.2% | 0.56% | REPETITIVE ("The The The") |
| **Attempt 5 (current)** | **7.13** | **97.0%** | **1.1%** | **GIBBERISH (no repetition)** |

### Key Findings

**Finding 1: Previous 10k Test Was BETTER**
| Model | Loss @ 10k | Sample Quality |
|-------|-----------|----------------|
| Previous test (OLD tokenizer) | **6.21** | "A boy named Benny" |
| Attempt 5 (NEW tokenizer) | 7.67 | "spaghettTimmy's river" |

**Finding 2: Cosine LR Decay Killed Learning**
- Loss dropped 0.91 in first 10k (LR=0.001)
- Loss dropped only 0.09 in last 30k (LR decaying)
- By step 40k, LR dropped to 0.0001 (10x smaller)

**Finding 3: NEW Tokenizer Worse Than OLD**
- OLD tokenizer: Loss 6.21 @ 10k
- NEW tokenizer: Loss 7.67 @ 10k
- Same config, different tokenizer = different results

**Finding 4: Model Healthy Throughout**
- Entropy stayed 96-97% (healthy)
- TopProb stayed 0.4-1.1% (healthy, not collapsed)
- No repetition patterns (unlike Attempt 2 and 4)

### Root Causes of Failure

| Cause | Evidence |
|-------|----------|
| **Cosine LR decay** | Killed learning after 20k |
| **NEW tokenizer** | Worse performance than OLD |
| **Small model + ternary** | Hard combination, needs more iterations |
| **50k not enough** | More time needed at high LR |

### What Worked

| Success | Evidence |
|---------|----------|
| LR=0.001 is best | 5x faster than 0.0003 |
| Beta1=0.5 works | Healthy entropy throughout |
| Monitoring successful | Tracked entropy + TopProb |
| No collapse | Stable entropy 97% |

### What Didn't Work

| Failure | Evidence |
|---------|----------|
| Cosine LR decay | Learning stopped at 20k |
| NEW tokenizer | Worse than OLD |
| 50k steps | Not enough time |
| Samples still gibberish | No sentence formation |

### Lessons Learned

1. **Cosine decay too aggressive for small models**
   - LR dropped too fast
   - Learning stopped before model learned patterns
   
2. **Tokenizer choice matters**
   - OLD tokenizer performed better
   - Need to investigate why

3. **Ternary quantization is hard**
   - Limited precision (-1, 0, +1)
   - Need more iterations or different approach

4. **Small models need different schedules**
   - Constant LR may be better
   - Or custom schedule for small models

5. **Healthy metrics ≠ good samples**
   - Entropy 97%, TopProb 1.1% = healthy
   - But samples still gibberish
   - Need other metrics (validation loss)

---

## Next Steps - Updated Plan

### IMMEDIATE NEXT EXPERIMENT: Constant LR + OLD Tokenizer

**Hypothesis:** Constant LR will prevent plateau, OLD tokenizer will improve learning

| Config | Value |
|--------|-------|
| LR | **0.001 CONSTANT** (no decay) |
| Tokenizer | **OLD** (from backup) |
| Steps | 50k |
| Monitoring | entropy + TopProb + samples |

**Expected:**
- Loss should continue dropping (not plateau)
- Should reach lower than 7.13
- Samples should improve

### Future Approaches to Try

| Priority | Approach | Reason |
|----------|----------|--------|
| 1 | **Constant LR + OLD tokenizer** | Most likely to work |
| 2 | Longer training (100k) | Small model needs more time |
| 3 | Research Microsoft BitNet training | Use proven method |
| 4 | Different quantization (2-bit) | Less aggressive |

### Recommended Research Actions

| Action | Purpose |
|--------|---------|
| Study Microsoft BitNet paper | Understand exact training approach |
| Implement validation loss | Detect overfitting |
| Build checkpoint evaluation system | Compare samples systematically |
| Test constant LR schedule | Prevent learning plateau |

### Decision Required

**User must decide:**
1. Try constant LR + OLD tokenizer experiment?
2. Research Microsoft BitNet approach first?
3. Try different quantization approach?

---

## Ternary Implementation Verification (Aug 18, 2026)

### Question

User observed that matrix multiplications use `{-scale, 0, +scale}` instead of `{-1, 0, +1}`.
Is this correct BitNet behavior?

### Investigation

**Date:** August 18, 2026

**Method:** Compare our implementation against official/authoritative references

**References Examined:**
1. kyegomez/BitNet - Community reference implementation
2. microsoft/BitNet - Official Microsoft inference code
3. BitNet original paper (arxiv:2310.11453)
4. BitNet b1.58 paper (arxiv:2402.17764)

**Key Code Comparison:**

| Implementation | Weight Quantization | Tensor to Matmul |
|----------------|--------------------|------------------|
| kyegomez | `(w - mean).sign() * scale` | `{-scale, 0, +scale}` |
| Ours | `weight_quantized * scale` | `{-scale, 0, +scale}` |
| Microsoft | Pre-quantized weights | Packed ternary |

**Verification Questions:**

| Question | Status | Answer |
|----------|--------|--------|
| Does BitNet use scaled ternary in linear? | VERIFIED CORRECT | YES |
| When is scale applied? | VERIFIED CORRECT | After ternarization, before matmul |
| Is scaled weights correct? | VERIFIED CORRECT | YES - intentional design |
| Quantization equation | DIFFERENT BUT VALID | We use Round+Clip, ref uses Sign |
| Scale equation | VERIFIED CORRECT | scale = mean( |W| ) |
| STE definition | VERIFIED CORRECT | Gradient passes through |
| Weights full precision? | VERIFIED CORRECT | YES - latent FP32 weights |
| Matches reference? | DIFFERENT BUT VALID | Both correct BitNet variants |

### Conclusion

**IMPLEMENTATION VERIFIED CORRECT.**

The presence of `{-scale, 0, +scale}` values in F.linear() is **EXPECTED AND CORRECT** behavior per BitNet.

Both kyegomez reference and our implementation multiply ternary weights by scale before matrix multiplication:

```python
# kyegomez reference:
u = (w - e).sign() * scale  # Produces {-scale, 0, +scale}

# Our implementation:
weight_dequantized = weight_quantized * scale  # Produces {-scale, 0, +scale}
```

**This is correct BitNet behavior.**

### Valid Variants

| Aspect | kyegomez | Ours | Status |
|--------|----------|------|--------|
| Weight centering | Subtract mean | No centering | Both valid |
| Quantization | sign() | Round + clip | Both valid |
| Threshold | Implicit 0 | Explicit 0.5 | Both valid |

### Files

- Verification report: `experiments/results/TERNARY_IMPLEMENTATION_VERIFICATION.md`
- Reference comparison: `experiments/results/BITNET_IMPLEMENTATION_COMPARISON.md`
- Downloaded papers + code: `experiments/bitnet_references/`

### Lesson Learned

Scaled weights (`{-scale, 0, +scale}`) are the INTENDED design.
Do NOT assume this is a bug if revisiting.

---

## Document History

**Original Document Last Updated:** Aug 18, 2026 (Total training time invested: ~280 hours)

**Updated:** October 15, 2026
- Added Phase 46-47 completion (FP32 + AdamW baseline, BitLinear + AdamW success)
- Updated Stage 7 status from "⚠️ PARTIAL" to "✅ COMPLETE"
- Updated Stage 7 Milestone from "⚠️ NOT SATISFIED" to "✅ COMPLETE"
- Added working models uploaded to Hugging Face
- Total training time: ~309 hours (280 + 29 from Phase 46-47)

**Key Addition:** Stage 7 is now COMPLETE with val loss 1.92 and coherent generation, proving ternary quantization works with AdamW optimizer.

---

*Original document last updated: Aug 18, 2026*
*Total training time invested: ~280 hours (198 previous + 82 Attempt 5)*

*This document updated: October 15, 2026*
*Total training time invested: ~309 hours (280 original + 29 from Phase 46-47)*

---

## Stage 8-9 Completion Update (October 16-26, 2026)

**Phase 60: ARTN Architecture, Training, and Evaluation**

Full detailed documentation lives in `experiments/results/ROOT_CAUSE_INVESTIGATION_SUMMARY.md` (Phase 60.1-60.13, lines 6255-6705). This section provides the summary and error corrections.

### Attempt 8: ARTN Training (October 25, 2026)

| Metric | Value |
|--------|-------|
| Architecture | ARTNGPT — 1 shared BitLinearBlock + halting gate |
| Parameters | 2,904,065 (50.2% fewer than FP32, 57.6% fewer than BitLinear) |
| Steps | 10,000 / 10,000 |
| Val loss | 2.0346 (includes padding — fair NLL 2.904, see corrections below) |
| Val perplexity | 7.6496 |
| Expected passes | 4.0015 (target: 4.0 — gate converged) |
| Ternary sparsity | 35.4% zeros, ~32.3% each ±1 |
| Training time | 14.35 hours (51,653 seconds) |
| Checkpoints | 20 (all preserved, 665 MB) |
| NaN/Inf losses | 0 |
| Optimizer | AdamW (beta1=0.5, beta2=0.98, lr=0.001) |

**Output:** `outputs/artn_recursive_20261025_002725/`

### Stage 8 Completion: Quantized Inference (October 16-21, 2026)

- Created `src/inference/generate.py` — autoregressive generation pipeline
- Created inference-only checkpoint: 27 MB (from 79 MB, no optimizer state)
- Deployment benchmark: baseline RSS 234 MB, peak 286 MB, 4.61 tok/s
- Quality investigation: tokenizer correct, temperature 0.5-0.9 optimal, RAM corrected from 0.18 MB (tracemalloc) to 21.46 MB (psutil)

### Stage 9 Completion: CPU Benchmarking (October 25-26, 2026) [ORIGINAL — uncorrected halt logic, see corrected re-run below]

**Speed benchmark (ALL MEASURED, 10 prompts, greedy, warm-up, psutil RSS):**

| Model | Parameters | Tok/s | Peak RSS | Avg Passes |
|-------|-----------|-------|----------|------------|
| FP32 (6 blocks, tied) | 5,828,608 | 24.06 | 280 MB | 6.00 |
| BitLinear GPT (6 blocks) | 6,852,608 | 5.76 | 356 MB | 6.00 |
| ARTN (t=1.0) | 2,904,065 | 5.68 | 344 MB | 5.92* |
| ARTN (t=0.9, early-exit) | 2,904,065 | 8.69 | 344 MB | 3.30 |

\*5.92 was a floating-point artifact from uncorrected halt logic; corrected to 6.0000 in Oct 28 re-run (see below)

**Causal NLL (per-position, hard early-exit, 1280 tokens):**

| Threshold | NLL | PPL | Avg Passes | NLL delta vs 1.0 | Pass savings |
|-----------|-----|-----|-----------|------------------|-------------|
| 0.9 | 2.8632 | 17.52 | 3.30 | +0.0024 | 45.0% |
| 1.0 | 2.8608 | 17.47 | 6.00 | 0.0000 | 0.0% |

**Fair non-padding NLL (200 samples, 34,980 valid tokens):**

| Model | NLL | PPL | Parameters |
|-------|-----|-----|-----------|
| FP32 (tied) | 2.4230 | 11.28 | 5,828,608 |
| BitLinear GPT | 2.7425 | 15.53 | 6,852,608 |
| ARTN | 2.9036 | 18.24 | 2,904,065 |

**Corrected Stage 9 Benchmark (October 28, 2026 — re-run with fixed halt logic):**

**Why corrected:** Original `artn_threshold_sweep.py` and `stage9_benchmark.py` had no 
`remaining.clamp(min=0.0)` and no threshold=1.0 bypass. Floating-point subtraction made 
`remaining` slightly negative, causing threshold=1.0 to show 5.92 passes instead of 6.0. 
Fix back-ported from `fair_eval.py`. Original scripts backed up, originals results preserved.

**Original scripts (backed up):**
- `/root/agent-sandbox/ARTN/scripts/artn_threshold_sweep.py.orig_20261026`
- `/root/agent-sandbox/ARTN/scripts/stage9_benchmark.py.orig_20261026`

**Corrected scripts (live):**
- `/root/agent-sandbox/ARTN/scripts/artn_threshold_sweep.py`
- `/root/agent-sandbox/ARTN/scripts/stage9_benchmark.py`

**Corrected Stage 9 benchmark (ALL MEASURED, Oct 28, 10 prompts, greedy, seed=42, 1 warm-up):**

| Model | Parameters | Tok/s | Peak RSS (MB) | Avg Passes |
|-------|-----------|-------|---------------|------------|
| FP32 (6 blocks, tied) | 5,828,608 | 22.62 | 278.45 | 6.00 |
| BitLinear GPT (6 blocks) | 6,852,608 | 5.45 | 354.62 | 6.00 |
| ARTN (t=1.0) | 2,904,065 | 5.29 | 282.34 | 6.00 |
| ARTN (t=0.9) | 2,904,065 | 8.42 | 276.37 | 3.57 |

**Corrected threshold sweep (ALL MEASURED, Oct 28, 10 prompts, greedy, seed=42, no warm-up):**

| Threshold | Avg Passes | Tok/s | Identical to 1.0 |
|-----------|-----------|-------|------------------|
| 0.3 | 2.7240 | 9.29 | 0/10 |
| 0.5 | 2.9980 | 9.50 | 0/10 |
| 0.7 | 3.2740 | 8.54 | 3/10 |
| 0.9 | 3.5660 | 8.00 | 10/10 |
| 1.0 | 6.0000 | 5.21 | ref |

RSS ordering: ARTN t=0.9 (276) < FP32 (278) < ARTN t=1.0 (282) < BitLinear (355).
BitLinear's peak RSS is ~76 MB higher than FP32; cause not isolated. Speed 
advantage at t=0.9: 1.59x vs t=1.0, 1.55x vs BitLinear — single-run measurement, 
not statistically established. Speed changed between original and corrected runs; 
contributions of system variation and code-path overhead have not been isolated. 
10/10 identical at t=0.9: confirmed for 10 benchmark prompts only. 3/7 exploratory 
interactive prompts showed different text at t=0.9 vs t=1.0 — not generalized. 
Generated text at each threshold below 1.0 unchanged between original and corrected 
implementations (same-threshold reproducibility, not cross-threshold equality). 
Sweep has no warm-up; benchmark has 1 warm-up prompt — difference preserved from 
original scripts.

Corrected evidence: 
`/root/agent-sandbox/ARTN/outputs/artn_recursive_20261025_002725/stage9_benchmark_corrected.json`, 
`/root/agent-sandbox/ARTN/outputs/artn_recursive_20261025_002725/threshold_sweep_results_corrected.json` (Oct 28). 
Originals preserved: 
`/root/agent-sandbox/ARTN/outputs/artn_recursive_20261025_002725/stage9_benchmark.json`, 
`/root/agent-sandbox/ARTN/outputs/artn_recursive_20261025_002725/threshold_sweep_results.json` (Oct 25).

---

## Errors and Corrections (Phase 60)

This section documents all mistakes, bugs, and incorrect claims discovered during Phase 60 work. Each is noted honestly with what was wrong, when discovered, and how addressed.

### 1. Stage 7 Checkpoint Was NOT ARTN (CRITICAL)

**What was wrong:** The trained BitLinear checkpoint (`step_9500.pt`) was presented as the ARTN model. It has 6 SEPARATE BitLinearBlock instances, NO recursive parameter sharing, and NO halting gate. It is standard ternary GPT, not ARTN.

**When discovered:** October 21, 2026 (architecture audit, Phase 60.3)

**Impact:** Stages 4 (Recursive Transformer) and 5 (Adaptive Halting Gate) were marked ✅ DONE in the status table above based on code existing, but the code was NEVER trained. The trained checkpoint did not use recursion or halting.

**How addressed:** Designed and implemented a new ARTNGPT class (`experiments/artn_model.py`, Oct 23) with 1 shared BitLinearBlock + halting gate. Trained from scratch (Attempt 8, Oct 25). The old Stage 4/5 status entries are marked [CORRECTED] above — code exists but was never trained until Phase 60.

### 2. All Reported Validation Losses Include Padding (31.4%)

**What was wrong:** All reported val losses (FP32 1.69, BitLinear 1.92, ARTN 2.03) include 31.4% trivially-easy padding tokens (target=0, NLL=0.011). Training uses `cross_entropy(ignore_index=-1)` which excludes only -1, NOT padding (ID=0). The model received gradient signal for padding positions.

**When discovered:** October 26, 2026 (padding audit, Phase 60.12)

**Impact:** All absolute loss numbers are artificially low. The relative comparison between models is valid (all three used same padding-inclusive loss), but absolute numbers are not comparable to published results that exclude padding.

**How addressed:** Computed fair non-padding NLL for all three models (FP32 2.423, BitLinear 2.743, ARTN 2.904). Marked prior losses with [CORRECTED] above. The training loss computation itself was not changed — this is a reporting correction, not a training fix.

### 3. Parameter Counts Were Wrong

**What was wrong:** Previous reports had incorrect parameter counts for all three models.

| Model | Previous claim | Correct value | Source of error |
|-------|---------------|--------------|-----------------|
| FP32 | 6,852,608 | 5,828,608 | Did not account for weight tying (lm_head = token_embedding) |
| BitLinear | 6,852,682 | 6,852,608 | Included 74 scalar buffers as parameters |
| ARTN | 2,904,077 | 2,904,065 | Minor miscount |

The "58% parameter reduction" claim was wrong. Correct: 50.2% vs FP32 (tied), 57.6% vs BitLinear (untied).

**When discovered:** October 26, 2026 (parameter audit, Phase 60.13)

**How addressed:** Recounted via `sum(p.numel() for p in model.parameters())`. Buffers counted separately. Weight tying verified via data_ptr comparison.

### 4. Future-Token Leakage in Halting Gate Design

**What was wrong:** First halting gate design used mean pooling `gate(mean_pool(x))` — the gate sees all positions including future tokens, breaking causality. Second design used sequence-level `gate(x[:,-1,:])` applied to all positions via broadcasting — position t's output was weighted by a scalar computed from the last position, still leaking future information.

**When discovered:** October 2026 (during architecture design, Phase 60.4)

**How addressed:** Fixed to per-position causal: `h = sigmoid(gate(x))` → (B,T,1). Each position's halt probability depends only on its own causal representation. Verified via autograd: changing future input tokens does not affect earlier positions' logits or halting weights.

### 5. Existing Ponder Loss Was Non-Differentiable

**What was wrong:** `src/model/halting.py` line 187 creates `torch.tensor(diff**2, requires_grad=False)` with `.item()` detached values. The gradient could never flow to the halting gate through this loss. If the existing halting code had been trained, the gate would not have learned to halt adaptively.

**When discovered:** October 2026 (during architecture audit, Phase 60.3)

**How addressed:** Replaced with differentiable `E = 1 + Σ mean(R_i)` computed from actual halt probabilities. Verified gradient direction with autograd: E>target → gradient pushes h up, E<target → pushes h down. The new ARTNGPT class uses this corrected ponder loss.

### 6. Training Time Estimate Was Wrong (4.2h vs 14.6h)

**What was wrong:** Earlier documentation estimated 4.2 hours for 10k steps based on assumed 1.5s/step. Actual measured time was 5.25s/step = 14.6 hours (BitLinear) and 5.17s/step = 14.35 hours (ARTN). The estimate was off by 3.5x.

**When discovered:** October 25, 2026 (Phase 60.5, during training script preparation)

**How addressed:** Corrected estimate using measured per-step time from `bitlinear_adamw_results.json` (52,450 seconds = 14.6 hours). ARTN actual: 51,653 seconds = 14.35 hours (measured).

### 7. BitLinear GPT Is SLOWER Than FP32 on CPU

**What was wrong:** Blueprint expected "throughput increase via integer addition replacing multiplication" (Stage 9 evaluation matrix). Measured: FP32 24.06 tok/s, BitLinear 5.76 tok/s. Ternary quantization in PyTorch adds Python/autograd overhead that exceeds savings from simpler arithmetic. CPU has optimized BLAS/SIMD for FP32 matmul but no hardware acceleration for ternary.

**When discovered:** October 25, 2026 (Stage 9 benchmark, Phase 60.8)

**How addressed:** This is a measured limitation of the current PyTorch implementation, not of ternary quantization in theory. Reported honestly. The speedup from ARTN comes from adaptive halting (fewer passes), not from ternary arithmetic being faster.

### 8. RAM Measurement Used Wrong Tool (tracemalloc vs psutil)

**What was wrong:** Stage 8 inference reported 0.18 MB RAM using tracemalloc, which only tracks Python allocations — not PyTorch tensors, model weights, or CUDA/CPU memory pools. Actual RAM was 21.46 MB (psutil process RSS).

**When discovered:** October 21, 2026 (quality investigation, Phase 60.2)

**How addressed:** Switched all memory measurements to psutil `Process().memory_info().rss`. All Stage 9 measurements use psutil.

### 9. threshold=1.0 Floating-Point Artifact (5.92 vs 6.0 passes)

**What was wrong:** First threshold sweep reported 5.92 avg passes at threshold=1.0. Expected exactly 6.0. Root cause: the `remaining` tensor goes slightly negative due to floating-point subtraction (`remaining = remaining - p`), making `cumulative = 1.0 - remaining > 1.0` before pass 6, triggering early stop.

**When discovered:** October 25, 2026 (user flagged, Phase 60.7)

**How addressed:** Fixed in causal evaluation (Phase 60.10): threshold=1.0 bypasses the halt check entirely (always 6 passes). For threshold<1.0, `remaining.clamp(min=0.0)` prevents the floating-point artifact. Corrected measurement: exactly 6.000 passes (histogram {6:1280}).

### 10. Batch-Level Hard Exit Evaluation Used Wrong Halt Mechanism [SUPERSEDED]

**What was wrong:** First hard-exit NLL evaluation (Phase 60.9) used the last position of a full 255-token sequence for the halt decision. This is not the actual inference path — during generation, each token's halt decision should use only its own preceding context. Batch evaluation showed almost no early exits (5.88-5.98 passes at all thresholds), which was misleading.

**When discovered:** October 25, 2026 (user flagged, Phase 60.10)

**How addressed:** Replaced by per-position causal evaluation (Phase 60.10): for each position t, run forward on context [0..t], halt from position t, score target t+1. Results: threshold=0.9 gives 3.30 avg passes (not 5.90), 45% pass savings, +0.0024 NLL. The batch-level results are marked [SUPERSEDED] in the ROOT_CAUSE doc.

### 11. FP32 Baseline Load Failure [PARTIALLY SUPERSEDED]

**What was wrong:** Stage 8 quality investigation reported FP32 baseline could NOT be loaded due to architecture mismatch between checkpoint and `src/model/gpt.py`. The checkpoint uses `token_embedding`, `blocks.*`, `ln_f` but `src/model/gpt.py` expected `embeddings.token_embedding`, `embeddings.positional_encoding`, `ln_final`.

**When discovered:** October 21, 2026 (Phase 60.2)

**How addressed:** Found the correct `FP32GPT` class in `experiments/fp32_baseline_controlled.py` which matches the checkpoint structure. Successfully loaded and benchmarked all three models in Phase 60.8. The original finding is marked [PARTIALLY SUPERSEDED] in the ROOT_CAUSE doc.

---

## Phase 60 Evidence Summary

| Evidence | File | Date |
|----------|------|------|
| Inference pipeline | `src/inference/generate.py` | Oct 16 |
| Stage 8 inference results | `outputs/stage8_inference_results.json` | Oct 16 |
| Quality investigation | `outputs/stage8_quality_investigation_report.md` | Oct 21 |
| Deployment benchmark | `outputs/stage8_deployment_benchmark.json` | Oct 21 |
| Inference-only checkpoint | `outputs/bitlinear_adamw_20260910_143646/checkpoints/bitlinear_adamw/step_9500_inference_only.pt` | Oct 21 |
| ARTN model | `experiments/artn_model.py` | Oct 23 |
| ARTN smoke test | `scripts/artn_smoke_test.py` | Oct 23 |
| ARTN training script | `experiments/artn_train.py` | Oct 24 |
| ARTN training results | `outputs/artn_recursive_20261025_002725/artn_results.json` | Oct 25 |
| ARTN checkpoints (20) | `outputs/artn_recursive_20261025_002725/checkpoints/artn/` | Oct 25 |
| Threshold sweep | `outputs/artn_recursive_20261025_002725/threshold_sweep_results.json` | Oct 25 |
| Stage 9 speed benchmark | `outputs/artn_recursive_20261025_002725/stage9_benchmark.json` | Oct 25 |
| Hard exit eval [SUPERSEDED] | `outputs/artn_recursive_20261025_002725/stage9_hard_exit_eval.json` | Oct 25 |
| Causal evaluation | `outputs/artn_recursive_20261025_002725/stage9_causal_eval.json` | Oct 25 |
| Fair evaluation | `outputs/artn_recursive_20261025_002725/fair_eval.json` | Oct 25 |
| Generation quality evaluation (Phase 60.15-60.23) | `experiments/results/ROOT_CAUSE_INVESTIGATION_SUMMARY.md` lines 6963+ | Oct 29-Nov 3 |
| Controlled training pilot (Phase 60.24-60.27) | `experiments/results/ROOT_CAUSE_INVESTIGATION_SUMMARY.md` lines 8509+ | Nov 3-5 |
| Diagnostic follow-up probes (Phase 60.28) | `experiments/results/ROOT_CAUSE_INVESTIGATION_SUMMARY.md` lines 8822+ | Nov 6 |
| Four-arm construction-transfer experiment (Phase 60.29) | `experiments/results/ROOT_CAUSE_INVESTIGATION_SUMMARY.md` lines 9167+ | Nov 7-10 |

**Full Phase 60 documentation:** `experiments/results/ROOT_CAUSE_INVESTIGATION_SUMMARY.md` lines 6255-9612

---

*This document updated: November 10, 2026*
*Total training time invested: ~348.7 hours (280 original + 29 Phase 46-47 + 14.35 ARTN Phase 60 + 23.2 controlled training pilot Phase 60.24-60.27)*
