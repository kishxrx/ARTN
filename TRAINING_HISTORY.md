# ARTN Training History - Complete Documentation

## Overview

This document records ALL training attempts, their configurations, results, and lessons learned.

---

## Training Attempts Summary

| Attempt | Script | Steps | Loss | Entropy | Result | Status |
|---------|--------|-------|------|---------|--------|--------|
| 1 | unknown | 5,000 | 8.45 | Not monitored | SUCCESS | Verified pipeline works |
| 2 | train_optimized.py | 50,000 | 3.16 | 8.2% | COLLAPSED | Failed |
| 3 | train_correct.py | 2,000 | 8.6143 | 94.8% | Healthy but incomplete | Stopped (wrong accumulation) |
| 4 | train_optimized.py (fixed) | 50,000 | 7.56 | 96.2% | HEALTHY but not useful | COMPLETED (gibberish output) |

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

*Document last updated: Aug 10, 2026*
*Training status: Attempt 4 COMPLETED - planning Attempt 5*
