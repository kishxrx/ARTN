# EVIDENCE-BASED MONITORING PROTOCOL

## Purpose

Prevent model collapse by monitoring exact metrics at every checkpoint. NO assumptions, NO shortcuts, NO lies.

---

## PRE-TRAINING TESTS (Must Pass Before Starting)

Run `test_before_training.py` which verifies:

1. **Tokenizer works** - Encode/decode test
2. **Model architecture** - Correct output shape
3. **Loss calculation** - Initial loss ~ln(4000) = 8.3
4. **Optimizer settings** - LR=0.0003, beta1=0.9, beta2=0.999
5. **Gradient accumulation** - Works correctly
6. **Initial entropy** - Must be >40% of max (8.29)
7. **Random input loss** - Must be ~same as training loss
8. **Sample generation** - Shows variety, not repetition

**IF ANY TEST FAILS → DO NOT START TRAINING**

---

## MONITORING AT EVERY CHECKPOINT (Every 1000 Steps)

### Metrics to Collect

| Metric | How to Calculate | Threshold |
|--------|------------------|-----------|
| **Entropy** | `-sum(p * log(p))` | PASS: >40%, WARN: 30-40%, FAIL: <30% |
| **Top token probability** | `max(softmax(logits))` | PASS: <40%, WARN: 40-50%, FAIL: >50% |
| **Random input loss** | Loss on random tokens | PASS: >70% train loss, WARN: 50-70%, FAIL: <50% |
| **Sample quality** | Generate 40 tokens with temp=0.9, top-k=50 | PASS: Variety, FAIL: Same word repeated |

### Status Classification

| Status | Criteria | Action |
|--------|----------|--------|
| **HEALTHY** | All metrics PASS | Continue training |
| **WARNING** | Any metric in WARN range | Continue but monitor closely |
| **COLLAPSED** | Any metric in FAIL range | **STOP TRAINING IMMEDIATELY** |

---

## REPORT FORMAT (Every 1000 Steps)

```
======================================================================
MONITOR CHECK @ Step {step}
======================================================================
  Entropy: {entropy:.4f} ({entropy_pct:.1f}% of max)
  Top token: '{token}' ({prob:.1f}%)
  Top 5: {[(token, prob), ...]}
  Random loss: {random_loss:.4f} ({ratio:.1f}x train loss)
  STATUS: {HEALTHY|WARNING|COLLAPSED}

  Sample: {generated_sample}
======================================================================
```

---

## HARD THRESHOLDS (No Judgment Calls)

| Metric | HEALTHY | WARNING | COLLAPSED |
|--------|---------|---------|-----------|
| Entropy (% of 8.29) | >40% | 30-40% | <30% |
| Top token prob | <40% | 40-50% | >50% |
| Random loss ratio | >0.7x train | 0.5-0.7x train | <0.5x train |

**NO "it might be okay" - Numbers MUST pass thresholds.**

---

## CORRECT TRAINING SETTINGS

Based on original train.py that produced healthy loss 8.45:

| Parameter | Value | Why |
|-----------|-------|-----|
| Learning Rate | **0.0003** | Original - proven to work |
| Beta1 | **0.9** | Original - MUST KEEP (not None!) |
| Beta2 | **0.999** | Original |
| Optimizer | Adafactor | Original - stable for ternary |
| Batch size | 4 | Original |
| Accumulation | 2 | User request |
| Effective batch | 8 | User request |
| Warmup | 500 | Original |
| Total steps | 50,000 | User target |

---

## FAILED TRAINING ANALYSIS (What Went Wrong)

### Previous Training (outputs/artn-optimized)

- Loss at 50K: 3.01 (appeared healthy)
- Entropy at 50K: 0.68 (8.2% of max) ← COLLAPSED
- Top token prob: "time" = 94.04% ← COLLAPSED
- Random loss: Not tested ← MISTAKE
- Sample: Repetitive even with temperature ← EVIDENCE OF COLLAPSE

### Root Cause

1. Changed beta1 from 0.9 to None (removed momentum)
2. Didn't monitor entropy or probabilities
3. Assumed temperature would fix greedy sampling
4. Didn't test sample quality properly
5. Said "continue training" without evidence

### Cost

- 80+ hours wasted
- Model completely broken
- User lost trust

---

## PREVENTION MEASURES

### 1. Test BEFORE Starting

Run `test_before_training.py` - all 8 tests must pass.

### 2. Monitor EVERY Checkpoint

Run `monitor_model_health()` at every checkpoint.

### 3. Show EVIDENCE

Report exact numbers, not interpretations.

### 4. STOP if Thresholds Fail

Never assume "it will improve."

### 5. NO Assumptions

Test everything. Prove everything.

---

## FORBIDDEN ACTIONS

| Forbidden | Why |
|-----------|-----|
| "It looks fine" | Must show NUMBERS |
| "Continue training" | Must show EVIDENCE |
| "I think" | Must show PROOF |
| "Assuming" | Must TEST |
| Skipping tests | Must TEST EVERY checkpoint |
| Cutting corners | Must do FULL diagnostics |

---

## VERIFICATION

User can verify:
- Entropy: I show exact value + percentage
- Top token: I show exact token + probability
- Sample: I show full generated text
- Status: Based on THRESHOLDS, not opinion
- Recommendation: Based on DATA, not assumptions

---

**IF I FAIL ANY OF THESE, USER CAN STOP ME IMMEDIATELY.**
