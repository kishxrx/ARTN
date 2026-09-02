# LR Experiment Commands - FINAL VERSION

**Date:** August 18, 2026

## Prerequisites

Run static diagnostics first (already PASSED):

```bash
cd /root/agent-sandbox/ARTN
/root/agent-sandbox/.venv/bin/python experiments/static_diagnostics.py
```

**Status:** 9 PASS, 0 FAIL, 2 UNKNOWN (minor)

---

## LR Experiments (SEQUENTIAL - Run in Order)

Each experiment takes approximately 1 hour on this CPU machine.

### Experiment 1: Baseline (LR=0.001, constant)

```bash
cd /root/agent-sandbox/ARTN
PYTHONUNBUFFERED=1 /root/agent-sandbox/.venv/bin/python experiments/lr_experiment.py --config lr_001_constant
```

**Purpose:** Baseline for comparison (same LR as original training)

**Output:** `experiments/results/lr_experiments/lr_001_constant_<timestamp>/`

---

### Experiment 2: Higher LR (LR=0.003, constant)

```bash
cd /root/agent-sandbox/ARTN
PYTHONUNBUFFERED=1 /root/agent-sandbox/.venv/bin/python experiments/lr_experiment.py --config lr_003_constant
```

**Purpose:** Test H1 (LR too low)

**Output:** `experiments/results/lr_experiments/lr_003_constant_<timestamp>/`

---

### Experiment 3: Higher LR with Cosine (LR=0.003, cosine)

```bash
cd /root/agent-sandbox/ARTN
PYTHONUNBUFFERED=1 /root/agent-sandbox/.venv/bin/python experiments/lr_experiment.py --config lr_003_cosine
```

**Purpose:** Test H2 (Schedule wrong)

**Output:** `experiments/results/lr_experiments/lr_003_cosine_<timestamp>/`

---

## Results Comparison (After All Experiments)

```bash
cd /root/agent-sandbox/ARTN
/root/agent-sandbox/.venv/bin/python experiments/compare_lr_results.py --results_dir experiments/results/lr_experiments
```

**Output:**
- Comparison table
- Hypothesis ranking (STRONGLY SUPPORTED / POSSIBLE / UNLIKELY / RULED OUT / UNKNOWN)
- Evidence for each hypothesis

---

## What Is Compared

| Signal | Purpose |
|--------|---------|
| Training loss trajectory | Is loss decreasing? |
| Perplexity | Language modeling quality |
| Gradient norms | Gradient health |
| Fixed-prompt generations | Coherence/structure |
| Repetition rate | Output quality |
| Coherent ratio | Generation readability |

**Critical:** Analysis does NOT use arbitrary thresholds (e.g., "loss < 2 = success").

---

## How Hypotheses Are Evaluated

| Status | Definition |
|--------|------------|
| **STRONGLY SUPPORTED** | Multiple independent signals improve, especially validation loss AND generations |
| **POSSIBLE** | Some signals improve but evidence is mixed |
| **UNLIKELY** | Little/no improvement or conflicting signals |
| **RULED OUT** | Controlled comparison clearly contradicts the hypothesis |
| **UNKNOWN** | Insufficient evidence |

---

## Hypotheses Tested

| Hypothesis | Experiments Compared |
|------------|---------------------|
| H1: LR too low | lr_001_constant vs lr_003_constant |
| H2: Schedule wrong | lr_003_constant vs lr_003_cosine |
| H3: Model too small | UNKNOWN (needs architecture experiment) |
| H4: Ternary limitation | UNKNOWN (needs full-precision baseline) |
| H5: Not enough training | UNKNOWN (needs longer run) |

---

## Estimated Time

- Experiment 1: ~1 hour
- Experiment 2: ~1 hour
- Experiment 3: ~1 hour
- Comparison: 10 seconds

**TOTAL: ~3 hours**

---

## Expected Findings

### If H1 is STRONGLY SUPPORTED:
- lr_003_constant shows lower loss AND better generations
- Action: Use higher LR for future training

### If H2 is STRONGLY SUPPORTED:
- cosine decay shows worse results than constant
- Action: Use constant schedule

### If neither is supported:
- Issue is NOT hyperparameters
- Consider: architecture, data, training duration

---

## No Automatic Changes

This experiment:
- Does NOT modify model architecture
- Does NOT modify training strategy
- Does NOT update master blueprint
- Only REPORTS evidence and ranks hypotheses

User decides what action to take based on evidence.

---

## After Running

1. Review comparison output
2. Check hypothesis rankings
3. Examine sample generations
4. Read evidence for each hypothesis
5. Decide next steps based on evidence

---

## Files Summary

| File | Purpose |
|------|---------|
| `static_diagnostics.py` | Non-training checks |
| `lr_experiment.py` | Training experiment runner |
| `compare_lr_results.py` | Evidence-based comparison |
| `LR_EXPERIMENT_COMMANDS.md` | This file |

---

## Quick Reference

```bash
# Run all three experiments (sequential):
cd /root/agent-sandbox/ARTN

PYTHONUNBUFFERED=1 /root/agent-sandbox/.venv/bin/python experiments/lr_experiment.py --config lr_001_constant
PYTHONUNBUFFERED=1 /root/agent-sandbox/.venv/bin/python experiments/lr_experiment.py --config lr_003_constant
PYTHONUNBUFFERED=1 /root/agent-sandbox/.venv/bin/python experiments/lr_experiment.py --config lr_003_cosine

# Compare results:
/root/agent-sandbox/.venv/bin/python experiments/compare_lr_results.py --results_dir experiments/results/lr_experiments
```
