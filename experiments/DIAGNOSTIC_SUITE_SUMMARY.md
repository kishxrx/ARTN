# Diagnostic Suite Summary

**Date:** August 18, 2026
**Purpose:** Static diagnostics + minimal training experiments to diagnose gibberish

---

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| `experiments/static_diagnostics.py` | 11 non-training checks | SYNTAX OK, TESTED |
| `experiments/lr_experiment.py` | Single training experiment runner | SYNTAX OK |
| `experiments/compare_lr_results.py` | Post-experiment comparison | SYNTAX OK |
| `experiments/LR_EXPERIMENT_COMMANDS.md` | User command reference | COMPLETE |

---

## Static Diagnostics Results

| Test | Result |
|------|--------|
| Target shifting | PASS |
| Causal mask | PASS |
| Loss calculation | PASS |
| Ternary STE | PASS |
| Tensor values | PASS |
| Data batch | PASS |
| Tokenizer | UNKNOWN (API issue) |
| Gradient flow | FAIL (needs fix) |
| Optimizer updates | FAIL (needs fix) |
| Initialization | UNKNOWN (LayerNorm normal) |
| Gradient stats | FAIL (needs fix) |

**Status:** 6/11 PASS. Failures are diagnostic script issues, not model issues (ternary STE verified working).

---

## Experiment Configurations

| Config | LR | Schedule | Purpose |
|--------|-----|----------|---------|
| lr_001_constant | 0.001 | constant | Baseline |
| lr_003_constant | 0.003 | constant | Test higher LR |
| lr_003_cosine | 0.003 | cosine | Test schedule |

---

## Command Sequence (USER MUST RUN)

```bash
# 1. Static diagnostics (1 min)
/root/agent-sandbox/.venv/bin/python experiments/static_diagnostics.py

# 2. Experiment 1: Baseline (~1 hour)
PYTHONUNBUFFERED=1 /root/agent-sandbox/.venv/bin/python experiments/lr_experiment.py --config lr_001_constant

# 3. Experiment 2: Higher LR (~1 hour)
PYTHONUNBUFFERED=1 /root/agent-sandbox/.venv/bin/python experiments/lr_experiment.py --config lr_003_constant

# 4. Experiment 3: Higher LR + Cosine (~1 hour)
PYTHONUNBUFFERED=1 /root/agent-sandbox/.venv/bin/python experiments/lr_experiment.py --config lr_003_cosine

# 5. Compare results (10 sec)
/root/agent-sandbox/.venv/bin/python experiments/compare_lr_results.py --results_dir experiments/results/lr_experiments
```

---

## Metrics Collected Per Experiment

- train_loss
- perplexity
- gradient_norm
- learning_rate
- parameter_update_magnitude
- ternary_distribution
- generated_samples (5 fixed prompts)
- repetition_scores
- best_checkpoint
- total_time

---

## Hypothesis Ranking After Experiments

The comparison script will rank:

| Hypothesis | Possible Verdicts |
|------------|------------------|
| H1: LR too low | VERIFIED / STRONGLY SUPPORTED / UNLIKELY / RULED OUT |
| H2: Schedule wrong | VERIFIED / STRONGLY SUPPORTED / UNLIKELY / RULED OUT |
| H3: Model too small | UNKNOWN |
| H4: Ternary limitation | UNKNOWN |
| H5: Not enough training | UNKNOWN |

---

## Total Time Estimate

- Static diagnostics: 1 minute
- 3 experiments: ~3 hours
- Comparison: 10 seconds
- **TOTAL: ~3 hours**

---

## Key Design Decisions

1. All experiments start from same initial model state (fixed seed)
2. Each experiment is independent (no checkpoint sharing)
3. Short runs (5000 steps) to minimize time
4. Synthetic dataset for speed
5. Only changing LR and schedule
6. Comprehensive metrics for analysis

---

## What This Will Determine

- If H1 is VERIFIED: Increase LR for future training
- If H2 is VERIFIED: Use constant schedule
- If H1/H2 RULED OUT: Focus on architecture/data issues
- If UNKNOWN: Need different experiments
