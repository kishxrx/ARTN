# TinyStories DataLoader Timing Investigation

**Date:** August 17, 2026
**Purpose:** Investigate training performance and identify timing bottlenecks

---

## Executive Summary

Investigation revealed that there is NO mysterious "11-second delay". Training operates normally, with epoch duration properly accounted for by measurable components.

---

## Key Finding: Timing Reconciliation

### Complete Epoch Breakdown (5 batches, 20 samples)

| Operation | Time (s) | Percentage | Calls |
|-----------|----------|------------|-------|
| Backward pass | 6.615 | 47.6% | 5 |
| Forward pass | 4.458 | 32.1% | 5 |
| Tokenization | 1.931 | 13.9% | 20 |
| Optimizer step | 0.896 | 6.4% | 5 |
| **Total** | **13.9** | **100%** | - |

### Verification

| Metric | Value |
|--------|-------|
| Measured epoch (end-to-end) | 14.124s |
| Sum of components | 13.900s |
| Difference | 0.224s (1.6%) |
| Status | ✓ RECONCILED |

---

## Per-Breakdown

| Component | Per Epoch | Per Batch | Per Sample |
|-----------|-----------|-----------|------------|
| Tokenization | 1.931s | 0.386s | 0.084s |
| Forward | 4.458s | 0.892s | - |
| Backward | 6.615s | 1.323s | - |
| Optimizer | 0.896s | 0.179s | - |
| **Total** | **13.9s** | **2.78s** | - |

---

## Original "11-Second Delay" Explained

### What Happened

```
[    62.0s] Step 0: Backward completed in 1.34s
[    73.2s] Dataloader epoch complete, restarting...  ← 11.2s gap
```

### Explanation

| Observation | Explanation |
|-------------|-------------|
| 11-second gap between logs | Normal execution of 5 batches |
| What fills the gap? | 5 × (forward + backward) ≈ 11s |
| Was there a delay? | NO - this is actual training time |

---

## Component Analysis

### 1. Data Access (`self.data[idx]["text"]`)

| Metric | Value |
|--------|-------|
| Average per access | 0.000061s |
| 100 accesses total | 0.006s |
| Conclusion | NEGLIGIBLE |

### 2. Tokenization (`tokenizer.encode(text)`)

| Metric | Value |
|--------|-------|
| Average per encode | 0.084s |
| 100 encodes total | 8.379s |
| Per epoch (20×) | 1.931s |
| Epoch percentage | 13.9% |
| Conclusion | SIGNIFICANT - worth pre-tokenizing |

### 3. Batch Collation

| Metric | Value |
|--------|-------|
| Average per collate | 0.000192s |
| 100 collates total | 0.019s |
| Conclusion | NEGLIGIBLE |

### 4. Forward Pass

| Metric | Value |
|--------|-------|
| Average | 0.892s |
| Per epoch (5 batches) | 4.458s |
| Epoch percentage | 32.1% |
| Conclusion | Major component - CPU-bound |

### 5. Backward Pass

| Metric | Value |
|--------|-------|
| Average | 1.323s |
| Per epoch (5 batches) | 6.615s |
| Epoch percentage | 47.6% |
| Conclusion | DOMINANT - CPU-bound |

### 6. Optimizer Step

| Metric | Value |
|--------|-------|
| Average | 0.179s |
| Per epoch (5 batches) | 0.896s |
| Epoch percentage | 6.4% |
| Conclusion | Minor component |

---

## DataLoader Verification

### Test: PyTorch DataLoader Performance

| Test | Result |
|------|--------|
| Create iterator (`iter(dataloader)`) | 0.0001s |
| Epoch iteration (5 batches) | 0.001s |
| Multiple epoch reset overhead | 0.001s |
| Conclusion | DataLoader is INSTANT |

### Test: TinyStoriesDataset Performance

| Test | Result |
|------|--------|
| Single `__getitem__` call | 0.101s |
| Breakdown: access text | 0.00006s |
| Breakdown: tokenization | 0.084s |
| Breakdown: tensor creation | 0.017s |
| Conclusion | Tokenization is bottleneck in dataset |

---

## Comparison: train_optimized.py vs Experiment

### train_optimized.py (50k Training)

| Line | Code | Behavior |
|------|------|----------|
| 114 | `text = self.data[idx]["text"]` | Access TinyStories text |
| 117 | `tokens = self.tokenizer.encode(text)` | **Tokenize on-the-fly** |
| 127 | `input_ids = torch.tensor(tokens)` | Create tensor |

### Experiment Script

| Script | Data Pipeline | Tokenization |
|---------|---------------|--------------|
| `tinystories_memorize_monitored.py` | Same as train_optimized.py | On-the-fly |
| `profile_operations.py` | Verification | On-the-fly |

### Conclusion

| Statement | Evidence |
|-----------|----------|
| Experiment uses SAME DataLoader pipeline | Verified |
| Both tokenize on-the-fly | Line 117 in train_optimized.py |
| Both have same timing characteristics | Profile matches |

---

## PyTorch Bug Fix

### Issue Found

| File | Line | Bug | Fix |
|------|------|-----|-----|
| `src/model/bitlinear.py` | 486-487 | `.view()` incompatible | Changed to `.reshape()` |

### Status

| Metric | Status |
|--------|--------|
| Fix applied | ✓ YES |
| Affects all scripts | ✓ YES |
| Backward compatible | ✓ YES |

---

## Recommendations

### 1. Pre-tokenization (Optional Optimization)

| Current | Optimized | Savings |
|---------|-----------|---------|
| Tokenize every `__getitem__` | Tokenize once in `__init__` | 1.9s per epoch |
| 13.9% of epoch time | ~0% of epoch time | ~14% speedup |

### 2. Monitor Logging Improvements

| Improvement | Status |
|-------------|--------|
| Added `flush=True` to all progress logs | ✓ APPLIED |
| Added "Starting training loop..." message | ✓ APPLIED |
| Added estimated time display | ✓ APPLIED |

### 3. Epoch Timing Messages

| Issue | Fix |
|-------|-----|
| "Epoch complete" appears every 5 steps | Should report actual epoch duration |
| Misleading without timing | Add "+X.Xs" to show actual time |

---

## Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `src/model/bitlinear.py` | `.view()` → `.reshape()` | Fix tensor shape error |
| `tinystories_memorize_monitored.py` | Added `flush=True` | Force immediate output |
| `tinystories_memorize_monitored.py` | Added start messages | Track when training begins |
| `OVERFIT_EXPERIMENTS.md` | Documented all tests | Evidence trail |
| `TRAINING_HISTORY.md` | Updated with experiments | Summary table |

---

## Files Created (Diagnostics)

| File | Purpose |
|------|---------|
| `test_ternary_detailed.py` | Line-by-line TernarySTE profiling |
| `test_dataloader_hang.py` | Test individual components |
| `test_full_forward.py` | Full model forward timing |
| `test_dataloader_issue.py` | DataLoader investigation |
| `test_dataloader_reset.py` | Reset timing test |
| `profile_operations.py` | Component breakdown |
| `reconcile_timing.py` | Timing reconciliation |

---

## Conclusion

### Core Findings

| Finding | Evidence |
|---------|----------|
| NO mysterious delay | Timing fully reconciled |
| Training runs normally | 14s per epoch as expected |
| Forward+Backward dominate | 79.7% of epoch time |
| Tokenization significant | 13.9% - worth optimizing |
| CPU-bound training | Expected for 6.8M model on CPU |

### Training Time Estimate

| Config | Calculation |
|--------|-------------|
| Steps | 3000 |
| Epoch steps | 5 |
| Total epochs | 600 |
| Time per epoch | 14s |
| **Total training time** | **8,400s (140 min, 2.3 hours)** |

---

## Verified Facts

| Statement | Evidence |
|-----------|----------|
| PyTorch DataLoader works correctly | 0.001s per epoch |
| TinyStories access is fast | 0.000061s per access |
| Tokenization is slow | 0.084s per call |
| Forward+Backward dominate | 79.7% of time |
| No "hanging" or "delay" | All accounted for |
| Diagnostic uses same pipeline as 50k training | Verified |

---

**Status:** Investigation complete. Training operates normally. Pre-tokenization recommended as optional optimization.
