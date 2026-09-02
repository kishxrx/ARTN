# Overfit Experiment Results

**Date:** August 16-17, 2026
**Purpose:** Investigate why 50k training produces gibberish by testing optimizer/LR configurations

---

## Background

### Previous Training Results

| Attempt | Date | Steps | Optimizer | LR | Final Loss | Entropy | Result |
|---------|------|-------|-----------|-----|------------|---------|--------|
| 5 | Aug 1-16, 2026 | 50,000 | Adafactor | 0.001 | 7.13 | 97.0% | Gibberish output |

**Observation:** 50k training completed with healthy entropy (97%) but outputs are incoherent.

---

## Discovery: Optimizer/LR Mismatch

### Initial Finding (Aug 16, 2026)

**Files examined:**
- `scripts/overfit_test.py` (random token test)
- `scripts/train_optimized.py` (real training)

**Configuration comparison:**

| Component | Random Overfit Test | 50k Real Training |
|-----------|---------------------|-------------------|
| Optimizer | `torch.optim.Adam` | Adafactor |
| Learning Rate | 0.01 | 0.001 |
| Beta1 | 0.9 (default) | 0.5 |
| Beta2 | 0.999 (default) | 0.98 |
| Weight Decay | None | 0.01 |

**Source evidence:**
- `scripts/overfit_test.py` line 105: `optimizer = torch.optim.Adam(model.parameters(), lr=lr)`
- `scripts/overfit_test.py` lines 50, 189, 203: `lr: float = 0.01`

---

## Experiment Series

### Test 1: Random Token Overfit (Baseline Verification)

**Purpose:** Verify original random token test still works

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| LR | 0.01 |
| Data | Random synthetic tokens |
| Vocab size | 100 |
| Samples | 5 |
| Steps | 500 |
| Model | BitLinearGPT (smaller) |
| Parameters | 115,044 |

**Results:**
| Metric | BitLinear (Ternary) | Baseline (FP32) |
|--------|---------------------|-----------------|
| Initial loss | 5.1455 | 4.6224 |
| Final loss | 0.0001 | 0.0002 |
| Loss reduction | 100% | 100% |
| Status | PASSED | PASSED |

**Evidence:**
- Script ran: `scripts/overfit_test.py`
- Output logs captured
- Loss trajectory: 5.15 → 0.0001 over 500 steps

**What this establishes:**
- Ternary BitLinear model CAN memorize data
- Adam + LR=0.01 works for memorization task
- Gradient flow is functional
- No architectural blocker

---

### Test 2: TinyStories Overfit with Adafactor + LR=0.001

**Purpose:** Test with SAME config as real 50k training

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adafactor |
| LR | 0.001 |
| Beta1 | 0.5 |
| Beta2 | 0.98 |
| Weight Decay | 0.01 |
| Data | TinyStories (20 stories) |
| Vocab size | 4000 |
| Steps | 1000 |
| Model | BitLinearGPT (full size) |
| Parameters | 6,856,608 |

**Results:**
| Metric | Value |
|--------|-------|
| Initial loss | 8.7828 |
| Final loss | 8.4564 |
| Loss reduction | 3.7% |
| Status | FAILED |

**Loss trajectory:**
| Step | Loss |
|------|------|
| 0 | 8.7828 |
| 200 | 8.6080 |
| 400 | 8.6725 |
| 600 | 8.5417 |
| 800 | 8.5175 |
| 999 | 8.4564 |

**Samples at step 999:**
```
Story 1: "...gave ke, herselYour..."
Story 2: "...just ³ure e..."
Story 3: "...bear. other u..."
```

**What this establishes:**
- Adafactor + LR=0.001 does NOT memorize TinyStories
- Loss barely decreased (3.7%)
- Generated text remains incoherent

---

### Test 3: TinyStories Overfit with Adafactor + LR=0.01

**Purpose:** Test if higher LR fixes Adafactor

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adafactor |
| LR | 0.01 (10x higher) |
| Other params | Same as Test 2 |

**Results:**
| Metric | Value |
|--------|-------|
| Initial loss | 8.7828 |
| Final loss | 7.3897 |
| Loss reduction | 15.9% |
| Best loss | 7.39 @ step 700 |
| Status | FAILED |

**Loss trajectory:**
| Step | Loss |
|------|------|
| 0 | 8.7828 |
| 200 | 8.0882 |
| 400 | 7.8686 |
| 600 | 7.5190 |
| 800 | 7.4444 |
| 999 | 7.3897 |

**Samples at step 999:**
```
Story 1: "...gave and and and th..."
Story 2: "...just a a ro..."
Story 3: "...a a a a smile..."
```

**Repetition pattern observed:**
- Generated text shows repetition: "and and and", "a a a a"
- LR=0.001 test did NOT show repetition
- This may indicate early mode collapse

**What this establishes:**
- Higher LR (0.01) improves Adafactor performance (15.9% vs 3.7%)
- Still fails to memorize (loss stayed ~7.4, not near zero)
- Repetition suggests possible stability issues at high LR

---

### Test 4: TinyStories Overfit with Adam + LR=0.001

**Purpose:** Test Adam optimizer (same LR as real training)

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| LR | 0.001 |
| Beta1 | 0.5 |
| Beta2 | 0.98 |
| Weight Decay | 0.01 |
| Other params | Same as Test 2 |

**Results:**
| Metric | Value |
|--------|-------|
| Initial loss | 8.7828 |
| Final loss | 7.9970 |
| Loss reduction | 8.9% |
| Best loss | 7.45 @ step 250 |
| Status | FAILED |

**Loss trajectory:**
| Step | Loss |
|------|------|
| 0 | 8.7828 |
| 100 | 7.6864 |
| 250 | 7.4520 (best) |
| 500 | 7.7712 |
| 750 | 7.57 |
| 999 | 7.9970 |

**Critical observation:**
- Loss decreased initially (8.78 → 7.45)
- Loss then INCREASED (7.45 → 7.99)
- Optimization became unstable after step 250

**What this establishes:**
- Adam + LR=0.001 shows initial learning
- Optimization becomes unstable (loss rebounds)
- Does NOT memorize

---

### Test 5: TinyStories Overfit with Adam + LR=0.01

**Purpose:** Test SAME config as random test that worked

**Configuration:**
| Parameter | Value | Matches Random Test? |
|-----------|-------|---------------------|
| Optimizer | Adam | ✅ YES |
| LR | 0.01 | ✅ YES |
| Beta1 | 0.5 | ❌ NO (random used 0.9) |
| Beta2 | 0.98 | ❌ NO (random used 0.999) |
| Weight Decay | 0.01 | ❌ NO (random had none) |
| Data | TinyStories (20 stories) | ❌ NO |
| Vocab size | 4000 | ❌ NO (random used 100) |
| Model size | 6.8M params | ❌ NO (random used 115K) |

**Results:**
| Metric | Value |
|--------|-------|
| Initial loss | 8.7828 |
| Final loss | 7.6189 |
| Loss reduction | 13.3% |
| Best loss | 7.27 @ step 6 |
| Status | FAILED |

**Loss trajectory:**
| Step | Loss |
|------|------|
| 0 | 8.7828 |
| 6 | 7.2691 (best) |
| 100 | 8.0233 |
| 200 | 7.7765 |
| 400 | 7.6044 |
| 600 | 7.5678 |
| 800 | 7.5894 |
| 999 | 7.6189 |

**Critical observation:**
- Best loss achieved at step 6 (very early)
- Loss INCREASED after step 6
- Never returned to best loss

**What this establishes:**
- Adam + LR=0.01 on TinyStories: 13.3% reduction
- Same optimizer/LR on Random: 100% reduction
- Config that works on random does NOT work on TinyStories

---

## Summary Table

| Test | Optimizer | LR | Data | Vocab | Model Size | Final Loss | Reduction | Status |
|------|-----------|-----|------|-------|------------|------------|-----------|--------|
| 1 (Random) | Adam | 0.01 | Random tokens | 100 | 115K | 0.0001 | **100%** | PASSED |
| 2 | Adafactor | 0.001 | TinyStories | 4000 | 6.8M | 8.4564 | 3.7% | FAILED |
| 3 | Adafactor | 0.01 | TinyStories | 4000 | 6.8M | 7.3897 | 15.9% | FAILED |
| 4 | Adam | 0.001 | TinyStories | 4000 | 6.8M | 7.9970 | 8.9% | FAILED |
| 5 | Adam | 0.01 | TinyStories | 4000 | 6.8M | 7.6189 | 13.3% | FAILED |

---

## Verified Facts

### Fact 1: Random Overfit Test Works
- **Evidence:** Script ran, output captured
- **Config:** Adam + LR=0.01
- **Result:** Loss 5.15 → 0.0001 (100% reduction)
- **File:** `scripts/overfit_test.py` line 105 confirms Adam optimizer

### Fact 2: Ternary Architecture Can Memorize
- **Evidence:** BitLinear (ternary) passed random test
- **Result:** Loss → 0.0001
- **Conclusion:** Ternary quantization does NOT prevent learning

### Fact 3: Same Optimizer/LR Fails on TinyStories
- **Evidence:** Test 5 used Adam + LR=0.01 (same as random test)
- **Result:** Only 13.3% reduction on TinyStories (vs 100% on random)
- **Conclusion:** Config alone does NOT explain the difference

### Fact 4: Optimization Instability
- **Evidence:** Test 4 (Adam+0.001): loss decreased then increased
- **Evidence:** Test 5 (Adam+0.01): best loss at step 6, then worse
- **Observation:** Loss rebounds after initial decrease

### Fact 5: Repetition Pattern
- **Evidence:** Test 3 (Adafactor+0.01) generated "and and and", "a a a a"
- **Evidence:** Test 2 (Adafactor+0.001) did NOT show repetition
- **Observation:** Higher LR may cause instability

---

## Differences Between Random and TinyStories Tests

### Configuration Differences

| Factor | Random Test | TinyStories Test |
|--------|-------------|------------------|
| Vocab size | 100 | 4000 |
| Model parameters | 115K | 6.8M |
| Data type | Random synthetic tokens | Real language stories |
| Tokenizer | None (random integers) | BPE tokenizer (4000 vocab) |
| Sequence structure | None (random) | Structured language |
| Beta1 | 0.9 (default) | 0.5 |
| Beta2 | 0.999 (default) | 0.98 |
| Weight decay | None | 0.01 |

### Data Characteristics

| Factor | Random Tokens | TinyStories |
|--------|---------------|-------------|
| Structure | None | Grammatical sentences |
| Repetition | None | Story patterns |
| Token relationships | None | Semantic meaning |
| Memorization type | Pattern matching | Language learning |

---

## What We Do NOT Know

### Unknown 1: Why Random Succeeds and TinyStories Fails
- **Observation:** Same optimizer/LR, different results
- **Possible factors:** Vocab size, model size, data structure
- **Status:** Unexplained

### Unknown 2: Why Optimization Becomes Unstable
- **Observation:** Loss rebounds after initial decrease
- **Possible causes:** Learning dynamics, gradient issues, data characteristics
- **Status:** Unexplained

### Unknown 3: Role of Vocab Size
- **Observation:** Random used vocab=100, TinyStories used vocab=4000
- **Possible impact:** Loss calculation, gradient scale
- **Status:** Not tested

### Unknown 4: Role of Model Size
- **Observation:** Random used 115K params, TinyStories used 6.8M params
- **Possible impact:** Optimization dynamics
- **Status:** Not tested

### Unknown 5: Role of Tokenizer
- **Observation:** Random uses no tokenizer, TinyStories uses BPE
- **Possible impact:** Token distribution, sequence structure
- **Status:** Not tested

---

## Files Modified/Created

| File | Action | Date |
|------|--------|------|
| `scripts/overfit_test.py` | Ran (no changes) | Aug 17, 2026 |
| `experiments/tinystories_overfit.py` | Created | Aug 16, 2026 |
| `src/model/bitlinear.py` | Added `generate` method | Aug 16, 2026 |
| `TRAINING_HISTORY.md` | Updated with findings | Aug 16, 2026 |
| `OVERFIT_EXPERIMENTS.md` | Created (this file) | Aug 17, 2026 |

---

## Next Steps (Not Recommendations, Just Options)

| Option | Purpose | Status |
|--------|---------|--------|
| Test with vocab=100 on TinyStories | Isolate vocab effect | Not done |
| Test with smaller model | Isolate model size effect | Not done |
| Examine token distribution | Understand data characteristics | Not done |
| Test without tokenizer | Isolate tokenizer effect | Not done |
| Compare gradient norms between tests | Understand optimization difference | Not done |

---

## Evidence Trail

### Random Test Evidence
1. Script exists: `scripts/overfit_test.py`
2. Optimizer: Line 105 `torch.optim.Adam`
3. LR: Lines 50, 189, 203 `lr: float = 0.01`
4. Ran successfully: Output captured Aug 17, 2026

### TinyStories Test Evidence
1. Script exists: `experiments/tinystories_overfit.py`
2. Results saved: `experiments/results/results.json`
3. Output captured: Terminal logs
4. All 4 configurations tested and documented above

---

**Document created:** August 17, 2026
**Author:** Hermes Agent
**Status:** Factual record of experiments conducted
