# Padding Inflation Discovery: The Critical Audit Finding

**Date:** August 19, 2026
**Status:** CRITICAL DISCOVERY
**Importance:** This finding changed our entire understanding of the problem

---

## Executive Summary

**The apparent gap between FP32 (26.89%) and Ternary (3.56%) held-out accuracy was an ILLUSION caused by padding token inflation.**

**On REAL language tokens, both models perform SIMILARLY (~5% accuracy).**

**This RULES OUT ternary quantization as the root cause of gibberish generation.**

---

## The Misleading Initial Results

### Initial FP32 vs Ternary Comparison

Reported in `FP32_VS_TERNARY_COMPARISON.md` (SUPERSEDED):

| Model | Held-Out Top-1 Accuracy | Interpretation |
|-------|------------------------|----------------|
| FP32 5k | **26.89%** | appeared to learn |
| Ternary 50k | **3.56%** | appeared to fail |
| Difference | **23.33%** | FP32 appeared 7.5x better |

**Initial (INCORRECT) Conclusion:** FP32 clearly outperforms Ternary → Ternary quantization limits learning

---

## The Audit That Changed Everything

### Audit Script: `experiments/audit_heldout_evaluation.py`

**Audit Date:** August 19, 2026
**Purpose:** Verify the 26.89% FP32 held-out accuracy

**What the audit checked:**
1. Manual Top-1 accuracy calculation from raw logits
2. Manual CrossEntropyLoss calculation
3. Target/prediction position alignment
4. Padding token handling
5. Zero overlap between training and held-out sets
6. Same input tokens for FP32 and Ternary

**Audit Results:**

```
Total accuracy (incl. padding): 26.89%
Manual calculation verification: 26.89% ✓ MATCHES

BUT THEN:

Breakdown by token type:
  Padding positions: 102/102 = 100.00% correct
  Real positions: 19/348 = 5.46% correct
  
Total tokens: 450
Padding tokens: 102 (22.7%)
Real tokens: 348 (77.3%)
```

---

## The Critical Discovery

### FP32 Performance Breakdown

| Token Type | Count | Correct | Accuracy |
|------------|-------|---------|----------|
| **Padding (token 0)** | 102 | 102 | **100.00%** |
| **Real language tokens** | 348 | 19 | **5.46%** |
| **Total** | 450 | 121 | **26.89%** |

**FP32 achieved 100% accuracy on padding positions!**

### Ternary Performance Breakdown

| Token Type | Count | Correct | Accuracy |
|------------|-------|---------|----------|
| **Padding (token 0)** | 102 | 0 | **0.00%** |
| **Real language tokens** | 348 | 16 | **4.60%** |
| **Total** | 450 | 16 | **3.56%** |

**Ternary achieved 0% on padding, but 4.60% on real tokens!**

---

## Why This Happened

### The Padding Issue

**Evaluation code (identical for both models):**

```python
# No ignore_index specified in CrossEntropyLoss
criterion = nn.CrossEntropyLoss(reduction='mean')

# Padding positions included in accuracy calculation
correct = (preds == targets).sum().item()
accuracy = 100 * correct / targets.numel()
```

**What happened:**

1. **Held-out sequences had padding** (22.7% of tokens)
2. **Padding token ID = 0**
3. **FP32 learned to predict token 0** for end-of-sequence positions
4. **This gave FP32 100% accuracy on padding**
5. **Padding inflated total accuracy from 5.46% to 26.89%**

**Ternary:**
1. **Never learned to predict padding**
2. **0% accuracy on padding**
3. **Total accuracy stayed at 3.56%**

---

## The Corrected Comparison

### Real Token Accuracy (Excluding Padding)

| Model | Real Token Accuracy | Total Accuracy | Inflation |
|--------|---------------------|----------------|-----------|
| FP32 5k | **5.46%** | 26.89% | +21.43% |
| Ternary 50k | **4.60%** | 3.56% | -1.04% |
| **Difference** | **0.86%** | 23.33% | - |

**The real difference is only 0.86 percentage points!**

---

## Evidence Verification

### Manual Calculation Evidence

**From audit output:**

```
--- Manual Calculation (First 30 positions) ---
Pos      Input      Target     Pred       Correct?
--------------------------------------------------
0,0      258        272        3762       ✗
0,1      272        113        489        ✗
...
0,10     0          109        ✗
0,11     0          0          ✓          ← Padding predicted correctly!
0,12     0          0          ✓          ← Padding predicted correctly!
...

Manual accuracy (first 30 positions): 8/30 = 26.67%
Full accuracy: 121/450 = 26.89% ✓ MATCHES

Real token accuracy (excl. padding): 19/348 = 5.46%
Padding accuracy: 102/102 = 100.00%
```

### Ternary Verification

**Re-calculated in same audit:**

```
Ternary 50k Held-Out Accuracy:
  Total accuracy (incl. padding): 3.56%
  Padding accuracy: 0.00%
  Real token accuracy (excl. padding): 4.60%
```

---

## Why FP32 Predicts Padding

### Hypothesis

FP32 model learned:
1. End of sequence often has token 0 (padding)
2. After certain patterns → predict padding
3. This is a form of memorization/overfitting to sequence structure

Ternary model:
1. Did not learn this pattern
2. Acts as regularization - cannot memorize sequence structure
3. But also cannot learn language patterns

---

## Impact on Our Investigation

### What This Means

**BEFORE this discovery:**
- Thought FP32 >> Ternary on held-out (26.89% vs 3.56%)
- Concluded ternary quantization limits learning
- Would have investigated quantization further

**AFTER this discovery:**
- FP32 ≈ Ternary on real tokens (5.46% vs 4.60%)
- Ternary quantization is RULED OUT
- Both models fail to learn language patterns
- Problem is something that affects BOTH architectures

---

## Files and Evidence

### Primary Evidence Files

| File | Content |
|------|---------|
| `experiments/audit_heldout_evaluation.py` | Audit script that discovered inflation |
| `experiments/results/heldout_evaluation_results.json` | Raw evaluation data |
| `experiments/heldout_evaluation.py` | Original evaluation script |

### Related Documents

| Document | Relationship |
|----------|--------------|
| `FP32_VS_TERNARY_COMPARISON.md` | SUPERSEDED - contains incorrect conclusion |
| `FINAL_HYPOTHESIS_RANKING.md` | Updated hypothesis status based on this finding |
| `ROOT_CAUSE_INVESTIGATION_SUMMARY.md` | Consolidated findings including this |

---

## Key Lessons Learned

### 1. Always Exclude Padding from Accuracy

**Wrong:**
```python
accuracy = (preds == targets).sum() / targets.numel()
```

**Correct:**
```python
mask = targets != padding_token_id
accuracy = (preds[mask] == targets[mask]).sum() / mask.sum()
```

### 2. Audit Surprising Results

26.89% vs 3.56% seemed too large a gap. The audit revealed the truth.

### 3. Break Down Metrics by Token Type

Different token types (padding, rare words, common words) may have different accuracy patterns.

---

## Conclusion

**Classification: VERIFIED**

**The padding inflation discovery is VERIFIED with:**
- Manual calculations matching reported values
- Breakdown by token type (padding vs real)
- Same audit methodology applied to both FP32 and Ternary
- Evidence saved in `heldout_evaluation_results.json`

**Impact:** This finding RULED OUT ternary quantization as the root cause and redirected investigation toward factors that affect both FP32 and Ternary equally.

---

## References

1. **Audit script:** `experiments/audit_heldout_evaluation.py`
2. **Evaluation data:** `experiments/results/heldout_evaluation_results.json`
3. **Superseded document:** `FP32_VS_TERNARY_COMPARISON.md` (marked as SUPERSEDED)
4. **Forward pass verification:** `FORWARD_PASS_VERIFICATION_WITH_EVIDENCE.md`
5. **Training data overlap verification:** See audit script line 50-85
