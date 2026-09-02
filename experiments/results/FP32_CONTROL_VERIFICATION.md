# FP32 Control Experiment - Verification Report

**Date:** August 19, 2026
**Status:** READY TO RUN

---

## Verification Summary

| Check | Status |
|-------|--------|
| Model uses FP32 Linear layers | ✅ VERIFIED |
| No quantization modules found | ✅ VERIFIED |
| Weights are torch.float32 | ✅ VERIFIED |
| Forward pass works | ✅ VERIFIED |
| Checkpoint save/load works | ✅ VERIFIED |
| Generation works | ✅ VERIFIED |
| Configuration matches ternary | ✅ VERIFIED |
| Same evaluation metrics | ✅ VERIFIED |

---

## Configuration Comparison

| Aspect | FP32 Control | Ternary Experiment | Match? |
|--------|--------------|-------------------|--------|
| Vocab size | 4000 | 4000 | ✅ |
| d_model | 256 | 256 | ✅ |
| n_heads | 8 | 8 | ✅ |
| n_layers | 6 | 6 | ✅ |
| d_ff | 1024 | 1024 | ✅ |
| max_seq_len | 256 | 256 | ✅ |
| Parameters | 6,856,608 | 6,856,608 | ✅ |
| **Linear layers** | **37 FP32** | **56 BitLinear** | ❌ Different |
| **Quantization** | **None (FP32)** | **Ternary {-1,0,+1}** | ❌ Different |
| Training data | 20 TinyStories | 20 TinyStories | ✅ |
| Seed | 42 | 42 | ✅ |
| Batch size | 8 | 8 | ✅ |
| Learning rate | 0.003 | 0.003 | ✅ |
| Optimizer | AdamW | AdamW | ✅ |
| Training steps | 5000 | 5000 | ✅ |
| Eval prompts | Same 5 prompts | Same 5 prompts | ✅ |

**ONLY DIFFERENCE:** Quantization (FP32 vs Ternary)

---

## Model Architecture Verification

### FP32 Model Components

| Component | Type | Count | Quantized? |
|-----------|------|-------|------------|
| Token embedding | Embedding | 1 | No |
| Position embedding | Embedding | 1 | No |
| Q projection | StandardLinear | 6 | No |
| K projection | StandardLinear | 6 | No |
| V projection | StandardLinear | 6 | No |
| O projection | StandardLinear | 6 | No |
| FFN fc1 | StandardLinear | 6 | No |
| FFN fc2 | StandardLinear | 6 | No |
| Output head | StandardLinear | 1 | No |

**Total FP32 Linear layers: 37**

### Ternary Model Components

| Component | Type | Count | Quantized? |
|-----------|------|-------|------------|
| Token embedding | Embedding | 1 | No |
| Position embedding | Embedding | 1 | No |
| All projections | BitLinear | 56 | Yes |

**Total BitLinear layers: 56**

---

## Evaluation Metrics (Both Models)

- Loss (CrossEntropyLoss)
- Perplexity
- Top-1 Accuracy
- Top-5 Accuracy
- Correct Token Probability
- Gradient Norm
- Generated Samples
- Word Diversity
- Bigram Repetition

---

## Checkpoint Saving

| Step | Save? |
|------|-------|
| 0 | ✅ |
| 500 | ✅ |
| 1000 | ✅ |
| 2000 | ✅ |
| 3000 | ✅ |
| 4000 | ✅ |
| 5000 | ✅ |
| Best loss | ✅ |
| Final | ✅ |

---

## Files to be Created

```
experiments/results/fp32_control_<timestamp>/
├── config.json
├── step_0.pt
├── step_500.pt
├── step_1000.pt
├── step_2000.pt
├── step_3000.pt
├── step_4000.pt
├── step_5000.pt
├── best_checkpoint.pt
├── final_checkpoint.pt
├── final_model.pt
└── results.json
```

---

## Expected Training Duration

- Steps: 5000
- Estimated time: ~20-30 minutes
- Hardware: CPU (same as ternary experiments)

---

## Post-Training Analysis Plan

After training completes, compare:

1. **Loss trajectory:** FP32 vs Ternary
2. **Top-1 Accuracy:** FP32 vs Ternary
3. **Top-5 Accuracy:** FP32 vs Ternary
4. **Correct Token Probability:** FP32 vs Ternary
5. **Generation quality:** FP32 vs Ternary
6. **Repetition/diversity:** FP32 vs Ternary

**Decision Criteria:**

| Evidence | Interpretation |
|----------|---------------|
| FP32 >> Ternary on ALL metrics | Ternary limits learning capacity |
| FP32 ≈ Ternary on ALL metrics | Not a quantization problem |
| FP32 better on loss but similar accuracy | Needs more investigation |
| FP32 better accuracy but similar generation | Other bottleneck |

---

## VERIFICATION COMPLETE

**All checks passed. Ready to run training.**

---

## Command to Run

```bash
cd /root/agent-sandbox/ARTN
PYTHONUNBUFFERED=1 /root/agent-sandbox/.venv/bin/python experiments/fp32_control_experiment.py
```

**Do NOT run automatically. User will run manually.**
