# Root Cause Analysis: LR Experiment Loss Discrepancy

**Date:** August 19, 2026
**Status:** ROOT CAUSE IDENTIFIED

---

## Executive Summary

**ROOT CAUSE: Loss calculation is SUMMING losses over all tokens instead of averaging.**

- Expected loss: ~8.7 (with CrossEntropyLoss reduction='mean')
- Actual loss: ~4368 (loss * batch_size * seq_len / accumulation_factor)
- Ratio: ~500x higher than normal

---

## Evidence

### One-Batch Diagnostic Results

| Metric | Synthetic Data | Real Data |
|--------|----------------|-----------|
| **Loss** | **8.73** | **8.64** |
| Gradient Norm | 0.147 | 0.376 |
| Params with Gradients | 96/102 | 96/102 |

**Key finding:** Both data types produce normal loss ~8.7!

### LR Experiment Results

| Step | Loss | Gradient Norm |
|------|------|---------------|
| 500 | 4368.79 | 0.000000 |
| 1000 | 4365.80 | 0.000000 |
| 1500 | 4362.85 | 0.000000 |
| ... | ... | ... |
| 5000 | 4343.53 | 0.000000 |

**Key finding:**
1. Loss is ~4368 (500x higher than normal)
2. Gradients are all 0.0 (but one-batch diagnostic shows healthy gradients)

---

## Root Cause Analysis

### Loss Scale Factor

```
Experiment loss: 4368.79
Normal loss: 8.73
Ratio: 500.4x
```

```
Batch size: 4
Seq len: 256
Tokens per batch: 1024
Accumulation steps: 2
```

**Hypothesis:** Loss accumulation logic is wrong.

---

## Loss Accumulation Logic (lr_experiment.py)

### Lines 360-391:

```python
# Forward pass
output = model(input_ids)
logits = output['logits']

loss = nn.CrossEntropyLoss()(
    logits.view(-1, MODEL_CONFIG["vocab_size"]),
    target_ids.view(-1)
)

# Scale loss for accumulation
loss = loss / TRAINING_CONFIG["accumulation_steps"]
accumulation_loss += loss.item()

# Backward pass
loss.backward()

# Log metrics
if step % 500 == 0:
    actual_loss = accumulation_loss * TRAINING_CONFIG["accumulation_steps"]
```

### The Bug:

1. **CrossEntropyLoss default: reduction='mean'**
   - Returns average loss per token
   - Normal value: ~8.7

2. **Line 366: loss = loss / accumulation_steps**
   - Scales loss for gradient accumulation
   - Correct for backward pass
   - But `.item()` extracts the scaled value

3. **Line 367: accumulation_loss += loss.item()**
   - Accumulates scaled loss values
   - Over 500 steps with accumulation_steps=2:
     - 250 optimizer updates
     - Each update: 2 batches accumulated
     - accumulation_loss = sum of 500 scaled losses

4. **Line 391: actual_loss = accumulation_loss * accumulation_steps**
   - Tries to un-scale
   - But accumulation_loss already contains 500 values!
   - Result: actual_loss = sum of 500 * loss * acc_steps / acc_steps = sum of 500 losses

---

## Why Gradients = 0.0?

### Gradient norm is computed AFTER optimizer.step()

Lines 397-5:
```python
# Optimizer step
if (step + 1) % TRAINING_CONFIG["accumulation_steps"] == 0:
    optimizer.step()
    optimizer.zero_grad()  # ← Gradients cleared!

# Compute gradient norm
grad_norm = get_gradient_norm(model)  # ← Returns 0.0!
```

**The gradient is cleared BEFORE being logged!**

---

## Comparison: Verified Training Pipeline

### tinystories_memorize_observability.py Lines 518-530:

```python
# Forward
outputs = model(batch)
logits = outputs['logits']

# Compute loss manually
targets = batch[:, 1:].clone()
targets[targets == 0] = -100  # mask padding

criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
loss = criterion(logits[:, :-1, :].reshape(-1, logits.size(-1)), targets.reshape(-1))

# Backward
loss.backward()

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# Optimizer step
optimizer.step()
```

**Key differences:**
1. Loss is NOT divided by accumulation_steps
2. Gradient norm is computed BEFORE optimizer.step()
3. Uses proper padding mask (ignore_index=-100)

---

## Why One-Batch Diagnostic Shows Normal Loss?

In one_batch_diagnostic.py:
```python
loss1 = nn.CrossEntropyLoss()(
    logits.reshape(-1, VOCAB_SIZE),
    synthetic_target.reshape(-1)
)
```

**No accumulation logic!** Direct loss calculation → normal value ~8.7

---

## Scale Factor Calculation

```
Normal loss: 8.73

In lr_experiment:
- 500 steps logged
- Each accumulated over 2 batches
- accumulation_loss = 500 * (loss / 2)
- actual_loss = accumulation_loss * 2 = 500 * loss

If loss ~ 8.73, then:
  actual_loss = 500 * 8.73 = 4365

Experiment shows: 4368.79

Match: YES!
```

---

## Conclusion

**ROOT CAUSE IDENTIFIED:**

1. **Loss accumulation logic incorrectly sums losses over 500 steps**
   - Line 367: accumulates scaled loss values
   - Line 391: tries to un-scale but multiplies sum of 500 values
   - Result: logged loss = 500 * actual_loss

2. **Gradient norm computed after zero_grad()**
   - Line 385: optimizer.zero_grad()
   - Line 397: get_gradient_norm(model)
   - Result: gradient norm = 0.0 always

3. **One-batch diagnostic shows correct results**
   - No accumulation logic
   - Direct loss calculation
   - Result: loss ~8.7, gradients healthy

---

## Fix

1. **Fix loss logging:**
   ```python
   # Track unscaled loss for logging
   if step % 500 == 0:
       # Compute average loss over last 500 steps
       actual_loss = accumulation_loss / 500
       accumulation_loss = 0.0
   ```

2. **Fix gradient logging:**
   ```python
   # Compute gradient norm BEFORE optimizer.step()
   grad_norm = get_gradient_norm(model)
   
   optimizer.step()
   optimizer.zero_grad()
   ```

---

## Classification

| Question | Answer | Evidence |
|----------|--------|----------|
| Is loss calculation correct? | **MODEL CALCULATION: YES** | One-batch shows ~8.7 |
| Is loss logging correct? | **NO** | Sums 500 values → 500x scale |
| Is gradient norm logging correct? | **NO** | Computed after zero_grad() |
| Are gradients actually zero? | **NO** | One-batch shows healthy gradients |
| Is learning happening? | **YES** | Loss decreases from 4368→4343 |

---

## Final Answer

**ROOT CAUSE:**

1. **Loss logging bug:** Line 391 computes `accumulation_loss * accumulation_steps`, but `accumulation_loss` contains sum of 500 scaled losses. Result: 500x inflated loss.

2. **Gradient logging bug:** Line 397 computes gradient norm after Line 385 clears gradients. Result: always 0.0.

**The model is learning correctly**, but the metrics are logged incorrectly.
