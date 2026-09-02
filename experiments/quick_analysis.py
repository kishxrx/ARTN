#!/usr/bin/env python3
"""
Analyze results WITHOUT re-running evaluation.
Just analyzes training trajectory and classifies pattern.
"""

print("="*80)
print("ANALYSIS WITHOUT RE-EVALUATION")
print("="*80)

# Results from your run
train_loss = 7.14  # Approximate from output
heldout_loss = train_loss + 0.0139
loss_gap = 0.0139
acc_gap = 0.01

print(f"\nMetrics from your run:")
print(f"  Loss gap: {loss_gap:.4f}")
print(f"  Accuracy gap: {acc_gap:.2f}%")

# Training trajectory from logs
trajectory = [
    (500, 8.5819),
    (1000, 8.4483),
    (5000, 7.9383),
    (10000, 7.6746),
    (20000, 7.2160),
    (30000, 7.2291),
    (40000, 7.0645),
    (50000, 7.1267),
]

print(f"\n{'='*80}")
print("LOSS TRAJECTORY ANALYSIS")
print("="*80)

print(f"\n{'Steps':<15} {'Loss':>10} {'Δ Loss':>10}")
print(f"{'-'*35}")

for i, (step, loss) in enumerate(trajectory):
    if i == 0:
        print(f"{step:<15,} {loss:>10.4f} {'-':>10}")
    else:
        prev_step, prev_loss = trajectory[i-1]
        delta = loss - prev_loss
        print(f"{step:<15,} {loss:>10.4f} {delta:>10.4f}")

# Calculate rates
early_rate = (trajectory[0][1] - trajectory[3][1]) / (5000 - 500) * 1000
late_rate = (trajectory[4][1] - trajectory[7][1]) / (50000 - 20000) * 1000

print(f"\nLoss reduction rates:")
print(f"  Early (500-5k): {early_rate:.4f}/k steps")
print(f"  Late (20k-50k): {late_rate:.4f}/k steps")
print(f"  Ratio: {abs(early_rate/late_rate):.1f}x slower")

# Pattern classification
print(f"\n{'='*80}")
print("PATTERN CLASSIFICATION")
print("="*80)

if abs(loss_gap) < 0.5 and train_loss > 7.0:
    pattern = "C: Both similar and high (underfitting/weak signal)"
elif late_rate < 0.01:
    pattern = "E: Both plateaued at high loss"
else:
    pattern = "D: Both still improving"

print(f"\nPattern: {pattern}")
print(f"\nEvidence:")
print(f"  - Training and held-out loss nearly identical (gap = {loss_gap:.4f})")
print(f"  - Training and held-out accuracy nearly identical (gap = {acc_gap:.2f}%)")
print(f"  - Both losses are HIGH (~7.1)")
print(f"  - Model plateaued after step 20,000")

# Hypothesis ranking
print(f"\n{'='*80}")
print("HYPOTHESIS RANKING")
print("="*80)

print(f"\n{'Hypothesis':<35} {'Status':<25} {'Evidence'}")
print(f"{'-'*100}")

print(f"{'1. Insufficient data':<35} {'POSSIBLE':<25} {'1.61 tok/param (marginal)'}")
print(f"{'2. Insufficient training':<35} {'RULED OUT':<25} {'Plateaued at 20k (not learning more)'}")
print(f"{'3. Optimization/training dynamics':<35} {'STRONGLY SUPPORTED':<25} {'Early learned, late stopped'}")
print(f"{'4. Ternary quantization':<35} {'POSSIBLE':<25} {'Need FP32 comparison'}")
print(f"{'5. Model capacity':<35} {'POSSIBLE':<25} {'Both train/held-out high loss'}")
print(f"{'6. Tokenizer/data issue':<35} {'RULED OUT':<25} {'Vocab 99.2%, pipeline verified'}")
print(f"{'7. Other implementation':<35} {'UNLIKELY':<25} {'Forward pass verified'}")

# Conclusion
print(f"\n{'='*80}")
print("CONCLUSION")
print("="*80)

print(f"""
KEY FINDINGS:

1. TRAINING ≈ HELD-OUT (not overfitting)
   - Loss gap: {loss_gap:.4f}
   - Accuracy gap: {acc_gap:.2f}%
   - Model is NOT memorizing training data

2. BOTH ARE HIGH LOSS (~7.1)
   - Random baseline loss: ~8.3
   - Model improved only ~1.2 from random
   - Still very high loss

3. MODEL PLATEAUED AT STEP 20,000
   - Early rate: {early_rate:.4f}/k steps
   - Late rate: {late_rate:.4f}/k steps
   - {abs(early_rate/late_rate):.1f}x slower in late phase

DIAGNOSIS:

The model is UNDERFITTING - it cannot properly fit even the training data.

This is NOT an overfitting problem. The model is NOT memorizing.

POSSIBLE CAUSES:

1. Optimization dynamics issue (learning stalled)
2. Ternary quantization limiting learning capacity
3. Model capacity too small for the data complexity
4. Marginal data-to-parameter ratio (1.61)

HIGHEST VALUE NEXT STEP:

Compare FP32 control model on same metrics to isolate ternary vs optimization issue.
""")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print("="*80)
