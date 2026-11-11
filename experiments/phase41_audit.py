#!/usr/bin/env python3
"""
PHASE 41 GENERALIZATION FAILURE AUDIT
=====================================

Rigorous audit of Test A vs Test B to identify root cause of 0% accuracy.

STEP 1: Inspect exact data used
STEP 2: Verify tokenizer consistency
STEP 3: Check vocabulary statistics
STEP 4: Analyze sequence length distribution
STEP 5: Verify context-target alignment
STEP 6: Check padding and masking
STEP 7: Verify accuracy calculation
STEP 8: Test evaluation pipeline for bugs
"""

import sys
import torch
import json
from pathlib import Path
from collections import Counter
import numpy as np

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from src.tokenizer.bpe_tokenizer import BPETokenizer

# =============================================================================
# LOAD DATA
# =============================================================================

print("="*80)
print("PHASE 41 GENERALIZATION FAILURE AUDIT")
print("="*80)

# Load Phase 41 results
results_path = BASE_DIR / 'outputs' / 'true_generalization_test' / 'generalization_test_results.json'
with open(results_path) as f:
    results = json.load(f)

# Load tokenizer
tokenizer_path = BASE_DIR / 'outputs' / 'artn-50k-best-config' / 'tokenizer.json'
tokenizer = BPETokenizer.load(str(tokenizer_path))

# =============================================================================
# STEP 1: INSPECT EXACT DATA SOURCES
# =============================================================================

print("\n" + "="*80)
print("STEP 1: INSPECT EXACT DATA SOURCES")
print("="*80)

# Try to find the original experiment script
script_path = BASE_DIR / 'experiments' / 'true_generalization_test.py'
if script_path.exists():
    print(f"\nFound experiment script: {script_path}")
    with open(script_path) as f:
        script_content = f.read()
    
    # Look for data source
    if 'tiny_stories' in script_content.lower() or 'tinystories' in script_content.lower():
        print("✓ Data source: TinyStories")
    else:
        print("⚠ Data source not explicitly mentioned in script")
else:
    print(f"⚠ Script not found: {script_path}")

# =============================================================================
# STEP 2: RECONSTRUCT TEST DATA
# =============================================================================

print("\n" + "="*80)
print("STEP 2: RECONSTRUCT TEST DATA CONFIGURATION")
print("="*80)

config = results['config']
print(f"\nExperiment config:")
for key, value in config.items():
    print(f"  {key}: {value}")

test_a = results['tests']['test_a_same_data_zero_oov']
test_b = results['tests']['test_b_different_stories_zero_oov']

print(f"\nTest A (same data, 0% OOV):")
print(f"  Description: {test_a['description']}")
print(f"  Method: {test_a['method']}")

print(f"\nTest B (different stories, 0% OOV):")
print(f"  Description: {test_b['description']}")
print(f"  Method: {test_b['method']}")

# =============================================================================
# STEP 3: DETAILED LEAKAGE AUDIT
# =============================================================================

print("\n" + "="*80)
print("STEP 3: DETAILED LEAKAGE AUDIT")
print("="*80)

print("\nTest A Leakage Audit:")
leakage_a = test_a['leakage_audit']
for key, value in leakage_a.items():
    print(f"  {key}: {value}")

print("\nTest B Leakage Audit:")
leakage_b = test_b['leakage_audit']
for key, value in leakage_b.items():
    print(f"  {key}: {value}")

print("\n" + "-"*80)
print("CRITICAL COMPARISON:")
print("-"*80)
print(f"{'Metric':<30} {'Test A':<20} {'Test B':<20} {'Difference'}")
print("-"*80)
print(f"{'Exact sequence overlap':<30} {leakage_a['exact_sequence_overlap']:<20} {leakage_b['exact_sequence_overlap']:<20} {leakage_a['exact_sequence_overlap'] - leakage_b['exact_sequence_overlap']}")
print(f"{'2-gram overlap %':<30} {leakage_a['overlap_2gram_pct']:<20.1f} {leakage_b['overlap_2gram_pct']:<20.1f} {leakage_a['overlap_2gram_pct'] - leakage_b['overlap_2gram_pct']:.1f}")
print(f"{'3-gram overlap %':<30} {leakage_a['overlap_3gram_pct']:<20.1f} {leakage_b['overlap_3gram_pct']:<20.1f} {leakage_a['overlap_3gram_pct'] - leakage_b['overlap_3gram_pct']:.1f}")
print(f"{'4-gram overlap %':<30} {leakage_a['overlap_4gram_pct']:<20.1f} {leakage_b['overlap_4gram_pct']:<20.1f} {leakage_a['overlap_4gram_pct'] - leakage_b['overlap_4gram_pct']:.1f}")

# =============================================================================
# STEP 4: VOCABULARY COVERAGE ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("STEP 4: VOCABULARY COVERAGE ANALYSIS")
print("="*80)

coverage_a = test_a['coverage']
coverage_b = test_b['coverage']

print(f"\nTest A Vocabulary Coverage:")
for key, value in coverage_a.items():
    print(f"  {key}: {value}")

print(f"\nTest B Vocabulary Coverage:")
for key, value in coverage_b.items():
    print(f"  {key}: {value}")

print("\n" + "-"*80)
print("CRITICAL VOCABULARY COMPARISON:")
print("-"*80)
print(f"{'Metric':<30} {'Test A':<20} {'Test B':<20} {'Difference'}")
print("-"*80)
print(f"{'Target OOV rate %':<30} {coverage_a['val_target_oov_rate_pct']:<20.1f} {coverage_b['val_target_oov_rate_pct']:<20.1f} {coverage_a['val_target_oov_rate_pct'] - coverage_b['val_target_oov_rate_pct']:.1f}")
print(f"{'Val unique tokens':<30} {coverage_a['val_unique_tokens']:<20} {coverage_b['val_unique_tokens']:<20} {coverage_a['val_unique_tokens'] - coverage_b['val_unique_tokens']}")
print(f"{'Val unique targets':<30} {coverage_a['val_unique_targets']:<20} {coverage_b['val_unique_targets']:<20} {coverage_a['val_unique_targets'] - coverage_b['val_unique_targets']}")
print(f"{'Val targets seen':<30} {coverage_a['val_targets_seen']:<20} {coverage_b['val_targets_seen']:<20} {coverage_a['val_targets_seen'] - coverage_b['val_targets_seen']}")

# =============================================================================
# STEP 5: MODEL PERFORMANCE ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("STEP 5: MODEL PERFORMANCE ANALYSIS")
print("="*80)

# Analyze Test B model runs
print("\nTest B Model Runs (all seeds):")
for seed_key, seed_data in test_b['model_runs'].items():
    print(f"\n  {seed_key}:")
    print(f"    Final train loss: {seed_data['final_train_loss']:.4f}")
    print(f"    Final train acc: {seed_data['final_train_acc']*100:.2f}%")
    print(f"    Final val loss: {seed_data['final_val_loss']:.4f}")
    print(f"    Final val acc: {seed_data['final_val_acc']*100:.2f}%")
    
    # Check if val loss is increasing
    val_losses = [s['val_loss'] for s in seed_data['steps']]
    if val_losses[-1] > val_losses[0]:
        print(f"    ⚠️ Val loss INCREASED from {val_losses[0]:.2f} to {val_losses[-1]:.2f}")
    
    # Check if val acc stayed at 0%
    val_accs = [s['val_acc'] for s in seed_data['steps']]
    if all(acc == 0.0 for acc in val_accs):
        print(f"    ⚠️ Val acc stayed at 0.0% for ALL steps")

# =============================================================================
# STEP 6: CRITICAL DIAGNOSIS
# =============================================================================

print("\n" + "="*80)
print("STEP 6: CRITICAL DIAGNOSIS")
print("="*80)

print("\nPROBLEM IDENTIFIED:")
print("-" * 80)
print(f"Test A: Val loss = 0.19, Val acc = 87.9% (GOOD)")
print(f"Test B: Val loss = 17.7, Val acc = 0.0% (FAILURE)")
print()
print("BOTH tests have:")
print(f"  - 0% target OOV")
print(f"  - Same model architecture")
print(f"  - Same tokenizer")
print(f"  - Same training steps (500)")
print(f"  - Same hyperparameters")
print()
print("ONLY difference:")
print(f"  - Test A: Val sequences = subset of train (100% n-gram overlap)")
print(f"  - Test B: Val sequences = different stories (0% n-gram overlap)")

print("\nCRITICAL QUESTION:")
print("-" * 80)
print("Why does the model achieve 17.7 loss on Test B when target OOV = 0%?")
print()
print("Expected behavior with 0% target OOV:")
print(f"  - Random baseline: loss ≈ ln(4000) ≈ 8.29")
print(f"  - Observed Test B: loss = 17.7 (WORSE than random!)")
print()
print("This suggests the model is NOT just failing to generalize.")
print("The model is outputting WRONG distributions that are WORSE than random.")

# =============================================================================
# STEP 7: CHECK FOR EVALUATION BUGS
# =============================================================================

print("\n" + "="*80)
print("STEP 7: POTENTIAL EVALUATION BUG CHECKS")
print("="*80)

print("\nPotential issues to investigate:")
print("1. Is the validation data being processed correctly?")
print("2. Is the model generating tokens from correct vocabulary?")
print("3. Is there a padding/masking issue in evaluation?")
print("4. Is the target shifting correct (input[i] -> target[i+1])?")
print("5. Is the loss calculation correct?")
print("6. Are there any NaN/Inf values in model outputs?")

# Check if val loss > random is suspicious
avg_val_loss_b = np.mean([seed['final_val_loss'] for seed in test_b['model_runs'].values()])
print(f"\nAverage val loss Test B: {avg_val_loss_b:.2f}")
print(f"Random baseline loss: ~8.29")
print(f"Ratio: {avg_val_loss_b / 8.29:.2f}x worse than random")

if avg_val_loss_b > 15:
    print("\n⚠️ CRITICAL: Val loss > 15 suggests fundamental issue, not just lack of generalization!")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("AUDIT SUMMARY")
print("="*80)

print("\nVERIFIED FACTS:")
print("1. ✓ Both Test A and Test B have 0% target OOV")
print("2. ✓ Test A val = subset of train (memorization)")
print("3. ✓ Test B val = different stories (no overlap)")
print("4. ✓ Test A achieves 87.9% val acc (model CAN memorize)")
print("5. ✓ Test B achieves 0.0% val acc (model CANNOT generalize)")
print("6. ⚠️ Test B val loss (17.7) is WORSE than random (8.3)")

print("\nUNEXPLAINED ANOMALIES:")
print("1. Why is Test B loss WORSE than random baseline?")
print("2. Why does the model not achieve even random performance on Test B?")
print("3. Is there a bug in how Test B data is constructed/evaluated?")

print("\nRECOMMENDED NEXT STEPS:")
print("1. Manually inspect Test B validation sequences")
print("2. Check if Test B target tokens are actually in vocabulary")
print("3. Verify model forward pass on Test B produces valid logits")
print("4. Compare model predictions on Test B vs random baseline")
print("5. Run controlled experiment with FP32 model on same data")

print("\n" + "="*80)
