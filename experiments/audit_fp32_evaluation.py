#!/usr/bin/env python3
"""
FP32 Evaluation Pipeline Audit

Investigates why step_0 FP32 model shows 92.7% accuracy (impossible for untrained model).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.fp32_control_experiment import StandardGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*70)
print("FP32 EVALUATION PIPELINE AUDIT")
print("="*70)

# Load tokenizer
tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
print(f"\nTokenizer vocab_size: {tokenizer.vocab_size}")

# ============================================================
# 1. VERIFY STEP_0 CHECKPOINT IS FRESHLY INITIALIZED
# ============================================================

print(f"\n{'='*70}")
print("1. STEP_0 CHECKPOINT VERIFICATION")
print(f"{'='*70}")

# Load step_0 checkpoint
fp32_dir = Path("experiments/results/fp32_control_20260819_232646")
step0_path = fp32_dir / "step_0.pt"

print(f"\nLoading: {step0_path}")
step0_ckpt = torch.load(step0_path, map_location='cpu')

print(f"\nCheckpoint contents:")
print(f"  Step: {step0_ckpt['step']}")
print(f"  Loss: {step0_ckpt.get('loss', 'N/A')}")

# Create fresh model with SAME SEED
print(f"\nCreating fresh FP32 model with seed=42...")
torch.manual_seed(42)
fresh_model = StandardGPT(4000, 256, 8, 6, 1024, 256)

# Load step_0 weights into another model
step0_model = StandardGPT(4000, 256, 8, 6, 1024, 256)
step0_model.load_state_dict(step0_ckpt['model_state_dict'])

# Compare parameters
print(f"\nComparing fresh model vs step_0 checkpoint:")
param_diffs = []
for (name1, p1), (name2, p2) in zip(fresh_model.named_parameters(), step0_model.named_parameters()):
    diff = (p1 - p2).abs().max().item()
    param_diffs.append((name1, diff))

max_diff = max(d for _, d in param_diffs)
print(f"\n  Max parameter difference: {max_diff:.10f}")

if max_diff < 1e-10:
    print(f"  ✓ VERIFIED: step_0 checkpoint is IDENTICAL to fresh initialization")
    print(f"  (Parameters match within 1e-10 tolerance)")
else:
    print(f"  ✗ ANOMALY: step_0 checkpoint differs from fresh initialization!")
    print(f"\n  Top 5 differing parameters:")
    sorted_diffs = sorted(param_diffs, key=lambda x: x[1], reverse=True)[:5]
    for name, diff in sorted_diffs:
        print(f"    {name}: {diff:.10f}")

# ============================================================
# 2. VERIFY TARGET ALIGNMENT
# ============================================================

print(f"\n{'='*70}")
print("2. TARGET ALIGNMENT VERIFICATION")
print(f"{'='*70}")

# The evaluation sentences used in training
eval_sentences = [
    "Once upon a time there was a little girl named Sarah.",
    "The little girl loved to play in the garden every day.",
    "One day she found a magic flower in the garden.",
    "The flower glowed with beautiful colors.",
    "Sarah picked the flower and made a wish.",
    "Suddenly a fairy appeared before her eyes.",
    "The fairy granted Sarah three special wishes.",
    "Sarah wished for happiness and peace.",
    "The fairy smiled and disappeared into the light.",
    "From that day on Sarah was always happy.",
]

# Tokenize
sequences = []
for sentence in eval_sentences:
    tokens = tokenizer.encode(sentence)
    sequences.append(tokens)

max_len = max(len(s) for s in sequences)
print(f"\nMax sequence length: {max_len}")

# Pad sequences
padded_sequences = []
for seq in sequences:
    padded = seq + [0] * (max_len - len(seq))
    padded_sequences.append(padded)

eval_input_ids = torch.tensor(padded_sequences)
print(f"Evaluation input shape: {eval_input_ids.shape}")

# How were targets constructed?
print(f"\n--- How targets were constructed ---")
print(f"Original code from fp32_control_experiment.py:")
print(f"  targets = torch.cat([eval_input_ids[:, 1:], torch.zeros(eval_input_ids.size(0), 1, dtype=torch.long)], dim=1)")

# Construct targets EXACTLY as in the script
targets = torch.cat([eval_input_ids[:, 1:], torch.zeros(eval_input_ids.size(0), 1, dtype=torch.long)], dim=1)
print(f"\nTargets shape: {targets.shape}")

# Manual verification: print input/target pairs
print(f"\n--- Input/Target Token Pairs (First 3 sentences, first 8 positions) ---")

for sent_idx in range(3):
    print(f"\nSentence {sent_idx}: \"{eval_sentences[sent_idx][:40]}...\"")
    print(f"{'Pos':<5} {'Input Token':<20} {'Target Token':<20} {'Correct?'}")
    print(f"{'-'*60}")
    
    for pos in range(min(8, max_len-1)):
        input_token = eval_input_ids[sent_idx, pos].item()
        target_token = targets[sent_idx, pos].item()
        
        input_word = tokenizer.decode([input_token])
        target_word = tokenizer.decode([target_token])
        
        # Verify: target[i] should be input[i+1]
        expected_target = eval_input_ids[sent_idx, pos+1].item() if pos+1 < max_len else 0
        
        is_correct = "✓" if target_token == expected_target else "✗"
        
        print(f"{pos:<5} {input_token:<5} '{input_word}'{'':<12} {target_token:<5} '{target_word}'{'':<12} {is_correct}")

# ============================================================
# 3. CRITICAL CHECK: IS target[i] = input[i] ???
# ============================================================

print(f"\n{'='*70}")
print("3. CRITICAL: TARGET POSITION CHECK")
print(f"{'='*70}")

print(f"\nChecking if targets are ALIGNED with inputs (bug):")
print(f"  If target[i] = input[i], model just needs to copy input to get 100% accuracy!")
print(f"  If target[i] = input[i+1], this is correct next-token prediction.")

# Check alignment
same_count = 0
shifted_count = 0
total_positions = 0

for i in range(eval_input_ids.size(0)):
    for j in range(eval_input_ids.size(1) - 1):
        input_token = eval_input_ids[i, j].item()
        target_token = targets[i, j].item()
        next_input_token = eval_input_ids[i, j+1].item() if j+1 < eval_input_ids.size(1) else -1
        
        if target_token == input_token:
            same_count += 1
        if target_token == next_input_token:
            shifted_count += 1
        
        total_positions += 1

print(f"\n  Positions where target[i] = input[i]: {same_count} / {total_positions} ({100*same_count/total_positions:.1f}%)")
print(f"  Positions where target[i] = input[i+1]: {shifted_count} / {total_positions} ({100*shifted_count/total_positions:.1f}%)")

if same_count > shifted_count:
    print(f"\n  ✗ CRITICAL BUG DETECTED: Targets are aligned with INPUTS, not shifted!")
    print(f"  This means the model can achieve 100% accuracy by copying input tokens!")
elif shifted_count > same_count:
    print(f"\n  ✓ Targets are correctly shifted (next-token prediction)")
else:
    print(f"\n  ? Mixed alignment - need manual verification")

# ============================================================
# 4. VERIFY PADDING HANDLING
# ============================================================

print(f"\n{'='*70}")
print("4. PADDING/IGNORE_INDEX HANDLING")
print(f"{'='*70}")

print(f"\nPadding token ID: 0")
print(f"\nSequence lengths (before padding):")
for i, seq in enumerate(sequences):
    print(f"  Sentence {i}: {len(seq)} tokens")

print(f"\nChecking if padding positions are included in accuracy calculation:")
print(f"  Loss function: CrossEntropyLoss(reduction='mean')")
print(f"  No ignore_index specified - ALL positions counted including padding!")

padding_count = 0
total_tokens = 0
for i in range(eval_input_ids.size(0)):
    seq_len = len(sequences[i])
    padding_in_seq = max_len - seq_len
    padding_count += padding_in_seq
    total_tokens += max_len

print(f"\n  Total tokens: {total_tokens}")
print(f"  Padding tokens: {padding_count} ({100*padding_count/total_tokens:.1f}%)")
print(f"  Real tokens: {total_tokens - padding_count} ({100*(total_tokens-padding_count)/total_tokens:.1f}%)")

if padding_count > 0:
    print(f"\n  ⚠ Padding tokens are included in accuracy calculation!")
    print(f"  If model predicts token 0 (padding) for all positions, it gets {100*padding_count/total_tokens:.1f}% accuracy on padding!")

# ============================================================
# 5. MANUAL ACCURACY CALCULATION FROM RAW LOGITS
# ============================================================

print(f"\n{'='*70}")
print("5. MANUAL ACCURACY CALCULATION")
print(f"{'='*70}")

step0_model.eval()

with torch.no_grad():
    # Get logits
    output = step0_model(eval_input_ids)
    logits = output['logits']
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits dtype: {logits.dtype}")
    print(f"Logits mean: {logits.mean().item():.6f}")
    print(f"Logits std: {logits.std().item():.6f}")
    
    # Get predictions
    preds = logits.argmax(dim=-1)
    print(f"\nPredictions shape: {preds.shape}")
    
    # Manual calculation
    correct = 0
    total = 0
    
    print(f"\n--- First 20 positions (detailed) ---")
    print(f"{'Pos':<8} {'Input':<10} {'Target':<10} {'Pred':<10} {'Correct?'}")
    print(f"{'-'*50}")
    
    detailed_correct = 0
    detailed_total = 0
    
    for i in range(min(2, eval_input_ids.size(0))):  # First 2 sentences
        for j in range(min(10, eval_input_ids.size(1))):  # First 10 positions
            input_token = eval_input_ids[i, j].item()
            target_token = targets[i, j].item()
            pred_token = preds[i, j].item()
            
            is_correct = "✓" if pred_token == target_token else "✗"
            
            if pred_token == target_token:
                detailed_correct += 1
            detailed_total += 1
            
            print(f"{i},{j:<5} {input_token:<10} {target_token:<10} {pred_token:<10} {is_correct}")
    
    # Full calculation
    for i in range(eval_input_ids.size(0)):
        for j in range(eval_input_ids.size(1)):
            if preds[i, j].item() == targets[i, j].item():
                correct += 1
            total += 1
    
    manual_accuracy = 100 * correct / total
    detailed_accuracy = 100 * detailed_correct / detailed_total
    
    print(f"\n--- Accuracy Calculation ---")
    print(f"  Manual calculation (all positions): {correct}/{total} = {manual_accuracy:.2f}%")
    print(f"  Detailed (first 20 positions): {detailed_correct}/{detailed_total} = {detailed_accuracy:.2f}%")
    print(f"  Script reported: 92.73%")

# ============================================================
# 6. PREDICTION ANALYSIS: WHAT IS MODEL PREDICTING?
# ============================================================

print(f"\n{'='*70}")
print("6. PREDICTION ANALYSIS")
print(f"{'='*70}")

print(f"\nWhat tokens is the model predicting?")

pred_counts = {}
for i in range(preds.size(0)):
    for j in range(preds.size(1)):
        p = preds[i, j].item()
        pred_counts[p] = pred_counts.get(p, 0) + 1

sorted_preds = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 10 most predicted tokens:")
for token_id, count in sorted_preds[:10]:
    token_word = tokenizer.decode([token_id])
    print(f"  Token {token_id} ('{token_word}'): {count} times ({100*count/preds.numel():.1f}%)")

# Check if predicting mostly padding (token 0)
padding_pred_count = pred_counts.get(0, 0)
print(f"\n  Predictions of token 0 (padding): {padding_pred_count} / {preds.numel()} ({100*padding_pred_count/preds.numel():.1f}%)")

# ============================================================
# 7. TEST ON RANDOM/UNSEEN INPUT
# ============================================================

print(f"\n{'='*70}")
print("7. TEST ON RANDOM INPUT")
print(f"{'='*70}")

torch.manual_seed(999)  # Different seed
random_input = torch.randint(0, 4000, (3, 10))
print(f"\nRandom input shape: {random_input.shape}")
print(f"Random input tokens: {random_input[0, :5].tolist()}")

with torch.no_grad():
    random_output = step0_model(random_input)
    random_logits = random_output['logits']
    random_preds = random_logits.argmax(dim=-1)

print(f"\nRandom input predictions: {random_preds[0, :5].tolist()}")

# Create random targets (also random)
random_targets = torch.randint(0, 4000, (3, 10))

random_correct = (random_preds == random_targets).sum().item()
random_total = random_targets.numel()
random_accuracy = 100 * random_correct / random_total

print(f"\nAccuracy on random input/target: {random_correct}/{random_total} = {random_accuracy:.2f}%")
print(f"Expected for random model: {100/4000:.4f}%")

if random_accuracy < 1.0:
    print(f"\n  ✓ Model behaves randomly on unseen input (expected behavior)")
else:
    print(f"\n  ✗ Model shows high accuracy on random input (ANOMALY)")

# ============================================================
# 8. COMPARE FP32 AND TERNARY EVALUATION ON SAME FUNCTION
# ============================================================

print(f"\n{'='*70}")
print("8. FP32 VS TERNARY ON SAME EVALUATION")
print(f"{'='*70}")

from src.model.bitlinear import BitLinearGPT

# Load ternary step_0 model
ternary_step0_path = Path("experiments/results/checkpoints/step_0.pt")
print(f"\nLoading ternary step_0: {ternary_step0_path}")

ternary_ckpt = torch.load(ternary_step0_path, map_location='cpu')
ternary_model = BitLinearGPT(4000, 256, 8, 6, 1024, 256)
ternary_model.load_state_dict(ternary_ckpt['model_state_dict'])
ternary_model.eval()

# Evaluate both on SAME data
print(f"\nEvaluating both models on same evaluation set...")

with torch.no_grad():
    # FP32
    fp32_output = step0_model(eval_input_ids)
    fp32_logits = fp32_output['logits']
    fp32_preds = fp32_logits.argmax(dim=-1)
    fp32_correct = (fp32_preds == targets).sum().item()
    fp32_acc = 100 * fp32_correct / targets.numel()
    
    # Ternary
    ternary_output = ternary_model(eval_input_ids)
    ternary_logits = ternary_output['logits']
    ternary_preds = ternary_logits.argmax(dim=-1)
    ternary_correct = (ternary_preds == targets).sum().item()
    ternary_acc = 100 * ternary_correct / targets.numel()

print(f"\n  FP32 step_0 accuracy: {fp32_acc:.2f}%")
print(f"  Ternary step_0 accuracy: {ternary_acc:.2f}%")
print(f"\n  Difference: {fp32_acc - ternary_acc:.2f}%")

if abs(fp32_acc - ternary_acc) < 5.0:
    print(f"\n  Both models have similar accuracy (expected for untrained models)")
else:
    print(f"\n  ✗ ANOMALY: Large difference between untrained FP32 and Ternary!")

# ============================================================
# SUMMARY
# ============================================================

print(f"\n{'='*70}")
print("AUDIT SUMMARY")
print(f"{'='*70}")

print(f"\nFindings:")
print(f"\n1. Step_0 checkpoint: {'✓ Verified freshly initialized' if max_diff < 1e-10 else '✗ Contains trained weights'}")
print(f"2. Target alignment: {same_count} positions where target=input, {shifted_count} where target=input+1")
print(f"3. Padding handling: {padding_count} padding tokens included in accuracy")
print(f"4. Manual accuracy: {manual_accuracy:.2f}%")
print(f"5. Random input accuracy: {random_accuracy:.2f}%")
print(f"6. FP32 vs Ternary: {fp32_acc:.2f}% vs {ternary_acc:.2f}%")

print(f"\n{'='*70}")
print("INVESTIGATION COMPLETE")
print(f"{'='*70}")
