#!/usr/bin/env python3
"""
CRITICAL SANITY CHECK: Loss Calculation Verification

Resolves discrepancy between:
- Training loss (from logs): 7.13
- Evaluation loss (from our script): 10.23

Then compares FP32 5k vs Ternary 50k using IDENTICAL methodology.
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*80)
print("CRITICAL SANITY CHECK: Loss Calculation Verification")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# BASELINE CALCULATION
# ============================================================

print(f"\n{'='*80}")
print("BASELINE CALCULATION")
print("="*80)

vocab_size = 4000
uniform_loss = torch.log(torch.tensor(vocab_size, dtype=torch.float))
print(f"\nUniform random baseline:")
print(f"  Vocab size: {vocab_size}")
print(f"  Loss = ln({vocab_size}) = {uniform_loss.item():.4f}")
print(f"  Perplexity = e^loss = {torch.exp(uniform_loss).item():.1f}")

# ============================================================
# LOAD TOKENIZER
# ============================================================

print(f"\n{'='*80}")
print("LOADING TOKENIZER")
print("="*80)

tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
print(f"Vocab: {tokenizer.vocab_size}")
print(f"PAD token: {tokenizer.pad_token_id}")
print(f"EOS token: {tokenizer.eos_token_id}")

# ============================================================
# SANITY CHECK 1: Manual Loss Calculation
# ============================================================

print(f"\n{'='*80}")
print("SANITY CHECK 1: Manual Loss Calculation")
print("="*80)

# Test on small example
test_tokens = [1, 5, 10, 20, 30, 40, 50, 100, 200, 300]
print(f"\nTest tokens: {test_tokens}")
print(f"Number of predictions: {len(test_tokens) - 1}")

# Uniform logits
uniform_logits = torch.zeros(vocab_size)
print(f"\nUniform logits shape: {uniform_logits.shape}")
print(f"Uniform logits sum: {uniform_logits.sum().item()}")

# Calculate loss for uniform predictions
manual_loss = 0.0
for i in range(len(test_tokens) - 1):
    target = test_tokens[i + 1]
    # Loss = -log(1/vocab_size) = log(vocab_size)
    manual_loss += uniform_loss.item()

avg_manual_loss = manual_loss / (len(test_tokens) - 1)
print(f"\nManual uniform loss: {avg_manual_loss:.4f}")
print(f"Should equal ln({vocab_size}) = {uniform_loss.item():.4f}")
print(f"Match: {abs(avg_manual_loss - uniform_loss.item()) < 0.001}")

# ============================================================
# SANITY CHECK 2: Training Loss Path vs Eval Loss Path
# ============================================================

print(f"\n{'='*80}")
print("SANITY CHECK 2: Training Loss vs Eval Loss Path")
print("="*80)

# Load 50k ternary checkpoint
checkpoint_path = "outputs/artn-optimized-backup-50k/final_model/model.pt"
print(f"\nLoading Ternary 50k checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)

model_config = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
}

model = BitLinearGPT(**model_config).to(device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
print(f"✓ Model loaded")

# Create a test batch (similar to training)
test_text = "Once upon a time there was a little girl named Sarah. She loved to play in the garden every day."
test_tokens = tokenizer.encode(test_text)
print(f"\nTest text: \"{test_text}\"")
print(f"Tokens: {len(test_tokens)}")

# Pad to 256
max_seq_len = 256
padded_tokens = test_tokens + [tokenizer.pad_token_id] * (max_seq_len - len(test_tokens))
input_ids = torch.tensor([padded_tokens], dtype=torch.long, device=device)

print(f"\nPadded length: {len(padded_tokens)}")
print(f"Real tokens: {len(test_tokens)}")
print(f"Padding tokens: {len(padded_tokens) - len(test_tokens)}")

# Method 1: Training path (using targets with ignore_index=-100)
print(f"\n--- Method 1: Training Path ---")
targets_train = input_ids.clone()
# Shift targets: target[i] = input[i+1]
targets_train = torch.cat([input_ids[:, 1:], torch.full((1, 1), tokenizer.pad_token_id, device=device)], dim=1)
# Mask padding
targets_train[targets_train == tokenizer.pad_token_id] = -100

output = model(input_ids, targets_train)
loss_train = output["loss"].item()

print(f"Loss (training path): {loss_train:.4f}")

# Count non-ignored tokens
non_ignored = (targets_train != -100).sum().item()
print(f"Non-ignored tokens: {non_ignored}")

# Method 2: Evaluation path (manual per-token calculation)
print(f"\n--- Method 2: Eval Path (Real Tokens Only) ---")
output = model(input_ids)
logits = output["logits"]

loss_eval = 0.0
count_eval = 0
for pos in range(len(test_tokens) - 1):
    target = test_tokens[pos + 1]
    pos_logits = logits[0, pos, :]
    loss = F.cross_entropy(pos_logits.unsqueeze(0), torch.tensor([target], device=device))
    loss_eval += loss.item()
    count_eval += 1

avg_loss_eval = loss_eval / count_eval if count_eval > 0 else 0

print(f"Loss (eval path): {avg_loss_eval:.4f}")
print(f"Tokens evaluated: {count_eval}")

# Compare
print(f"\n--- Comparison ---")
print(f"Training path loss: {loss_train:.4f}")
print(f"Evaluation path loss: {avg_loss_eval:.4f}")
print(f"Difference: {abs(loss_train - avg_loss_eval):.4f}")

if abs(loss_train - avg_loss_eval) < 0.1:
    print(f"✓ MATCH - Both methods give same loss")
else:
    print(f"✗ MISMATCH - Need to investigate!")

# ============================================================
# SANITY CHECK 3: Why is eval loss 10.23 vs training log 7.13?
# ============================================================

print(f"\n{'='*80}")
print("SANITY CHECK 3: Training Log Loss vs Our Eval Loss")
print("="*80)

print(f"\nTraining log (from monitor_log.txt):")
print(f"  Step 500: 8.58")
print(f"  Step 20000: 7.22")
print(f"  Step 50000: 7.13")

print(f"\nOur evaluation on 50k checkpoint:")
print(f"  Training data: 10.23")
print(f"  Held-out data: 10.24")

print(f"\nDifference: 10.23 - 7.13 = 3.10")

print(f"\nPossible causes:")
print(f"  1. Training loss calculated differently")
print(f"  2. Different token set (training vs eval)")
print(f"  3. Different batch composition")
print(f"  4. Training loss averaged over steps, not tokens")
print(f"  5. Checkpoint mismatch (wrong checkpoint loaded)")

# Check if we loaded the right checkpoint
print(f"\nCheckpoint verification:")
print(f"  Path: {checkpoint_path}")
if 'step' in checkpoint:
    print(f"  Step: {checkpoint['step']}")
if 'loss' in checkpoint:
    print(f"  Loss: {checkpoint['loss']:.4f}")

# ============================================================
# SANITY CHECK 4: Evaluate on SAME data as training
# ============================================================

print(f"\n{'='*80}")
print("SANITY CHECK 4: Evaluate on Training Data Sample")
print("="*80)

# Load same training data
dataset = load_dataset("roneneldan/TinyStories", split="train")
train_dataset = dataset.select(range(50000))

# Evaluate on small sample using EXACT training methodology
print(f"\nEvaluating on 100 training stories using training methodology...")

total_loss = 0.0
total_tokens = 0

for i in range(100):
    text = train_dataset[i]['text']
    tokens = tokenizer.encode(text)[:max_seq_len - 1]
    tokens = tokens + [tokenizer.eos_token_id]
    
    # Pad
    padding_len = max_seq_len - len(tokens)
    padded = tokens + [tokenizer.pad_token_id] * padding_len
    
    input_ids = torch.tensor([padded], dtype=torch.long, device=device)
    targets = input_ids.clone()
    targets = torch.cat([input_ids[:, 1:], torch.full((1, 1), tokenizer.pad_token_id, device=device)], dim=1)
    targets[targets == tokenizer.pad_token_id] = -100
    
    with torch.no_grad():
        output = model(input_ids, targets)
        loss = output["loss"].item()
        
        # Count non-ignored tokens
        non_ignored = (targets != -100).sum().item()
        if non_ignored > 0:
            total_loss += loss * non_ignored
            total_tokens += non_ignored
    
    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/100")

avg_loss_training_method = total_loss / total_tokens if total_tokens > 0 else 0
perplexity_training = torch.exp(torch.tensor(avg_loss_training_method)).item()

print(f"\nResults (training methodology):")
print(f"  Loss: {avg_loss_training_method:.4f}")
print(f"  Perplexity: {perplexity_training:.1f}")
print(f"  Real tokens: {total_tokens:,}")

# ============================================================
# SANITY CHECK 5: Evaluate on SAME data using eval methodology
# ============================================================

print(f"\n{'='*80}")
print("SANITY CHECK 5: Evaluate on Same Data (Eval Methodology)")
print("="*80)

print(f"\nEvaluating on same 100 stories using eval methodology...")

total_loss_eval = 0.0
total_tokens_eval = 0
correct_top1 = 0
correct_top5 = 0

for i in range(100):
    text = train_dataset[i]['text']
    tokens = tokenizer.encode(text)[:max_seq_len - 1]
    tokens = tokens + [tokenizer.eos_token_id]
    
    if len(tokens) < 2:
        continue
    
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output = model(input_ids)
        logits = output["logits"]
        
        for pos in range(len(tokens) - 1):
            target = tokens[pos + 1]
            
            if target == tokenizer.pad_token_id:
                continue
            
            pos_logits = logits[0, pos, :]
            loss = F.cross_entropy(pos_logits.unsqueeze(0), torch.tensor([target], device=device))
            total_loss_eval += loss.item()
            total_tokens_eval += 1
            
            probs = F.softmax(pos_logits, dim=-1)
            top1 = probs.argmax().item()
            top5 = probs.topk(5).indices.tolist()
            
            if top1 == target:
                correct_top1 += 1
            if target in top5:
                correct_top5 += 1
    
    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/100")

avg_loss_eval_method = total_loss_eval / total_tokens_eval if total_tokens_eval > 0 else 0
perplexity_eval = torch.exp(torch.tensor(avg_loss_eval_method)).item()
top1_acc = 100 * correct_top1 / total_tokens_eval if total_tokens_eval > 0 else 0
top5_acc = 100 * correct_top5 / total_tokens_eval if total_tokens_eval > 0 else 0

print(f"\nResults (eval methodology):")
print(f"  Loss: {avg_loss_eval_method:.4f}")
print(f"  Perplexity: {perplexity_eval:.1f}")
print(f"  Top-1: {top1_acc:.2f}%")
print(f"  Top-5: {top5_acc:.2f}%")
print(f"  Real tokens: {total_tokens_eval:,}")

# ============================================================
# COMPARISON
# ============================================================

print(f"\n{'='*80}")
print("METHODOLOGY COMPARISON")
print("="*80)

print(f"\n{'Method':<30} {'Loss':>10} {'Perplexity':>12} {'Top-1':>10}")
print(f"{'-'*65}")
print(f"{'Training log (step 50k)':<30} {'7.13':>10} {'~1240':>12} {'?':>10}")
print(f"{'Training methodology':<30} {avg_loss_training_method:>10.4f} {perplexity_training:>12.1f} {'?':>10}")
print(f"{'Eval methodology':<30} {avg_loss_eval_method:>10.4f} {perplexity_eval:>12.1f} {top1_acc:>10.2f}%")
print(f"{'Uniform baseline':<30} {uniform_loss.item():>10.4f} {torch.exp(uniform_loss).item():>12.1f} {'0.025%':>10}")

# ============================================================
# DIAGNOSIS
# ============================================================

print(f"\n{'='*80}")
print("DIAGNOSIS")
print("="*80)

loss_diff_methods = abs(avg_loss_training_method - avg_loss_eval_method)
loss_diff_training_log = abs(avg_loss_training_method - 7.13)

print(f"\nLoss difference between methods: {loss_diff_methods:.4f}")
print(f"Loss difference vs training log: {loss_diff_training_log:.4f}")

if loss_diff_methods < 0.5:
    print(f"\n✓ Training and eval methodologies give SIMILAR loss")
    print(f"  → Discrepancy is NOT in evaluation methodology")
    
    if loss_diff_training_log > 2.0:
        print(f"\n⚠ But both differ significantly from training log (7.13)")
        print(f"  → Checkpoint may be different from step 50k")
        print(f"  → OR training loss calculation differs")
else:
    print(f"\n✗ Training and eval methodologies give DIFFERENT loss")
    print(f"  → Issue in how loss is calculated")

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "uniform_baseline": {
        "loss": uniform_loss.item(),
        "perplexity": torch.exp(uniform_loss).item(),
    },
    "training_methodology": {
        "loss": avg_loss_training_method,
        "perplexity": perplexity_training,
        "real_tokens": total_tokens,
    },
    "eval_methodology": {
        "loss": avg_loss_eval_method,
        "perplexity": perplexity_eval,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "real_tokens": total_tokens_eval,
    },
    "training_log": {
        "loss_at_step_50000": 7.13,
    },
    "differences": {
        "method_gap": loss_diff_methods,
        "training_log_gap": loss_diff_training_log,
    }
}

with open("experiments/results/sanity_check_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*80}")
print("RESULTS SAVED: experiments/results/sanity_check_results.json")
print("="*80)
