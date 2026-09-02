#!/usr/bin/env python3
"""
HIGHEST-VALUE DIAGNOSTIC: Training vs Held-Out Analysis

Rigorous comparison using existing 50k checkpoint.
Padding EXCLUDED from all metrics.
"""

import sys
import json
import torch
import torch.nn.functional as F
import random
from pathlib import Path
from collections import Counter
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*80)
print("TRAINING vs HELD-OUT ANALYSIS - 50k CHECKPOINT")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# LOAD MODEL AND TOKENIZER
# ============================================================

print(f"\n{'='*80}")
print("LOADING MODEL AND TOKENIZER")
print("="*80)

tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
print(f"Tokenizer vocab: {tokenizer.vocab_size}")
print(f"PAD token: {tokenizer.pad_token_id}")
print(f"EOS token: {tokenizer.eos_token_id}")

# Load 50k checkpoint
checkpoint_path = "outputs/artn-optimized-backup-50k/final_model/model.pt"
print(f"\nLoading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)

# Create model
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

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# ============================================================
# PREPARE DATA
# ============================================================

print(f"\n{'='*80}")
print("PREPARING DATA")
print("="*80)

# Load TinyStories
dataset = load_dataset("roneneldan/TinyStories", split="train")

# Training data: first 50,000 stories
train_dataset = dataset.select(range(50000))

# Held-out data: next 1,000 stories (after training data)
heldout_dataset = dataset.select(range(50000, 51000))

print(f"Training stories: {len(train_dataset):,}")
print(f"Held-out stories: {len(heldout_dataset):,}")

# ============================================================
# EVALUATION FUNCTION (EXCLUDES PADDING)
# ============================================================

def evaluate_dataset(model, tokenizer, dataset, name, max_samples=1000, max_seq_len=256):
    """
    Evaluate model on dataset.
    Returns metrics on REAL tokens only (padding excluded).
    """
    
    model.eval()
    
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_real_tokens = 0
    all_correct_probs = []
    
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            text = dataset[i]['text']
            
            # Tokenize
            tokens = tokenizer.encode(text)[:max_seq_len - 1]
            tokens = tokens + [tokenizer.eos_token_id]
            
            if len(tokens) < 2:
                continue
            
            # Create input and target
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            
            # Forward pass
            output = model(input_ids)
            logits = output["logits"]  # Model returns dict
            
            # Calculate loss and accuracy on REAL tokens only
            # logits: [1, seq_len, vocab]
            # We predict token[i+1] from logits[i]
            
            for pos in range(len(tokens) - 1):
                target_token = tokens[pos + 1]
                
                # Skip padding tokens in target
                if target_token == tokenizer.pad_token_id:
                    continue
                
                # Get logits for this position
                pos_logits = logits[0, pos, :]  # [vocab_size]
                
                # Calculate loss
                loss = F.cross_entropy(pos_logits.unsqueeze(0), 
                                       torch.tensor([target_token], device=device))
                total_loss += loss.item()
                
                # Calculate accuracy
                probs = F.softmax(pos_logits, dim=-1)
                
                # Top-1
                top1_pred = probs.argmax().item()
                if top1_pred == target_token:
                    total_correct_top1 += 1
                
                # Top-5
                top5_preds = probs.topk(5).indices.tolist()
                if target_token in top5_preds:
                    total_correct_top5 += 1
                
                # Correct token probability
                correct_prob = probs[target_token].item()
                all_correct_probs.append(correct_prob)
                
                total_real_tokens += 1
            
            if (i + 1) % 100 == 0:
                print(f"  {name}: Processed {i+1}/{min(max_samples, len(dataset))}")
    
    # Calculate metrics
    avg_loss = total_loss / total_real_tokens if total_real_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    top1_acc = 100 * total_correct_top1 / total_real_tokens if total_real_tokens > 0 else 0
    top5_acc = 100 * total_correct_top5 / total_real_tokens if total_real_tokens > 0 else 0
    avg_correct_prob = sum(all_correct_probs) / len(all_correct_probs) if all_correct_probs else 0
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "correct_prob": avg_correct_prob,
        "total_real_tokens": total_real_tokens,
    }

# ============================================================
# EVALUATE TRAINING DATA
# ============================================================

print(f"\n{'='*80}")
print("EVALUATING TRAINING DATA (REAL TOKENS ONLY)")
print("="*80)

train_metrics = evaluate_dataset(model, tokenizer, train_dataset, "Training", max_samples=1000)

print(f"\nTRAINING METRICS:")
print(f"  Loss: {train_metrics['loss']:.4f}")
print(f"  Perplexity: {train_metrics['perplexity']:.2f}")
print(f"  Top-1 Accuracy: {train_metrics['top1_accuracy']:.2f}%")
print(f"  Top-5 Accuracy: {train_metrics['top5_accuracy']:.2f}%")
print(f"  Correct Token Prob: {train_metrics['correct_prob']:.4f}")
print(f"  Real Tokens Evaluated: {train_metrics['total_real_tokens']:,}")

# ============================================================
# EVALUATE HELD-OUT DATA
# ============================================================

print(f"\n{'='*80}")
print("EVALUATING HELD-OUT DATA (REAL TOKENS ONLY)")
print("="*80)

heldout_metrics = evaluate_dataset(model, tokenizer, heldout_dataset, "Held-Out", max_samples=1000)

print(f"\nHELD-OUT METRICS:")
print(f"  Loss: {heldout_metrics['loss']:.4f}")
print(f"  Perplexity: {heldout_metrics['perplexity']:.2f}")
print(f"  Top-1 Accuracy: {heldout_metrics['top1_accuracy']:.2f}%")
print(f"  Top-5 Accuracy: {heldout_metrics['top5_accuracy']:.2f}%")
print(f"  Correct Token Prob: {heldout_metrics['correct_prob']:.4f}")
print(f"  Real Tokens Evaluated: {heldout_metrics['total_real_tokens']:,}")

# ============================================================
# COMPARISON
# ============================================================

print(f"\n{'='*80}")
print("TRAINING vs HELD-OUT COMPARISON")
print("="*80)

print(f"\n{'Metric':<25} {'Training':>15} {'Held-Out':>15} {'Gap':>15}")
print(f"{'-'*70}")
print(f"{'Loss':<25} {train_metrics['loss']:>15.4f} {heldout_metrics['loss']:>15.4f} {heldout_metrics['loss']-train_metrics['loss']:>15.4f}")
print(f"{'Perplexity':<25} {train_metrics['perplexity']:>15.2f} {heldout_metrics['perplexity']:>15.2f} {heldout_metrics['perplexity']-train_metrics['perplexity']:>15.2f}")
print(f"{'Top-1 Accuracy':<25} {train_metrics['top1_accuracy']:>15.2f}% {heldout_metrics['top1_accuracy']:>15.2f}% {heldout_metrics['top1_accuracy']-train_metrics['top1_accuracy']:>15.2f}%")
print(f"{'Top-5 Accuracy':<25} {train_metrics['top5_accuracy']:>15.2f}% {heldout_metrics['top5_accuracy']:>15.2f}% {heldout_metrics['top5_accuracy']-train_metrics['top5_accuracy']:>15.2f}%")
print(f"{'Correct Token Prob':<25} {train_metrics['correct_prob']:>15.4f} {heldout_metrics['correct_prob']:>15.4f} {heldout_metrics['correct_prob']-train_metrics['correct_prob']:>15.4f}")

# ============================================================
# LOSS TRAJECTORY ANALYSIS
# ============================================================

print(f"\n{'='*80}")
print("LOSS TRAJECTORY ANALYSIS")
print("="*80)

# From training logs
trajectory = [
    (500, 8.5819),
    (1000, 8.4483),
    (1500, 8.3178),
    (2000, 8.2750),
    (2500, 8.1989),
    (3000, 8.0949),
    (3500, 8.1613),
    (4000, 8.0193),
    (4500, 7.9459),
    (5000, 7.9383),
    (10000, 7.6746),
    (15000, 7.3747),
    (20000, 7.2160),
    (25000, 7.2822),
    (30000, 7.2291),
    (35000, 7.0503),
    (40000, 7.0645),
    (45000, 7.1178),
    (50000, 7.1267),
]

print(f"\nLoss by interval:")
print(f"{'Steps':<15} {'Loss':>10} {'Δ Loss':>10} {'Rate':>15}")
print(f"{'-'*55}")

for i, (step, loss) in enumerate(trajectory):
    if i == 0:
        print(f"{step:<15,} {loss:>10.4f} {'-':>10} {'-':>15}")
    else:
        prev_step, prev_loss = trajectory[i-1]
        delta = loss - prev_loss
        rate = delta / (step - prev_step) * 1000
        print(f"{step:<15,} {loss:>10.4f} {delta:>10.4f} {rate:>15.4f}/k steps")

# Calculate slope at different phases
print(f"\nLoss reduction rate by phase:")

phases = [
    ("0-5k", trajectory[0], trajectory[9]),
    ("5k-10k", trajectory[9], trajectory[10]),
    ("10k-20k", trajectory[10], trajectory[12]),
    ("20k-30k", trajectory[12], trajectory[14]),
    ("30k-40k", trajectory[14], trajectory[16]),
    ("40k-50k", trajectory[16], trajectory[18]),
]

for phase_name, (start_step, start_loss), (end_step, end_loss) in phases:
    reduction = start_loss - end_loss
    rate = reduction / (end_step - start_step) * 1000
    print(f"  {phase_name}: {reduction:.4f} ({rate:.4f}/k steps)")

# Determine if plateaued
early_rate = (trajectory[0][1] - trajectory[9][1]) / (5000) * 1000
late_rate = (trajectory[12][1] - trajectory[18][1]) / (30000) * 1000

print(f"\nEarly rate (0-5k): {early_rate:.4f}/k steps")
print(f"Late rate (20k-50k): {late_rate:.4f}/k steps")
print(f"Ratio: {early_rate/late_rate:.1f}x slower")

# ============================================================
# GENERATION TEST
# ============================================================

print(f"\n{'='*80}")
print("GENERATION TEST")
print("="*80)

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, device='cpu'):
    """Generate text from prompt."""
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(input_ids)
            logits = output["logits"]  # Model returns dict
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            
            if next_token == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated)

prompts = [
    "Once upon a time",
    "There was a little",
    "The cat sat on",
    "A boy named",
]

print(f"\nGenerating from 50k checkpoint:")
for prompt in prompts:
    generated = generate(model, tokenizer, prompt, max_new_tokens=30, temperature=0.8, device=device)
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Generated: \"{generated}\"")

# ============================================================
# PATTERN ANALYSIS
# ============================================================

print(f"\n{'='*80}")
print("PATTERN ANALYSIS")
print("="*80)

loss_gap = heldout_metrics['loss'] - train_metrics['loss']
acc_gap = train_metrics['top1_accuracy'] - heldout_metrics['top1_accuracy']

print(f"\nLoss gap (held-out - training): {loss_gap:.4f}")
print(f"Accuracy gap (training - held-out): {acc_gap:.2f}%")

print(f"\nPattern Classification:")

if train_metrics['loss'] > heldout_metrics['loss'] + 1.0:
    pattern = "A: Train >> Held-out (underfitting/optimization problem)"
elif train_metrics['loss'] < heldout_metrics['loss'] - 1.0:
    if acc_gap > 10:
        pattern = "B: Train << Held-out (overfitting)"
    else:
        pattern = "B-MILD: Train < Held-out (mild overfitting)"
elif abs(loss_gap) < 0.5 and train_metrics['loss'] > 7.0:
    pattern = "C: Both similar and high (underfitting/weak signal)"
elif late_rate < 0.01:
    pattern = "E: Both plateaued at high loss"
else:
    pattern = "D: Both still improving"

print(f"\n{pattern}")

# ============================================================
# HYPOTHESIS RANKING
# ============================================================

print(f"\n{'='*80}")
print("HYPOTHESIS RANKING (Based on Evidence)")
print("="*80)

hypotheses = []

# 1. Insufficient data
# Evidence: 1.61 tokens/param, loss high, both train/held-out similar
if train_metrics['top1_accuracy'] < 10:
    hypotheses.append(("Insufficient data", "STRONGLY SUPPORTED", 
                      f"Top-1: {train_metrics['top1_accuracy']:.1f}%, Loss: {train_metrics['loss']:.2f}, 1.61 tok/param"))
else:
    hypotheses.append(("Insufficient data", "POSSIBLE", 
                      f"Marginal tokens/param (1.61)"))

# 2. Insufficient training
# Evidence: Late rate nearly zero
if late_rate < 0.01:
    hypotheses.append(("Insufficient training", "RULED OUT", 
                      f"Plateaued (rate: {late_rate:.4f}/k steps)"))
else:
    hypotheses.append(("Insufficient training", "POSSIBLE", 
                      "Could try more steps"))

# 3. Optimization/training dynamics
# Evidence: Loss plateaued, early learning worked
if early_rate > 0.1 and late_rate < 0.01:
    hypotheses.append(("Optimization dynamics", "STRONGLY SUPPORTED", 
                      f"Early: {early_rate:.2f}/k, Late: {late_rate:.2f}/k"))
else:
    hypotheses.append(("Optimization dynamics", "POSSIBLE", 
                      "Learning slowed"))

# 4. Ternary quantization
# Evidence: Need FP32 comparison
hypotheses.append(("Ternary quantization", "POSSIBLE", 
                  "Need FP32 control comparison"))

# 5. Model capacity
# Evidence: Both train and held-out similar and high
if abs(loss_gap) < 0.5 and train_metrics['loss'] > 7.0:
    hypotheses.append(("Model capacity", "POSSIBLE", 
                      "Both train/held-out high, may need more capacity"))
else:
    hypotheses.append(("Model capacity", "UNLIKELY", 
                      "Learning occurred"))

# 6. Tokenizer/data issue
# Evidence: Vocab 99.2% used, no duplicates, contiguity verified
hypotheses.append(("Tokenizer/data issue", "RULED OUT", 
                  "Vocab 99.2%, no dupes, pipeline verified"))

# 7. Other implementation
# Evidence: Forward pass verified, gradients exist
hypotheses.append(("Other implementation", "UNLIKELY", 
                  "Forward pass verified, loss decreasing"))

print(f"\n{'Hypothesis':<30} {'Status':<20} {'Evidence'}")
print(f"{'-'*100}")

for hyp, status, evidence in hypotheses:
    print(f"{hyp:<30} {status:<20} {evidence}")

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "training_metrics": train_metrics,
    "heldout_metrics": heldout_metrics,
    "comparison": {
        "loss_gap": loss_gap,
        "accuracy_gap": acc_gap,
    },
    "trajectory": {
        "early_rate": early_rate,
        "late_rate": late_rate,
        "ratio": early_rate/late_rate if late_rate > 0 else float('inf'),
    },
    "pattern": pattern,
    "hypotheses": [(h, s, e) for h, s, e in hypotheses],
}

output_path = Path("experiments/results/training_vs_heldout_analysis.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*80}")
print(f"RESULTS SAVED: {output_path}")
print("="*80)
