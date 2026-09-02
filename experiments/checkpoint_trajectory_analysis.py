#!/usr/bin/env python3
"""
CHECKPOINT TRAJECTORY ANALYSIS - READ ONLY

Analyze ALL checkpoints from 1k to 50k to determine:
- Did the model EVER learn contextual information?
- When did learning plateau?
- When did context learning appear/disappear?

CRITICAL: This is READ-ONLY. No training, no checkpoint modification.
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from datasets import load_dataset
import random
import math

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Add project to path for imports
import sys
sys.path.insert(0, "/root/agent-sandbox/ARTN")

print("="*100)
print("CHECKPOINT TRAJECTORY ANALYSIS - READ ONLY")
print("="*100)

# Device
device = "cpu"
print(f"Device: {device}")

# ==============================================================================
# PATHS - SAME AS VALIDATED FORENSIC DIAGNOSTIC
# ==============================================================================

checkpoint_dir = Path("outputs/artn-50k-best-config/checkpoints")
tokenizer_path = Path("outputs/artn-optimized/tokenizer.json")
output_dir = Path("experiments/results/checkpoint_trajectory_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# LOAD TOKENIZER - SAME AS VALIDATED DIAGNOSTIC
# ==============================================================================

from src.tokenizer.bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer.load(str(tokenizer_path))
vocab_size = tokenizer.vocab_size
print(f"Tokenizer loaded: vocab={vocab_size}")

# ==============================================================================
# LOAD MODEL CONFIG - SAME AS VALIDATED DIAGNOSTIC
# ==============================================================================

from src.model import BitLinearGPT, create_bitlinear_gpt

# Model parameters - same as validated diagnostic
vocab_size = 4000
d_model = 256
n_layers = 6
n_heads = 8
max_seq_len = 256

print(f"Model config: vocab={vocab_size}, d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")

# ==============================================================================
# LOAD HELD-OUT DATA - SAME AS VALIDATED DIAGNOSTIC
# ==============================================================================

print("\nLoading held-out evaluation data...")

dataset = load_dataset("roneneldan/TinyStories", split="train")

# Use stories 50000+ (not in training)
held_out_stories = []
for i, item in enumerate(dataset):
    if i >= 50000 and i < 50500:  # 500 held-out stories
        held_out_stories.append(item["text"])

print(f"Held-out stories: {len(held_out_stories)}")

# Tokenize held-out data
held_out_tokens = []
for story in held_out_stories:
    tokens = tokenizer.encode(story)[:256]  # BPETokenizer returns list directly
    if len(tokens) >= 10:
        held_out_tokens.append(tokens)

print(f"Held-out sequences for evaluation: {len(held_out_tokens)}")

# ==============================================================================
# BUILD UNIGRAM BASELINE FROM TRAINING DATA
# ==============================================================================

print("\nBuilding unigram baseline from training data...")

train_tokens_list = []
for i, item in enumerate(dataset):
    if i < 50000:  # Training data
        tokens = tokenizer.encode(item["text"])  # BPETokenizer returns list directly
        train_tokens_list.extend(tokens)
    if i >= 10000:  # Sample 10k stories for baseline
        break

token_counts = Counter(train_tokens_list)
total_tokens = sum(token_counts.values())
unigram_probs = {t: c / total_tokens for t, c in token_counts.items()}

print(f"Unigram baseline built from {total_tokens} tokens")

# ==============================================================================
# BIGRAM BASELINE
# ==============================================================================

print("Building bigram baseline...")

bigram_counts = Counter()
unigram_for_bigram = Counter()

for i, item in enumerate(dataset):
    if i < 50000:
        tokens = tokenizer.encode(item["text"])  # BPETokenizer returns list directly
        for j in range(len(tokens) - 1):
            bigram_counts[(tokens[j], tokens[j+1])] += 1
            unigram_for_bigram[tokens[j]] += 1
    if i >= 5000:
        break

# Convert to probabilities
bigram_probs = {}
for (t1, t2), count in bigram_counts.items():
    if unigram_for_bigram[t1] > 0:
        bigram_probs[(t1, t2)] = count / unigram_for_bigram[t1]

print(f"Bigram baseline built from {len(bigram_probs)} pairs")

# ==============================================================================
# EVALUATION FUNCTIONS - SAME AS VALIDATED DIAGNOSTIC
# ==============================================================================

def evaluate_checkpoint(model, tokens_list, max_context=256):
    """Evaluate model on held-out data with READ-ONLY operations."""
    model.eval()
    
    total_loss = 0.0
    total_predictions = 0
    correct_top1 = 0
    correct_top5 = 0
    
    with torch.no_grad():
        for tokens in tokens_list:
            if len(tokens) < 2:
                continue
            
            # Use only specified context
            if max_context < 256:
                tokens = tokens[-max_context:]
            
            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
            targets = torch.tensor([tokens[1:]], dtype=torch.long)
            
            # Forward pass
            output = model(input_ids)
            logits = output["logits"]
            
            # Calculate loss
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            
            # Get predictions
            probs = F.softmax(logits, dim=-1)
            top1_pred = torch.argmax(probs, dim=-1)
            top5_pred = torch.topk(probs, k=5, dim=-1).indices
            
            # Calculate accuracy (exclude padding token 0)
            mask = targets != 0
            correct_top1 += ((top1_pred == targets) & mask).sum().item()
            correct_top5 += ((top5_pred == targets.unsqueeze(-1)).any(dim=-1) & mask).sum().item()
            
            total_loss += loss.item() * targets.size(1)
            total_predictions += mask.sum().item()
    
    avg_loss = total_loss / total_predictions if total_predictions > 0 else float('inf')
    top1_acc = correct_top1 / total_predictions if total_predictions > 0 else 0
    top5_acc = correct_top5 / total_predictions if total_predictions > 0 else 0
    
    return avg_loss, top1_acc, top5_acc


def evaluate_unigram_baseline(tokens_list):
    """Evaluate unigram baseline on same data."""
    total_loss = 0.0
    total_predictions = 0
    correct_top1 = 0
    
    for tokens in tokens_list:
        if len(tokens) < 2:
            continue
        
        targets = tokens[1:]
        
        for target in targets:
            if target == 0:  # Skip padding
                continue
            
            # Unigram predicts most frequent token
            most_freq_token = max(unigram_probs.keys(), key=lambda t: unigram_probs[t])
            
            if most_freq_token == target:
                correct_top1 += 1
            
            # Cross-entropy loss
            prob = unigram_probs.get(target, 1e-10)
            total_loss += -math.log(prob)
            total_predictions += 1
    
    avg_loss = total_loss / total_predictions if total_predictions > 0 else float('inf')
    top1_acc = correct_top1 / total_predictions if total_predictions > 0 else 0
    
    return avg_loss, top1_acc


def evaluate_bigram_baseline(tokens_list):
    """Evaluate bigram baseline on same data."""
    correct_top1 = 0
    total_predictions = 0
    
    for tokens in tokens_list:
        if len(tokens) < 2:
            continue
        
        for i in range(len(tokens) - 1):
            context_token = tokens[i]
            target = tokens[i + 1]
            
            if target == 0:  # Skip padding
                continue
            
            # Find most likely next token given context
            candidates = [(t2, prob) for (t1, t2), prob in bigram_probs.items() if t1 == context_token]
            
            if candidates:
                best_token = max(candidates, key=lambda x: x[1])[0]
            else:
                # Fallback to unigram
                best_token = max(unigram_probs.keys(), key=lambda t: unigram_probs[t])
            
            if best_token == target:
                correct_top1 += 1
            total_predictions += 1
    
    top1_acc = correct_top1 / total_predictions if total_predictions > 0 else 0
    return top1_acc


def evaluate_context_shuffling(model, tokens_list):
    """Evaluate with shuffled context."""
    model.eval()
    
    total_loss = 0.0
    total_predictions = 0
    
    with torch.no_grad():
        for tokens in tokens_list:
            if len(tokens) < 2:
                continue
            
            # Shuffle context (keep target unchanged)
            context = tokens[:-1]
            target = tokens[-1]
            
            # Shuffle context tokens
            shuffled_context = context.copy()
            random.shuffle(shuffled_context)
            
            input_ids = torch.tensor([shuffled_context], dtype=torch.long)
            targets = torch.tensor([[target]], dtype=torch.long)
            
            output = model(input_ids)
            logits = output["logits"][:, -1, :]  # Last position
            
            loss = F.cross_entropy(logits, targets.view(-1))
            
            total_loss += loss.item()
            total_predictions += 1
    
    return total_loss / total_predictions if total_predictions > 0 else float('inf')


# ==============================================================================
# FIND ALL CHECKPOINTS
# ==============================================================================

print("\n" + "="*100)
print("FINDING CHECKPOINTS")
print("="*100)

checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
print(f"Found {len(checkpoints)} checkpoints")

if len(checkpoints) == 0:
    print("ERROR: No checkpoints found!")
    sys.exit(1)

# Extract step numbers
checkpoint_steps = []
for ckpt in checkpoints:
    step = int(ckpt.stem.split("_")[-1])
    checkpoint_steps.append(step)

print(f"Steps: {checkpoint_steps[:5]}...{checkpoint_steps[-5:]}")

# ==============================================================================
# ANALYZE EACH CHECKPOINT
# ==============================================================================

print("\n" + "="*100)
print("ANALYZING CHECKPOINTS (READ-ONLY)")
print("="*100)

results = []

for i, (ckpt_path, step) in enumerate(zip(checkpoints, checkpoint_steps)):
    print(f"\n[{i+1}/{len(checkpoints)}] Analyzing checkpoint step {step}...")
    
    # Load checkpoint (READ-ONLY)
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Create model with same parameters
    model = BitLinearGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 4,
        max_seq_len=max_seq_len,
        dropout=0.0
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Evaluate with full context
    print(f"  Evaluating full context (256 tokens)...")
    full_loss, full_top1, full_top5 = evaluate_checkpoint(model, held_out_tokens, max_context=256)
    
    # Evaluate with limited context
    print(f"  Evaluating limited context...")
    ctx8_loss, ctx8_top1, _ = evaluate_checkpoint(model, held_out_tokens, max_context=8)
    ctx4_loss, ctx4_top1, _ = evaluate_checkpoint(model, held_out_tokens, max_context=4)
    ctx1_loss, ctx1_top1, _ = evaluate_checkpoint(model, held_out_tokens, max_context=1)
    
    # Context benefit
    context_benefit = full_loss - ctx1_loss
    
    # Evaluate with shuffled context
    print(f"  Evaluating shuffled context...")
    shuffled_loss = evaluate_context_shuffling(model, held_out_tokens)
    shuffle_effect = full_loss - shuffled_loss
    
    # Calculate unigram baseline (only once)
    if i == 0:
        print(f"  Computing unigram baseline...")
        unigram_loss, unigram_top1 = evaluate_unigram_baseline(held_out_tokens)
    
    # Calculate bigram baseline (only once)
    if i == 0:
        print(f"  Computing bigram baseline...")
        bigram_top1 = evaluate_bigram_baseline(held_out_tokens)
    
    # Model vs baselines
    model_vs_unigram = full_top1 - unigram_top1
    model_vs_bigram = full_top1 - bigram_top1
    
    # Store results
    result = {
        "step": step,
        "loss": full_loss,
        "perplexity": math.exp(full_loss) if full_loss < 10 else float('inf'),
        "top1_accuracy": full_top1 * 100,
        "top5_accuracy": full_top5 * 100,
        "context_benefit": context_benefit,
        "ctx1_loss": ctx1_loss,
        "ctx4_loss": ctx4_loss,
        "ctx8_loss": ctx8_loss,
        "shuffle_effect": shuffle_effect,
        "shuffled_loss": shuffled_loss,
        "model_vs_unigram": model_vs_unigram * 100,
        "model_vs_bigram": model_vs_bigram * 100,
    }
    
    results.append(result)
    
    print(f"  Loss: {full_loss:.4f}, Top-1: {full_top1*100:.2f}%, Context Benefit: {context_benefit:.4f}")

# Add baseline results
baseline_results = {
    "unigram_loss": unigram_loss,
    "unigram_top1": unigram_top1 * 100,
    "bigram_top1": bigram_top1 * 100,
}

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

output_data = {
    "checkpoints": results,
    "baselines": baseline_results,
    "config": {
        "held_out_stories": len(held_out_stories),
        "max_seq_len": 256,
        "vocab_size": vocab_size,
    }
}

output_file = output_dir / "checkpoint_trajectory_analysis.json"
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nResults saved: {output_file}")

# ==============================================================================
# PRINT SUMMARY TABLE
# ==============================================================================

print("\n" + "="*100)
print("SUMMARY TABLE")
print("="*100)

print(f"\n{'Step':<8} {'Loss':<10} {'PPL':<10} {'Top-1%':<10} {'Top-5%':<10} {'Ctx Ben':<10} {'M-U Gain':<10} {'M-B Gap':<10}")
print("-"*100)

for r in results:
    print(f"{r['step']:<8} {r['loss']:<10.4f} {r['perplexity']:<10.1f} {r['top1_accuracy']:<10.2f} {r['top5_accuracy']:<10.2f} {r['context_benefit']:<10.4f} {r['model_vs_unigram']:<10.2f} {r['model_vs_bigram']:<10.2f}")

print("-"*100)
print(f"\nBASELINES:")
print(f"  Unigram Top-1: {unigram_top1*100:.2f}%")
print(f"  Bigram Top-1: {bigram_top1*100:.2f}%")

# ==============================================================================
# ANALYZE TRAJECTORY
# ==============================================================================

print("\n" + "="*100)
print("TRAJECTORY ANALYSIS")
print("="*100)

# Find when context benefit existed
context_benefits = [r['context_benefit'] for r in results]
max_context_benefit = max(context_benefits)
max_context_step = results[context_benefits.index(max_context_benefit)]['step']

print(f"\nContext Benefit Analysis:")
print(f"  Maximum context benefit: {max_context_benefit:.4f} at step {max_context_step}")
print(f"  Context benefit range: {min(context_benefits):.4f} to {max(context_benefits):.4f}")

# Find when model-vs-unigram gain existed
model_unigram_gains = [r['model_vs_unigram'] for r in results]
max_gain = max(model_unigram_gains)
max_gain_step = results[model_unigram_gains.index(max_gain)]['step']

print(f"\nModel vs Unigram Analysis:")
print(f"  Maximum gain over unigram: {max_gain:.2f}% at step {max_gain_step}")
print(f"  Model always close to unigram: {'YES' if abs(max_gain) < 1.0 else 'NO'}")

# Find when learning plateaued
losses = [r['loss'] for r in results]
min_loss = min(losses)
min_loss_step = results[losses.index(min_loss)]['step']

print(f"\nLoss Trajectory:")
print(f"  Minimum loss: {min_loss:.4f} at step {min_loss_step}")
print(f"  Loss range: {min(losses):.4f} to {max(losses):.4f}")

# ==============================================================================
# KEY FINDINGS
# ==============================================================================

print("\n" + "="*100)
print("KEY FINDINGS")
print("="*100)

# Did model EVER learn context?
if max_context_benefit < 0.01:
    print("\n1. CONTEXT LEARNING: ❌ NEVER EXISTED")
    print(f"   Maximum context benefit was only {max_context_benefit:.4f}")
    print("   Model never learned to use contextual information.")
else:
    print(f"\n1. CONTEXT LEARNING: ⚠️ MINIMAL")
    print(f"   Maximum context benefit was {max_context_benefit:.4f} at step {max_context_step}")
    print("   Context benefit was present but very small.")

# Model vs unigram
if abs(max_gain) < 0.5:
    print("\n2. MODEL vs UNIGRAM: ≈ IDENTICAL")
    print("   Model performance matches unigram baseline at all checkpoints.")
    print("   Model learned only word frequencies, not contextual patterns.")
else:
    print(f"\n2. MODEL vs UNIGRAM: Different by {max_gain:.2f}%")
    print("   Model shows some deviation from unigram baseline.")

# Model vs bigram
avg_model_bigram_gap = sum([r['model_vs_bigram'] for r in results]) / len(results)
if avg_model_bigram_gap < -10:
    print("\n3. MODEL vs BIGRAM: ❌ CONSISTENTLY WORSE")
    print(f"   Average gap: {avg_model_bigram_gap:.2f}%")
    print("   Model consistently underperforms simple bigram baseline.")
else:
    print(f"\n3. MODEL vs BIGRAM: Gap of {avg_model_bigram_gap:.2f}%")

# When did learning plateau?
print(f"\n4. LEARNING PLATEAU:")
# Find when loss stopped improving significantly
loss_improvements = []
for i in range(1, len(losses)):
    improvement = losses[i-1] - losses[i]
    loss_improvements.append(improvement)

# Find where improvements become negligible
plateau_step = None
for i, imp in enumerate(loss_improvements):
    if i > 5 and abs(imp) < 0.01:
        plateau_step = results[i]['step']
        break

if plateau_step:
    print(f"   Learning plateaued around step {plateau_step}")
else:
    print("   Learning continued improving throughout training")

# ==============================================================================
# CONCLUSIONS
# ==============================================================================

print("\n" + "="*100)
print("CONCLUSIONS")
print("="*100)

print("""
VERIFIED FINDINGS (based on data):

1. Model performance trajectory has been recorded across all checkpoints.

2. Context benefit was measured at each checkpoint using:
   - Full context (256 tokens) vs limited context (1/4/8 tokens)
   - Normal vs shuffled context

3. Model was compared against:
   - Unigram baseline (word frequency predictor)
   - Bigram baseline (simple n-gram model)

4. Key question answered: Did the model EVER learn contextual information?

HYPOTHESES (not verified, need further investigation):

- If model never exceeded unigram baseline, the issue is fundamental to training/setup.
- If model improved then plateaued, capacity or data limit was reached.
- If model degraded over time, catastrophic forgetting or LR issues.

NEXT STEPS:

- Review the trajectory data to identify any critical transition points.
- If context learning never existed, investigate training/evaluation pipeline.
- If context learning existed then disappeared, investigate training dynamics.
""")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
