#!/usr/bin/env python3
"""
CAUSAL EXPERIMENT: Does Restoring LR Restore Learning?

Uses EXISTING 50k checkpoint as starting point.
Tests different learning rates to determine if LR decay caused the plateau.

This is a CONTROLLED CAUSAL EXPERIMENT:
- Same checkpoint
- Same architecture
- Same data
- Same everything except LR

Interpretation:
A) Higher LR improves held-out loss → LR decay caused plateau
B) Higher LR increases movement but no loss improvement → LR not sufficient
C) Higher LR causes instability → LR stability problem
D) All LR values remain flat → LR decay not the bottleneck
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
import numpy as np
from collections import defaultdict
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT, BitLinear
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*80)
print("CAUSAL EXPERIMENT: Does Restoring LR Restore Learning?")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_50K = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")
TOKENIZER_PATH = "outputs/artn-optimized/tokenizer.json"
OUTPUT_DIR = Path("experiments/results/lr_causal_experiment")

MAX_SEQ_LEN = 256
BATCH_SIZE = 4
GRAD_ACCUM = 2
CONTINUATION_STEPS = 3000  # Short experiment
EVAL_INTERVAL = 500

# Learning rates to test
LR_CONFIGS = {
    "control": {"lr": 3e-5, "name": "Control (3e-5 - same as 50k)"},
    "experiment_a": {"lr": 1e-4, "name": "Experiment A (1e-4 - 3x higher)"},
    "experiment_b": {"lr": 3e-4, "name": "Experiment B (3e-4 - 10x higher)"},
}

# ============================================================
# LOAD TOKENIZER AND DATA
# ============================================================

print(f"\n{'='*80}")
print("LOADING TOKENIZER AND DATA")
print("="*80)

tokenizer = BPETokenizer.load(TOKENIZER_PATH)
print(f"Tokenizer: vocab={tokenizer.vocab_size}, PAD={tokenizer.pad_token_id}")

dataset = load_dataset("roneneldan/TinyStories", split="train")
train_data = dataset.select(range(50000))
heldout_data = dataset.select(range(50000, 50200))
print(f"Training data: 50,000 stories")
print(f"Held-out data: 200 stories")

# ============================================================
# MODEL CONFIG
# ============================================================

model_config = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
}

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_model(model, tokenizer, dataset, name="", max_samples=200):
    """Evaluate model on dataset."""
    model.eval()
    
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_correct_prob = 0.0
    total_real_tokens = 0
    
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            text = dataset[i]['text']
            tokens = tokenizer.encode(text)[:MAX_SEQ_LEN-1]
            tokens = tokens + [tokenizer.eos_token_id]
            
            if len(tokens) < 2:
                continue
            
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            output = model(input_ids)
            logits = output["logits"]
            
            for pos in range(len(tokens) - 1):
                target = tokens[pos + 1]
                
                if target == tokenizer.pad_token_id:
                    continue
                
                pos_logits = logits[0, pos, :]
                loss = F.cross_entropy(pos_logits.unsqueeze(0), torch.tensor([target], device=device))
                total_loss += loss.item()
                
                probs = F.softmax(pos_logits, dim=-1)
                top1 = probs.argmax().item()
                top5 = probs.topk(5).indices.tolist()
                
                if top1 == target:
                    total_correct_top1 += 1
                if target in top5:
                    total_correct_top5 += 1
                
                total_correct_prob += probs[target].item()
                total_real_tokens += 1
    
    return {
        "loss": total_loss / total_real_tokens if total_real_tokens > 0 else float('inf'),
        "top1_accuracy": 100 * total_correct_top1 / total_real_tokens if total_real_tokens > 0 else 0,
        "top5_accuracy": 100 * total_correct_top5 / total_real_tokens if total_real_tokens > 0 else 0,
        "correct_prob": 100 * total_correct_prob / total_real_tokens if total_real_tokens > 0 else 0,
        "real_tokens": total_real_tokens,
    }

def get_ternary_stats(model):
    """Get ternary weight statistics."""
    stats = {
        "mean_abs_w": 0.0,
        "ternary_neg": 0.0,
        "ternary_zero": 0.0,
        "ternary_pos": 0.0,
        "total_params": 0,
    }
    
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            weight = module.weight.data
            
            mean_abs = weight.abs().mean().item()
            stats["mean_abs_w"] += mean_abs * weight.numel()
            stats["total_params"] += weight.numel()
            
            # Ternary distribution
            scale = weight.abs().mean().item()
            weight_scaled = weight / (scale + 1e-8)
            
            neg = (weight_scaled < -module.threshold).float().mean().item() * 100
            zero = ((weight_scaled >= -module.threshold) & (weight_scaled <= module.threshold)).float().mean().item() * 100
            pos = (weight_scaled > module.threshold).float().mean().item() * 100
            
            stats["ternary_neg"] += neg
            stats["ternary_zero"] += zero
            stats["ternary_pos"] += pos
    
    n_layers = stats["total_params"] / 65536  # Approximate
    stats["ternary_neg"] /= n_layers
    stats["ternary_zero"] /= n_layers
    stats["ternary_pos"] /= n_layers
    stats["mean_abs_w"] /= stats["total_params"]
    
    return stats

def get_logit_stats(model, tokenizer, sample_text):
    """Get logit statistics."""
    model.eval()
    
    tokens = tokenizer.encode(sample_text)[:MAX_SEQ_LEN-1]
    tokens = tokens + [tokenizer.eos_token_id]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output = model(input_ids)
        logits = output["logits"]
    
    logit_flat = logits[0].flatten()
    
    # Entropy
    entropies = []
    for pos in range(min(50, logits.shape[1])):
        pos_logits = logits[0, pos, :]
        probs = F.softmax(pos_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(entropy)
    
    return {
        "mean": logit_flat.mean().item(),
        "std": logit_flat.std().item(),
        "entropy": np.mean(entropies),
    }

def generate_sample(model, tokenizer, prompt="Once upon a time", max_new_tokens=30):
    """Generate text sample."""
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(input_ids)
            logits = output["logits"]
            next_logits = logits[0, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            if len(generated) >= 100:
                break
    
    return tokenizer.decode(generated)

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_continuation(model, optimizer, train_data, tokenizer, steps, lr_name, eval_interval=500):
    """Train model for continuation steps."""
    model.train()
    
    step = 0
    losses = []
    
    # Create data loader
    data_iter = iter(train_data)
    
    while step < steps:
        # Get batch
        batch_texts = []
        for _ in range(BATCH_SIZE * GRAD_ACCUM):
            try:
                batch_texts.append(next(data_iter)['text'])
            except StopIteration:
                data_iter = iter(train_data)
                batch_texts.append(next(data_iter)['text'])
        
        # Tokenize and pad
        batch_tokens = []
        for text in batch_texts:
            tokens = tokenizer.encode(text)[:MAX_SEQ_LEN-1]
            tokens = tokens + [tokenizer.eos_token_id]
            batch_tokens.append(tokens)
        
        # Pad to same length
        max_len = max(len(t) for t in batch_tokens)
        batch_padded = []
        for tokens in batch_tokens:
            padding_len = max_len - len(tokens)
            batch_padded.append(tokens + [tokenizer.pad_token_id] * padding_len)
        
        # Create input and target
        input_ids = torch.tensor(batch_padded, dtype=torch.long, device=device)
        targets = input_ids.clone()
        targets[targets == tokenizer.pad_token_id] = -100
        
        # Forward pass
        output = model(input_ids, targets)
        loss = output["loss"] / GRAD_ACCUM
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            losses.append(loss.item() * GRAD_ACCUM)
            
            if (step + 1) % eval_interval == 0:
                print(f"    {lr_name} Step {step+1}/{steps}: Loss = {losses[-1]:.4f}")
        
        step += 1
    
    return losses

# ============================================================
# RUN EXPERIMENTS
# ============================================================

print(f"\n{'='*80}")
print("LOADING 50k CHECKPOINT (BASELINE)")
print("="*80)

# Load baseline model
baseline_model = BitLinearGPT(**model_config).to(device)
checkpoint = torch.load(CHECKPOINT_50K, map_location=device)
if 'model_state_dict' in checkpoint:
    baseline_model.load_state_dict(checkpoint['model_state_dict'])
else:
    baseline_model.load_state_dict(checkpoint)

# Evaluate baseline
print(f"\nEvaluating baseline (50k checkpoint)...")
baseline_metrics = evaluate_model(baseline_model, tokenizer, heldout_data, "Baseline")
baseline_ternary = get_ternary_stats(baseline_model)
baseline_logit = get_logit_stats(baseline_model, tokenizer, heldout_data[0]['text'])
baseline_gen = generate_sample(baseline_model, tokenizer)

print(f"\nBaseline Results:")
print(f"  Held-out loss: {baseline_metrics['loss']:.4f}")
print(f"  Top-1: {baseline_metrics['top1_accuracy']:.2f}%")
print(f"  Top-5: {baseline_metrics['top5_accuracy']:.2f}%")
print(f"  Mean|W|: {baseline_ternary['mean_abs_w']:.6f}")
print(f"  Ternary: {baseline_ternary['ternary_neg']:.1f}% / {baseline_ternary['ternary_zero']:.1f}% / {baseline_ternary['ternary_pos']:.1f}%")
print(f"  Logit std: {baseline_logit['std']:.4f}")
print(f"  Generation: {baseline_gen[:80]}...")

# ============================================================
# RUN EACH LR CONFIG
# ============================================================

results = {
    "baseline": {
        "metrics": baseline_metrics,
        "ternary": baseline_ternary,
        "logit": baseline_logit,
        "generation": baseline_gen,
    }
}

for exp_name, lr_config in LR_CONFIGS.items():
    print(f"\n{'='*80}")
    print(f"RUNNING: {lr_config['name']}")
    print("="*80)
    
    # Clone baseline model
    model = copy.deepcopy(baseline_model)
    
    # Setup optimizer with NEW learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_config['lr'],
        betas=(0.5, 0.98),
        weight_decay=0.01,
    )
    
    print(f"  LR: {lr_config['lr']}")
    print(f"  Steps: {CONTINUATION_STEPS}")
    
    # Train continuation
    train_losses = train_continuation(
        model, optimizer, train_data, tokenizer,
        steps=CONTINUATION_STEPS,
        lr_name=lr_config['name'],
        eval_interval=EVAL_INTERVAL,
    )
    
    # Evaluate after continuation
    print(f"\nEvaluating after continuation...")
    after_metrics = evaluate_model(model, tokenizer, heldout_data, exp_name)
    after_ternary = get_ternary_stats(model)
    after_logit = get_logit_stats(model, tokenizer, heldout_data[0]['text'])
    after_gen = generate_sample(model, tokenizer)
    
    print(f"\nAfter Continuation Results:")
    print(f"  Held-out loss: {after_metrics['loss']:.4f}")
    print(f"  Top-1: {after_metrics['top1_accuracy']:.2f}%")
    print(f"  Top-5: {after_metrics['top5_accuracy']:.2f}%")
    print(f"  Mean|W|: {after_ternary['mean_abs_w']:.6f}")
    print(f"  Ternary: {after_ternary['ternary_neg']:.1f}% / {after_ternary['ternary_zero']:.1f}% / {after_ternary['ternary_pos']:.1f}%")
    print(f"  Logit std: {after_logit['std']:.4f}")
    print(f"  Generation: {after_gen[:80]}...")
    
    # Calculate changes
    loss_change = baseline_metrics['loss'] - after_metrics['loss']
    top1_change = after_metrics['top1_accuracy'] - baseline_metrics['top1_accuracy']
    mean_w_change = baseline_ternary['mean_abs_w'] - after_ternary['mean_abs_w']
    
    print(f"\nChanges from Baseline:")
    print(f"  Loss: {loss_change:+.4f}")
    print(f"  Top-1: {top1_change:+.2f}%")
    print(f"  Mean|W|: {mean_w_change:+.6f}")
    
    # Store results
    results[exp_name] = {
        "lr": lr_config['lr'],
        "train_losses": train_losses,
        "metrics_before": baseline_metrics,
        "metrics_after": after_metrics,
        "ternary_before": baseline_ternary,
        "ternary_after": after_ternary,
        "logit_before": baseline_logit,
        "logit_after": after_logit,
        "generation_before": baseline_gen,
        "generation_after": after_gen,
        "changes": {
            "loss": loss_change,
            "top1": top1_change,
            "mean_abs_w": mean_w_change,
        }
    }
    
    # Clean up
    del model
    del optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================
# ANALYSIS
# ============================================================

print(f"\n{'='*80}")
print("CAUSAL ANALYSIS")
print("="*80)

print(f"\n{'Experiment':<25} {'LR':>10} {'Loss Δ':>10} {'Top-1 Δ':>10} {'Result':>20}")
print("-"*80)

for exp_name, lr_config in LR_CONFIGS.items():
    r = results[exp_name]
    loss_delta = r['changes']['loss']
    top1_delta = r['changes']['top1']
    
    # Determine result
    if loss_delta > 0.05:
        result = "IMPROVEMENT"
    elif abs(loss_delta) < 0.01:
        result = "FLAT"
    elif loss_delta < -0.05:
        result = "WORSE"
    else:
        result = "MARGINAL"
    
    print(f"{lr_config['name']:<25} {r['lr']:>10.0e} {loss_delta:>+10.4f} {top1_delta:>+10.2f}% {result:>20}")

# ============================================================
# INTERPRETATION
# ============================================================

print(f"\n{'='*80}")
print("INTERPRETATION")
print("="*80)

control_loss = results['control']['changes']['loss']
exp_a_loss = results['experiment_a']['changes']['loss']
exp_b_loss = results['experiment_b']['changes']['loss']

print(f"\nLoss changes:")
print(f"  Control (3e-5):   {control_loss:+.4f}")
print(f"  Experiment A (1e-4): {exp_a_loss:+.4f}")
print(f"  Experiment B (3e-4): {exp_b_loss:+.4f}")

print(f"\nInterpretation:")

if exp_a_loss > 0.05 or exp_b_loss > 0.05:
    if control_loss < 0.01:
        print(f"\n  ✅ RESULT A: Higher LR produces improvement, control remains flat")
        print(f"     → STRONG causal evidence that LR decay caused the plateau")
    else:
        print(f"\n  ⚠️ RESULT: Both higher LR and control show improvement")
        print(f"     → Partial evidence for LR decay, but other factors involved")
elif exp_a_loss < -0.05 or exp_b_loss < -0.05:
    print(f"\n  ❌ RESULT C: Higher LR causes degradation")
    print(f"     → LR is an optimization stability problem")
else:
    if control_loss < 0.01:
        print(f"\n  ⚠️ RESULT D: All LR values remain flat")
        print(f"     → LR decay is NOT the primary bottleneck")
    else:
        print(f"\n  ❓ RESULT B: Movement but no loss improvement")
        print(f"     → LR is not sufficient, investigate other factors")

# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_DIR / "lr_causal_experiment.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*80}")
print(f"RESULTS SAVED: {OUTPUT_DIR / 'lr_causal_experiment.json'}")
print("="*80)
