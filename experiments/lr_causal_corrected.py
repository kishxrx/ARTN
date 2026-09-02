#!/usr/bin/env python3
"""
LR CAUSAL EXPERIMENT (CORRECTED): Using Custom AdaFactor

This experiment tests whether the LR schedule caused the plateau.

STARTING POINT: Exact 50k checkpoint with restored optimizer state

CONTINUATION: 1000 steps with ONLY LR changed

CONDITIONS:
  CONTROL:     LR = 3e-5  (same as 50k end)
  EXPERIMENT A: LR = 1e-4 (3x higher)
  EXPERIMENT B: LR = 3e-4 (10x higher)

EVERYTHING ELSE IDENTICAL:
  - Same custom AdaFactor optimizer
  - Same restored optimizer state
  - Same model architecture
  - Same tokenizer
  - Same dataset
  - Same batch size
  - Same sequence length
  - Same masking
  - Same loss function
  - Same evaluation pipeline

CRITICAL:
  - Uses custom AdaFactor (NOT AdamW)
  - Restores optimizer state from checkpoint
  - Changes ONLY the LR parameter
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
import numpy as np
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT, BitLinear
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.adafactor import Adafactor  # CUSTOM AdaFactor

print("="*80)
print("LR CAUSAL EXPERIMENT (CORRECTED): AdaFactor + Restored State")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_50K = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")
TOKENIZER_PATH = "outputs/artn-optimized/tokenizer.json"
OUTPUT_DIR = Path("experiments/results/lr_causal_corrected")

MAX_SEQ_LEN = 256
BATCH_SIZE = 4
GRAD_ACCUM = 2
CONTINUATION_STEPS = 1000
EVAL_INTERVAL = 200

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
heldout_data = dataset.select(range(50000, 50500))  # 500 for evaluation
print(f"Training data: 50,000 stories")
print(f"Held-out data: 500 stories (for evaluation)")

# ============================================================
# MODEL CONFIG (EXACTLY SAME AS ORIGINAL)
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
# EVALUATION FUNCTION (SAME AS ORIGINAL)
# ============================================================

def evaluate_model(model, tokenizer, dataset, max_samples=500):
    """Evaluate model on held-out data - EXACTLY as original training."""
    model.eval()
    
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
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
                
                total_real_tokens += 1
    
    model.train()
    
    return {
        "loss": total_loss / total_real_tokens if total_real_tokens > 0 else float('inf'),
        "top1_accuracy": 100 * total_correct_top1 / total_real_tokens if total_real_tokens > 0 else 0,
        "top5_accuracy": 100 * total_correct_top5 / total_real_tokens if total_real_tokens > 0 else 0,
        "real_tokens": total_real_tokens,
    }

def get_model_stats(model):
    """Get model statistics."""
    stats = {
        "mean_abs_w": 0.0,
        "quantization_scale_avg": 0.0,
        "ternary_neg": 0.0,
        "ternary_zero": 0.0,
        "ternary_pos": 0.0,
        "n_layers": 0,
    }
    
    scales = []
    
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            weight = module.weight.data
            mean_abs = weight.abs().mean().item()
            scales.append(mean_abs)
            
            scale = weight.abs().mean().item()
            weight_scaled = weight / (scale + 1e-8)
            
            neg_pct = (weight_scaled < -module.threshold).float().mean().item() * 100
            zero_pct = ((weight_scaled >= -module.threshold) & (weight_scaled <= module.threshold)).float().mean().item() * 100
            pos_pct = (weight_scaled > module.threshold).float().mean().item() * 100
            
            stats["ternary_neg"] += neg_pct
            stats["ternary_zero"] += zero_pct
            stats["ternary_pos"] += pos_pct
            stats["n_layers"] += 1
    
    if stats["n_layers"] > 0:
        stats["ternary_neg"] /= stats["n_layers"]
        stats["ternary_zero"] /= stats["n_layers"]
        stats["ternary_pos"] /= stats["n_layers"]
        stats["quantization_scale_avg"] = np.mean(scales)
    
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
    
    entropies = []
    for pos in range(min(50, logits.shape[1])):
        pos_logits = logits[0, pos, :]
        probs = F.softmax(pos_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        entropies.append(entropy)
    
    model.train()
    
    return {
        "mean": logit_flat.mean().item(),
        "std": logit_flat.std().item(),
        "entropy": np.mean(entropies),
    }

def generate_sample(model, tokenizer, prompt="Once upon a time", max_new_tokens=20):
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
    
    model.train()
    
    return tokenizer.decode(generated)

# ============================================================
# LOAD BASELINE MODEL AND CHECKPOINT
# ============================================================

print(f"\n{'='*80}")
print("LOADING BASELINE (50k checkpoint)")
print("="*80)

checkpoint = torch.load(CHECKPOINT_50K, map_location='cpu')
print(f"Checkpoint keys: {list(checkpoint.keys())}")

# Verify optimizer state
if 'optimizer_state_dict' in checkpoint:
    opt_state = checkpoint['optimizer_state_dict']
    pg = opt_state['param_groups'][0]
    print(f"\n✓ Optimizer state FOUND in checkpoint")
    print(f"  lr: {pg.get('lr')}")
    print(f"  beta1: {pg.get('beta1')}")
    print(f"  beta2: {pg.get('beta2')}")
else:
    print(f"\n✗ ERROR: No optimizer state in checkpoint!")
    sys.exit(1)

# Load model
baseline_model = BitLinearGPT(**model_config).to(device)
baseline_model.load_state_dict(checkpoint['model_state_dict'])
print(f"\n✓ Model loaded")

# Baseline evaluation
print(f"\n{'='*80}")
print("BASELINE EVALUATION (Step 50k)")
print("="*80)

baseline_metrics = evaluate_model(baseline_model, tokenizer, heldout_data)
baseline_stats = get_model_stats(baseline_model)
baseline_logit = get_logit_stats(baseline_model, tokenizer, heldout_data[0]['text'])
baseline_gen = generate_sample(baseline_model, tokenizer)

print(f"\nHeld-out loss: {baseline_metrics['loss']:.4f}")
print(f"Top-1: {baseline_metrics['top1_accuracy']:.2f}%")
print(f"Top-5: {baseline_metrics['top5_accuracy']:.2f}%")
print(f"Ternary: {baseline_stats['ternary_neg']:.1f}% / {baseline_stats['ternary_zero']:.1f}% / {baseline_stats['ternary_pos']:.1f}%")
print(f"Quant scale: {baseline_stats['quantization_scale_avg']:.6f}")
print(f"Logit std: {baseline_logit['std']:.4f}")
print(f"Generation: {baseline_gen[:80]}...")

# ============================================================
# RUN EACH LR CONFIG
# ============================================================

results = {
    "baseline": {
        "metrics": baseline_metrics,
        "stats": baseline_stats,
        "logit": baseline_logit,
        "generation": baseline_gen,
    }
}

for exp_name, lr_config in LR_CONFIGS.items():
    print(f"\n{'='*80}")
    print(f"RUNNING: {lr_config['name']}")
    print("="*80)
    
    # Load FRESH model from checkpoint
    model = BitLinearGPT(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Setup optimizer with CUSTOM AdaFactor (EXACTLY as original training)
    optimizer = Adafactor(
        model.parameters(),
        lr=lr_config['lr'],  # ONLY THIS CHANGES
        weight_decay=0.01,
    )
    
    # Restore optimizer state
    print(f"\nRestoring optimizer state...")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # CHANGE ONLY THE LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_config['lr']
    
    print(f"✓ Optimizer state restored, LR set to {lr_config['lr']}")
    
    # Verify
    print(f"\nOptimizer verification:")
    print(f"  lr: {optimizer.param_groups[0]['lr']}")
    print(f"  beta1: {optimizer.param_groups[0]['beta1']}")
    print(f"  beta2: {optimizer.param_groups[0]['beta2']}")
    
    # Training loop
    print(f"\nTraining for {CONTINUATION_STEPS} steps...")
    model.train()
    
    train_losses = []
    eval_metrics_history = []
    
    data_iter = iter(train_data)
    
    for step in range(CONTINUATION_STEPS):
        # Get batch
        batch_texts = []
        for _ in range(BATCH_SIZE * GRAD_ACCUM):
            try:
                batch_texts.append(next(data_iter)['text'])
            except StopIteration:
                data_iter = iter(train_data)
                batch_texts.append(next(data_iter)['text'])
        
        # Tokenize
        batch_tokens = []
        for text in batch_texts:
            tokens = tokenizer.encode(text)[:MAX_SEQ_LEN-1]
            tokens = tokens + [tokenizer.eos_token_id]
            batch_tokens.append(tokens)
        
        # Pad
        max_len = max(len(t) for t in batch_tokens)
        batch_padded = []
        for tokens in batch_tokens:
            padding_len = max_len - len(tokens)
            batch_padded.append(tokens + [tokenizer.pad_token_id] * padding_len)
        
        input_ids = torch.tensor(batch_padded, dtype=torch.long, device=device)
        targets = input_ids.clone()
        targets[targets == tokenizer.pad_token_id] = -100
        
        # Forward
        output = model(input_ids, targets)
        loss = output["loss"] / GRAD_ACCUM
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        train_losses.append(loss.item() * GRAD_ACCUM)
        
        # Log
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{CONTINUATION_STEPS}: Train loss = {loss.item() * GRAD_ACCUM:.4f}")
        
        # Evaluate at intervals
        if (step + 1) % EVAL_INTERVAL == 0:
            eval_metrics = evaluate_model(model, tokenizer, heldout_data)
            eval_metrics_history.append({
                "step": step + 1,
                "metrics": eval_metrics,
            })
            print(f"    → Held-out loss: {eval_metrics['loss']:.4f} (baseline: {baseline_metrics['loss']:.4f})")
    
    # Final evaluation
    print(f"\nFinal evaluation...")
    final_metrics = evaluate_model(model, tokenizer, heldout_data)
    final_stats = get_model_stats(model)
    final_logit = get_logit_stats(model, tokenizer, heldout_data[0]['text'])
    final_gen = generate_sample(model, tokenizer)
    
    print(f"\nFinal Results:")
    print(f"  Held-out loss: {final_metrics['loss']:.4f} (baseline: {baseline_metrics['loss']:.4f})")
    print(f"  Top-1: {final_metrics['top1_accuracy']:.2f}% (baseline: {baseline_metrics['top1_accuracy']:.2f}%)")
    print(f"  Top-5: {final_metrics['top5_accuracy']:.2f}% (baseline: {baseline_metrics['top5_accuracy']:.2f}%)")
    print(f"  Generation: {final_gen[:80]}...")
    
    # Calculate changes
    loss_change = final_metrics['loss'] - baseline_metrics['loss']
    top1_change = final_metrics['top1_accuracy'] - baseline_metrics['top1_accuracy']
    
    print(f"\nChanges from Baseline:")
    print(f"  Loss: {loss_change:+.4f}")
    print(f"  Top-1: {top1_change:+.2f}%")
    
    # Store results
    results[exp_name] = {
        "lr": lr_config['lr'],
        "train_losses": train_losses,
        "eval_history": eval_metrics_history,
        "metrics_baseline": baseline_metrics,
        "metrics_final": final_metrics,
        "stats_baseline": baseline_stats,
        "stats_final": final_stats,
        "logit_baseline": baseline_logit,
        "logit_final": final_logit,
        "generation_baseline": baseline_gen,
        "generation_final": final_gen,
        "changes": {
            "loss": loss_change,
            "top1": top1_change,
            "top5": final_metrics['top5_accuracy'] - baseline_metrics['top5_accuracy'],
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

print(f"\n{'Experiment':<25} {'LR':>10} {'Loss Δ':>12} {'Top-1 Δ':>12} {'Result':>25}")
print("-"*85)

for exp_name, lr_config in LR_CONFIGS.items():
    r = results[exp_name]
    loss_delta = r['changes']['loss']
    top1_delta = r['changes']['top1']
    
    # Determine result
    if loss_delta < -0.05:
        result = "✅ IMPROVEMENT (LR helps)"
    elif abs(loss_delta) < 0.02:
        result = "⚠️ STABLE (no change)"
    elif loss_delta > 0.05:
        result = "❌ DEGRADATION (LR hurts)"
    else:
        result = "➖ MARGINAL"
    
    print(f"{lr_config['name']:<25} {r['lr']:>10.0e} {loss_delta:>+12.4f} {top1_delta:>+12.2f}% {result:>25}")

# ============================================================
# INTERPRETATION
# ============================================================

print(f"\n{'='*80}")
print("INTERPRETATION")
print("="*80)

control_loss = results['control']['changes']['loss']
exp_a_loss = results['experiment_a']['changes']['loss']
exp_b_loss = results['experiment_b']['changes']['loss']

print(f"\nHeld-out loss changes:")
print(f"  Control (3e-5):     {control_loss:+.4f}")
print(f"  Experiment A (1e-4): {exp_a_loss:+.4f}")
print(f"  Experiment B (3e-4): {exp_b_loss:+.4f}")

print(f"\nConclusion:")

if exp_a_loss < -0.05 or exp_b_loss < -0.05:
    print(f"\n  ✅ RESULT A: Higher LR IMPROVES held-out performance")
    print(f"     → LR decay was LIMITING learning")
    print(f"     → Plateau was caused by LR being too low")
    print(f"     → Recommend: Continue training with higher min_lr")
    
    if control_loss < -0.02:
        print(f"\n  ⚠️ NOTE: Control also improved slightly")
        print(f"     → Model may not have fully plateaued yet")
    
elif exp_a_loss > 0.05 or exp_b_loss > 0.05:
    print(f"\n  ❌ RESULT B: Higher LR DEGRADES held-out performance")
    print(f"     → LR decay was PROTECTING generalization")
    print(f"     → Plateau was NOT caused by low LR")
    print(f"     → Model has reached its generalization limit")
    
    if control_loss < -0.02:
        print(f"\n  ⚠️ NOTE: Control improved")
        print(f"     → Model may still be learning at 3e-5 LR")
    
else:
    print(f"\n  ⚠️ RESULT C: All LR values produce similar results")
    print(f"     → LR is not the primary factor")
    print(f"     → Investigate other hypotheses (data, capacity, ternary)")

# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_DIR / "lr_causal_corrected.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*80}")
print(f"RESULTS SAVED: {OUTPUT_DIR / 'lr_causal_corrected.json'}")
print("="*80)
