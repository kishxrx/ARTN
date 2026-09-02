#!/usr/bin/env python3
"""
100-STEP DIAGNOSTIC: Why Does Continuation Cause Catastrophic Degradation?

Tracks EVERYTHING every 10 steps to identify exactly when/why degradation begins.
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT, BitLinear
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*80)
print("100-STEP DIAGNOSTIC: Continuation Failure Analysis")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_50K = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")
TOKENIZER_PATH = "outputs/artn-optimized/tokenizer.json"

MAX_SEQ_LEN = 256
BATCH_SIZE = 4
GRAD_ACCUM = 2
DIAGNOSTIC_STEPS = 100
LOG_INTERVAL = 10

LR = 3e-5  # Same as end of original training

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
print(f"Held-out data: 200 stories (for evaluation)")

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
# DIAGNOSTIC FUNCTIONS
# ============================================================

def evaluate_model(model, tokenizer, dataset, max_samples=200):
    """Evaluate model on held-out data."""
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
    """Get comprehensive model statistics."""
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
            
            # Weight statistics
            mean_abs = weight.abs().mean().item()
            scales.append(mean_abs)
            
            # Ternary distribution
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

def get_gradient_stats(model):
    """Get gradient statistics."""
    total_norm = 0.0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
            param_count += 1
    
    total_norm = total_norm ** 0.5
    
    return {
        "grad_norm": total_norm,
        "param_count": param_count,
    }

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
# INSPECT CHECKPOINT
# ============================================================

print(f"\n{'='*80}")
print("INSPECTING CHECKPOINT")
print("="*80)

checkpoint = torch.load(CHECKPOINT_50K, map_location='cpu')

print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

if 'model_state_dict' in checkpoint:
    print(f"\nModel state dict keys (first 10): {list(checkpoint['model_state_dict'].keys())[:10]}")
    
    # Check for latent weights
    for key in checkpoint['model_state_dict'].keys():
        if 'latent' in key.lower() or 'fp_weight' in key.lower():
            print(f"  Found latent/FP weight: {key}")

if 'optimizer_state_dict' in checkpoint:
    print(f"\n✓ Optimizer state dict FOUND in checkpoint")
    print(f"  Optimizer keys: {list(checkpoint['optimizer_state_dict'].keys())}")
else:
    print(f"\n✗ Optimizer state dict NOT FOUND in checkpoint")
    print(f"  This may explain continuation failure!")

if 'scheduler_state_dict' in checkpoint:
    print(f"\n✓ Scheduler state dict FOUND in checkpoint")
else:
    print(f"\n✗ Scheduler state dict NOT FOUND in checkpoint")

# ============================================================
# LOAD MODEL
# ============================================================

print(f"\n{'='*80}")
print("LOADING MODEL")
print("="*80)

model = BitLinearGPT(**model_config).to(device)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

print(f"Model loaded successfully")

# Verify ternary distribution
model_stats = get_model_stats(model)
print(f"\nTernary distribution after loading:")
print(f"  Neg: {model_stats['ternary_neg']:.2f}%")
print(f"  Zero: {model_stats['ternary_zero']:.2f}%")
print(f"  Pos: {model_stats['ternary_pos']:.2f}%")
print(f"  Sum: {model_stats['ternary_neg'] + model_stats['ternary_zero'] + model_stats['ternary_pos']:.2f}%")
print(f"  Quantization scale avg: {model_stats['quantization_scale_avg']:.6f}")

# Baseline evaluation
print(f"\n{'='*80}")
print("BASELINE EVALUATION (Step 0)")
print("="*80)

baseline_metrics = evaluate_model(model, tokenizer, heldout_data)
print(f"Held-out loss: {baseline_metrics['loss']:.4f}")
print(f"Top-1: {baseline_metrics['top1_accuracy']:.2f}%")
print(f"Top-5: {baseline_metrics['top5_accuracy']:.2f}%")

baseline_logit = get_logit_stats(model, tokenizer, heldout_data[0]['text'])
print(f"Logit std: {baseline_logit['std']:.4f}")
print(f"Logit entropy: {baseline_logit['entropy']:.4f}")

baseline_gen = generate_sample(model, tokenizer)
print(f"Generation: {baseline_gen[:80]}...")

# ============================================================
# SETUP OPTIMIZER (WITH OR WITHOUT STATE)
# ============================================================

print(f"\n{'='*80}")
print("SETTING UP OPTIMIZER")
print("="*80)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    betas=(0.5, 0.98),
    weight_decay=0.01,
)

if 'optimizer_state_dict' in checkpoint:
    print("Found optimizer state in checkpoint...")
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state restored successfully")
    except Exception as e:
        print(f"✗ Failed to restore optimizer state: {e}")
        print(f"  Continuing with FRESH optimizer state")
        print(f"  This may explain continuation degradation!")

# ============================================================
# RUN DIAGNOSTIC
# ============================================================

print(f"\n{'='*80}")
print(f"RUNNING {DIAGNOSTIC_STEPS} STEP DIAGNOSTIC")
print("="*80)

results = []

model.train()
data_iter = iter(train_data)

for step in range(DIAGNOSTIC_STEPS):
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
    
    # Forward pass
    output = model(input_ids, targets)
    loss = output["loss"] / GRAD_ACCUM
    
    # Backward pass
    loss.backward()
    
    # Log every step for first 20, then every LOG_INTERVAL
    if step < 20 or (step + 1) % LOG_INTERVAL == 0:
        # Get gradient stats
        grad_stats = get_gradient_stats(model)
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nStep {step+1}/{DIAGNOSTIC_STEPS}:")
        print(f"  Train loss: {loss.item() * GRAD_ACCUM:.4f}")
        print(f"  Grad norm: {grad_stats['grad_norm']:.4f}")
        print(f"  LR: {current_lr:.6f}")
    
    # Gradient accumulation
    if (step + 1) % GRAD_ACCUM == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    # Detailed logging every 10 steps
    if (step + 1) % 10 == 0:
        print(f"\n{'='*80}")
        print(f"DETAILED DIAGNOSTIC AT STEP {step+1}")
        print("="*80)
        
        # Evaluate
        eval_metrics = evaluate_model(model, tokenizer, heldout_data)
        print(f"\nHeld-out loss: {eval_metrics['loss']:.4f} (baseline: {baseline_metrics['loss']:.4f})")
        print(f"Top-1: {eval_metrics['top1_accuracy']:.2f}% (baseline: {baseline_metrics['top1_accuracy']:.2f}%)")
        print(f"Top-5: {eval_metrics['top5_accuracy']:.2f}% (baseline: {baseline_metrics['top5_accuracy']:.2f}%)")
        
        # Model stats
        model_stats = get_model_stats(model)
        print(f"\nTernary: {model_stats['ternary_neg']:.1f}% / {model_stats['ternary_zero']:.1f}% / {model_stats['ternary_pos']:.1f}%")
        print(f"Quant scale: {model_stats['quantization_scale_avg']:.6f}")
        
        # Logit stats
        logit_stats = get_logit_stats(model, tokenizer, heldout_data[0]['text'])
        print(f"Logit std: {logit_stats['std']:.4f} (baseline: {baseline_logit['std']:.4f})")
        print(f"Logit entropy: {logit_stats['entropy']:.4f}")
        
        # Generation
        gen = generate_sample(model, tokenizer)
        print(f"\nGeneration: {gen[:80]}...")
        
        # Store results
        results.append({
            "step": step + 1,
            "train_loss": loss.item() * GRAD_ACCUM,
            "heldout_loss": eval_metrics['loss'],
            "top1": eval_metrics['top1_accuracy'],
            "top5": eval_metrics['top5_accuracy'],
            "grad_norm": grad_stats['grad_norm'],
            "lr": optimizer.param_groups[0]['lr'],
            "ternary_neg": model_stats['ternary_neg'],
            "ternary_zero": model_stats['ternary_zero'],
            "ternary_pos": model_stats['ternary_pos'],
            "quant_scale": model_stats['quantization_scale_avg'],
            "logit_std": logit_stats['std'],
            "logit_entropy": logit_stats['entropy'],
            "generation": gen[:100],
        })

# ============================================================
# FINAL SUMMARY
# ============================================================

print(f"\n{'='*80}")
print("DIAGNOSTIC SUMMARY")
print("="*80)

print(f"\n{'Step':<8} {'Train Loss':>12} {'Held-Out':>12} {'Top-1':>8} {'Grad Norm':>12}")
print("-"*55)

for r in results:
    print(f"{r['step']:<8} {r['train_loss']:>12.4f} {r['heldout_loss']:>12.4f} {r['top1']:>8.2f}% {r['grad_norm']:>12.4f}")

# Calculate degradation rate
if len(results) >= 2:
    initial_loss = results[0]['heldout_loss']
    final_loss = results[-1]['heldout_loss']
    degradation = final_loss - initial_loss
    
    print(f"\nHeld-out loss change: {initial_loss:.4f} → {final_loss:.4f} ({degradation:+.4f})")
    
    if degradation > 0.1:
        print(f"\n⚠️ DEGRADATION DETECTED WITHIN FIRST {results[-1]['step']} STEPS!")
        print(f"   This suggests a continuation mechanism problem, not gradual overfitting.")
    else:
        print(f"\n✓ No significant degradation in first {results[-1]['step']} steps.")

# ============================================================
# SAVE RESULTS
# ============================================================

output_path = Path("experiments/results/diagnostic_100step.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*80}")
print(f"RESULTS SAVED: {output_path}")
print("="*80)
