#!/usr/bin/env python3
"""
CORRECT Evaluation: FP32 5k vs Ternary 50k

Uses the CORRECT 50k checkpoint: outputs/artn-50k-best-config/
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
print("CORRECT EVALUATION: FP32 5k vs Ternary 50k")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# BASELINE
# ============================================================

vocab_size = 4000
uniform_loss = torch.log(torch.tensor(vocab_size, dtype=torch.float))
print(f"\nUniform baseline: Loss = {uniform_loss.item():.4f}, Perplexity = {torch.exp(uniform_loss).item():.0f}")

# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
print(f"Tokenizer: vocab={tokenizer.vocab_size}, PAD={tokenizer.pad_token_id}, EOS={tokenizer.eos_token_id}")

# ============================================================
# LOAD MODELS
# ============================================================

model_config = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
}

# Ternary 50k - CORRECT checkpoint
print(f"\n{'='*80}")
print("LOADING TERNARY 50k (CORRECT CHECKPOINT)")
print("="*80)

ternary_path = "outputs/artn-50k-best-config/final_model/model.pt"
print(f"Path: {ternary_path}")

ternary_50k = BitLinearGPT(**model_config).to(device)
checkpoint = torch.load(ternary_path, map_location=device)
if 'model_state_dict' in checkpoint:
    ternary_50k.load_state_dict(checkpoint['model_state_dict'])
else:
    ternary_50k.load_state_dict(checkpoint)
ternary_50k.eval()
print(f"✓ Loaded")

# FP32 5k
print(f"\n{'='*80}")
print("LOADING FP32 5k")
print("="*80)

fp32_path = "experiments/results/fp32_control_20260819_232646/checkpoints/step_5000/model.pt"
print(f"Path: {fp32_path}")

# Check if exists
import os
if os.path.exists(fp32_path):
    fp32_5k = BitLinearGPT(**model_config).to(device)
    checkpoint = torch.load(fp32_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        fp32_5k.load_state_dict(checkpoint['model_state_dict'])
    else:
        fp32_5k.load_state_dict(checkpoint)
    fp32_5k.eval()
    print(f"✓ Loaded")
else:
    print(f"✗ FP32 checkpoint not found at this path")
    print(f"   Looking for alternatives...")
    # Find FP32 checkpoint
    import glob
    fp32_candidates = glob.glob("experiments/results/fp32*/checkpoints/step_*/model.pt")
    if fp32_candidates:
        print(f"   Found: {fp32_candidates}")
        fp32_path = fp32_candidates[0]
        fp32_5k = BitLinearGPT(**model_config).to(device)
        checkpoint = torch.load(fp32_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            fp32_5k.load_state_dict(checkpoint['model_state_dict'])
        else:
            fp32_5k.load_state_dict(checkpoint)
        fp32_5k.eval()
        print(f"✓ Loaded: {fp32_path}")
    else:
        print(f"✗ No FP32 checkpoint found, will skip FP32 evaluation")
        fp32_5k = None

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, tokenizer, dataset, name, max_samples=500, max_seq_len=256):
    """Evaluate model on dataset."""
    model.eval()
    
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_real_tokens = 0
    
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            text = dataset[i]['text']
            tokens = tokenizer.encode(text)[:max_seq_len - 1]
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
        
        if (i + 1) % 100 == 0:
            print(f"  {name}: {i+1}/{max_samples}")
    
    avg_loss = total_loss / total_real_tokens if total_real_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    top1_acc = 100 * total_correct_top1 / total_real_tokens if total_real_tokens > 0 else 0
    top5_acc = 100 * total_correct_top5 / total_real_tokens if total_real_tokens > 0 else 0
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "real_tokens": total_real_tokens,
    }

# ============================================================
# LOAD DATA
# ============================================================

print(f"\n{'='*80}")
print("LOADING DATA")
print("="*80)

dataset = load_dataset("roneneldan/TinyStories", split="train")
train_data = dataset.select(range(50000))
heldout_data = dataset.select(range(50000, 51000))

print(f"Training: 500 stories (from 50k)")
print(f"Held-out: 500 stories")

# ============================================================
# EVALUATE TERNARY 50k
# ============================================================

print(f"\n{'='*80}")
print("EVALUATING TERNARY 50k (CORRECT CHECKPOINT)")
print("="*80)

train_ternary = evaluate_model(ternary_50k, tokenizer, train_data, "Train", max_samples=500)
heldout_ternary = evaluate_model(ternary_50k, tokenizer, heldout_data, "Held-out", max_samples=500)

print(f"\nTERNARY 50k Results:")
print(f"{'Metric':<20} {'Training':>15} {'Held-Out':>15}")
print(f"{'-'*50}")
print(f"{'Loss':<20} {train_ternary['loss']:>15.4f} {heldout_ternary['loss']:>15.4f}")
print(f"{'Perplexity':<20} {train_ternary['perplexity']:>15.1f} {heldout_ternary['perplexity']:>15.1f}")
print(f"{'Top-1 Accuracy':<20} {train_ternary['top1_accuracy']:>15.2f}% {heldout_ternary['top1_accuracy']:>15.2f}%")
print(f"{'Top-5 Accuracy':<20} {train_ternary['top5_accuracy']:>15.2f}% {heldout_ternary['top5_accuracy']:>15.2f}%")
print(f"{'Real Tokens':<20} {train_ternary['real_tokens']:>15,} {heldout_ternary['real_tokens']:>15,}")

# ============================================================
# EVALUATE FP32 5k (if available)
# ============================================================

if fp32_5k is not None:
    print(f"\n{'='*80}")
    print("EVALUATING FP32 5k")
    print("="*80)
    
    train_fp32 = evaluate_model(fp32_5k, tokenizer, train_data, "Train", max_samples=500)
    heldout_fp32 = evaluate_model(fp32_5k, tokenizer, heldout_data, "Held-out", max_samples=500)
    
    print(f"\nFP32 5k Results:")
    print(f"{'Metric':<20} {'Training':>15} {'Held-Out':>15}")
    print(f"{'-'*50}")
    print(f"{'Loss':<20} {train_fp32['loss']:>15.4f} {heldout_fp32['loss']:>15.4f}")
    print(f"{'Perplexity':<20} {train_fp32['perplexity']:>15.1f} {heldout_fp32['perplexity']:>15.1f}")
    print(f"{'Top-1 Accuracy':<20} {train_fp32['top1_accuracy']:>15.2f}% {heldout_fp32['top1_accuracy']:>15.2f}%")
    print(f"{'Top-5 Accuracy':<20} {train_fp32['top5_accuracy']:>15.2f}% {heldout_fp32['top5_accuracy']:>15.2f}%")
    print(f"{'Real Tokens':<20} {train_fp32['real_tokens']:>15,} {heldout_fp32['real_tokens']:>15,}")

# ============================================================
# COMPARISON
# ============================================================

print(f"\n{'='*80}")
print("COMPARISON: FP32 5k vs Ternary 50k")
print("="*80)

print(f"\n{'Metric':<20} {'Uniform':>12} {'FP32 5k':>15} {'Ternary 50k':>15}")
print(f"{'-'*65}")
print(f"{'Loss':<20} {uniform_loss.item():>12.4f} {train_fp32['loss'] if fp32_5k else 'N/A':>15} {train_ternary['loss']:>15.4f}")
print(f"{'Perplexity':<20} {torch.exp(uniform_loss).item():>12.0f} {train_fp32['perplexity'] if fp32_5k else 'N/A':>15.1f} {train_ternary['perplexity']:>15.1f}")
print(f"{'Top-1 Accuracy':<20} {'0.025%':>12} {train_fp32['top1_accuracy'] if fp32_5k else 'N/A':>15.2f}% {train_ternary['top1_accuracy']:>15.2f}%")

# ============================================================
# GENERATION TEST
# ============================================================

print(f"\n{'='*80}")
print("GENERATION TEST")
print("="*80)

def generate(model, tokenizer, prompt, max_new_tokens=30, device='cpu'):
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
            if next_token == tokenizer.eos_token_id:
                break
    return tokenizer.decode(generated)

prompts = ["Once upon a time", "There was a little", "The cat sat on"]

print(f"\nTernary 50k generations:")
for prompt in prompts:
    gen = generate(ternary_50k, tokenizer, prompt, max_new_tokens=20, device=device)
    print(f"  '{prompt}' → \"{gen}\"")

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "uniform_baseline": {"loss": uniform_loss.item(), "perplexity": torch.exp(uniform_loss).item()},
    "ternary_50k": {"training": train_ternary, "heldout": heldout_ternary},
}

if fp32_5k is not None:
    results["fp32_5k"] = {"training": train_fp32, "heldout": heldout_fp32}

with open("experiments/results/correct_evaluation_results.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*80}")
print("RESULTS SAVED: experiments/results/correct_evaluation_results.json")
print("="*80)
