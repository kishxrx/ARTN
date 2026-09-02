#!/usr/bin/env python3
"""
20-STEP QUICK DIAGNOSTIC: Using CUSTOM AdaFactor Optimizer
"""
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT, BitLinear
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.adafactor import Adafactor

print("="*80)
print("20-STEP QUICK DIAGNOSTIC: CUSTOM AdaFactor")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

CHECKPOINT_50K = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")
TOKENIZER_PATH = "outputs/artn-optimized/tokenizer.json"
MAX_SEQ_LEN = 256
BATCH_SIZE = 4
DIAGNOSTIC_STEPS = 20
LR = 3e-5

# Load tokenizer and data
tokenizer = BPETokenizer.load(TOKENIZER_PATH)
dataset = load_dataset("roneneldan/TinyStories", split="train")
train_data = dataset.select(range(50000))
heldout_data = dataset.select(range(50000, 50100))

# Model config
model_config = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
}

# Evaluation function
def evaluate_model(model, tokenizer, dataset, max_samples=100):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
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
                total_tokens += 1
    
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

# Load checkpoint and model
print(f"\nLoading checkpoint...")
checkpoint = torch.load(CHECKPOINT_50K, map_location='cpu')

model = BitLinearGPT(**model_config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Model loaded")

# Baseline evaluation
print(f"\n{'='*80}")
print("BASELINE (Step 0)")
print("="*80)
baseline_loss = evaluate_model(model, tokenizer, heldout_data)
print(f"Held-out loss: {baseline_loss:.4f}")

# Setup optimizer
print(f"\n{'='*80}")
print("SETTING UP CUSTOM AdaFactor")
print("="*80)

optimizer = Adafactor(model.parameters(), lr=LR, weight_decay=0.01)
print(f"Created custom AdaFactor")

# Restore optimizer state
if 'optimizer_state_dict' in checkpoint:
    print(f"Restoring optimizer state...")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"✓ Optimizer state restored (50k steps of momentum/variance)")
else:
    print(f"✗ No optimizer state - using FRESH state")

# Run diagnostic
print(f"\n{'='*80}")
print(f"RUNNING {DIAGNOSTIC_STEPS} STEPS")
print("="*80)

results = []
model.train()
data_iter = iter(train_data)

for step in range(DIAGNOSTIC_STEPS):
    # Get batch
    batch_texts = []
    for _ in range(BATCH_SIZE):
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
    batch_padded = [t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in batch_tokens]
    
    input_ids = torch.tensor(batch_padded, dtype=torch.long, device=device)
    targets = input_ids.clone()
    targets[targets == tokenizer.pad_token_id] = -100
    
    # Forward
    output = model(input_ids, targets)
    loss = output["loss"]
    
    # Backward
    loss.backward()
    
    # Optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    # Log
    print(f"Step {step+1}/{DIAGNOSTIC_STEPS}: Train loss = {loss.item():.4f}")
    
    # Evaluate every 5 steps
    if (step + 1) % 5 == 0:
        heldout_loss = evaluate_model(model, tokenizer, heldout_data)
        print(f"  → Held-out loss: {heldout_loss:.4f} (baseline: {baseline_loss:.4f})")
        results.append({
            "step": step + 1,
            "train_loss": loss.item(),
            "heldout_loss": heldout_loss,
        })

# Final summary
print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)

if results:
    print(f"\n{'Step':<8} {'Train Loss':>12} {'Held-Out':>12}")
    print("-"*35)
    for r in results:
        print(f"{r['step']:<8} {r['train_loss']:>12.4f} {r['heldout_loss']:>12.4f}")
    
    # Calculate change
    final_loss = results[-1]['heldout_loss']
    change = final_loss - baseline_loss
    
    print(f"\nHeld-out loss change: {baseline_loss:.4f} → {final_loss:.4f} ({change:+.4f})")
    
    if abs(change) < 0.05:
        print(f"\n✓ STABLE! Optimizer state restoration WORKS!")
        print(f"  Degradation was caused by WRONG OPTIMIZER (AdamW instead of custom AdaFactor)")
    elif change > 0.1:
        print(f"\n⚠️ DEGRADATION STILL OCCURS")
    else:
        print(f"\n✓ MINIMAL CHANGE")

# Save results
output_path = Path("experiments/results/diagnostic_20step_adafactor.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {output_path}")
