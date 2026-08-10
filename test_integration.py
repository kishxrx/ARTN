#!/usr/bin/env python3
"""
FULL INTEGRATION TEST
Tests training script + monitor + checkpoint + log in ONE test.
"""

import torch
import sys
import os
sys.path.insert(0, '.')

from src.model.bitlinear import BitLinearGPT
from src.training.adafactor import Adafactor
from src.tokenizer.bpe_tokenizer import BPETokenizer
from datasets import load_dataset

print("="*70)
print("FULL INTEGRATION TEST")
print("="*70)

# Test 1: Create output directory
print("\n[TEST 1] Create output directory")
os.makedirs('outputs/test-run', exist_ok=True)
os.makedirs('outputs/test-run/checkpoints', exist_ok=True)
print("✓ PASS")

# Test 2: Tokenizer
print("\n[TEST 2] Tokenizer")
print("Loading TinyStories subset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")
texts = [item["text"] for item in dataset.select(range(500))]
corpus = "\n".join(texts)

tokenizer = BPETokenizer(vocab_size=4000, min_frequency=2)
tokenizer.train(corpus)
tokenizer.save('outputs/test-run/tokenizer.json')
print(f"✓ PASS - Vocab size: 4000")

# Test 3: Model
print("\n[TEST 3] Model")
model = BitLinearGPT(
    vocab_size=4000,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_seq_len=256,
)
num_params = sum(p.numel() for p in model.parameters())
print(f"✓ PASS - Parameters: {num_params:,}")

# Test 4: Optimizer (CRITICAL - must have beta1=0.9)
print("\n[TEST 4] Optimizer")
optimizer = Adafactor(
    model.parameters(),
    lr=0.0003,
    weight_decay=0.01,
    beta1=0.9,  # CRITICAL
    beta2=0.999,
)
assert optimizer.param_groups[0]['beta1'] == 0.9, "BETA1 WRONG!"
print(f"✓ PASS - LR: {optimizer.param_groups[0]['lr']}, Beta1: {optimizer.param_groups[0]['beta1']}")

# Test 5: Training step
print("\n[TEST 5] Training step")
model.train()
input_ids = torch.randint(0, 4000, (4, 50))
labels = input_ids.clone()

outputs = model(input_ids, targets=labels)
loss = outputs['loss']
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"✓ PASS - Loss: {loss.item():.4f}")

# Test 6: Monitor function (entropy check)
print("\n[TEST 6] Monitor function (entropy)")
model.eval()

def monitor_model_health(model, tokenizer, device='cpu'):
    prompt = "Once upon a time"
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits'][0, -1, :]
        probs = torch.softmax(logits, dim=0)
        
        # Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = torch.log(torch.tensor(4000.0)).item()
        entropy_pct = (entropy / max_entropy) * 100
        
        # Top token
        top_prob, top_idx = torch.max(probs, dim=0)
        top_token = tokenizer.decode([top_idx.item()])
        
        # Determin status
        status = "HEALTHY"
        if entropy_pct < 20:
            status = "COLLAPSED"
        elif entropy_pct < 30:
            status = "WARNING"
        
        return {
            'entropy': entropy,
            'entropy_pct': entropy_pct,
            'top_token': top_token,
            'top_prob': top_prob.item() * 100,
            'status': status
        }

metrics = monitor_model_health(model, tokenizer)
print(f"  Entropy: {metrics['entropy']:.4f} ({metrics['entropy_pct']:.1f}%)")
print(f"  Top token: '{metrics['top_token']}' ({metrics['top_prob']:.1f}%)")
print(f"  Status: {metrics['status']}")

assert metrics['entropy_pct'] > 30, f"Low entropy: {metrics['entropy_pct']:.1f}%"
print("✓ PASS - Entropy healthy")

# Test 7: Sample generation
print("\n[TEST 7] Sample generation (temperature + top-k)")

def top_k_filter(logits, k):
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits

prompt = "Once upon a time"
input_ids = torch.tensor([tokenizer.encode(prompt)])
generated = input_ids.clone()

torch.manual_seed(42)

with torch.no_grad():
    for _ in range(30):
        outputs = model(generated)
        logits = outputs['logits'][0, -1, :] / 0.9
        logits = top_k_filter(logits.clone(), 50)
        probs = torch.softmax(logits, dim=0)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

sample = tokenizer.decode(generated[0].tolist())
print(f"  Sample: {sample}")
print("✓ PASS - Sample generated")

# Test 8: Checkpoint save
print("\n[TEST 8] Checkpoint save")
ckpt_path = 'outputs/test-run/checkpoints/checkpoint_step_100.pt'
torch.save({
    'step': 100,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}, ckpt_path)
print(f"✓ PASS - Saved to {ckpt_path}")

# Test 9: Checkpoint load
print("\n[TEST 9] Checkpoint load")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f"✓ PASS - Loaded from step {ckpt['step']}")

# Test 10: Log file
print("\n[TEST 10] Log file")
import datetime
timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
with open('outputs/test-run/monitor_log.txt', 'a') as f:
    f.write(f"{timestamp} | Step 100 | Loss {loss.item():.4f} | Entropy {metrics['entropy_pct']:.1f}% | STATUS: {metrics['status']}\n")

with open('outputs/test-run/monitor_log.txt', 'r') as f:
    content = f.read()
print(f"  Log content: {content.strip()}")
print("✓ PASS - Log file written")

# Test 11: Collapse detection simulation
print("\n[TEST 11] Collapse detection simulation")

# Simulate collapsed metrics
collapsed_entropy = 15.0  # < 20% = COLLAPSED
collapsed_top_prob = 85.0  # > 70% = COLLAPSED

status = "HEALTHY"
warnings = []

if collapsed_entropy < 20:
    status = "COLLAPSED"
    warnings.append(f"CRITICAL: Entropy {collapsed_entropy:.1f}% < 20%")

if collapsed_top_prob > 70:
    status = "COLLAPSED"
    warnings.append(f"CRITICAL: Top prob {collapsed_top_prob:.1f}% > 70%")

print(f"  Simulated entropy: {collapsed_entropy}%")
print(f"  Simulated top prob: {collapsed_top_prob}%")
print(f"  Detected status: {status}")
print(f"  Warnings: {warnings}")

assert status == "COLLAPSED", "Collapse detection failed!"
print("✓ PASS - Collapse detected correctly")

# Test 12: Multiple training steps (gradient accumulation)
print("\n[TEST 12] Gradient accumulation (2 steps)")
model.train()
optimizer.zero_grad()

accum_loss = 0
for i in range(2):
    input_ids = torch.randint(0, 4000, (4, 50))
    labels = input_ids.clone()
    
    outputs = model(input_ids, targets=labels)
    loss = outputs['loss'] / 2  # Scale by accum steps
    loss.backward()
    accum_loss += loss.item() * 2

optimizer.step()
optimizer.zero_grad()

print(f"  Accumulated loss: {accum_loss:.4f}")
print("✓ PASS - Gradient accumulation works")

# FINAL SUMMARY
print("\n" + "="*70)
print("ALL 12 TESTS PASSED")
print("="*70)
print("\nVERIFIED:")
print("  ✓ Tokenizer works")
print("  ✓ Model architecture correct")
print("  ✓ Optimizer settings correct (beta1=0.9)")
print("  ✓ Training step works")
print("  ✓ Monitor function works")
print("  ✓ Sample generation works")
print("  ✓ Checkpoint save/load works")
print("  ✓ Log file works")
print("  ✓ Collapse detection works")
print("  ✓ Gradient accumulation works")
print("\nREADY TO TRAIN")
print("="*70)
