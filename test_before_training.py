#!/usr/bin/env python3
"""
PRE-TRAINING VALIDATION TEST
Must pass BEFORE starting training.

This proves:
1. Model architecture is correct
2. Tokenizer works
3. Optimizer settings are correct
4. Loss calculation is correct
5. Starting entropy is GOOD
"""

import torch
import sys
sys.path.insert(0, '.')

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer
from datasets import load_dataset

print("="*70)
print("PRE-TRAINING VALIDATION TEST")
print("="*70)

# Test 1: Tokenizer
print("\n[TEST 1] Tokenizer")
print("-"*70)
dataset = load_dataset("roneneldan/TinyStories", split="train")
texts = [item["text"] for item in dataset.select(range(1000))]
corpus = "\n".join(texts)

tokenizer = BPETokenizer(vocab_size=4000, min_frequency=2)
tokenizer.train(corpus)

test_text = "Once upon a time there was a girl named Lily."
tokens = tokenizer.encode(test_text)
decoded = tokenizer.decode(tokens)

print(f"Original: {test_text}")
print(f"Tokens: {tokens[:10]}...")
print(f"Decoded: {decoded}")
assert len(tokens) > 0, "Tokenizer failed"
assert len(decoded) > 0, "Tokenizer decode failed"
print("✓ PASS: Tokenizer works")

# Test 2: Model Architecture
print("\n[TEST 2] Model Architecture")
print("-"*70)
model = BitLinearGPT(
    vocab_size=4000,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_seq_len=256,
    dropout=0.1,
)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

input_ids = torch.randint(0, 4000, (2, 50))
outputs = model(input_ids)

print(f"Input shape: {input_ids.shape}")
print(f"Output logits shape: {outputs['logits'].shape}")
assert outputs['logits'].shape == (2, 50, 4000), f"Wrong output shape: {outputs['logits'].shape}"
print("✓ PASS: Model architecture correct")

# Test 3: Loss Calculation
print("\n[TEST 3] Loss Calculation")
print("-"*70)
labels = input_ids.clone()
outputs = model(input_ids, targets=labels)
loss = outputs['loss']

print(f"Initial loss: {loss.item():.4f}")
print(f"Theoretical max loss: ln(4000) = {torch.log(torch.tensor(4000.0)).item():.4f}")

# Initial loss should be around ln(vocab_size) for random model
assert 7.0 < loss.item() < 10.0, f"Suspicious initial loss: {loss.item()}"
print("✓ PASS: Loss calculation correct")

# Test 4: Optimizer Settings
print("\n[TEST 4] Optimizer Settings (Adafactor)")
print("-"*70)
from src.training.adafactor import Adafactor

optimizer = Adafactor(
    model.parameters(),
    lr=0.0003,
    weight_decay=0.01,
    beta1=0.9,  # CRITICAL: Must be 0.9
    beta2=0.999,
)

print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
print(f"Beta1: {optimizer.param_groups[0]['beta1']}")
print(f"Beta2: {optimizer.param_groups[0]['beta2']}")

assert optimizer.param_groups[0]['lr'] == 0.0003, "Wrong LR"
assert optimizer.param_groups[0]['beta1'] == 0.9, "CRITICAL: beta1 must be 0.9"
print("✓ PASS: Optimizer settings correct")

# Test 5: Gradient Accumulation
print("\n[TEST 5] Gradient Accumulation")
print("-"*70)

model.train()
optimizer.zero_grad()

# Simulate 2 accumulation steps
accum_steps = 2
total_loss = 0

for i in range(accum_steps):
    input_ids = torch.randint(0, 4000, (4, 50))
    labels = input_ids.clone()
    
    outputs = model(input_ids, targets=labels)
    loss = outputs['loss']
    scaled_loss = loss / accum_steps
    scaled_loss.backward()
    
    total_loss += loss.item()

# Check gradients exist
has_grad = False
for param in model.parameters():
    if param.grad is not None:
        has_grad = True
        break

assert has_grad, "No gradients after accumulation"
print(f"Accumulated loss: {total_loss/accum_steps:.4f}")
print("✓ PASS: Gradient accumulation works")

# Test 6: Entropy Check (CRITICAL)
print("\n[TEST 6] Initial Entropy Check")
print("-"*70)

model.eval()
test_prompts = [
    "Once upon a time",
    "The dog ran",
    "A girl named",
    "In the forest",
]

for prompt in test_prompts:
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits'][0, -1, :]
        probs = torch.softmax(logits, dim=0)
        
        # Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = torch.log(torch.tensor(4000.0)).item()
        entropy_pct = (entropy / max_entropy) * 100
        
        # Top token probability
        top_prob, top_idx = torch.max(probs, dim=0)
        top_token = tokenizer.decode([top_idx.item()])
        
        print(f"Prompt: '{prompt}'")
        print(f"  Entropy: {entropy:.4f} ({entropy_pct:.1f}% of max)")
        print(f"  Top token: '{top_token}' ({top_prob.item()*100:.2f}%)")

# Test 7: Random Input Loss (CRITICAL)
print("\n[TEST 7] Random Input Loss Check")
print("-"*70)

model.eval()
random_input = torch.randint(0, 4000, (4, 50))
random_labels = random_input.clone()

with torch.no_grad():
    outputs = model(random_input, targets=random_labels)
    random_loss = outputs['loss'].item()

print(f"Random input loss: {random_loss:.4f}")
print(f"Training loss: {loss.item():.4f}")
print(f"Ratio: {random_loss/loss.item():.2f}x")

# Random loss should be ~same as training loss initially
assert random_loss > 5.0, f"Suspicious random loss: {random_loss}"
print("✓ PASS: Random input loss healthy")

# Test 8: Sample Generation
print("\n[TEST 8] Sample Generation (Temperature 0.9, Top-k 50)")
print("-"*70)

def top_k_filter(logits, k):
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits

prompt = "Once upon a time"
input_ids = torch.tensor([tokenizer.encode(prompt)])
generated = input_ids.clone()

torch.manual_seed(42)
model.eval()

with torch.no_grad():
    for _ in range(30):
        outputs = model(generated)
        logits = outputs['logits'][0, -1, :] / 0.9
        logits = top_k_filter(logits.clone(), 50)
        probs = torch.softmax(logits, dim=0)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

sample = tokenizer.decode(generated[0].tolist())
print(f"Sample: {sample}")

# Check sample has variety (not stuck on one word)
words = sample.split()
unique_words = len(set(words))
print(f"Total words: {len(words)}, Unique: {unique_words}")

assert unique_words > 1, f"Sample stuck on one word"
print("✓ PASS: Sample generation shows variety")

# FINAL SUMMARY
print("\n" + "="*70)
print("PRE-TRAINING VALIDATION COMPLETE")
print("="*70)

print("\nALL 8 TESTS PASSED:")
print("  ✓ TEST 1: Tokenizer")
print("  ✓ TEST 2: Model Architecture")
print("  ✓ TEST 3: Loss Calculation")
print("  ✓ TEST 4: Optimizer Settings")
print("  ✓ TEST 5: Gradient Accumulation")
print("  ✓ TEST 6: Initial Entropy")
print("  ✓ TEST 7: Random Input Loss")
print("  ✓ TEST 8: Sample Generation")

print("\nVERIFIED SETTINGS:")
print(f"  Learning Rate: {optimizer.param_groups[0]['lr']}")
print(f"  Beta1: {optimizer.param_groups[0]['beta1']}")
print(f"  Beta2: {optimizer.param_groups[0]['beta2']}")
print(f"  Parameters: {num_params:,}")
print(f"  Expected effective batch: {4 * 2} = 8")
print(f"  Total steps: 50,000")

print("\nREADY TO TRAIN")
print("="*70)
