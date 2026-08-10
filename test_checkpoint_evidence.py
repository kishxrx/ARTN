#!/usr/bin/env python3
"""Test checkpoint quality with REAL evidence."""
import torch
import math
import sys
sys.path.insert(0, '.')

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print('='*70)
print('EVIDENCE TEST - Testing Current Model')
print('='*70)

# Load model
model = BitLinearGPT(vocab_size=4000, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_seq_len=256)
ckpt = torch.load('outputs/artn-optimized/checkpoints/checkpoint_step_29000.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

tokenizer = BPETokenizer.load('outputs/artn-optimized-backup-50k/tokenizer.json')

print(f'\nCheckpoint: Step {ckpt["step"]}')
print(f'Loss: {ckpt["loss"]:.4f}')

# Test entropy
print('\n' + '='*70)
print('ENTROPY CHECK')
print('='*70)

prompt = 'Once upon a time'
input_ids = torch.tensor([tokenizer.encode(prompt)[:20]])
out = model(input_ids)
probs = torch.softmax(out['logits'][0, -1, :], dim=0)
entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
entropy_pct = (entropy / math.log(4000)) * 100

print(f'Entropy: {entropy_pct:.1f}%')
print(f'Status: {"HEALTHY" if entropy_pct > 30 else "COLLAPSED"}')

# Top 10
print('\n' + '='*70)
print('TOP 10 TOKENS')
print('='*70)
top10 = torch.topk(probs, 10)
for i, (p, idx) in enumerate(zip(top10.values, top10.indices)):
    tok = tokenizer.decode([idx.item()])
    print(f'{i+1}. "{tok}" - {p.item()*100:.2f}%')

# Generate 5 samples WITHOUT fixed seed
print('\n' + '='*70)
print('5 DIFFERENT SAMPLES (No Fixed Seed)')
print('='*70)

for n in range(5):
    gen = input_ids.clone()
    for _ in range(25):
        out = model(gen)
        logits = out['logits'][0, -1, :] / 0.85
        topk_val, topk_idx = torch.topk(logits, 40)
        probs = torch.softmax(topk_val, dim=0)
        choice = torch.multinomial(probs, 1)
        tok = topk_idx[choice].unsqueeze(0)
        gen = torch.cat([gen, tok], dim=1)
    txt = tokenizer.decode(gen[0].tolist())
    # Count unique words
    words = txt.split()
    unique = len(set(words))
    total = len(words)
    print(f'\nSample {n+1}:')
    print(f'{txt}')
    print(f'Words: {total}, Unique: {unique} ({unique/total*100:.1f}%)')

print('\n' + '='*70)
print('EVIDENCE COMPLETE')
print('='*70)
