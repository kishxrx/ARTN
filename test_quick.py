#!/usr/bin/env python3
"""ESSENTIAL GENERATION TEST - Quick and Focused"""
import torch
import sys
sys.path.insert(0, '.')
from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print('ESSENTIAL GENERATION TEST')
print('='*70)

# Load
print('Loading model...')
tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
model = BitLinearGPT(vocab_size=4000, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_seq_len=256)
ckpt = torch.load('outputs/artn-optimized/checkpoints/checkpoint_step_50000.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Loaded: Loss {ckpt["loss"]:.4f}')

def top_k_filter(logits, k):
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits

def generate(prompt, max_tokens=40, temp=0.9, top_k=50):
    torch.manual_seed(42)
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(generated)
            logits = outputs['logits'][0, -1, :] / temp
            logits = top_k_filter(logits.clone(), top_k)
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.view(1, 1)], dim=1)
    
    return tokenizer.decode(generated[0].tolist())

# Test 1: Entropy check
print('\n[1] ENTROPY CHECK')
prompt = 'Once upon a time'
input_ids = torch.tensor([tokenizer.encode(prompt)])
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs['logits'][0, -1, :]
    probs = torch.softmax(logits, dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    max_entropy = torch.log(torch.tensor(4000.0)).item()
    print(f'Entropy: {entropy:.4f} / {max_entropy:.4f} ({entropy/max_entropy*100:.1f}%)')
    
    top5 = torch.topk(probs, k=5)
    print('Top 5:')
    for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
        token = tokenizer.decode([idx.item()])
        print(f'  {i+1}. "{token}" ({prob.item()*100:.2f}%)')

# Test 2: Temperature sampling
print('\n[2] TEMPERATURE SAMPLING (Temp=0.9, Top-k=50)')
prompts = ['Once upon a time', 'The dog', 'A girl named', 'In the forest']
for p in prompts:
    text = generate(p, max_tokens=30, temp=0.9, top_k=50)
    print(f'{p}: {text}')

# Test 3: Different temperatures
print('\n[3] DIFFERENT TEMPERATURES (Top-k=50)')
prompt = 'Once upon a time'
for temp in [0.5, 0.7, 1.0]:
    text = generate(prompt, max_tokens=30, temp=temp, top_k=50)
    print(f'Temp {temp}: {text}')

# Test 4: Different top-k values
print('\n[4] DIFFERENT TOP-K VALUES (Temp=0.9)')
prompt = 'Once upon a time'
for k in [10, 30, 100]:
    text = generate(prompt, max_tokens=30, temp=0.9, top_k=k)
    print(f'Top-k {k}: {text}')

print('\nDONE')
