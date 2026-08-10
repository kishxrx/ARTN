#!/usr/bin/env python3
"""
Interactive Model Tester
Test your trained model with custom prompts while training runs.

Usage:
    python test_model_interactive.py
    
This loads the LATEST checkpoint and lets you prompt the model.
"""
import torch
import sys
import glob
import os
sys.path.insert(0, '.')

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print('='*70)
print('INTERACTIVE MODEL TESTER')
print('='*70)

# Find latest checkpoint (sorted by step number)
checkpoint_dir = 'outputs/artn-optimized/checkpoints'
checkpoints = glob.glob(f'{checkpoint_dir}/checkpoint_step_*.pt')
if not checkpoints:
    print('\nERROR: No checkpoints found!')
    sys.exit(1)

# Sort by step number (extract number from filename)
def get_step(filepath):
    import re
    match = re.search(r'checkpoint_step_(\d+)\.pt', filepath)
    return int(match.group(1)) if match else 0

checkpoints = sorted(checkpoints, key=get_step)
latest_checkpoint = checkpoints[-1]

# Load model
print(f'\nLoading: {os.path.basename(latest_checkpoint)}')
model = BitLinearGPT(vocab_size=4000, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_seq_len=256)
ckpt = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f'Step: {ckpt["step"]}')
print(f'Loss: {ckpt.get("loss", "N/A"):.4f}')

# Load tokenizer
tokenizer = BPETokenizer.load('outputs/artn-optimized-backup-50k/tokenizer.json')

# Check entropy
prompt = 'Once upon a time'
input_ids = torch.tensor([tokenizer.encode(prompt)[:20]])
out = model(input_ids)
probs = torch.softmax(out['logits'][0, -1, :], dim=0)
entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
entropy_pct = (entropy / 8.294) * 100
print(f'Entropy: {entropy_pct:.1f}%')
print('='*70)

def generate(prompt, temperature=0.8, top_k=40, max_tokens=50):
    """Generate text from prompt."""
    input_ids = torch.tensor([tokenizer.encode(prompt)[:50]])
    generated = input_ids.clone()
    
    for _ in range(max_tokens):
        out = model(generated)
        logits = out['logits'][0, -1, :] / temperature
        
        # Top-k filtering
        topk_val, topk_idx = torch.topk(logits, top_k)
        probs = torch.softmax(topk_val, dim=0)
        choice = torch.multinomial(probs, 1)
        next_token = topk_idx[choice].unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)
    
    return tokenizer.decode(generated[0].tolist())

# Interactive loop
print('\n' + '='*70)
print('INTERACTIVE MODE')
print('='*70)
print('Commands:')
print('  - Type a prompt to generate text')
print('  - Type "test" to run 5 test prompts')
print('  - Type "quit" to exit')
print('='*70)

test_prompts = [
    'Once upon a time',
    'The little dog',
    'A girl named',
    'In the forest',
    'Once there was a'
]

while True:
    print()
    user_input = input('Prompt (or "test"/"quit"): ').strip()
    
    if user_input.lower() == 'quit':
        print('Goodbye!')
        break
    
    if user_input.lower() == 'test':
        print('\n' + '='*70)
        print('RUNNING 5 TEST PROMPTS')
        print('='*70)
        for i, prompt in enumerate(test_prompts):
            output = generate(prompt, temperature=0.8, top_k=40, max_tokens=40)
            print(f'\nTest {i+1}: "{prompt}"')
            print(f'Output: {output}')
        continue
    
    if user_input:
        output = generate(user_input, temperature=0.8, top_k=40, max_tokens=50)
        print(f'\nOutput: {output}')
