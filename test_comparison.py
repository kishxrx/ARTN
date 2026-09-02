#!/usr/bin/env python3
"""Test sample generation for multiple models."""

import torch
import sys
sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

def test_samples(checkpoint_path, tokenizer_path, label):
    print(f'\n=== {label} ===\n')
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = {
        'vocab_size': 4000,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'max_seq_len': 256,
    }
    model = BitLinearGPT(**config)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Load tokenizer
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    # Generate samples
    prompts = ['Once upon a time', 'The little girl', 'A boy named']
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)[:256]
        
        with torch.no_grad():
            for _ in range(30):
                input_tensor = torch.tensor([tokens], dtype=torch.long)
                logits = model(input_tensor)
                logits_tensor = logits['logits'] if isinstance(logits, dict) else logits
                next_token_logits = logits_tensor[0, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                tokens.append(next_token)
                if next_token == 0:
                    break
        
        output = tokenizer.decode(tokens)
        print(f'Prompt: {prompt}')
        print(f'Output: {output}')
        print()

if __name__ == '__main__':
    print("=" * 70)
    print("ARTN MODEL COMPARISON - SAMPLE GENERATION")
    print("=" * 70)
    
    # Test beta1=0.5, LR=0.001, Batch=8
    test_samples(
        '/root/agent-sandbox/ARTN/outputs/beta1-0.5-batch8-lr001/final_model.pt',
        '/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json',
        'NEW: beta1=0.5, LR=0.001, Batch=8 @ 10k'
    )
    
    # Test beta1=0.7, LR=0.001, Batch=8
    test_samples(
        '/root/agent-sandbox/ARTN/outputs/beta1-0.7-batch8-lr001/final_model.pt',
        '/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json',
        'NEW: beta1=0.7, LR=0.001, Batch=8 @ 10k'
    )
