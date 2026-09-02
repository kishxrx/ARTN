#!/usr/bin/env python3
"""Test sample generation."""

import torch
import sys
sys.path.insert(0, '.')

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
    import sys
    
    # Get which model to test from command line
    model_choice = sys.argv[1] if len(sys.argv) > 1 else 'current'
    
    if model_choice == 'current':
        test_samples(
            'outputs/beta1-0.5-batch8-lr001/final_model.pt',
            'outputs/artn-optimized/tokenizer.json',
            'CURRENT: beta1=0.5, LR=0.001, Batch=8 @ 10k (Loss 6.21)'
        )
    elif model_choice == 'healthy':
        test_samples(
            'outputs/artn-optimized-backup-50k/final_model/model.pt',
            'outputs/artn-optimized-backup-50k/tokenizer.json',
            'PREVIOUS 50k: beta1=0.9, LR=0.0003 (Loss 7.56, HEALTHY but gibberish)'
        )
    elif model_choice == 'collapsed':
        test_samples(
            'outputs/artn-optimized/final_model/model.pt',
            'outputs/artn-optimized/tokenizer.json',
            'COLLAPSED 50k: beta1=None (Loss 3.16, Entropy 8.2%, REPETITIVE)'
        )
