#!/usr/bin/env python3
"""Test sample generation for 50k training model."""

import torch
import sys
sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer
import json

def generate_samples(checkpoint_path, tokenizer_path, prompts):
    """Generate samples from model."""
    
    # Load tokenizer
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    # Model config (from training script)
    model = BitLinearGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n" + "="*60)
    print(f"SAMPLES FROM: {checkpoint_path}")
    print("="*60 + "\n")
    
    for prompt in prompts:
        # Tokenize prompt
        tokens = tokenizer.encode(prompt)
        tokens = tokens[:30]  # Truncate if too long
        
        # Generate
        with torch.no_grad():
            for _ in range(30):
                input_tensor = torch.tensor([tokens], dtype=torch.long)
                output = model(input_tensor)
                logits = output['logits']
                logits = logits[:, -1, :] / 0.8  # Temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                tokens.append(next_token)
        
        # Decode
        output = tokenizer.decode(tokens)
        print(f"Prompt: \"{prompt}\"")
        print(f"Output: \"{output}\"")
        print()

if __name__ == '__main__':
    prompts = [
        "Once upon a time",
        "The little girl",
        "A boy named",
        "There was a",
        "One day",
    ]
    
    # Test checkpoint at 50k (FINAL)
    generate_samples(
        '/root/agent-sandbox/ARTN/outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt',
        '/root/agent-sandbox/ARTN/outputs/artn-50k-best-config/tokenizer.json',
        prompts
    )
