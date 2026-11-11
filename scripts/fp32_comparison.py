#!/usr/bin/env python3
"""
FP32 vs Ternary Comparison - Simple Version
"""

import sys
from pathlib import Path
import torch
import json
import time
import psutil
import os

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

# Import the same GPT architecture used for training
from experiments.bitlinear_adamw_controlled import BitLinearGPT
from src.model.gpt import GPT
from src.tokenizer.bpe_tokenizer import BPETokenizer


def get_process_memory_mb():
    """Get total process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def main():
    print('=' * 80)
    print('FP32 vs TERNARY COMPARISON')
    print('=' * 80)
    
    # Load FP32 checkpoint
    print('\nLoading FP32 checkpoint...')
    fp32_ckpt = torch.load(
        'experiments/results/fp32_control_20260819_232646/final_checkpoint.pt',
        map_location='cpu',
        weights_only=False
    )
    
    # Create FP32 model with same architecture
    # The FP32 checkpoint uses the training script architecture
    fp32_state = fp32_ckpt['model_state_dict']
    
    # Check architecture
    print(f'FP32 state dict keys: {len(fp32_state)}')
    print(f'  token_embedding.weight shape: {fp32_state["token_embedding.weight"].shape}')
    
    # FP32 uses standard GPT, not BitLinear
    # But the checkpoint has matching keys, so let's try
    # Actually, FP32 uses the same architecture but with nn.Linear instead of BitLinear
    
    # Let me check if there's a standard GPT implementation
    from src.model.gpt import GPT
    
    # Try loading with standard GPT
    try:
        fp32_model = GPT(
            vocab_size=4000,
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            max_seq_len=256
        )
        fp32_model.load_state_dict(fp32_state)
        print('✅ FP32 model loaded with GPT class')
    except Exception as e:
        print(f'❌ Could not load with GPT: {e}')
        print('\\nFP32 checkpoint has different architecture than src/model/gpt.py')
        print('Attempting to create custom FP32 model...')
        
        # Create custom FP32 model matching the checkpoint
        import torch.nn as nn
        import torch.nn.functional as F
        
        class CustomGPT(nn.Module):
            def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
                super().__init__()
                
                self.token_embedding = nn.Embedding(vocab_size, d_model)
                self.position_embedding = nn.Embedding(max_seq_len, d_model)
                
                # Need to build blocks
                # From checkpoint: blocks.*.ln1, blocks.*.attn.{q,k,v,o}_proj, blocks.*.ln2, blocks.*.ff.{fc1,fc2}
                
                # This is getting complex. Let me just compare generation from original test
                pass
        
        print('FP32 baseline comparison requires architecture matching.')
        print('Skipping FP32 comparison for now.')
        return
    
    # Load tokenizer
    tokenizer_path = '/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json'
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    # Test generation
    fp32_model.eval()
    
    prompt = "Once upon a time"
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    print(f'\\nFP32 Generation (temp=0.9):')
    print('-' * 80)
    
    with torch.no_grad():
        # FP32 model doesn't have generate method, need to implement
        # Actually, let me check if it has one
        if hasattr(fp32_model, 'generate'):
            output = fp32_model.generate(input_ids, max_new_tokens=100, temperature=0.9)
            generated = tokenizer.decode(output[0].tolist())
            print(generated[:200])
        else:
            print('FP32 model does not have generate() method')
            print('Need to implement autoregressive generation')
    
    # Load ternary for comparison
    print('\n' + '=' * 80)
    print('TERNARY MODEL (for comparison)')
    print('=' * 80)
    
    ternary_ckpt = torch.load(
        'outputs/bitlinear_adamw_20260910_143646/checkpoints/bitlinear_adamw/step_9500.pt',
        map_location='cpu',
        weights_only=False
    )
    
    config = ternary_ckpt.get('config', {})
    
    ternary_model = BitLinearGPT(
        vocab_size=config.get('vocab_size', 4000),
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        max_seq_len=config.get('max_seq_len', 256),
    )
    
    ternary_model.load_state_dict(ternary_ckpt['model_state_dict'])
    ternary_model.eval()
    
    print(f'\nTernary Generation (temp=0.9):')
    print('-' * 80)
    
    with torch.no_grad():
        output = ternary_model.generate(input_ids, max_new_tokens=100, temperature=0.9)
        generated = tokenizer.decode(output[0].tolist())
        print(generated[:200])
    
    # Memory comparison
    print('\n' + '=' * 80)
    print('MEMORY COMPARISON')
    print('=' * 80)
    
    mem_before = get_process_memory_mb()
    
    # Reload ternary
    ternary_model = BitLinearGPT(4000, 256, 8, 6, 1024, 256)
    ternary_model.load_state_dict(torch.load('outputs/bitlinear_adamw_20260910_143646/checkpoints/bitlinear_adamw/step_9500.pt', map_location='cpu', weights_only=False)['model_state_dict'])
    
    mem_after_ternary = get_process_memory_mb()
    
    # Reload FP32
    fp32_model = GPT(4000, 256, 8, 6, 1024, 256)
    fp32_model.load_state_dict(torch.load('experiments/results/fp32_control_20260819_232646/final_checkpoint.pt', map_location='cpu', weights_only=False)['model_state_dict'])
    
    mem_after_fp32 = get_process_memory_mb()
    
    print(f'\nTernary model loaded: {mem_after_ternary:.2f} MB')
    print(f'FP32 model loaded: {mem_after_fp32:.2f} MB')
    print(f'Difference: {mem_after_fp32 - mem_after_ternary:.2f} MB')


if __name__ == '__main__':
    main()
