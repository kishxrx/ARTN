#!/usr/bin/env python3
"""
Stage 8 Quality Investigation

1. Test temperature/top-k settings
2. Correct RAM measurement
3. Compare with FP32 baseline
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

from experiments.bitlinear_adamw_controlled import BitLinearGPT as BitLinearGPT_Ternary
from src.model.gpt import GPT as GPT_FP32
from src.tokenizer.bpe_tokenizer import BPETokenizer


def get_process_memory_mb():
    """Get total process memory in MB using psutil."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def load_ternary_model(checkpoint_path, tokenizer_path, device='cpu'):
    """Load BitLinear model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = BitLinearGPT_Ternary(
        vocab_size=config.get('vocab_size', 4000),
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        max_seq_len=config.get('max_seq_len', 256),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    return model, tokenizer


def load_fp32_model(checkpoint_path, tokenizer_path, device='cpu'):
    """Load FP32 model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # FP32 model config
    model = GPT_FP32(
        vocab_size=4000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    return model, tokenizer


def generate_with_memory(model, tokenizer, prompt, max_tokens=100, temperature=0.9, device='cpu'):
    """Generate with proper memory tracking."""
    # Memory before
    mem_before = get_process_memory_mb()
    
    # Encode
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_tokens, temperature)
    
    end_time = time.time()
    
    # Memory after (peak)
    mem_after = get_process_memory_mb()
    mem_peak = mem_after - mem_before
    
    # Decode
    generated = tokenizer.decode(output_ids[0].tolist())
    
    return {
        'generated': generated,
        'time': end_time - start_time,
        'mem_peak_mb': mem_peak,
        'tokens': max_tokens,
    }


def test_temperature_sweep(model, tokenizer, prompt, device='cpu'):
    """Test different temperatures."""
    temperatures = [0.5, 0.7, 0.9, 1.1, 1.3]
    
    results = []
    
    for temp in temperatures:
        print(f'  Testing temperature {temp}...')
        result = generate_with_memory(
            model, tokenizer, prompt,
            max_tokens=100,
            temperature=temp,
            device=device
        )
        result['temperature'] = temp
        results.append(result)
    
    return results


def main():
    print('=' * 80)
    print('STAGE 8 QUALITY INVESTIGATION')
    print('=' * 80)
    print()
    
    # =========================================================================
    # PART 1: Temperature Sweep on Ternary Model
    # =========================================================================
    print('PART 1: Temperature Sweep (Ternary Model)')
    print('=' * 80)
    
    ternary_ckpt = 'outputs/bitlinear_adamw_20260910_143646/checkpoints/bitlinear_adamw/step_9500.pt'
    tokenizer_path = '/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json'
    
    print('Loading ternary model...')
    model_ternary, tokenizer = load_ternary_model(ternary_ckpt, tokenizer_path)
    
    prompt = "Once upon a time"
    print(f'Prompt: "{prompt}"')
    print()
    
    temp_results = test_temperature_sweep(model_ternary, tokenizer, prompt)
    
    print('\nTemperature Results:')
    print('-' * 80)
    
    for r in temp_results:
        print(f'\nTemp {r["temperature"]}:')
        print(f'  Generated: {r["generated"][:100]}...')
        print(f'  Time: {r["time"]:.2f}s | Peak RAM: {r["mem_peak_mb"]:.2f} MB')
    
    # =========================================================================
    # PART 2: FP32 Baseline Comparison
    # =========================================================================
    print('\n' + '=' * 80)
    print('PART 2: FP32 Baseline Comparison')
    print('=' * 80)
    
    fp32_ckpt = 'experiments/results/fp32_control_20260819_232646/final_checkpoint.pt'
    
    try:
        print('Loading FP32 model...')
        model_fp32, tokenizer_fp32 = load_fp32_model(fp32_ckpt, tokenizer_path)
        
        print('\nGenerating with FP32 baseline (temp=0.9):')
        fp32_result = generate_with_memory(
            model_fp32, tokenizer_fp32, prompt,
            max_tokens=100,
            temperature=0.9
        )
        
        print(f'\nFP32 Generated:')
        print(f'  {fp32_result["generated"]}')
        print(f'  Time: {fp32_result["time"]:.2f}s | Peak RAM: {fp32_result["mem_peak_mb"]:.2f} MB')
        
        # Compare memory
        print('\n' + '=' * 80)
        print('MEMORY COMPARISON')
        print('=' * 80)
        
        # Load both and measure
        mem_before = get_process_memory_mb()
        model_ternary, _ = load_ternary_model(ternary_ckpt, tokenizer_path)
        mem_after_ternary = get_process_memory_mb()
        
        mem_before = get_process_memory_mb()
        model_fp32, _ = load_fp32_model(fp32_ckpt, tokenizer_path)
        mem_after_fp32 = get_process_memory_mb()
        
        print(f'\nTernary model load: {mem_after_ternary:.2f} MB')
        print(f'FP32 model load: {mem_after_fp32:.2f} MB')
        print(f'Difference: {mem_after_fp32 - mem_after_ternary:.2f} MB')
        
        fp32_available = True
        
    except Exception as e:
        print(f'\n❌ Could not load FP32 baseline: {e}')
        import traceback
        traceback.print_exc()
        fp32_available = False
    
    # =========================================================================
    # PART 3: Save Results
    # =========================================================================
    print('\n' + '=' * 80)
    print('SAVING RESULTS')
    print('=' * 80)
    
    results = {
        'temperature_sweep': temp_results,
        'fp32_comparison': {
            'available': fp32_available,
            'result': fp32_result if fp32_available else None,
        } if 'fp32_result' in locals() else {'available': False},
    }
    
    output_file = BASE_DIR / 'outputs' / 'stage8_quality_investigation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'✅ Results saved to: {output_file}')


if __name__ == '__main__':
    main()
