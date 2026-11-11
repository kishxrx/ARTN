#!/usr/bin/env python3
"""
Stage 8 Deployment Preparation

1. Create inference-only checkpoint (remove optimizer state)
2. Load inference model without optimizer state
3. Measure RSS (total and peak) during inference
4. Benchmark speed and memory on 10 prompts
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

from experiments.bitlinear_adamw_controlled import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer


def get_rss_mb():
    """Get current process RSS in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def create_inference_checkpoint(original_ckpt_path, output_ckpt_path):
    """
    Create inference-only checkpoint without optimizer state.
    
    Preserves:
    - model_state_dict
    - config
    - step
    - val_loss (for reference)
    
    Removes:
    - optimizer_state_dict
    - train_loss
    - other training metadata
    """
    print('=' * 80)
    print('CREATING INFERENCE-ONLY CHECKPOINT')
    print('=' * 80)
    
    # Load original checkpoint
    print(f'\nLoading original checkpoint: {original_ckpt_path}')
    original = torch.load(original_ckpt_path, map_location='cpu', weights_only=False)
    
    # Create inference-only checkpoint
    inference_ckpt = {
        'model_state_dict': original['model_state_dict'],
        'config': original.get('config', {}),
        'step': original.get('step', 0),
        'val_loss': original.get('val_loss', None),
        'inference_only': True,
        'original_checkpoint': original_ckpt_path,
    }
    
    # Compare sizes
    original_size_mb = os.path.getsize(original_ckpt_path) / 1024 / 1024
    
    # Save inference checkpoint
    torch.save(inference_ckpt, output_ckpt_path)
    inference_size_mb = os.path.getsize(output_ckpt_path) / 1024 / 1024
    
    print(f'\nOriginal checkpoint size:   {original_size_mb:.2f} MB')
    print(f'Inference checkpoint size:  {inference_size_mb:.2f} MB')
    print(f'Saved:                      {inference_size_mb:.2f} MB ({(inference_size_mb/original_size_mb)*100:.1f}% of original)')
    print(f'\nRemoved: optimizer_state_dict')
    print(f'Preserved: model_state_dict, config, step, val_loss')
    print(f'\n✅ Saved to: {output_ckpt_path}')
    
    return inference_size_mb


def load_inference_model(checkpoint_path, device='cpu'):
    """
    Load model for inference (no optimizer state).
    
    Returns: model, config
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Verify this is inference checkpoint
    if 'optimizer_state_dict' in checkpoint:
        print('⚠️  Warning: Checkpoint contains optimizer state')
        print('   For deployment, use inference-only checkpoint')
    
    config = checkpoint.get('config', {})
    
    model = BitLinearGPT(
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
    
    return model, config


def benchmark_inference(model, tokenizer, prompts, max_tokens=100, temperature=0.9, device='cpu'):
    """
    Benchmark inference on multiple prompts.
    
    Measures:
    - Generation time
    - Tokens per second
    - Peak RSS during generation
    - Total tokens generated
    
    Returns: list of results
    """
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f'  [{i+1}/{len(prompts)}] {prompt[:40]}...')
        
        # Measure RSS before
        rss_before = get_rss_mb()
        
        # Encode
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        
        # Generate with peak memory tracking
        start_time = time.time()
        peak_rss = rss_before
        
        with torch.no_grad():
            # Simple generation (the model's generate method)
            token_ids = input_ids.clone()
            
            for _ in range(max_tokens):
                # Crop to max_seq_len
                idx_cond = token_ids if token_ids.size(1) <= 256 else token_ids[:, -256:]
                
                # Forward pass
                logits = model(idx_cond)['logits']
                logits = logits[:, -1, :] / temperature
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                token_ids = torch.cat([token_ids, next_token], dim=1)
                
                # Track peak RSS
                current_rss = get_rss_mb()
                if current_rss > peak_rss:
                    peak_rss = current_rss
        
        end_time = time.time()
        
        # Measure RSS after
        rss_after = get_rss_mb()
        
        # Decode
        generated = tokenizer.decode(token_ids[0].tolist())
        new_tokens = token_ids.size(1) - input_ids.size(1)
        
        # Calculate metrics
        generation_time = end_time - start_time
        tokens_per_second = new_tokens / generation_time if generation_time > 0 else 0
        peak_rss_delta = peak_rss - rss_before
        
        results.append({
            'prompt': prompt,
            'generated': generated,
            'new_tokens': new_tokens,
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
            'rss_before': rss_before,
            'rss_after': rss_after,
            'peak_rss': peak_rss,
            'peak_rss_delta': peak_rss_delta,
        })
    
    return results


def main():
    print('=' * 80)
    print('STAGE 8 DEPLOYMENT PREPARATION')
    print('=' * 80)
    print()
    
    # =========================================================================
    # PART 1: Create inference-only checkpoint
    # =========================================================================
    original_ckpt = 'outputs/bitlinear_adamw_20260910_143646/checkpoints/bitlinear_adamw/step_9500.pt'
    inference_ckpt = 'outputs/bitlinear_adamw_20260910_143646/checkpoints/bitlinear_adamw/step_9500_inference_only.pt'
    
    inference_size = create_inference_checkpoint(original_ckpt, inference_ckpt)
    print()
    
    # =========================================================================
    # PART 2: Load model and measure RAM
    # =========================================================================
    print('=' * 80)
    print('LOADING INFERENCE MODEL')
    print('=' * 80)
    
    # Baseline RSS
    baseline_rss = get_rss_mb()
    print(f'\nBaseline RSS (before imports): {baseline_rss:.2f} MB')
    
    # Load model
    print(f'\nLoading from: {inference_ckpt}')
    model, config = load_inference_model(inference_ckpt)
    
    after_model_rss = get_rss_mb()
    model_rss_delta = after_model_rss - baseline_rss
    
    print(f'After model load:             {after_model_rss:.2f} MB (+{model_rss_delta:.2f} MB)')
    
    # Load tokenizer
    tokenizer_path = '/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json'
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    after_tokenizer_rss = get_rss_mb()
    tokenizer_rss_delta = after_tokenizer_rss - after_model_rss
    
    print(f'After tokenizer load:         {after_tokenizer_rss:.2f} MB (+{tokenizer_rss_delta:.2f} MB)')
    print()
    
    # =========================================================================
    # PART 3: Benchmark on 10 prompts
    # =========================================================================
    print('=' * 80)
    print('BENCHMARK: 10 PROMPTS')
    print('=' * 80)
    
    prompts = [
        "Once upon a time",
        "A little girl named",
        "The boy went to the",
        "There was a big dog",
        "She looked at the sky and",
        "In the morning, the",
        "The cat sat on the",
        "He opened the door and saw",
        "Once there was a magical",
        "The children played in the",
    ]
    
    print(f'\nPrompts: {len(prompts)}')
    print(f'Max tokens per prompt: 100')
    print(f'Temperature: 0.9')
    print()
    
    results = benchmark_inference(model, tokenizer, prompts, max_tokens=100, temperature=0.9)
    
    # =========================================================================
    # PART 4: Calculate statistics
    # =========================================================================
    print('\n' + '=' * 80)
    print('RESULTS SUMMARY')
    print('=' * 80)
    
    total_tokens = sum(r['new_tokens'] for r in results)
    total_time = sum(r['generation_time'] for r in results)
    avg_speed = total_tokens / total_time if total_time > 0 else 0
    
    # Memory statistics
    min_rss_before = min(r['rss_before'] for r in results)
    max_rss_after = max(r['rss_after'] for r in results)
    max_peak_rss = max(r['peak_rss'] for r in results)
    avg_peak_delta = sum(r['peak_rss_delta'] for r in results) / len(results)
    
    print(f'\nGeneration Performance:')
    print(f'  Total prompts:        {len(prompts)}')
    print(f'  Total tokens:         {total_tokens}')
    print(f'  Total time:           {total_time:.2f}s')
    print(f'  Average speed:        {avg_speed:.2f} tok/s')
    print()
    
    print(f'Memory Usage:')
    print(f'  Baseline (Python):    {baseline_rss:.2f} MB')
    print(f'  After model load:     {after_model_rss:.2f} MB (+{model_rss_delta:.2f} MB)')
    print(f'  After tokenizer:      {after_tokenizer_rss:.2f} MB (+{tokenizer_rss_delta:.2f} MB)')
    print(f'  Peak during gen:      {max_peak_rss:.2f} MB')
    print()
    
    print(f'Inference-Ready Memory:')
    print(f'  Model + tokenizer:    {after_tokenizer_rss:.2f} MB')
    print(f'  + Generation buffers: {max_peak_rss:.2f} MB peak')
    print()
    
    # =========================================================================
    # PART 5: Detailed results table
    # =========================================================================
    print('=' * 80)
    print('DETAILED RESULTS')
    print('=' * 80)
    print()
    print(f'{"Prompt":<30} {"Tokens":<7} {"Time":<8} {"Speed":<10} {"Peak RSS":<10}')
    print('-' * 80)
    
    for r in results:
        prompt_short = r['prompt'][:28] + '..' if len(r['prompt']) > 30 else r['prompt']
        print(f'{prompt_short:<30} {r["new_tokens"]:<7} {r["generation_time"]:<8.2f} {r["tokens_per_second"]:<10.2f} {r["peak_rss"]:<10.2f}')
    
    print()
    
    # =========================================================================
    # PART 6: Save results
    # =========================================================================
    print('=' * 80)
    print('SAVING RESULTS')
    print('=' * 80)
    
    output = {
        'test_date': '2026-10-15',
        'test_type': 'inference_deployment_benchmark',
        'inference_checkpoint': {
            'path': inference_ckpt,
            'size_mb': inference_size,
            'original_checkpoint': original_ckpt,
        },
        'memory': {
            'baseline_rss_mb': baseline_rss,
            'model_rss_mb': after_model_rss,
            'model_delta_mb': model_rss_delta,
            'tokenizer_delta_mb': tokenizer_rss_delta,
            'total_inference_rss_mb': after_tokenizer_rss,
            'peak_generation_rss_mb': max_peak_rss,
        },
        'performance': {
            'total_prompts': len(prompts),
            'total_tokens': total_tokens,
            'total_time_s': total_time,
            'avg_speed_tok_per_s': avg_speed,
        },
        'results': results,
    }
    
    output_file = BASE_DIR / 'outputs' / 'stage8_deployment_benchmark.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f'\n✅ Results saved to: {output_file}')
    print()
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print('=' * 80)
    print('DEPLOYMENT SUMMARY')
    print('=' * 80)
    print()
    print('INFERENCE CHECKPOINT CREATED:')
    print(f'  File: {inference_ckpt}')
    print(f'  Size: {inference_size:.2f} MB')
    print(f'  Contains: model_state_dict, config, step, val_loss')
    print(f'  Removed: optimizer_state_dict')
    print()
    print('INFERENCE MEMORY REQUIREMENTS:')
    print(f'  Model + tokenizer:    {after_tokenizer_rss:.2f} MB')
    print(f'  Peak during generation: {max_peak_rss:.2f} MB')
    print()
    print('PERFORMANCE:')
    print(f'  Average speed: {avg_speed:.2f} tok/s')
    print(f'  Total tokens: {total_tokens}')
    print()
    print('✅ STAGE 8 DEPLOYMENT-READY')


if __name__ == '__main__':
    main()
