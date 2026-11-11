#!/usr/bin/env python3
"""
RAM Measurement Verification for Stage 8

Measures total process RSS at each stage:
1. Baseline (before imports)
2. After imports
3. After model loading
4. During generation
5. After generation

Uses psutil to measure actual RSS (Resident Set Size).
"""

import sys
from pathlib import Path
import psutil
import os

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

def get_rss_mb():
    """Get current process RSS in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_all_memory_info():
    """Get detailed memory info."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        'shared_mb': mem_info.shared / 1024 / 1024 if hasattr(mem_info, 'shared') else 0,
    }


print('=' * 80)
print('RAM MEASUREMENT VERIFICATION')
print('=' * 80)
print()

# ===========================================================================
# STAGE 1: Baseline (minimal Python)
# ===========================================================================
print('STAGE 1: Baseline (minimal Python, no imports)')
print('-' * 80)
baseline_mem = get_all_memory_info()
print(f'RSS: {baseline_mem["rss_mb"]:.2f} MB')
print(f'VMS: {baseline_mem["vms_mb"]:.2f} MB')
print()

# ===========================================================================
# STAGE 2: After PyTorch import
# ===========================================================================
print('STAGE 2: After PyTorch import')
print('-' * 80)
import torch
after_torch_mem = get_all_memory_info()
print(f'RSS: {after_torch_mem["rss_mb"]:.2f} MB (delta: +{after_torch_mem["rss_mb"] - baseline_mem["rss_mb"]:.2f} MB)')
print()

# ===========================================================================
# STAGE 3: After loading model
# ===========================================================================
print('STAGE 3: After loading BitLinear model')
print('-' * 80)

from experiments.bitlinear_adamw_controlled import BitLinearGPT

checkpoint_path = 'outputs/bitlinear_adamw_20260910_143646/checkpoints/bitlinear_adamw/step_9500.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

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
model.eval()

after_model_mem = get_all_memory_info()
print(f'RSS: {after_model_mem["rss_mb"]:.2f} MB (delta: +{after_model_mem["rss_mb"] - after_torch_mem["rss_mb"]:.2f} MB)')
print()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {total_params:,}')

# Calculate expected FP32 size
expected_fp32_mb = (total_params * 4) / 1024 / 1024  # 4 bytes per FP32 param
print(f'Expected FP32 size: {expected_fp32_mb:.2f} MB')
print(f'Actual delta: {after_model_mem["rss_mb"] - after_torch_mem["rss_mb"]:.2f} MB')

# Check if checkpoint has optimizer state
if 'optimizer_state_dict' in checkpoint:
    opt_state = checkpoint['optimizer_state_dict']
    print(f'\nOptimizer state loaded: YES')
    # Try to estimate optimizer state size
    if 'state' in opt_state:
        num_params_with_state = len(opt_state['state'])
        print(f'  Parameters with optimizer state: {num_params_with_state}')
else:
    print(f'\nOptimizer state loaded: NO')

print()

# ===========================================================================
# STAGE 4: Load tokenizer
# ===========================================================================
print('STAGE 4: After loading tokenizer')
print('-' * 80)

from src.tokenizer.bpe_tokenizer import BPETokenizer
tokenizer = BPETokenizer.load('/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json')

after_tokenizer_mem = get_all_memory_info()
print(f'RSS: {after_tokenizer_mem["rss_mb"]:.2f} MB (delta: +{after_tokenizer_mem["rss_mb"] - after_model_mem["rss_mb"]:.2f} MB)')
print()

# ===========================================================================
# STAGE 5: During generation
# ===========================================================================
print('STAGE 5: During generation')
print('-' * 80)

prompt = "Once upon a time"
input_ids = torch.tensor([tokenizer.encode(prompt)])

# Measure before generation
before_gen_mem = get_all_memory_info()

with torch.no_grad():
    # Generate and measure peak
    output_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.9)
    
    # Measure during generation
    during_gen_mem = get_all_memory_info()

after_gen_mem = get_all_memory_info()

print(f'Before generation: {before_gen_mem["rss_mb"]:.2f} MB')
print(f'Peak during gen:   {during_gen_mem["rss_mb"]:.2f} MB (delta: +{during_gen_mem["rss_mb"] - before_gen_mem["rss_mb"]:.2f} MB)')
print(f'After generation:  {after_gen_mem["rss_mb"]:.2f} MB')
print()

# ===========================================================================
# STAGE 6: Memory breakdown
# ===========================================================================
print('STAGE 6: Memory Breakdown')
print('=' * 80)
print()

print('MEMORY DELTAS:')
print(f'  Baseline (Python):      {baseline_mem["rss_mb"]:.2f} MB')
print(f'  + PyTorch import:       {after_torch_mem["rss_mb"]:.2f} MB (+{after_torch_mem["rss_mb"] - baseline_mem["rss_mb"]:.2f} MB)')
print(f'  + Model load:           {after_model_mem["rss_mb"]:.2f} MB (+{after_model_mem["rss_mb"] - after_torch_mem["rss_mb"]:.2f} MB)')
print(f'  + Tokenizer:            {after_tokenizer_mem["rss_mb"]:.2f} MB (+{after_tokenizer_mem["rss_mb"] - after_model_mem["rss_mb"]:.2f} MB)')
print(f'  + Generation buffers:   {after_gen_mem["rss_mb"]:.2f} MB (+{after_gen_mem["rss_mb"] - after_tokenizer_mem["rss_mb"]:.2f} MB)')
print()

print('TOTAL MEMORY USAGE:')
print(f'  Final RSS: {after_gen_mem["rss_mb"]:.2f} MB')
print()

print('EXPECTED vs ACTUAL:')
print(f'  Model params (FP32):    {expected_fp32_mb:.2f} MB expected')
print(f'  Actual model delta:     {after_model_mem["rss_mb"] - after_torch_mem["rss_mb"]:.2f} MB')
print(f'  Difference:             {(after_model_mem["rss_mb"] - after_torch_mem["rss_mb"]) - expected_fp32_mb:.2f} MB')
print()

# ===========================================================================
# ANALYSIS
# ===========================================================================
print('ANALYSIS:')
print('-' * 80)

model_mem_delta = after_model_mem["rss_mb"] - after_torch_mem["rss_mb"]

if abs(model_mem_delta - expected_fp32_mb) < 10:
    print(f'✅ Memory usage is CLOSE to expected FP32 size')
    print(f'   Model appears to store latent FP32 weights')
elif model_mem_delta < expected_fp32_mb * 0.5:
    print(f'⚠️  Memory usage is MUCH LOWER than expected FP32 size')
    print(f'   Model may be using quantized storage')
else:
    print(f'⚠️  Memory usage is HIGHER than expected FP32 size')
    print(f'   Additional overhead present (optimizer state, buffers, etc.)')

print()

# Check parameter types
fp32_params = 0
quantized_params = 0
for name, param in model.named_parameters():
    if param.dtype == torch.float32:
        fp32_params += param.numel()
    else:
        quantized_params += param.numel()

print(f'PARAMETER STORAGE:')
print(f'  FP32 parameters:  {fp32_params:,} ({fp32_params/total_params*100:.1f}%)')
print(f'  Other dtypes:      {quantized_params:,} ({quantized_params/total_params*100:.1f}%)')
print()

# Check buffer sizes
buffer_mem = 0
for name, buffer in model.named_buffers():
    buffer_mem += buffer.numel() * 4  # Assume FP32

print(f'MODEL BUFFERS:')
print(f'  Buffer memory: ~{buffer_mem / 1024 / 1024:.2f} MB')
print()

print('=' * 80)
print('CONCLUSION')
print('=' * 80)
print()
print(f'Total inference RAM: {after_gen_mem["rss_mb"]:.2f} MB')
print(f'  - Python baseline: {baseline_mem["rss_mb"]:.2f} MB')
print(f'  - PyTorch runtime: {after_torch_mem["rss_mb"] - baseline_mem["rss_mb"]:.2f} MB')
print(f'  - Model weights:   {model_mem_delta:.2f} MB')
print(f'  - Tokenizer:       {after_tokenizer_mem["rss_mb"] - after_model_mem["rss_mb"]:.2f} MB')
print(f'  - Generation:      {after_gen_mem["rss_mb"] - after_tokenizer_mem["rss_mb"]:.2f} MB')
print()

print('Measurement Method: psutil.Process.memory_info().rss')
print('  RSS = Resident Set Size = actual physical RAM used by process')
print('  Includes: code, data, shared libraries, all memory allocations')
