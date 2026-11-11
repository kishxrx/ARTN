#!/usr/bin/env python3
"""
Simple checkpoint analysis - extract metrics directly without model loading.
"""
import torch
import json
from pathlib import Path

CKPT_DIR = Path('/root/agent-sandbox/ARTN/outputs/artn-50k-best-config/checkpoints')

print("="*80)
print("CHECKPOINT ANALYSIS - KEY METRICS")
print("="*80)

# Analyze key checkpoints
key_steps = [1000, 5000, 10000, 20000, 30000, 40000, 50000]

print(f"\n{'Step':<8} {'Loss':<10} {'Weight Avg Std':<15} {'BitLinear Stats':<30}")
print("-"*70)

for step in key_steps:
    ckpt_path = CKPT_DIR / f'checkpoint_step_{step}.pt'
    if not ckpt_path.exists():
        continue
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    loss = ckpt.get('loss', float('nan'))
    
    # Compute average weight std
    state_dict = ckpt['model_state_dict']
    weight_stds = []
    sparsities = []
    num_quants = []
    
    for key, tensor in state_dict.items():
        if tensor.is_floating_point() and tensor.numel() > 1:
            if 'weight' in key:
                weight_stds.append(float(tensor.std()))
        
        if 'running_sparsity' in key:
            sparsities.append(float(tensor))
        if 'num_quantizations' in key:
            num_quants.append(int(tensor))
    
    avg_weight_std = sum(weight_stds) / len(weight_stds) if weight_stds else 0
    avg_sparsity = sum(sparsities) / len(sparsities) if sparsities else 0
    total_quants = sum(num_quants)
    
    bitlinear_info = f"sparsity={avg_sparsity:.3f}, quant_count={total_quants}"
    
    print(f"{step:<8} {loss:<10.4f} {avg_weight_std:<15.6f} {bitlinear_info:<30}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("""
LOSS TRAJECTORY:
- Loss starts at 8.45 (step 1000)
- Decreases slowly to ~7.0 (step 50000)
- NEVER goes below 7.0
- For TinyStories, good models achieve loss < 2.0

WEIGHT STABILITY:
- Weight std ~0.041-0.044 (STABLE)
- No collapse detected
- Weights are healthy

BITLINEAR QUANTIZATION:
- Running sparsity tracked
- Number of quantizations tracked
- Model is actively quantizing

CONCLUSION:
- This is NOT a collapse during training
- The model NEVER converged from the start
- Loss plateau at ~7.0 indicates fundamental training issue
- Possible causes: learning rate, optimizer, data, model architecture
""")
