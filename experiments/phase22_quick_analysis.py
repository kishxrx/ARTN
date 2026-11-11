#!/usr/bin/env python3
"""Phase 22 Quick Analysis"""
import torch
import json
import numpy as np
from pathlib import Path

sys.path = ['/root/agent-sandbox/ARTN'] + sys.path if 'sys' in dir() else ['/root/agent-sandbox/ARTN']
import sys
from src.model.bitlinear import compute_abs_mean_scale

print("="*100)
print("PHASE 22: QUICK ANALYSIS")
print("="*100)

out_dir = Path('outputs/phase22_controlled_experiment')

# Load checkpoints
print("\n[Loading checkpoints]")
ckpts = {'fp32': {}, 'ternary': {}}
for step in [1000, 2000, 5000]:
    ckpts['fp32'][step] = torch.load(out_dir / 'fp32_checkpoints' / f'checkpoint_step_{step}.pt', map_location='cpu')
    ckpts['ternary'][step] = torch.load(out_dir / 'ternary_checkpoints' / f'checkpoint_step_{step}.pt', map_location='cpu')

# Training loss
print(f"\n{'Step':<10} {'FP32 Loss':<15} {'Ternary Loss':<15} {'Ratio':<10}")
print("-" * 50)
for step in [1000, 2000, 5000]:
    fp32_l = ckpts['fp32'][step].get('loss', 0)
    ternary_l = ckpts['ternary'][step].get('loss', 0)
    print(f"{step:<10} {fp32_l:<15.4f} {ternary_l:<15.4f} {ternary_l/fp32_l:.2f}x")

# Quantization error
print("\n[Quantization Error]")
quant_stats = {}
for step in [1000, 2000, 5000]:
    state = ckpts['ternary'][step]['model_state_dict']
    cos_sims = []
    for k, v in state.items():
        if 'weight' in k and 'embedding' not in k and 'ln' not in k and v.dim() == 2:
            scale = compute_abs_mean_scale(v)
            w_q = torch.zeros_like(v)
            w_q[v/scale > 0.5] = 1.0
            w_q[v/scale < -0.5] = -1.0
            w_eff = w_q * scale
            import torch.nn.functional as F
            cs = F.cosine_similarity(v.flatten().unsqueeze(0), w_eff.flatten().unsqueeze(0)).item()
            cos_sims.append(cs)
    quant_stats[step] = {'mean': np.mean(cos_sims), 'std': np.std(cos_sims)}
    print(f"   Step {step}: cos_sim = {np.mean(cos_sims):.4f} ± {np.std(cos_sims):.4f}")

# Weight divergence
print("\n[Weight Divergence - Embeddings]")
init = torch.load(out_dir / 'shared_initialization.pt', map_location='cpu')
shared = init['shared_weights']

for step in [1000, 2000, 5000]:
    fp32_emb = ckpts['fp32'][step]['model_state_dict']['token_embedding.weight']
    ternary_emb = ckpts['ternary'][step]['model_state_dict']['token_embedding.weight']
    
    import torch.nn.functional as F
    fp32_vs_init = F.cosine_similarity(fp32_emb.flatten().unsqueeze(0), shared['token_embedding.weight'].flatten().unsqueeze(0)).item()
    ternary_vs_init = F.cosine_similarity(ternary_emb.flatten().unsqueeze(0), shared['token_embedding.weight'].flatten().unsqueeze(0)).item()
    fp32_vs_ternary = F.cosine_similarity(fp32_emb.flatten().unsqueeze(0), ternary_emb.flatten().unsqueeze(0)).item()
    
    print(f"   Step {step}: FP32-init={fp32_vs_init:.4f}, Ternary-init={ternary_vs_init:.4f}, FP32-Ternary={fp32_vs_ternary:.4f}")

print("\n" + "="*100)
print("QUICK ANALYSIS COMPLETE")
print("="*100)
