#!/usr/bin/env python3
"""
Quick checkpoint inspection to understand structure.
"""
import torch
from pathlib import Path

# Load one checkpoint
ckpt_path = Path('/root/agent-sandbox/ARTN/outputs/artn-50k-best-config/checkpoints/checkpoint_step_1000.pt')

print(f"Loading: {ckpt_path.name}")
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"\nCheckpoint type: {type(ckpt)}")
print(f"\nCheckpoint keys:")

if isinstance(ckpt, dict):
    for key in ckpt.keys():
        val = ckpt[key]
        if isinstance(val, torch.Tensor):
            print(f"  {key}: Tensor {val.shape}")
        elif isinstance(val, dict):
            print(f"  {key}: dict with {len(val)} keys")
            # Show first few keys
            for i, k in enumerate(list(val.keys())[:5]):
                v = val[k]
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: Tensor {v.shape}")
                else:
                    print(f"    {k}: {type(v).__name__}")
            if len(val) > 5:
                print(f"    ... and {len(val) - 5} more keys")
        elif isinstance(val, (int, float, str)):
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: {type(val).__name__}")
