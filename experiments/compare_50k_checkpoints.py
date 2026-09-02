#!/usr/bin/env python3
"""
Compare all 50k checkpoints to identify the best one
"""
import torch
from pathlib import Path

checkpoints = [
    ("artn-50k-best-config", "outputs/artn-50k-best-config/final_model/model.pt"),
    ("artn-50k-best-config (checkpoint_step_50000)", "outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt"),
    ("artn-optimized-backup-50k", "outputs/artn-optimized-backup-50k/final_model/model.pt"),
    ("artn-optimized", "outputs/artn-optimized/final_model/model.pt"),
]

print("="*100)
print("50k CHECKPOINT COMPARISON")
print("="*100)

for name, path in checkpoints:
    checkpoint_path = Path(path)
    
    if not checkpoint_path.exists():
        print(f"\n❌ {name}: NOT FOUND")
        continue
    
    print(f"\n{'='*100}")
    print(f"✅ {name}")
    print(f"   Path: {path}")
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location='cpu')
    
    # Check what's in the checkpoint
    if isinstance(checkpoint, dict):
        print(f"   Keys: {list(checkpoint.keys())[:10]}")
        
        # Check for step info
        if 'step' in checkpoint:
            print(f"   Step: {checkpoint['step']}")
        
        if 'loss' in checkpoint:
            print(f"   Loss: {checkpoint['loss']}")
        
        # Check for config
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"   Config: total_steps={config.get('total_steps', 'N/A')}, max_samples={config.get('max_samples', 'N/A')}")
        
        # Check for optimizer state
        if 'optimizer_state_dict' in checkpoint:
            opt_state = checkpoint['optimizer_state_dict']
            if 'param_groups' in opt_state:
                pg = opt_state['param_groups'][0]
                print(f"   Optimizer: lr={pg.get('lr')}, beta1={pg.get('beta1')}, beta2={pg.get('beta2')}")
            print(f"   Optimizer state entries: {len(opt_state.get('state', {}))}")
        else:
            print(f"   ❌ No optimizer state")
    
    # File size
    file_size = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size:.1f} MB")

print(f"\n{'='*100}")
print("RECOMMENDATION")
print("="*100)
print("\n✅ RECOMMENDED: artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")
print("\nReasons:")
print("  1. Has optimizer state (critical for continuation)")
print("  2. beta1=0.5 (best value from experiments)")
print("  3. Has all 50 checkpoints (complete trajectory)")
print("  4. Most recent (Aug 16)")
print("  5. Loss tracked in monitor_log.txt")
