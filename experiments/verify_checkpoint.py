#!/usr/bin/env python3
import torch

checkpoint = torch.load('/root/agent-sandbox/ARTN/outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt', map_location='cpu')

print("="*80)
print("CHECKPOINT STRUCTURE")
print("="*80)

print(f"\nTop-level keys: {list(checkpoint.keys())}")

# Check optimizer state
if 'optimizer_state_dict' in checkpoint:
    opt_state = checkpoint['optimizer_state_dict']
    print(f"\n✓ optimizer_state_dict FOUND")
    print(f"  State entries: {len(opt_state.get('state', {}))}")
    
    # Check param_groups
    if 'param_groups' in opt_state and len(opt_state['param_groups']) > 0:
        pg = opt_state['param_groups'][0]
        print(f"\n  Param groups[0] keys: {list(pg.keys())}")
        print(f"\n  Optimizer parameters:")
        print(f"    lr: {pg.get('lr', 'N/A')}")
        print(f"    beta1: {pg.get('beta1', 'N/A')}")
        print(f"    beta2: {pg.get('beta2', 'N/A')}")
        print(f"    eps1: {pg.get('eps1', 'N/A')}")
        print(f"    eps2: {pg.get('eps2', 'N/A')}")
        print(f"    weight_decay: {pg.get('weight_decay', 'N/A')}")
        print(f"    clip_threshold: {pg.get('clip_threshold', 'N/A')}")
        
    # Check state structure (first entry)
    if 'state' in opt_state and len(opt_state['state']) > 0:
        first_key = list(opt_state['state'].keys())[0]
        first_state = opt_state['state'][first_key]
        print(f"\n  State[0] keys: {list(first_state.keys())}")
        print(f"  State[0] structure:")
        for k, v in first_state.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: Tensor {v.shape}")
            else:
                print(f"    {k}: {type(v).__name__} = {v}")
else:
    print(f"\n✗ optimizer_state_dict NOT FOUND")

print(f"\n  Step: {checkpoint.get('step', 'N/A')}")
print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
