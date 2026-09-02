#!/usr/bin/env python3
"""
Investigate AdaFactor parameter format mismatch
"""
import torch

# Load checkpoint
checkpoint = torch.load('outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt', map_location='cpu')

print("="*80)
print("CHECKPOINT OPTIMIZER PARAM GROUPS")
print("="*80)

opt_state = checkpoint['optimizer_state_dict']
pg = opt_state['param_groups'][0]

print(f"\nKeys in param_groups[0]:")
for k, v in pg.items():
    if k == 'params':
        print(f"  {k}: [{len(v)} param IDs]")
    else:
        print(f"  {k}: {v}")

print("\n" + "="*80)
print("PYTORCH AdaFactor EXPECTED PARAMETERS")
print("="*80)

import inspect
sig = inspect.signature(torch.optim.Adafactor.__init__)
print(f"\nAdaFactor.__init__ signature:")
for param_name, param in sig.parameters.items():
    if param_name != 'params':
        print(f"  {param_name}: default={param.default}")

print("\n" + "="*80)
print("ATTEMPTING TO CREATE MATCHING OPTIMIZER")
print("="*80)

# Extract parameters from checkpoint
lr = pg.get('lr', 1e-3)
beta1 = pg.get('beta1', None)
beta2 = pg.get('beta2', None)
eps1 = pg.get('eps1', 1e-30)
eps2 = pg.get('eps2', 1e-3)
weight_decay = pg.get('weight_decay', 0.0)
clip_threshold = pg.get('clip_threshold', 1.0)

print(f"\nExtracted from checkpoint:")
print(f"  lr: {lr}")
print(f"  beta1: {beta1}")
print(f"  beta2: {beta2}")
print(f"  eps1: {eps1}")
print(f"  eps2: {eps2}")
print(f"  weight_decay: {weight_decay}")
print(f"  clip_threshold: {clip_threshold}")

# Try creating with these exact params
try:
    # Create a dummy model
    dummy = torch.nn.Linear(10, 10)
    
    # Try different parameter combinations
    print(f"\n\nAttempt 1: Using eps as tuple (eps1, eps2)")
    try:
        opt = torch.optim.Adafactor(
            dummy.parameters(),
            lr=lr,
            eps=(eps1, eps2),
            weight_decay=weight_decay,
            clip_threshold=clip_threshold,
        )
        print(f"  ✓ SUCCESS with eps as tuple")
        
        # Check what the param_groups look like
        print(f"\n  Created param_groups:")
        for k, v in opt.param_groups[0].items():
            if k != 'params':
                print(f"    {k}: {v}")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    print(f"\n\nAttempt 2: Using beta1, beta2 separately")
    try:
        opt = torch.optim.Adafactor(
            dummy.parameters(),
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=(eps1, eps2),
            weight_decay=weight_decay,
            clip_threshold=clip_threshold,
        )
        print(f"  ✓ SUCCESS with beta1, beta2")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nThe checkpoint uses a DIFFERENT AdaFactor parameter format.")
print("We need to TRANSFORM the checkpoint format to match PyTorch's expected format.")
