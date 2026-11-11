import torch
import sys
sys.path.insert(0, '.')

# Load 50k checkpoint
ckpt_path = 'outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')

print("="*70)
print("50k CHECKPOINT METADATA")
print("="*70)

print(f"\nKeys in checkpoint: {list(ckpt.keys())}")

if 'step' in ckpt:
    print(f"Step: {ckpt['step']}")

if 'loss' in ckpt:
    print(f"Loss at checkpoint: {ckpt['loss']}")

if 'config' in ckpt:
    print("\n--- CONFIG ---")
    config = ckpt['config']
    for k, v in config.items():
        print(f"{k}: {v}")

if 'optimizer_state_dict' in ckpt:
    print("\n--- OPTIMIZER ---")
    opt_state = ckpt['optimizer_state_dict']
    print(f"Optimizer has {len(opt_state.get('state', {}))} parameter groups")
    if 'param_groups' in opt_state:
        for i, pg in enumerate(opt_state['param_groups']):
            print(f"Param group {i}: lr={pg.get('lr', 'N/A')}, betas={pg.get('betas', 'N/A')}")

# Check model state dict keys
if 'model_state_dict' in ckpt:
    print("\n--- MODEL STATE DICT ---")
    msd = ckpt['model_state_dict']
    print(f"Total parameters: {len(msd)}")
    
    # Show first few and last few keys
    keys = list(msd.keys())
    print(f"\nFirst 5 keys: {keys[:5]}")
    print(f"Last 5 keys: {keys[-5:]}")
    
    # Check for BitLinear-specific keys
    bitlinear_keys = [k for k in keys if 'running_sparsity' in k or 'num_quantizations' in k]
    if bitlinear_keys:
        print(f"\nBitLinear-specific keys found: {len(bitlinear_keys)}")
        print(f"Example: {bitlinear_keys[:3]}")
    else:
        print("\nNo BitLinear-specific keys (standard Linear layers)")

print("\n" + "="*70)
