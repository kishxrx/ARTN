import torch
import sys
import os
from pathlib import Path
sys.path.insert(0, '.')

print("="*80)
print("COMPLETE CHECKPOINT INVENTORY")
print("="*80)

# List all checkpoint directories
checkpoint_dirs = [
    'outputs/artn-50k-best-config',
    'outputs/phase22_controlled_experiment',
    'outputs/fp32-5k-comparison',
    'outputs/beta1-0.5-batch8-lr001',
    'outputs/beta1-0.7-batch8-lr001',
    'outputs/artn-optimized',
    'outputs/artn-core-v1',
]

def analyze_checkpoint(ckpt_path):
    """Extract key information from checkpoint"""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        return f"Error loading: {e}"
    
    info = {}
    
    # Config
    if 'config' in ckpt:
        cfg = ckpt['config']
        info['lr'] = cfg.get('max_lr', cfg.get('learning_rate', 'N/A'))
        info['weight_decay'] = cfg.get('weight_decay', 'N/A')
        info['beta1'] = cfg.get('beta1', 'N/A')
        info['beta2'] = cfg.get('beta2', 'N/A')
        info['steps'] = cfg.get('total_steps', ckpt.get('step', 'N/A'))
        info['warmup'] = cfg.get('warmup_steps', 'N/A')
        info['batch_size'] = cfg.get('batch_size', 'N/A')
        info['max_samples'] = cfg.get('max_samples', 'N/A')
    
    # Step
    if 'step' in ckpt:
        info['step'] = ckpt['step']
    
    # Loss
    if 'loss' in ckpt:
        info['loss'] = f"{ckpt['loss']:.4f}"
    
    # Model state dict - check for BitLinear
    if 'model_state_dict' in ckpt:
        msd = ckpt['model_state_dict']
        has_bitlinear = any('running_sparsity' in k for k in msd.keys())
        info['model_type'] = 'Ternary (BitLinear)' if has_bitlinear else 'FP32 (Standard)'
        
        # Token embedding stats
        if 'token_embedding.weight' in msd:
            emb = msd['token_embedding.weight']
            info['emb_std'] = f"{emb.std():.4f}"
            info['emb_mean'] = f"{emb.mean():.4f}"
    
    # Optimizer state
    if 'optimizer_state_dict' in ckpt:
        opt = ckpt['optimizer_state_dict']
        if 'param_groups' in opt and len(opt['param_groups']) > 0:
            pg = opt['param_groups'][0]
            info['opt_lr'] = pg.get('lr', 'N/A')
            info['opt_weight_decay'] = pg.get('weight_decay', 'N/A')
        
        # Check optimizer type by state keys
        if 'state' in opt and len(opt['state']) > 0:
            first_state = list(opt['state'].values())[0]
            if 'exp_avg_sq_row' in first_state:
                info['optimizer'] = 'Adafactor'
            elif 'exp_avg' in first_state and 'exp_avg_sq' in first_state:
                info['optimizer'] = 'AdamW/Adam'
            else:
                info['optimizer'] = 'Unknown'
    
    return info

# Analyze each checkpoint directory
print("\n" + "="*80)
print("CHECKPOINT ANALYSIS")
print("="*80)

for ckpt_dir in checkpoint_dirs:
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        print(f"\n{ckpt_dir}: NOT FOUND")
        continue
    
    print(f"\n{'='*80}")
    print(f"DIRECTORY: {ckpt_dir}")
    print(f"{'='*80}")
    
    # Find checkpoints
    checkpoint_files = list(ckpt_path.glob('checkpoints/checkpoint_step_*.pt'))
    
    if not checkpoint_files:
        # Try final_model
        final_model = ckpt_path / 'final_model' / 'model.pt'
        if final_model.exists():
            checkpoint_files = [final_model]
        else:
            print("  No checkpoints found")
            continue
    
    # Analyze key checkpoints
    key_steps = [1000, 5000, 10000, 50000]
    
    for ckpt_file in sorted(checkpoint_files):
        step = ckpt_file.stem.replace('checkpoint_step_', '').replace('model', 'final')
        
        # Skip intermediate steps for long runs
        if step.isdigit():
            step_num = int(step)
            # Only show key checkpoints
            if step_num not in [1000, 5000, 10000, 50000] and step_num % 10000 != 0:
                continue
        
        print(f"\n  Checkpoint: {ckpt_file.name}")
        info = analyze_checkpoint(str(ckpt_file))
        
        if isinstance(info, str):
            print(f"    {info}")
            continue
        
        print(f"    Step: {info.get('step', 'N/A')}")
        print(f"    Model: {info.get('model_type', 'N/A')}")
        print(f"    Optimizer: {info.get('optimizer', 'N/A')}")
        print(f"    LR: {info.get('lr', 'N/A')}")
        print(f"    Weight Decay: {info.get('weight_decay', 'N/A')}")
        print(f"    Beta1: {info.get('beta1', 'N/A')}")
        print(f"    Beta2: {info.get('beta2', 'N/A')}")
        print(f"    Loss: {info.get('loss', 'N/A')}")
        print(f"    Token Emb Std: {info.get('emb_std', 'N/A')}")
        print(f"    Token Emb Mean: {info.get('emb_mean', 'N/A')}")

print("\n" + "="*80)
print("PHASE 22 SPECIFIC ANALYSIS")
print("="*80)

# Phase 22 FP32
fp32_ckpt = 'outputs/phase22_controlled_experiment/fp32_checkpoints/checkpoint_step_5000.pt'
if Path(fp32_ckpt).exists():
    ckpt = torch.load(fp32_ckpt, map_location='cpu')
    print(f"\nPhase 22 FP32 @ 5000:")
    print(f"  Model state dict keys (first 5): {list(ckpt['model_state_dict'].keys())[:5]}")
    
    # Check if optimizer state exists
    if 'optimizer_state_dict' in ckpt:
        opt = ckpt['optimizer_state_dict']
        print(f"  Optimizer state exists: YES")
        if 'state' in opt and len(opt['state']) > 0:
            first_state = list(opt['state'].values())[0]
            print(f"  Optimizer state keys: {list(first_state.keys())}")
    else:
        print(f"  Optimizer state: NOT SAVED")

# Phase 22 Ternary
ternary_ckpt = 'outputs/phase22_controlled_experiment/ternary_checkpoints/checkpoint_step_5000.pt'
if Path(ternary_ckpt).exists():
    ckpt = torch.load(ternary_ckpt, map_location='cpu')
    print(f"\nPhase 22 Ternary @ 5000:")
    print(f"  Model state dict keys (first 5): {list(ckpt['model_state_dict'].keys())[:5]}")
    
    # Check for BitLinear
    has_bitlinear = any('running_sparsity' in k for k in ckpt['model_state_dict'].keys())
    print(f"  Has BitLinear: {has_bitlinear}")
    
    if 'optimizer_state_dict' in ckpt:
        opt = ckpt['optimizer_state_dict']
        print(f"  Optimizer state exists: YES")
    else:
        print(f"  Optimizer state: NOT SAVED")

print("\n" + "="*80)
print("TRAINING SCRIPT CHECK")
print("="*80)

# Check training scripts for Phase 22
phase22_script = 'experiments/phase22_full_training.py'
if Path(phase22_script).exists():
    with open(phase22_script) as f:
        content = f.read()
    
    # Find optimizer
    if 'AdamW' in content:
        print(f"\nPhase 22 script uses: AdamW")
        # Find the exact line
        for line in content.split('\n'):
            if 'AdamW' in line and 'optim' in line:
                print(f"  Line: {line.strip()}")
    
    # Find label construction
    for line in content.split('\n'):
        if 'labels' in line and 'clone' in line:
            print(f"  Label construction: {line.strip()}")
            break

print("\n" + "="*80)
