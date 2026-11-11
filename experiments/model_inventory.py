#!/usr/bin/env python3
"""
Comprehensive model inventory - find all trained models and their status.
"""
import torch
import json
from pathlib import Path
from datetime import datetime

outputs_dir = Path('/root/agent-sandbox/ARTN/outputs')

print("="*80)
print("COMPLETE MODEL INVENTORY")
print("="*80)

# Find all model directories
all_models = []

for d in outputs_dir.iterdir():
    if not d.is_dir():
        continue
    
    model_info = {
        'name': d.name,
        'path': str(d),
        'has_checkpoints': (d / 'checkpoints').exists(),
        'has_final_model': (d / 'final_model' / 'model.pt').exists(),
        'num_checkpoints': 0,
        'max_step': 0,
        'loss_start': None,
        'loss_end': None,
        'config': None,
    }
    
    # Count checkpoints
    if model_info['has_checkpoints']:
        ckpts = list((d / 'checkpoints').glob('checkpoint_step_*.pt'))
        model_info['num_checkpoints'] = len(ckpts)
        
        if ckpts:
            # Get step range and loss
            try:
                first = torch.load(ckpts[0], map_location='cpu')
                last = torch.load(ckpts[-1], map_location='cpu')
                
                model_info['max_step'] = last.get('step', 0)
                model_info['loss_start'] = first.get('loss')
                model_info['loss_end'] = last.get('loss')
                
                if 'config' in first:
                    model_info['config'] = {
                        'd_model': first['config'].get('d_model'),
                        'n_layers': first['config'].get('n_layers'),
                        'total_steps': first['config'].get('total_steps'),
                        'max_lr': first['config'].get('max_lr'),
                    }
            except Exception as e:
                model_info['error'] = str(e)
    
    # Check config file
    config_file = d / 'final_model' / 'config.json'
    if config_file.exists() and not model_info['config']:
        with open(config_file) as f:
            config = json.load(f)
            model_info['config'] = {
                'd_model': config.get('d_model'),
                'n_layers': config.get('n_layers'),
                'total_steps': config.get('total_steps'),
                'max_lr': config.get('max_lr'),
            }
    
    all_models.append(model_info)

# Sort by number of checkpoints
all_models.sort(key=lambda x: (-x['num_checkpoints'], x['name']))

# Print summary
print(f"\nFound {len(all_models)} model directories\n")

for m in all_models:
    print(f"\n{m['name']}:")
    print(f"  Checkpoints: {m['num_checkpoints']}")
    print(f"  Final model: {m['has_final_model']}")
    
    if m['max_step']:
        print(f"  Max step: {m['max_step']}")
    
    if m['loss_start'] and m['loss_end']:
        print(f"  Loss: {m['loss_start']:.4f} → {m['loss_end']:.4f}")
    
    if m['config']:
        print(f"  Config: d_model={m['config']['d_model']}, n_layers={m['config']['n_layers']}, "
              f"steps={m['config']['total_steps']}, lr={m['config']['max_lr']}")

# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE - MODELS WITH CHECKPOINTS")
print("="*80)

print(f"\n{'Name':<30} {'Steps':<10} {'Loss Start':<12} {'Loss End':<12} {'Config'}")
print("-"*100)

for m in all_models:
    if m['num_checkpoints'] > 0:
        name = m['name'][:28]
        steps = m['max_step'] or 'N/A'
        loss_s = f"{m['loss_start']:.4f}" if m['loss_start'] else 'N/A'
        loss_e = f"{m['loss_end']:.4f}" if m['loss_end'] else 'N/A'
        
        config_str = ''
        if m['config']:
            config_str = f"d={m['config']['d_model']}, L={m['config']['n_layers']}"
        
        print(f"{name:<30} {steps:<10} {loss_s:<12} {loss_e:<12} {config_str}")
