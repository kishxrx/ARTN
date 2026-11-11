#!/usr/bin/env python3
"""
CHECKPOINT DIFFERENTIAL ANALYSIS - FP32 GPT Model

Purpose: Analyze all checkpoints from 50k-step training run to find:
1. When gibberish/collapse starts
2. What metric changes first at the transition point
3. Root cause of degradation
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Setup paths
BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from src.model.gpt import GPT

# Checkpoint directory
CKPT_DIR = BASE_DIR / 'outputs' / 'artn-50k-best-config' / 'checkpoints'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'checkpoint_differential_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test data
TEST_TOKENS = torch.randint(0, 4000, (1, 32))  # Random test input


def analyze_weight_stats(state_dict: Dict) -> Dict:
    """Analyze all weight statistics from state dict."""
    stats = {}
    
    # Group by layer
    layer_stats = {}
    
    for key, tensor in state_dict.items():
        # Skip non-float tensors (int, long, etc.)
        if not tensor.is_floating_point():
            continue
        
        if tensor.numel() == 0:
            continue
        
        # Weight stats
        weight_mean = float(tensor.mean())
        weight_std = float(tensor.std() if tensor.numel() > 1 else 0.0)
        weight_norm = float(tensor.norm())
        weight_min = float(tensor.min())
        weight_max = float(tensor.max())
        
        stats[f"{key}_mean"] = weight_mean
        stats[f"{key}_std"] = weight_std
        stats[f"{key}_norm"] = weight_norm
        stats[f"{key}_min"] = weight_min
        stats[f"{key}_max"] = weight_max
        
        # Group by layer
        layer_name = '.'.join(key.split('.')[:3])  # e.g., "blocks.0.attn"
        if layer_name not in layer_stats:
            layer_stats[layer_name] = []
        layer_stats[layer_name].append(weight_std)
    
    # Compute per-layer average std
    for layer_name, stds in layer_stats.items():
        stats[f"{layer_name}_avg_std"] = np.mean(stds)
    
    # Overall statistics
    all_stds = [v for k, v in stats.items() if k.endswith('_std')]
    all_means = [v for k, v in stats.items() if k.endswith('_mean')]
    
    stats['overall_avg_weight_std'] = np.mean(all_stds)
    stats['overall_avg_weight_mean'] = np.mean(all_means)
    stats['overall_max_weight_std'] = np.max(all_stds)
    stats['overall_min_weight_std'] = np.min(all_stds)
    
    return stats


def analyze_activations(model: GPT, input_ids: torch.Tensor) -> Dict:
    """Analyze activations during forward pass."""
    stats = {}
    activations = {}
    
    # Hook to capture activations
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()
        return hook
    
    # Register hooks on key layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        try:
            _ = model(input_ids)
        except Exception as e:
            print(f"    Warning: forward pass error: {e}")
    
    # Analyze activations
    for name, act in activations.items():
        if act.numel() > 0:
            stats[f"act_{name}_mean"] = float(act.mean())
            stats[f"act_{name}_std"] = float(act.std())
            stats[f"act_{name}_min"] = float(act.min())
            stats[f"act_{name}_max"] = float(act.max())
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return stats


def analyze_logits(model: GPT, input_ids: torch.Tensor) -> Dict:
    """Analyze logits and softmax distribution."""
    model.eval()
    
    stats = {}
    
    with torch.no_grad():
        try:
            logits = model(input_ids)
            
            # Logits stats
            stats["logits_mean"] = float(logits.mean())
            stats["logits_std"] = float(logits.std())
            stats["logits_min"] = float(logits.min())
            stats["logits_max"] = float(logits.max())
            
            # Softmax analysis (last position)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            
            # Entropy
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs).item()
            max_entropy = math.log(logits.size(-1))
            
            stats["softmax_entropy"] = entropy
            stats["softmax_entropy_ratio"] = entropy / max_entropy
            stats["top1_prob"] = float(probs.max())
            stats["top5_prob_sum"] = float(probs.topk(5)[0].sum())
            
        except Exception as e:
            print(f"    Warning: logits analysis error: {e}")
            stats["logits_error"] = str(e)
    
    return stats


def analyze_gradient_norm(model: GPT, input_ids: torch.Tensor) -> Dict:
    """Analyze gradient norm on a diagnostic batch."""
    model.train()
    
    stats = {}
    
    try:
        # Forward pass
        logits = model(input_ids)
        targets = input_ids[:, 1:]
        logits_for_loss = logits[:, :-1, :]
        
        # Compute loss
        loss = F.cross_entropy(
            logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
            targets.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norms
        total_norm = 0.0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
        
        total_norm = total_norm ** 0.5
        
        stats["gradient_norm"] = total_norm
        stats["gradient_loss"] = float(loss.item())
        
        # Zero gradients
        model.zero_grad()
        
    except Exception as e:
        print(f"    Warning: gradient analysis error: {e}")
        stats["gradient_error"] = str(e)
    
    return stats


def analyze_checkpoint(ckpt_path: Path) -> Dict:
    """Comprehensive checkpoint analysis."""
    step = int(ckpt_path.stem.split('_')[-1])
    
    print(f"  Step {step}...", end='', flush=True)
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    results = {
        "step": step,
        "checkpoint_path": str(ckpt_path.name),
    }
    
    # Extract training loss
    if 'loss' in ckpt:
        results['training_loss'] = float(ckpt['loss'])
    
    # Extract config
    if 'config' in ckpt:
        config = ckpt['config']
        results['config_d_model'] = config.get('d_model')
        results['config_n_layers'] = config.get('n_layers')
        results['config_n_heads'] = config.get('n_heads')
    else:
        config = {
            "vocab_size": 4000,
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 1024,
            "max_seq_len": 256,
            "dropout": 0.0,
        }
    
    # Analyze weights directly from state dict
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        
        print(" [weights]", end='', flush=True)
        weight_stats = analyze_weight_stats(state_dict)
        results.update(weight_stats)
        
        # Load model for activation/logit analysis
        try:
            model_config = {
                "vocab_size": 4000,
                "d_model": config.get('d_model', 256),
                "n_heads": config.get('n_heads', 8),
                "n_layers": config.get('n_layers', 6),
                "d_ff": config.get('d_ff', 1024),
                "max_seq_len": config.get('max_seq_len', 256),
                "dropout": 0.0,
            }
            
            model = GPT(**model_config)
            model.load_state_dict(state_dict)
            model.eval()
            
            # Analyze activations
            print(" [activations]", end='', flush=True)
            act_stats = analyze_activations(model, TEST_TOKENS)
            results.update(act_stats)
            
            # Analyze logits
            print(" [logits]", end='', flush=True)
            logit_stats = analyze_logits(model, TEST_TOKENS)
            results.update(logit_stats)
            
            # Analyze gradients
            print(" [gradients]", end='', flush=True)
            grad_stats = analyze_gradient_norm(model, TEST_TOKENS)
            results.update(grad_stats)
            
        except Exception as e:
            print(f" [MODEL ERROR: {e}]", end='')
            results['model_error'] = str(e)
    
    print(" [done]")
    
    return results


def classify_checkpoint(loss: float, weight_std: float, logits_std: float) -> str:
    """Classify checkpoint health based on metrics."""
    if loss is None:
        return "UNKNOWN"
    
    # Classification based on loss and weight stability
    if loss < 2.0 and weight_std > 0.01 and logits_std > 0.1:
        return "HEALTHY"
    elif loss < 5.0 and weight_std > 0.001:
        return "DEGRADED"
    else:
        return "BAD"


def main():
    print("="*80)
    print("CHECKPOINT DIFFERENTIAL ANALYSIS - FP32 GPT")
    print("="*80)
    
    # Find all checkpoints
    checkpoints = sorted(CKPT_DIR.glob('checkpoint_step_*.pt'))
    
    print(f"\nFound {len(checkpoints)} checkpoints")
    print(f"Analyzing all checkpoints...\n")
    
    # Analyze all checkpoints
    all_results = []
    
    for i, ckpt_path in enumerate(checkpoints, 1):
        print(f"[{i}/{len(checkpoints)}]", end=' ')
        result = analyze_checkpoint(ckpt_path)
        
        # Classify
        loss = result.get('training_loss')
        weight_std = result.get('overall_avg_weight_std', 0)
        logits_std = result.get('logits_std', 0)
        result['classification'] = classify_checkpoint(loss, weight_std, logits_std)
        
        all_results.append(result)
        
        # Save intermediate results
        if i % 5 == 0:
            output_file = OUTPUT_DIR / 'checkpoint_analysis_progress.json'
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Save final results
    output_file = OUTPUT_DIR / 'checkpoint_analysis_final.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Create summary table
    print("\n" + "="*80)
    print("CHECKPOINT CLASSIFICATION TABLE")
    print("="*80)
    
    print(f"\n{'Step':<8} {'Loss':<10} {'Weight Std':<12} {'Logits Std':<12} {'Entropy':<10} {'Class':<12}")
    print("-"*70)
    
    for r in all_results:
        step = r['step']
        loss = r.get('training_loss', float('nan'))
        weight_std = r.get('overall_avg_weight_std', 0)
        logits_std = r.get('logits_std', 0)
        entropy = r.get('softmax_entropy', 0)
        classification = r.get('classification', 'UNKNOWN')
        
        print(f"{step:<8} {loss:<10.4f} {weight_std:<12.6f} {logits_std:<12.4f} {entropy:<10.2f} {classification:<12}")
    
    # Find transition points
    print("\n" + "="*80)
    print("TRANSITION POINT ANALYSIS")
    print("="*80)
    
    healthy_steps = [r for r in all_results if r.get('classification') == 'HEALTHY']
    degraded_steps = [r for r in all_results if r.get('classification') == 'DEGRADED']
    bad_steps = [r for r in all_results if r.get('classification') == 'BAD']
    
    if healthy_steps:
        last_healthy = max(healthy_steps, key=lambda x: x['step'])
        print(f"\nLast HEALTHY checkpoint: Step {last_healthy['step']}")
        print(f"  Loss: {last_healthy.get('training_loss', 'N/A'):.4f}")
        print(f"  Weight std: {last_healthy.get('overall_avg_weight_std', 'N/A'):.6f}")
        print(f"  Logits std: {last_healthy.get('logits_std', 'N/A'):.4f}")
    
    if degraded_steps:
        first_degraded = min(degraded_steps, key=lambda x: x['step'])
        print(f"\nFirst DEGRADED checkpoint: Step {first_degraded['step']}")
        print(f"  Loss: {first_degraded.get('training_loss', 'N/A'):.4f}")
        print(f"  Weight std: {first_degraded.get('overall_avg_weight_std', 'N/A'):.6f}")
        print(f"  Logits std: {first_degraded.get('logits_std', 'N/A'):.4f}")
    
    if bad_steps:
        first_bad = min(bad_steps, key=lambda x: x['step'])
        print(f"\nFirst BAD checkpoint: Step {first_bad['step']}")
        print(f"  Loss: {first_bad.get('training_loss', 'N/A'):.4f}")
        print(f"  Weight std: {first_bad.get('overall_avg_weight_std', 'N/A'):.6f}")
        print(f"  Logits std: {first_bad.get('logits_std', 'N/A'):.4f}")
    
    # Differential analysis
    if healthy_steps and (degraded_steps or bad_steps):
        print("\n" + "="*80)
        print("DIFFERENTIAL ANALYSIS: LAST HEALTHY vs FIRST DEGRADED/BAD")
        print("="*80)
        
        last_healthy = max(healthy_steps, key=lambda x: x['step'])
        first_bad_step = min(degraded_steps + bad_steps, key=lambda x: x['step']) if (degraded_steps or bad_steps) else None
        
        if first_bad_step:
            print(f"\nComparing Step {last_healthy['step']} (HEALTHY) vs Step {first_bad_step['step']} (first problem)")
            print(f"\n{'Metric':<40} {'HEALTHY':<15} {'PROBLEM':<15} {'Change':<15}")
            print("-"*85)
            
            # Compare key metrics
            metrics = ['training_loss', 'overall_avg_weight_std', 'logits_std', 'softmax_entropy', 'gradient_norm']
            
            for metric in metrics:
                h_val = last_healthy.get(metric)
                p_val = first_bad_step.get(metric)
                
                if h_val is not None and p_val is not None:
                    change = p_val - h_val
                    change_pct = (change / h_val * 100) if h_val != 0 else float('inf')
                    
                    print(f"{metric:<40} {h_val:<15.6f} {p_val:<15.6f} {change:>+14.6f} ({change_pct:>+.1f}%)")


if __name__ == "__main__":
    main()
