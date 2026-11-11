#!/usr/bin/env python3
"""
COMPREHENSIVE OPTIMIZER & UPDATE VERIFICATION

Investigates:
1. Why C/D show no weight change
2. Adam vs AdamW behavior
3. Safe weight decay regime
4. Multi-seed verification
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import BitLinear directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bitlinear", 
    Path(__file__).parent.parent / "src" / "model" / "bitlinear.py"
)
bitlinear_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bitlinear_module)
BitLinear = bitlinear_module.BitLinear
compute_abs_mean_scale = bitlinear_module.compute_abs_mean_scale


# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "optimizer_update_verification"

BASE_CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "num_steps": 100,
    "batch_size": 4,
    "seq_len": 32,
    "num_stories": 5,
    "lr": 0.01,
    "grad_clip": 1.0,
}

# Test configurations
OPTIMIZER_TESTS = [
    {"name": "Adam_WD0.01", "optimizer": "Adam", "weight_decay": 0.01},
    {"name": "Adam_WD0.001", "optimizer": "Adam", "weight_decay": 0.001},
    {"name": "Adam_WD0.0001", "optimizer": "Adam", "weight_decay": 0.0001},
    {"name": "Adam_WD0.0", "optimizer": "Adam", "weight_decay": 0.0},
    {"name": "AdamW_WD0.01", "optimizer": "AdamW", "weight_decay": 0.01},
    {"name": "AdamW_WD0.001", "optimizer": "AdamW", "weight_decay": 0.001},
    {"name": "AdamW_WD0.0001", "optimizer": "AdamW", "weight_decay": 0.0001},
    {"name": "AdamW_WD0.0", "optimizer": "AdamW", "weight_decay": 0.0},
]

SEEDS = [42, 43, 44]


# ============================================================================
# SIMPLE MODEL
# ============================================================================

class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.layer1 = BitLinear(config["d_model"], config["d_model"])
        self.layer2 = BitLinear(config["d_model"], config["d_model"])
        self.output = BitLinear(config["d_model"], config["vocab_size"], bias=False)
    
    def forward(self, x):
        h = self.embedding(x)
        h = F.gelu(self.layer1(h))
        h = F.gelu(self.layer2(h))
        return self.output(h)


# ============================================================================
# DATA
# ============================================================================

def generate_data(config: Dict, seed: int) -> torch.Tensor:
    """Generate TinyStories-like data."""
    torch.manual_seed(seed)
    
    # Zipf-like distribution
    vocab_size = config["vocab_size"]
    token_probs = torch.ones(vocab_size)
    for i in range(vocab_size):
        token_probs[i] = 1.0 / (i + 1) ** 0.5
    token_probs = token_probs / token_probs.sum()
    
    # Generate
    data = torch.zeros(config["num_stories"], config["seq_len"], dtype=torch.long)
    for i in range(config["num_stories"]):
        for j in range(config["seq_len"]):
            data[i, j] = torch.multinomial(token_probs, 1).item()
    
    return data


# ============================================================================
# DETAILED OPTIMIZER INSPECTION
# ============================================================================

def inspect_optimizer_and_parameters(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    name: str,
):
    """Detailed inspection of optimizer and parameters."""
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZER INSPECTION: {name}")
    print(f"{'='*80}")
    
    # 1. Optimizer type
    print(f"\n1. OPTIMIZER TYPE:")
    print(f"   Class: {type(optimizer).__name__}")
    print(f"   Module: {type(optimizer).__module__}")
    
    # 2. Optimizer config
    print(f"\n2. OPTIMIZER CONFIG:")
    for i, pg in enumerate(optimizer.param_groups):
        print(f"   Param Group {i}:")
        print(f"     lr: {pg['lr']}")
        print(f"     weight_decay: {pg['weight_decay']}")
        print(f"     betas: {pg['betas']}")
        print(f"     eps: {pg['eps']}")
        print(f"     num params: {len(pg['params'])}")
    
    # 3. BitLinear parameter inspection
    print(f"\n3. BITLINEAR PARAMETERS:")
    for layer_name in ['layer1', 'layer2', 'output']:
        layer = getattr(model, layer_name)
        weight = layer.weight
        
        print(f"\n   {layer_name}.weight:")
        print(f"     id: {id(weight)}")
        print(f"     is nn.Parameter: {isinstance(weight, nn.Parameter)}")
        print(f"     requires_grad: {weight.requires_grad}")
        print(f"     shape: {weight.shape}")
        print(f"     dtype: {weight.dtype}")
        
        # Check if in optimizer
        in_optimizer = False
        for pg in optimizer.param_groups:
            for p in pg['params']:
                if weight is p:  # Use identity comparison, not 'in'
                    in_optimizer = True
                    break
            if in_optimizer:
                break
        print(f"     in optimizer: {in_optimizer}")
    
    # 4. All parameters in optimizer
    print(f"\n4. ALL PARAMETERS IN OPTIMIZER:")
    total_params = 0
    for pg in optimizer.param_groups:
        for p in pg['params']:
            total_params += 1
    print(f"   Total parameters tracked: {total_params}")
    
    # Count by type
    bitlinear_count = 0
    other_count = 0
    for name_param, param in model.named_parameters():
        if 'layer' in name_param or 'output' in name_param:
            if 'weight' in name_param:
                bitlinear_count += 1
        else:
            other_count += 1
    print(f"   BitLinear weights: {bitlinear_count}")
    print(f"   Other params: {other_count}")


def measure_actual_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    step: int,
) -> Dict:
    """Measure actual update magnitudes."""
    
    results = {}
    
    # Get BitLinear layers
    bitlinear_layers = [
        ('layer1', model.layer1),
        ('layer2', model.layer2),
        ('output', model.output),
    ]
    
    # Store weights BEFORE
    weights_before = {}
    for name, layer in bitlinear_layers:
        weights_before[name] = layer.weight.data.clone()
    
    # Forward
    logits = model(batch)
    targets = batch[:, 1:].clone()
    logits_for_loss = logits[:, :-1, :]
    
    loss = F.cross_entropy(
        logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
        targets.reshape(-1)
    )
    
    # Zero grad
    optimizer.zero_grad()
    
    # Backward
    loss.backward()
    
    # Capture gradients
    gradient_info = {}
    for name, layer in bitlinear_layers:
        if layer.weight.grad is not None:
            gradient_info[name] = {
                "grad_norm": float(layer.weight.grad.norm().item()),
                "grad_mean": float(layer.weight.grad.mean().item()),
            }
        else:
            gradient_info[name] = {"grad_norm": 0.0, "grad_mean": 0.0}
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), BASE_CONFIG["grad_clip"])
    
    # Optimizer step
    optimizer.step()
    
    # Store weights AFTER
    weights_after = {}
    for name, layer in bitlinear_layers:
        weights_after[name] = layer.weight.data.clone()
    
    # Compute actual update
    update_info = {}
    for name in weights_before.keys():
        w_before = weights_before[name]
        w_after = weights_after[name]
        
        actual_update = w_after - w_before
        actual_update_norm = float(actual_update.norm().item())
        
        w_before_norm = float(w_before.norm().item())
        w_after_norm = float(w_after.norm().item())
        
        update_info[name] = {
            "weight_before_norm": w_before_norm,
            "weight_after_norm": w_after_norm,
            "actual_update_norm": actual_update_norm,
            "relative_update": actual_update_norm / (w_before_norm + 1e-8),
            "weight_changed": actual_update_norm > 1e-10,
        }
    
    return {
        "step": step,
        "loss": float(loss.item()),
        "gradients": gradient_info,
        "updates": update_info,
    }


# ============================================================================
# RUN SINGLE EXPERIMENT
# ============================================================================

def run_single_experiment(
    test_config: Dict,
    seed: int,
    verbose: bool = True,
) -> Dict:
    """Run single experiment."""
    
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Create model
    model = SimpleModel(BASE_CONFIG)
    
    # Create optimizer
    if test_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=BASE_CONFIG["lr"],
            weight_decay=test_config["weight_decay"],
            betas=(0.9, 0.999),
        )
    else:  # AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=BASE_CONFIG["lr"],
            weight_decay=test_config["weight_decay"],
            betas=(0.9, 0.999),
        )
    
    # Generate data
    data = generate_data(BASE_CONFIG, seed)
    
    # Inspect on first run
    if verbose and seed == 42 and test_config["name"] == "Adam_WD0.01":
        inspect_optimizer_and_parameters(model, optimizer, test_config["name"])
    
    # Track metrics
    results = {
        "config": test_config,
        "seed": seed,
        "steps": [],
    }
    
    # Training loop
    for step in range(BASE_CONFIG["num_steps"]):
        # Sample batch
        batch_idx = torch.randint(0, data.size(0), (BASE_CONFIG["batch_size"],))
        batch = data[batch_idx]
        
        # Measure update
        step_data = measure_actual_update(model, optimizer, batch, step)
        
        # Get layer1 weight stats
        with torch.no_grad():
            weight = model.layer1.weight
            scale = compute_abs_mean_scale(weight)
            weight_scaled = weight / scale.clamp(min=1e-8)
            
            weight_quantized = torch.zeros_like(weight)
            weight_quantized[weight_scaled > 0.5] = 1.0
            weight_quantized[weight_scaled < -0.5] = -1.0
            
            total = weight.numel()
            pct_zero = float((weight_quantized == 0).sum().item()) / total
            
            layer1_stats = {
                "weight_std": float(weight.std().item()),
                "scale": float(scale.item()),
                "ternary_zero": pct_zero,
            }
        
        # Compute accuracy
        with torch.no_grad():
            logits = model(batch)
            predictions = logits[:, :-1, :].argmax(dim=-1)
            targets = batch[:, 1:]
            accuracy = float((predictions == targets).float().mean().item())
        
        step_result = {
            "step": step,
            "loss": step_data["loss"],
            "accuracy": accuracy,
            "weight_std": layer1_stats["weight_std"],
            "scale": layer1_stats["scale"],
            "ternary_pct_zero": layer1_stats["ternary_zero"],
            "grad_norm_layer1": step_data["gradients"]["layer1"]["grad_norm"],
            "actual_update_norm_layer1": step_data["updates"]["layer1"]["actual_update_norm"],
            "weight_changed_layer1": step_data["updates"]["layer1"]["weight_changed"],
        }
        
        results["steps"].append(step_result)
        
        # Print progress
        if verbose and step in [0, 9, 49, 99]:
            print(f"  Step {step:3d}: loss={step_result['loss']:.4f}, "
                  f"w_std={step_result['weight_std']:.6f}, "
                  f"update_norm={step_result['actual_update_norm_layer1']:.2e}, "
                  f"changed={step_result['weight_changed_layer1']}")
    
    # Final summary
    final = results["steps"][-1]
    if verbose:
        print(f"\nFINAL: loss={final['loss']:.4f}, w_std={final['weight_std']:.6f}, "
              f"scale={final['scale']:.6f}, %0={final['ternary_pct_zero']:.1%}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("OPTIMIZER & UPDATE VERIFICATION")
    print("="*80)
    
    output_dir = Path(__file__).parent.parent / "outputs" / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Run all tests
    for test_config in OPTIMIZER_TESTS:
        print(f"\n{'='*80}")
        print(f"TEST: {test_config['name']}")
        print(f"{'='*80}")
        
        seed_results = {}
        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")
            results = run_single_experiment(test_config, seed, verbose=(seed == 42))
            seed_results[f"seed_{seed}"] = results
        
        all_results[test_config["name"]] = seed_results
    
    # Save results
    results_file = output_dir / "all_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {results_file}")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: SEED-AVERAGED RESULTS")
    print("="*80)
    
    print(f"\n{'Test':<20} {'Seed':<6} {'Final Loss':>10} {'Accuracy':>10} {'W_std':>10} {'Scale':>10} {'% Zeros':>10}")
    print("-"*80)
    
    for test_name, seed_results in all_results.items():
        for seed_key, results in seed_results.items():
            final = results["steps"][-1]
            seed_num = seed_key.split("_")[1]
            print(f"{test_name:<20} {seed_num:<6} {final['loss']:>10.4f} {final['accuracy']:>10.3%} "
                  f"{final['weight_std']:>10.6f} {final['scale']:>10.6f} {final['ternary_pct_zero']:>10.1%}")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    print("""
KEY QUESTIONS:
1. Is weight decay magnitude the cause? Compare WD=0.01 vs WD=0.001 vs WD=0.0
2. Is Adam vs AdamW different? Compare coupled L2 (Adam) vs decoupled (AdamW)
3. Is there a safe WD regime? Look for WD values that don't collapse
4. Do C/D from previous run have bugs? Check if weights actually change

EXPECTED FINDINGS:
- High WD (0.01) → Collapse for both Adam and AdamW
- Low WD (0.0-0.0001) → No collapse
- AdamW might be safer due to decoupled decay
- If weights don't change in ANY run, there's a bug
    """)


if __name__ == "__main__":
    main()
