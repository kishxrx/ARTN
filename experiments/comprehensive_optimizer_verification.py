#!/usr/bin/env python3
"""
COMPREHENSIVE OPTIMIZER VERIFICATION

Tests:
1. Exact Adam vs AdamW implementation difference
2. Longer stability tests (500 steps)
3. Boundary conditions for safe WD
4. Actual update component measurements
5. Seed stability verification
6. Train/validation split for generalization
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

EXPERIMENT_NAME = "comprehensive_optimizer_verification"

BASE_CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "num_steps": 500,  # Longer run
    "batch_size": 4,
    "seq_len": 32,
    "num_stories": 20,  # More data for train/val split
    "lr": 0.01,
    "grad_clip": 1.0,
}

# Test configurations - boundary conditions
OPTIMIZER_TESTS = [
    {"name": "AdamW_WD0.01", "optimizer": "AdamW", "weight_decay": 0.01},
    {"name": "AdamW_WD0.05", "optimizer": "AdamW", "weight_decay": 0.05},
    {"name": "AdamW_WD0.1", "optimizer": "AdamW", "weight_decay": 0.1},
    {"name": "Adam_WD0.0001", "optimizer": "Adam", "weight_decay": 0.0001},
    {"name": "Adam_WD0.00001", "optimizer": "Adam", "weight_decay": 0.00001},
]

SEEDS = [42, 43, 44]


# ============================================================================
# ADAM VS ADAMW IMPLEMENTATION VERIFICATION
# ============================================================================

def verify_adam_adamw_implementation():
    """
    Verify exact implementation difference between Adam and AdamW.
    
    Adam (coupled L2):
        grad = grad + wd * weight  # WD added to gradient
        Then apply Adam update
    
    AdamW (decoupled):
        Apply Adam update first (without WD in gradient)
        Then: weight = weight - lr * wd * weight  # Separate decay
    """
    
    print("="*80)
    print("ADAM vs ADAMW IMPLEMENTATION VERIFICATION")
    print("="*80)
    
    # Create simple test case
    weight = torch.randn(10, 10, requires_grad=True)
    grad = torch.randn(10, 10)
    
    lr = 0.01
    wd = 0.01
    
    print(f"\nTest parameters:")
    print(f"  LR: {lr}")
    print(f"  Weight decay: {wd}")
    print(f"  Weight norm: {weight.data.norm().item():.6f}")
    print(f"  Grad norm: {grad.norm().item():.6f}")
    
    # Manual Adam step with coupled WD
    print(f"\n--- Adam (coupled L2) ---")
    
    # Create Adam optimizer
    weight_adam = weight.clone().detach().requires_grad_(True)
    optimizer_adam = torch.optim.Adam([weight_adam], lr=lr, weight_decay=wd)
    
    # Simulate backward
    weight_adam.grad = grad.clone()
    
    # What Adam actually does internally:
    # grad_eff = grad + wd * weight
    grad_eff = grad + wd * weight_adam.data
    
    print(f"  Effective gradient norm (grad + wd*weight): {grad_eff.norm().item():.6f}")
    print(f"  WD contribution to gradient: {(wd * weight_adam.data).norm().item():.6f}")
    
    # Step
    weight_before = weight_adam.data.clone()
    optimizer_adam.step()
    weight_after = weight_adam.data.clone()
    
    adam_update = (weight_after - weight_before).norm().item()
    print(f"  Actual update norm: {adam_update:.6f}")
    
    # Manual AdamW step with decoupled WD
    print(f"\n--- AdamW (decoupled) ---")
    
    weight_adamw = weight.clone().detach().requires_grad_(True)
    optimizer_adamw = torch.optim.AdamW([weight_adamw], lr=lr, weight_decay=wd)
    
    # Simulate backward
    weight_adamw.grad = grad.clone()
    
    # What AdamW actually does:
    # 1. Adam update (without WD in gradient)
    # 2. Then: weight = weight - lr * wd * weight
    
    print(f"  Gradient norm (no WD added): {grad.norm().item():.6f}")
    print(f"  Decoupled decay contribution: {(lr * wd * weight_adamw.data).norm().item():.6f}")
    
    # Step
    weight_before = weight_adamw.data.clone()
    optimizer_adamw.step()
    weight_after = weight_adamw.data.clone()
    
    adamw_update = (weight_after - weight_before).norm().item()
    print(f"  Actual update norm: {adamw_update:.6f}")
    
    # Key difference
    print(f"\n--- KEY DIFFERENCE ---")
    print(f"  Adam update norm: {adam_update:.6f}")
    print(f"  AdamW update norm: {adamw_update:.6f}")
    print(f"  Ratio (Adam/AdamW): {adam_update/adamw_update:.2f}x")
    
    print(f"\n--- MECHANISM ---")
    print(f"  Adam:  WD added to gradient → scaled by Adam's adaptive LR")
    print(f"         Can be amplified by small v (gradient variance)")
    print(f"  AdamW: WD applied separately after Adam step")
    print(f"         Independent of adaptive LR, predictable magnitude")
    
    return {
        "adam_update_norm": adam_update,
        "adamw_update_norm": adamw_update,
        "ratio": adam_update / adamw_update,
    }


# ============================================================================
# MODEL
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
# DATA WITH TRAIN/VAL SPLIT
# ============================================================================

def generate_data(config: Dict, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate TinyStories-like data with train/val split."""
    torch.manual_seed(seed)
    
    vocab_size = config["vocab_size"]
    token_probs = torch.ones(vocab_size)
    for i in range(vocab_size):
        token_probs[i] = 1.0 / (i + 1) ** 0.5
    token_probs = token_probs / token_probs.sum()
    
    # Generate train
    train_data = torch.zeros(config["num_stories"], config["seq_len"], dtype=torch.long)
    for i in range(config["num_stories"]):
        for j in range(config["seq_len"]):
            train_data[i, j] = torch.multinomial(token_probs, 1).item()
    
    # Generate validation (different random stories)
    val_data = torch.zeros(config["num_stories"] // 2, config["seq_len"], dtype=torch.long)
    for i in range(config["num_stories"] // 2):
        for j in range(config["seq_len"]):
            val_data[i, j] = torch.multinomial(token_probs, 1).item()
    
    return train_data, val_data


# ============================================================================
# DETAILED UPDATE MEASUREMENT
# ============================================================================

def measure_update_components(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    optimizer_type: str,
    weight_decay: float,
    lr: float,
) -> Dict:
    """
    Measure ACTUAL update components:
    1. Task gradient update
    2. Weight decay contribution (Adam: coupled, AdamW: decoupled)
    3. Total update
    """
    
    # Get first BitLinear layer
    layer = model.layer1
    weight = layer.weight
    
    # Store before state
    weight_before = weight.data.clone()
    grad_before = weight.grad.clone() if weight.grad is not None else torch.zeros_like(weight)
    
    # Forward pass
    logits = model(batch)
    targets = batch[:, 1:].clone()
    logits_for_loss = logits[:, :-1, :]
    
    loss = F.cross_entropy(
        logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
        targets.reshape(-1)
    )
    
    # Zero grad and backward
    optimizer.zero_grad()
    loss.backward()
    
    # Capture task gradient
    task_grad_norm = weight.grad.norm().item()
    
    # Compute what the update WOULD be without optimizer
    # (This is the task gradient contribution)
    
    # Store optimizer state
    state = optimizer.state[weight]
    
    # Measure actual update
    torch.nn.utils.clip_grad_norm_(model.parameters(), BASE_CONFIG["grad_clip"])
    
    optimizer.step()
    
    weight_after = weight.data.clone()
    
    # Compute actual update
    actual_update = weight_after - weight_before
    actual_update_norm = actual_update.norm().item()
    
    # For AdamW: we can decompose the update
    # For Adam: the WD is mixed in, harder to separate
    
    # Estimate components
    if optimizer_type == "AdamW":
        # AdamW: update = adam_step + decay_step
        # decay_step = -lr * wd * weight_before
        decay_contribution = (lr * weight_decay * weight_before).norm().item()
        task_contribution_estimate = abs(actual_update_norm - decay_contribution)
    else:
        # Adam: WD is coupled, gradient is already modified
        # We measure the effective WD in the gradient
        # grad_eff = task_grad + wd * weight
        wd_in_grad = (weight_decay * weight_before).norm().item()
        decay_contribution = wd_in_grad  # This gets scaled by Adam
        task_contribution_estimate = task_grad_norm
    
    return {
        "loss": float(loss.item()),
        "actual_update_norm": float(actual_update_norm),
        "task_grad_norm": float(task_grad_norm),
        "decay_contribution": float(decay_contribution),
        "weight_before_norm": float(weight_before.norm().item()),
        "weight_after_norm": float(weight_after.norm().item()),
    }


# ============================================================================
# RUN SINGLE EXPERIMENT
# ============================================================================

def run_single_experiment(
    test_config: Dict,
    seed: int,
    verbose: bool = True,
) -> Dict:
    """Run single experiment with detailed monitoring."""
    
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
    train_data, val_data = generate_data(BASE_CONFIG, seed)
    
    # Track metrics
    results = {
        "config": test_config,
        "seed": seed,
        "steps": [],
        "update_measurements": [],
    }
    
    # Training loop
    for step in range(BASE_CONFIG["num_steps"]):
        # Sample batch
        batch_idx = torch.randint(0, train_data.size(0), (BASE_CONFIG["batch_size"],))
        batch = train_data[batch_idx]
        
        # Measure update at key steps
        if step in [0, 1, 10, 50, 100, 250, 499]:
            update_data = measure_update_components(
                model, optimizer, batch,
                test_config["optimizer"],
                test_config["weight_decay"],
                BASE_CONFIG["lr"],
            )
            results["update_measurements"].append({
                "step": step,
                **update_data,
            })
        
        # Standard training step
        logits = model(batch)
        targets = batch[:, 1:].clone()
        logits_for_loss = logits[:, :-1, :]
        
        loss = F.cross_entropy(
            logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
            targets.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), BASE_CONFIG["grad_clip"])
        optimizer.step()
        
        # Get weight stats
        with torch.no_grad():
            weight = model.layer1.weight
            scale = compute_abs_mean_scale(weight)
            weight_scaled = weight / scale.clamp(min=1e-8)
            
            weight_quantized = torch.zeros_like(weight)
            weight_quantized[weight_scaled > 0.5] = 1.0
            weight_quantized[weight_scaled < -0.5] = -1.0
            
            total = weight.numel()
            pct_neg = float((weight_quantized == -1).sum().item()) / total
            pct_zero = float((weight_quantized == 0).sum().item()) / total
            pct_pos = float((weight_quantized == 1).sum().item()) / total
            
            # Gradient norm
            grad_norm = 0.0
            if model.layer1.weight.grad is not None:
                grad_norm = float(model.layer1.weight.grad.norm().item())
            
            # Activation RMS (from layer1 output)
            with torch.no_grad():
                h = model.embedding(batch)
                h = model.layer1(h)
                activation_rms = float(h.norm().item() / math.sqrt(h.numel()))
        
        # Compute train accuracy
        with torch.no_grad():
            predictions = logits[:, :-1, :].argmax(dim=-1)
            targets = batch[:, 1:]
            train_accuracy = float((predictions == targets).float().mean().item())
        
        # Compute validation loss and accuracy
        with torch.no_grad():
            val_batch_idx = torch.randint(0, val_data.size(0), (BASE_CONFIG["batch_size"],))
            val_batch = val_data[val_batch_idx]
            
            val_logits = model(val_batch)
            val_targets = val_batch[:, 1:].clone()
            val_logits_for_loss = val_logits[:, :-1, :]
            
            val_loss = F.cross_entropy(
                val_logits_for_loss.reshape(-1, val_logits_for_loss.size(-1)),
                val_targets.reshape(-1)
            )
            
            val_predictions = val_logits[:, :-1, :].argmax(dim=-1)
            val_accuracy = float((val_predictions == val_targets).float().mean().item())
        
        step_result = {
            "step": step,
            "train_loss": float(loss.item()),
            "train_accuracy": train_accuracy,
            "val_loss": float(val_loss.item()),
            "val_accuracy": val_accuracy,
            "weight_std": float(weight.std().item()),
            "scale": float(scale.item()),
            "ternary_neg": pct_neg,
            "ternary_zero": pct_zero,
            "ternary_pos": pct_pos,
            "grad_norm": grad_norm,
            "activation_rms": activation_rms,
        }
        
        results["steps"].append(step_result)
        
        # Print progress
        if verbose and step in [0, 49, 99, 199, 299, 399, 499]:
            print(f"  Step {step:3d}: loss={step_result['train_loss']:.4f}, "
                  f"val_loss={step_result['val_loss']:.4f}, "
                  f"w_std={step_result['weight_std']:.6f}, "
                  f"scale={step_result['scale']:.6f}, "
                  f"%0={step_result['ternary_zero']:.1%}")
    
    # Final summary
    final = results["steps"][-1]
    if verbose:
        print(f"\nFINAL:")
        print(f"  Train loss: {final['train_loss']:.4f}, accuracy: {final['train_accuracy']:.3%}")
        print(f"  Val loss: {final['val_loss']:.4f}, accuracy: {final['val_accuracy']:.3%}")
        print(f"  Weight std: {final['weight_std']:.6f}, scale: {final['scale']:.6f}")
        print(f"  Ternary: -1={final['ternary_neg']:.1%}, 0={final['ternary_zero']:.1%}, +1={final['ternary_pos']:.1%}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("COMPREHENSIVE OPTIMIZER VERIFICATION")
    print("="*80)
    
    # 1. Verify Adam vs AdamW implementation
    print("\n" + "="*80)
    print("1. ADAM vs ADAMW IMPLEMENTATION VERIFICATION")
    print("="*80)
    
    impl_diff = verify_adam_adamw_implementation()
    
    # 2. Run experiments
    output_dir = Path(__file__).parent.parent / "outputs" / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    print("\n" + "="*80)
    print("2. STABILITY TESTS (500 steps, 3 seeds)")
    print("="*80)
    
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
    
    # 3. Save results
    results_file = output_dir / "comprehensive_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {results_file}")
    
    # 4. Summary table
    print("\n" + "="*80)
    print("SUMMARY: FINAL RESULTS (Step 499)")
    print("="*80)
    
    print(f"\n{'Test':<20} {'Seed':<6} {'Train Loss':>10} {'Val Loss':>10} {'Train Acc':>10} {'Val Acc':>10} {'W_std':>10}")
    print("-"*80)
    
    for test_name, seed_results in all_results.items():
        for seed_key, results in seed_results.items():
            final = results["steps"][-1]
            seed_num = seed_key.split("_")[1]
            print(f"{test_name:<20} {seed_num:<6} {final['train_loss']:>10.4f} {final['val_loss']:>10.4f} "
                  f"{final['train_accuracy']:>10.3%} {final['val_accuracy']:>10.3%} {final['weight_std']:>10.6f}")
    
    # 5. Update measurements
    print("\n" + "="*80)
    print("UPDATE COMPONENT MEASUREMENTS")
    print("="*80)
    
    for test_name, seed_results in all_results.items():
        print(f"\n{test_name}:")
        for seed_key, results in seed_results.items():
            if results["update_measurements"]:
                seed_num = seed_key.split("_")[1]
                print(f"\n  Seed {seed_num}:")
                for m in results["update_measurements"]:
                    print(f"    Step {m['step']:3d}: "
                          f"update_norm={m['actual_update_norm']:.2e}, "
                          f"task_grad={m['task_grad_norm']:.2e}, "
                          f"decay={m['decay_contribution']:.2e}")


if __name__ == "__main__":
    main()
