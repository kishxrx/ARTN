#!/usr/bin/env python3
"""
ROOT CAUSE ANALYSIS: Why do FP32 master weights collapse to zero?

This script investigates the EXACT mechanism causing weight collapse.

HYPOTHESES:
A. Weight decay / optimizer drives FP32 weights to zero
B. Quantization scale gradient drives weights to zero
C. STE implementation causes biased gradients toward zero
D. Thresholding creates irreversible dead zone
E. Interaction between multiple mechanisms

INVESTIGATION PLAN:
1. Optimizer analysis (lr, weight decay, parameter updates)
2. Gradient analysis (before STE, after STE, update ratios)
3. Quantizer analysis (scale behavior, gradient flow)
4. Ablation experiments (5 variants)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import random
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import BitLinear components directly
import importlib.util
spec = importlib.util.spec_from_file_location("bitlinear", Path(__file__).parent.parent / "src" / "model" / "bitlinear.py")
bitlinear_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bitlinear_module)
BitLinear = bitlinear_module.BitLinear
compute_abs_mean_scale = bitlinear_module.compute_abs_mean_scale


# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "root_cause_weight_collapse"

# Same as successful random test / failed TinyStories test
CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 32,
    "num_stories": 5,
    "batch_size": 4,
    "num_steps": 100,  # Short for analysis
    "seed": 42,
    "optimizer": "Adam",
    "lr": 0.01,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
}

# Ablation configurations
ABLATION_CONFIGS = {
    "A_baseline": {
        "weight_decay": 0.01,
        "scale_requires_grad": False,
        "fixed_scale": False,
        "per_channel_scale": False,
    },
    "B_no_weight_decay": {
        "weight_decay": 0.0,
        "scale_requires_grad": False,
        "fixed_scale": False,
        "per_channel_scale": False,
    },
    "C_scale_detached": {
        "weight_decay": 0.01,
        "scale_requires_grad": False,
        "fixed_scale": False,
        "per_channel_scale": False,
        "detach_scale": True,  # Already the case, but explicit
    },
    "D_fixed_scale": {
        "weight_decay": 0.01,
        "scale_requires_grad": False,
        "fixed_scale": True,  # Fix scale at initial value
        "per_channel_scale": False,
    },
    "E_per_channel_scale": {
        "weight_decay": 0.01,
        "scale_requires_grad": False,
        "fixed_scale": False,
        "per_channel_scale": True,
    },
}


# ============================================================================
# BITLINEAR WITH INSTRUMENTATION
# ============================================================================

class InstrumentedBitLinear(BitLinear):
    """BitLinear with detailed logging of weight/gradient/scale behavior."""
    
    def __init__(self, *args, fixed_scale=False, detach_scale=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_scale_value = None
        self.fixed_scale = fixed_scale
        self.detach_scale = detach_scale
        
        # Hooks for capturing gradients
        self.weight_gradient_before_ste = None
        self.weight_gradient_after_ste = None
        self.scale_gradient = None
        
    def quantize_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize with detailed logging."""
        
        # Compute scale
        if self.fixed_scale and self.fixed_scale_value is None:
            # Fix scale at initial value
            with torch.no_grad():
                self.fixed_scale_value = compute_abs_mean_scale(self.weight, self.eps).clone()
        
        if self.fixed_scale:
            scale = self.fixed_scale_value
        elif self.per_channel:
            scale = self.weight.abs().mean(dim=1, keepdim=True).clamp(min=self.eps)
        else:
            scale = self.weight.abs().mean().clamp(min=self.eps)
        
        if self.detach_scale:
            scale = scale.detach()
        
        # Quantize
        weight_scaled = self.weight / scale.clamp(min=1e-8)
        weight_quantized = torch.zeros_like(self.weight)
        weight_quantized[weight_scaled > self.threshold] = 1.0
        weight_quantized[weight_scaled < -self.threshold] = -1.0
        
        # Dequantize
        weight_dequantized = weight_quantized * scale
        
        return weight_dequantized, scale
    
    def get_detailed_stats(self) -> Dict:
        """Get detailed statistics about weights, scale, and quantization."""
        with torch.no_grad():
            scale = compute_abs_mean_scale(self.weight, self.eps)
            weight_scaled = self.weight / scale.clamp(min=1e-8)
            
            weight_quantized = torch.zeros_like(self.weight)
            weight_quantized[weight_scaled > self.threshold] = 1.0
            weight_quantized[weight_scaled < -self.threshold] = -1.0
            
            total = self.weight.numel()
            
            return {
                "weight_mean": self.weight.mean().item(),
                "weight_std": self.weight.std().item(),
                "weight_abs_mean": self.weight.abs().mean().item(),
                "weight_min": self.weight.min().item(),
                "weight_max": self.weight.max().item(),
                "scale": scale.item(),
                "ternary_neg": (weight_quantized == -1).sum().item() / total,
                "ternary_zero": (weight_quantized == 0).sum().item() / total,
                "ternary_pos": (weight_quantized == 1).sum().item() / total,
            }


# ============================================================================
# SIMPLE MODEL FOR TESTING
# ============================================================================

class SimpleTernaryModel(nn.Module):
    """Simple model with instrumented BitLinear layers."""
    
    def __init__(self, config: Dict, ablation_config: Dict = None):
        super().__init__()
        self.config = config
        self.ablation_config = ablation_config or {}
        
        # Embeddings (FP32)
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        
        # Simple stack of BitLinear layers
        self.layers = nn.ModuleList()
        for i in range(4):  # Just 4 layers for testing
            layer = InstrumentedBitLinear(
                config["d_model"],
                config["d_model"],
                bias=True,
                fixed_scale=self.ablation_config.get("fixed_scale", False),
                detach_scale=self.ablation_config.get("detach_scale", False),
                per_channel=self.ablation_config.get("per_channel_scale", False),
            )
            self.layers.append(layer)
        
        # Output projection
        self.output = InstrumentedBitLinear(
            config["d_model"],
            config["vocab_size"],
            bias=False,
            fixed_scale=self.ablation_config.get("fixed_scale", False),
            detach_scale=self.ablation_config.get("detach_scale", False),
            per_channel=self.ablation_config.get("per_channel_scale", False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed
        h = self.embedding(x)
        
        # Pass through layers
        for layer in self.layers:
            h = F.gelu(layer(h))
        
        # Output
        logits = self.output(h)
        return logits


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_tinystories_data(num_stories: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    """Generate synthetic TinyStories-like data (NOT random - more realistic)."""
    # Use a simple language model to generate more realistic sequences
    # For now, use a mixture of frequent and rare tokens
    
    torch.manual_seed(42)
    
    # Create token distribution (Zipf-like)
    token_probs = torch.ones(vocab_size)
    for i in range(vocab_size):
        token_probs[i] = 1.0 / (i + 1) ** 0.5  # Zipf-like
    token_probs = token_probs / token_probs.sum()
    
    # Generate sequences
    data = torch.zeros(num_stories, seq_len, dtype=torch.long)
    for i in range(num_stories):
        for j in range(seq_len):
            # Sample from Zipf distribution
            token = torch.multinomial(token_probs, 1).item()
            data[i, j] = token
    
    return data


# ============================================================================
# OPTIMIZER ANALYSIS
# ============================================================================

def analyze_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    step: int,
) -> Dict:
    """Analyze what happens during one optimizer step."""
    
    # Before forward/backward
    weight_stats_before = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            weight_stats_before[name] = {
                "norm": float(param.data.norm().item()),
                "mean": float(param.data.mean().item()),
                "std": float(param.data.std().item()),
                "abs_mean": float(param.data.abs().mean().item()),
            }
    
    # Forward
    logits = model(batch)
    
    # Compute loss (next-token prediction)
    targets = batch[:, 1:].clone()
    logits_for_loss = logits[:, :-1, :]
    
    loss = F.cross_entropy(
        logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
        targets.reshape(-1)
    )
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Backward
    loss.backward()
    
    # Capture gradients BEFORE optimizer step
    gradient_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            gradient_stats[name] = {
                "grad_norm": float(param.grad.norm().item()),
                "grad_mean": float(param.grad.mean().item()),
                "grad_std": float(param.grad.std().item()),
            }
    
    # Compute gradient clipping effect
    total_grad_norm = float(torch.sqrt(sum(
        p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
    )).item())
    grad_clip = CONFIG["grad_clip"]
    clip_factor = 1.0 if total_grad_norm <= grad_clip else grad_clip / total_grad_norm
    
    # Apply gradient clipping manually for analysis
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Compute update BEFORE applying
    update_stats = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None and 'weight' in name:
                # Adam update approximation
                # update = -lr * grad - lr * weight_decay * weight
                lr = optimizer.param_groups[0]['lr']
                weight_decay = optimizer.param_groups[0]['weight_decay']
                
                # Simple SGD-style update for analysis
                update = -lr * param.grad - lr * weight_decay * param.data
                update_norm = update.norm().item()
                weight_norm = param.data.norm().item()
                
                update_stats[name] = {
                    "update_norm": float(update_norm),
                    "weight_norm": float(weight_norm),
                    "update_ratio": float(update_norm / (weight_norm + 1e-8)),
                    "grad_contribution": float((-lr * param.grad).norm().item()),
                    "wd_contribution": float((-lr * weight_decay * param.data).norm().item()),
                }
    
    # Optimizer step
    optimizer.step()
    
    # After optimizer step
    weight_stats_after = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            weight_stats_before_stats = weight_stats_before[name]
            weight_stats_after[name] = {
                "norm": float(param.data.norm().item()),
                "mean": float(param.data.mean().item()),
                "std": float(param.data.std().item()),
                "abs_mean": float(param.data.abs().mean().item()),
                "norm_change": float(param.data.norm().item() - weight_stats_before_stats["norm"]),
            }
    
    return {
        "step": step,
        "loss": float(loss.item()),
        "weight_before": weight_stats_before,
        "weight_after": weight_stats_after,
        "gradients": gradient_stats,
        "updates": update_stats,
        "total_grad_norm": float(total_grad_norm),
        "clip_factor": float(clip_factor),
    }


# ============================================================================
# RUN ABLATION EXPERIMENT
# ============================================================================

def run_ablation_experiment(
    ablation_name: str,
    ablation_config: Dict,
    data: torch.Tensor,
    num_steps: int = 100,
) -> Dict:
    """Run a single ablation experiment."""
    
    print(f"\n{'='*80}")
    print(f"ABLATION: {ablation_name}")
    print(f"{'='*80}")
    print(f"Config: {ablation_config}")
    
    # Set seed
    torch.manual_seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    
    # Create model
    model = SimpleTernaryModel(CONFIG, ablation_config)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=ablation_config.get("weight_decay", CONFIG["weight_decay"]),
        betas=(0.9, 0.999),
    )
    
    # Track metrics
    results = {
        "ablation": ablation_name,
        "config": ablation_config,
        "steps": [],
    }
    
    # Training loop
    for step in range(num_steps):
        # Sample batch
        batch_idx = torch.randint(0, data.size(0), (CONFIG["batch_size"],))
        batch = data[batch_idx]
        
        # Analyze step
        step_analysis = analyze_optimizer_step(model, optimizer, batch, step)
        
        # Get ternary distribution from first layer
        layer_stats = model.layers[0].get_detailed_stats()
        
        # Compute accuracy
        with torch.no_grad():
            logits = model(batch)
            predictions = logits[:, :-1, :].argmax(dim=-1)
            targets = batch[:, 1:]
            accuracy = (predictions == targets).float().mean().item()
        
        # Get update stats for first layer
        first_layer_key = None
        for key in step_analysis["updates"].keys():
            if "layers.0" in key or "layer_0" in key:
                first_layer_key = key
                break
        
        if first_layer_key is None and step_analysis["updates"]:
            first_layer_key = list(step_analysis["updates"].keys())[0]
        
        if first_layer_key:
            update_data = step_analysis["updates"][first_layer_key]
            update_ratio = update_data.get("update_ratio", 0.0)
            grad_contribution = update_data.get("grad_contribution", 0.0)
            wd_contribution = update_data.get("wd_contribution", 0.0)
        else:
            update_ratio = 0.0
            grad_contribution = 0.0
            wd_contribution = 0.0
        
        # Record
        step_result = {
            "step": step,
            "loss": step_analysis["loss"],
            "accuracy": accuracy,
            "total_grad_norm": step_analysis["total_grad_norm"],
            "clip_factor": step_analysis["clip_factor"],
            
            # First layer weight stats
            "weight_std": layer_stats["weight_std"],
            "weight_abs_mean": layer_stats["weight_abs_mean"],
            "scale": layer_stats["scale"],
            "ternary_pct_zero": layer_stats["ternary_zero"],
            
            # First layer update stats
            "update_ratio": update_ratio,
            "grad_contribution": grad_contribution,
            "wd_contribution": wd_contribution,
        }
        
        results["steps"].append(step_result)
        
        # Print summary at key steps
        if step in [0, 9, 49, 99] or step % 10 == 0:
            print(f"Step {step:3d}: loss={step_result['loss']:.4f}, "
                  f"acc={accuracy:.3%}, "
                  f"w_std={layer_stats['weight_std']:.6f}, "
                  f"scale={layer_stats['scale']:.6f}, "
                  f"%0={layer_stats['ternary_zero']:.1%}")
    
    # Final summary
    final = results["steps"][-1]
    print(f"\nFINAL:")
    print(f"  Loss: {final['loss']:.4f}")
    print(f"  Accuracy: {final['accuracy']:.3%}")
    print(f"  Weight std: {final['weight_std']:.6f}")
    print(f"  Scale: {final['scale']:.6f}")
    print(f"  % zeros: {final['ternary_pct_zero']:.1%}")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("ROOT CAUSE ANALYSIS: Weight Collapse Mechanism")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Generate data
    print(f"\nGenerating TinyStories-like data...")
    data = generate_tinystories_data(
        CONFIG["num_stories"],
        CONFIG["max_seq_len"],
        CONFIG["vocab_size"]
    )
    print(f"  Data shape: {data.shape}")
    print(f"  Unique tokens: {len(torch.unique(data))}")
    
    # Run ablation experiments
    all_results = {}
    
    for ablation_name, ablation_config in ABLATION_CONFIGS.items():
        results = run_ablation_experiment(
            ablation_name,
            ablation_config,
            data,
            num_steps=CONFIG["num_steps"],
        )
        all_results[ablation_name] = results
    
    # Save results
    output_dir = Path(__file__).parent.parent / "outputs" / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("ABLATION COMPARISON")
    print("="*80)
    print(f"\n{'Ablation':<20} {'Final Loss':>10} {'Final Acc':>10} {'Weight Std':>12} {'Scale':>10} {'% Zeros':>10}")
    print("-"*80)
    
    for ablation_name, results in all_results.items():
        final = results["steps"][-1]
        print(f"{ablation_name:<20} {final['loss']:>10.4f} {final['accuracy']:>10.3%} "
              f"{final['weight_std']:>12.6f} {final['scale']:>10.6f} {final['ternary_pct_zero']:>10.1%}")
    
    print("\n" + "="*80)
    print("HYPOTHESIS RANKING")
    print("="*80)
    print("""
Based on ablation results:

Hypothesis A: Weight decay drives weights to zero
  - Compare A_baseline vs B_no_weight_decay
  - If B has significantly higher weight_std, weight decay is a major factor

Hypothesis B: Scale gradient drives weights to zero
  - Current implementation already detaches scale (no gradient)
  - Compare A_baseline vs C_scale_detached (should be identical)
  - If different, there's a gradient flow issue

Hypothesis C: STE causes biased gradients
  - Check gradient statistics in detailed analysis
  - If gradients are systematically negative/positive, bias exists

Hypothesis D: Thresholding creates irreversible dead zone
  - Compare A_baseline vs D_fixed_scale
  - If D maintains non-zero weights, threshold+scale interaction is the issue

Hypothesis E: Per-channel scale helps
  - Compare A_baseline vs E_per_channel_scale
  - If E maintains higher weight_std, per-channel helps
    """)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Review ablation results above
2. Identify which ablation prevents collapse
3. Trace back to the specific mechanism
4. Design minimal fix that addresses root cause
5. Verify fix doesn't break other properties
    """)


if __name__ == "__main__":
    main()
