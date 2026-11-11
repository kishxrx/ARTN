#!/usr/bin/env python3
"""
FORENSIC BITLINEAR DIAGNOSTIC

Purpose: Identify exactly WHAT in BitLinear is causing optimization failure.

Compare with FP32 baseline:
  - FP32: 98.7% accuracy
  - BitLinear: 9.0% accuracy

Measure at: Init, Step 1, 10, 50, 500

For each BitLinear layer:
  1. FP32 master weight statistics
  2. Ternary weight distribution
  3. Quantization/scale behavior
  4. Gradient flow
  5. Activation statistics
  6. Comparison with FP32 equivalent

Date: September 4, 2026
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import random
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT, BitLinear
from src.tokenizer.bpe_tokenizer import BPETokenizer


# =============================================================================
# CONFIGURATION - IDENTICAL TO TERNARY TEST
# =============================================================================

SEED = 42

MODEL_CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 32,
    "threshold": 0.5,
    "dropout": 0.0,
}

OPTIMIZER_CONFIG = {
    "optimizer": "Adam",
    "lr": 0.01,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}

TRAINING_CONFIG = {
    "num_stories": 5,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "total_steps": 500,
    "max_seq_len": 32,
}

TOKENIZER_PATH = "/root/agent-sandbox/ARTN/outputs/artn-50k-best-config/tokenizer.json"

OUTPUT_DIR = Path("/root/agent-sandbox/ARTN/outputs/forensic_bitlinear_diagnostic")


# =============================================================================
# DATASET
# =============================================================================

class TinyStoriesMemorizationDataset(Dataset):
    def __init__(self, stories, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []
        
        for story in stories:
            text = story["text"]
            tokens = tokenizer.encode(text)[:max_length - 1]
            tokens = tokens + [tokenizer.eos_token_id]
            padding_length = max_length - len(tokens)
            tokens = tokens + [tokenizer.pad_token_id] * padding_length
            self.sequences.append(tokens)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.sequences[idx], dtype=torch.long)
        inputs = input_ids[:-1].clone()
        targets = input_ids[1:].clone()
        targets[targets == self.tokenizer.pad_token_id] = -100
        return {"input_ids": inputs, "targets": targets}


# =============================================================================
# FORENSIC HOOKS
# =============================================================================

class BitLinearForensicTracker:
    """Track all BitLinear layer statistics during training."""
    
    def __init__(self, model):
        self.model = model
        self.layers = {}
        self.initial_weights = {}
        self.hooks = []
        
        # Find all BitLinear layers
        for name, module in model.named_modules():
            if isinstance(module, BitLinear):
                self.layers[name] = module
                # Store initial weights
                self.initial_weights[name] = module.weight.data.clone()
    
    def collect_stats(self, step, include_grads=False, include_activations=False):
        """Collect statistics from all BitLinear layers."""
        stats = {
            "step": step,
            "layers": {},
        }
        
        for name, layer in self.layers.items():
            layer_stats = {}
            
            # Get FP32 master weight
            weight = layer.weight.data
            
            # 1. FP32 master weight statistics
            layer_stats["weight_mean"] = weight.mean().item()
            layer_stats["weight_std"] = weight.std().item()
            layer_stats["weight_min"] = weight.min().item()
            layer_stats["weight_max"] = weight.max().item()
            
            # Weight change from initialization
            if name in self.initial_weights:
                weight_change = (weight - self.initial_weights[name]).abs().mean().item()
                layer_stats["weight_change_from_init"] = weight_change
            
            # 2. Ternary weight distribution
            with torch.no_grad():
                # Compute ternary weights
                scale = weight.abs().mean().clamp(min=1e-8)
                threshold = layer.threshold * scale
                ternary = torch.zeros_like(weight)
                ternary[weight > threshold] = 1.0
                ternary[weight < -threshold] = -1.0
                
                total = weight.numel()
                layer_stats["ternary_pct_neg1"] = (ternary == -1).sum().item() / total
                layer_stats["ternary_pct_0"] = (ternary == 0).sum().item() / total
                layer_stats["ternary_pct_pos1"] = (ternary == 1).sum().item() / total
                layer_stats["ternary_scale"] = scale.item()
            
            # 3. Quantization scale
            layer_stats["quant_scale"] = scale.item()
            
            # 4. Gradient statistics
            if include_grads and layer.weight.grad is not None:
                grad = layer.weight.grad.data
                layer_stats["grad_norm"] = grad.norm().item()
                layer_stats["grad_mean"] = grad.mean().item()
                layer_stats["grad_std"] = grad.std().item()
                layer_stats["grad_pct_zero"] = (grad == 0).sum().item() / grad.numel()
            elif include_grads:
                layer_stats["grad_norm"] = None
                layer_stats["grad_pct_zero"] = 1.0
            
            stats["layers"][name] = layer_stats
        
        return stats
    
    def register_activation_hooks(self):
        """Register forward hooks to capture activations."""
        self.activation_data = {}
        
        def make_hook(name):
            def hook(module, input, output):
                self.activation_data[name] = {
                    "input": input[0].detach() if len(input) > 0 else None,
                    "output": output.detach() if isinstance(output, torch.Tensor) else output[0].detach(),
                }
            return hook
        
        for name, layer in self.layers.items():
            hook = layer.register_forward_hook(make_hook(name))
            self.hooks.append(hook)
    
    def get_activation_stats(self):
        """Get activation statistics from the last forward pass."""
        stats = {}
        
        for name, data in self.activation_data.items():
            if data["input"] is not None:
                inp = data["input"]
                stats[f"{name}_input_mean"] = inp.mean().item()
                stats[f"{name}_input_std"] = inp.std().item()
                stats[f"{name}_input_rms"] = inp.pow(2).mean().sqrt().item()
            
            if data["output"] is not None:
                out = data["output"]
                stats[f"{name}_output_mean"] = out.mean().item()
                stats[f"{name}_output_std"] = out.std().item()
                stats[f"{name}_output_rms"] = out.pow(2).mean().sqrt().item()
        
        return stats
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# =============================================================================
# BITLINEAR vs FP32 COMPARISON
# =============================================================================

def compare_bitlinear_vs_fp32(bitlinear_layer, input_tensor):
    """Compare BitLinear forward pass with equivalent FP32 linear."""
    
    # Get BitLinear output
    with torch.no_grad():
        bitlinear_out = bitlinear_layer(input_tensor)
    
    # Create equivalent FP32 linear
    fp32_linear = nn.Linear(
        bitlinear_layer.in_features,
        bitlinear_layer.out_features,
        bias=False
    )
    
    # Use same underlying FP32 weights
    fp32_linear.weight.data = bitlinear_layer.weight.data.clone()
    
    # Get FP32 output
    with torch.no_grad():
        fp32_out = fp32_linear(input_tensor)
    
    # Compare
    diff = (bitlinear_out - fp32_out).abs()
    
    return {
        "bitlinear_output_rms": bitlinear_out.pow(2).mean().sqrt().item(),
        "fp32_output_rms": fp32_out.pow(2).mean().sqrt().item(),
        "output_diff_mean": diff.mean().item(),
        "output_diff_max": diff.max().item(),
        "output_diff_rms": diff.pow(2).mean().sqrt().item(),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("FORENSIC BITLINEAR DIAGNOSTIC")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print("="*80)
    
    torch.manual_seed(SEED)
    random.seed(SEED)
    
    # ==========================================================================
    # Load tokenizer and data
    # ==========================================================================
    print("\n[Tokenizer] Loading...")
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    
    print("\n[Dataset] Loading 5 TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    random.seed(SEED)
    indices = sorted(random.sample(range(len(dataset)), TRAINING_CONFIG["num_stories"]))
    stories = [dataset[i] for i in indices]
    
    train_dataset = TinyStoriesMemorizationDataset(
        stories, tokenizer, TRAINING_CONFIG["max_seq_len"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    
    # ==========================================================================
    # Create model
    # ==========================================================================
    print("\n[Model] Creating BitLinearGPT...")
    torch.manual_seed(SEED)
    model = BitLinearGPT(
        vocab_size=MODEL_CONFIG["vocab_size"],
        d_model=MODEL_CONFIG["d_model"],
        n_heads=MODEL_CONFIG["n_heads"],
        n_layers=MODEL_CONFIG["n_layers"],
        d_ff=MODEL_CONFIG["d_ff"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
        threshold=MODEL_CONFIG["threshold"],
        dropout=MODEL_CONFIG["dropout"],
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # ==========================================================================
    # Create forensic tracker
    # ==========================================================================
    print("\n[Forensic] Creating BitLinear tracker...")
    tracker = BitLinearForensicTracker(model)
    tracker.register_activation_hooks()
    
    print(f"  Found {len(tracker.layers)} BitLinear layers:")
    for name in tracker.layers.keys():
        print(f"    - {name}")
    
    # ==========================================================================
    # Create optimizer
    # ==========================================================================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=OPTIMIZER_CONFIG["lr"],
        betas=(OPTIMIZER_CONFIG["beta1"], OPTIMIZER_CONFIG["beta2"]),
        weight_decay=OPTIMIZER_CONFIG["weight_decay"],
    )
    
    # ==========================================================================
    # Collect initialization stats
    # ==========================================================================
    print("\n" + "="*80)
    print("INITIALIZATION STATISTICS")
    print("="*80)
    
    all_stats = []
    
    init_stats = tracker.collect_stats(step=0)
    all_stats.append(init_stats)
    
    # Print init stats
    print("\nLayer-by-layer initialization:")
    for name, layer_stats in init_stats["layers"].items():
        print(f"\n{name}:")
        print(f"  Weight mean: {layer_stats['weight_mean']:.6f}, std: {layer_stats['weight_std']:.6f}")
        print(f"  Ternary dist: -1={layer_stats['ternary_pct_neg1']*100:.1f}%, "
              f"0={layer_stats['ternary_pct_0']*100:.1f}%, "
              f"+1={layer_stats['ternary_pct_pos1']*100:.1f}%")
        print(f"  Scale: {layer_stats['quant_scale']:.6f}")
    
    # ==========================================================================
    # Training with forensic tracking
    # ==========================================================================
    print("\n" + "="*80)
    print("TRAINING WITH FORENSIC TRACKING")
    print("="*80)
    
    checkpoint_steps = [1, 10, 50, 500]
    
    results = {
        "config": {
            "model": MODEL_CONFIG,
            "optimizer": OPTIMIZER_CONFIG,
            "training": TRAINING_CONFIG,
        },
        "bitlinear_stats": {},
        "activation_stats": {},
        "comparison_stats": {},
        "training_metrics": {},
    }
    
    model.train()
    data_iter = iter(train_loader)
    
    for step in range(1, TRAINING_CONFIG["total_steps"] + 1):
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for _ in range(TRAINING_CONFIG["gradient_accumulation_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            input_ids = batch["input_ids"]
            targets = batch["targets"]
            
            outputs = model(input_ids, targets=targets)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            if torch.isnan(loss):
                print(f"\n[ERROR] NaN loss at step {step}")
                continue
            
            scaled_loss = loss / TRAINING_CONFIG["gradient_accumulation_steps"]
            scaled_loss.backward()
            accumulated_loss += loss.item()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), OPTIMIZER_CONFIG["max_grad_norm"]
        )
        
        optimizer.step()
        
        # Collect stats at checkpoint steps
        if step in checkpoint_steps:
            print(f"\n{'='*80}")
            print(f"STEP {step}")
            print(f"{'='*80}")
            
            # Training metrics
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                correct = (predictions == targets) & (targets != -100)
                accuracy = correct.sum().item() / (targets != -100).sum().item()
                perplexity = math.exp(accumulated_loss / TRAINING_CONFIG["gradient_accumulation_steps"])
            
            print(f"\nTraining metrics:")
            print(f"  Loss: {accumulated_loss / TRAINING_CONFIG['gradient_accumulation_steps']:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  Accuracy: {accuracy*100:.2f}%")
            print(f"  Grad norm: {grad_norm:.4f}")
            
            results["training_metrics"][step] = {
                "loss": accumulated_loss / TRAINING_CONFIG["gradient_accumulation_steps"],
                "perplexity": perplexity,
                "accuracy": accuracy,
                "grad_norm": grad_norm.item(),
            }
            
            # BitLinear stats
            stats = tracker.collect_stats(step=step, include_grads=True)
            all_stats.append(stats)
            results["bitlinear_stats"][step] = stats
            
            # Print layer stats
            print(f"\nBitLinear layer statistics:")
            for name, layer_stats in stats["layers"].items():
                print(f"\n{name}:")
                print(f"  Weight: mean={layer_stats['weight_mean']:.6f}, "
                      f"std={layer_stats['weight_std']:.6f}, "
                      f"change={layer_stats.get('weight_change_from_init', 0):.6f}")
                print(f"  Ternary: -1={layer_stats['ternary_pct_neg1']*100:.1f}%, "
                      f"0={layer_stats['ternary_pct_0']*100:.1f}%, "
                      f"+1={layer_stats['ternary_pct_pos1']*100:.1f}%")
                print(f"  Scale: {layer_stats['quant_scale']:.6f}")
                if layer_stats["grad_norm"] is not None:
                    print(f"  Gradient: norm={layer_stats['grad_norm']:.6f}, "
                          f"pct_zero={layer_stats['grad_pct_zero']*100:.1f}%")
            
            # Activation stats
            act_stats = tracker.get_activation_stats()
            results["activation_stats"][step] = act_stats
            
            print(f"\nActivation statistics (sample layer):")
            sample_layer = list(tracker.layers.keys())[0] if tracker.layers else None
            if sample_layer:
                inp_key = f"{sample_layer}_input_rms"
                out_key = f"{sample_layer}_output_rms"
                if inp_key in act_stats:
                    print(f"  Input RMS: {act_stats[inp_key]:.6f}")
                if out_key in act_stats:
                    print(f"  Output RMS: {act_stats[out_key]:.6f}")
            
            # Compare BitLinear vs FP32 for first layer
            if len(tracker.layers) > 0:
                first_layer_name = list(tracker.layers.keys())[0]
                first_layer = tracker.layers[first_layer_name]
                
                # Create sample input
                sample_input = torch.randn(1, 32, first_layer.in_features)
                
                comparison = compare_bitlinear_vs_fp32(first_layer, sample_input)
                results["comparison_stats"][step] = comparison
                
                print(f"\nBitLinear vs FP32 comparison ({first_layer_name}):")
                print(f"  BitLinear output RMS: {comparison['bitlinear_output_rms']:.6f}")
                print(f"  FP32 output RMS: {comparison['fp32_output_rms']:.6f}")
                print(f"  Output diff RMS: {comparison['output_diff_rms']:.6f}")
        
        # Progress
        if step % 100 == 0 and step not in checkpoint_steps:
            print(f"Step {step}: loss={accumulated_loss / TRAINING_CONFIG['gradient_accumulation_steps']:.4f}")
    
    # ==========================================================================
    # Final evaluation
    # ==========================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    model.eval()
    total_correct = 0
    total_active = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for i in range(len(train_dataset)):
            batch = train_dataset[i]
            input_ids = batch["input_ids"].unsqueeze(0)
            targets = batch["targets"].unsqueeze(0)
            
            outputs = model(input_ids, targets=targets)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            predictions = logits.argmax(dim=-1)
            correct = (predictions == targets) & (targets != -100)
            
            total_correct += correct.sum().item()
            total_active += (targets != -100).sum().item()
            total_loss += loss.item()
    
    final_loss = total_loss / len(train_dataset)
    final_accuracy = total_correct / total_active if total_active > 0 else 0
    
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Final accuracy: {final_accuracy*100:.2f}%")
    
    results["final_loss"] = final_loss
    results["final_accuracy"] = final_accuracy
    
    # ==========================================================================
    # Remove hooks
    # ==========================================================================
    tracker.remove_hooks()
    
    # ==========================================================================
    # Save results
    # ==========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "results.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Results] Saved to: {results_file}")
    
    # ==========================================================================
    # Analysis
    # ==========================================================================
    print("\n" + "="*80)
    print("FORENSIC ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n1. LAYER-BY-LAYER COMPARISON (Step 500 vs Init):")
    print("-" * 80)
    
    init_layers = all_stats[0]["layers"]
    final_layers = all_stats[-1]["layers"]
    
    for name in init_layers.keys():
        init = init_layers[name]
        final = final_layers[name]
        
        print(f"\n{name}:")
        print(f"  Weight change: {final.get('weight_change_from_init', 0):.6f}")
        print(f"  Ternary dist init: -1={init['ternary_pct_neg1']*100:.1f}%, "
              f"0={init['ternary_pct_0']*100:.1f}%, +1={init['ternary_pct_pos1']*100:.1f}%")
        print(f"  Ternary dist final: -1={final['ternary_pct_neg1']*100:.1f}%, "
              f"0={final['ternary_pct_0']*100:.1f}%, +1={final['ternary_pct_pos1']*100:.1f}%")
        print(f"  Scale init: {init['quant_scale']:.6f}, final: {final['quant_scale']:.6f}")
    
    print("\n2. FIRST POINT OF DIVERGENCE:")
    print("-" * 80)
    
    # Compare with FP32 baseline
    print("\nFP32 baseline: 100% accuracy at step 50, 98.7% final")
    print(f"BitLinear:     {results['training_metrics'].get(50, {}).get('accuracy', 0)*100:.1f}% accuracy at step 50, "
          f"{final_accuracy*100:.1f}% final")
    
    print("\n3. ROOT-CAUSE HYPOTHESES (ranked by evidence):")
    print("-" * 80)
    
    # Check for gradient flow
    grad_norms_500 = [s["grad_norm"] for s in final_layers.values() if s.get("grad_norm") is not None]
    if grad_norms_500:
        avg_grad = sum(grad_norms) / len(grad_norms) if (grad_norms := grad_norms_500) else 0
        print(f"\n[1] Gradient flow: avg grad norm at step 500 = {avg_grad:.6f}")
        if avg_grad < 0.001:
            print("    ⚠️ LOW gradient norm - possible vanishing gradients")
        elif avg_grad > 1.0:
            print("    ⚠️ HIGH gradient norm - possible exploding gradients")
        else:
            print("    ✅ Gradient norm in reasonable range")
    
    # Check ternary distribution
    pct_zeros = [s["ternary_pct_0"] for s in final_layers.values()]
    if pct_zeros:
        avg_sparsity = sum(pct_zeros) / len(pct_zeros)
        print(f"\n[2] Ternary sparsity: avg {avg_sparsity*100:.1f}% zeros")
        if avg_sparsity > 0.9:
            print("    ⚠️ VERY HIGH sparsity - most weights are 0")
        elif avg_sparsity > 0.5:
            print("    ⚠️ HIGH sparsity - many weights are 0")
        else:
            print("    ✅ Reasonable sparsity")
    
    # Check scale stability
    scales_init = [s["quant_scale"] for s in init_layers.values()]
    scales_final = [s["quant_scale"] for s in final_layers.values()]
    if scales_init and scales_final:
        print(f"\n[3] Quantization scale: init avg {sum(scales_init)/len(scales_init):.6f}, "
              f"final avg {sum(scales_final)/len(scales_final):.6f}")
        if abs(sum(scales_final)/len(scales_final) - sum(scales_init)/len(scales_init)) < 1e-6:
            print("    ⚠️ Scale BARELY CHANGED during training")
    
    # Check BitLinear vs FP32 output difference
    if results["comparison_stats"]:
        comp_500 = results["comparison_stats"].get(500, {})
        print(f"\n[4] BitLinear vs FP32 output difference (step 500):")
        print(f"    BitLinear RMS: {comp_500.get('bitlinear_output_rms', 0):.6f}")
        print(f"    FP32 RMS: {comp_500.get('fp32_output_rms', 0):.6f}")
        print(f"    Difference RMS: {comp_500.get('output_diff_rms', 0):.6f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
