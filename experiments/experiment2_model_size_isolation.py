#!/usr/bin/env python3
"""
EXPERIMENT 2: Model Size Isolation Test

Purpose: Isolate model size as the ONLY variable changed from successful random test.

SUCCESSFUL CONFIG (overfit_test.py):
  - vocab_size: 100
  - d_model: 64
  - n_heads: 4
  - n_layers: 2
  - d_ff: 256
  - max_seq_len: 32
  - num_samples: 5
  - lr: 0.01
  - optimizer: Adam (beta1=0.9 default, beta2=0.999 default, weight_decay=0 default)
  - grad_clip: 1.0
  - seed: 42
  - Model: BitLinearGPT
  - Objective: Next-token prediction
  - Data: Random tokens

THIS EXPERIMENT:
  - d_model: 64 → 256
  - n_heads: 4 → 8
  - n_layers: 2 → 6
  - d_ff: 256 → 1024
  - Model size: 115K → 6.8M (ONLY CHANGE)
  - Everything else: SAME (including vocab=100)

Date: September 4, 2026
"""

import sys
import torch
import torch.nn.functional as F
import json
import math
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT


# =============================================================================
# CONFIGURATION - EXACTLY MATCHES SUCCESSFUL RANDOM TEST EXCEPT model size
# =============================================================================

# SAME as successful random test
SEED = 42
VOCAB_SIZE = 100  # SAME - small vocab
MAX_SEQ_LEN = 32  # SAME
NUM_SAMPLES = 5   # SAME
NUM_STEPS = 500   # SAME
LR = 0.01         # SAME
GRAD_CLIP = 1.0   # SAME
WEIGHT_DECAY = 0.0  # Default (not specified in original)
BETA1 = 0.9       # Default (not specified in original)
BETA2 = 0.999     # Default (not specified in original)

# ONLY CHANGE: Model architecture (115K → 6.8M)
D_MODEL = 256     # CHANGED: 64 → 256
N_HEADS = 8       # CHANGED: 4 → 8
N_LAYERS = 6      # CHANGED: 2 → 6
D_FF = 1024       # CHANGED: 256 → 1024

# Output
OUTPUT_DIR = Path("/root/agent-sandbox/ARTN/outputs/experiment2_model_size_isolation")


# =============================================================================
# CREATE TINY DATASET - SAME AS SUCCESSFUL RANDOM TEST
# =============================================================================

def create_tiny_dataset(vocab_size: int, seq_len: int, num_samples: int = 5):
    """Create a tiny dataset that the model should memorize.
    
    SAME as successful random test (overfit_test.py lines 27-37).
    """
    # Use fixed seed for reproducibility - SAME
    torch.manual_seed(SEED)
    
    # Generate random sequences - SAME
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # Targets = shifted inputs (next token prediction) - SAME
    targets = torch.cat([inputs[:, 1:], torch.randint(0, vocab_size, (num_samples, 1))], dim=1)
    
    return inputs, targets


# =============================================================================
# HELPER FUNCTIONS FOR STATISTICS
# =============================================================================

def compute_stats(tensor, name="tensor"):
    """Compute statistics for a tensor."""
    if tensor is None:
        return None
    
    data = tensor.detach().float()
    return {
        "mean": data.mean().item(),
        "std": data.std().item(),
        "rms": data.pow(2).mean().sqrt().item(),
        "min": data.min().item(),
        "max": data.max().item(),
    }


def get_model_statistics(model):
    """Get embedding, activation, and logit statistics."""
    stats = {}
    
    # Embedding statistics
    emb_stats = compute_stats(model.token_embedding.weight, "token_embedding")
    stats["token_embedding"] = emb_stats
    
    pos_stats = compute_stats(model.position_embedding.weight, "position_embedding")
    stats["position_embedding"] = pos_stats
    
    # lm_head statistics
    lm_head_stats = compute_stats(model.lm_head.weight, "lm_head")
    stats["lm_head"] = lm_head_stats
    
    return stats


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    print("="*80)
    print("EXPERIMENT 2: MODEL SIZE ISOLATION TEST")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print("="*80)
    
    # Reset seed
    torch.manual_seed(SEED)
    
    # ==========================================================================
    # Verify configuration matches successful test
    # ==========================================================================
    print("\n[Configuration Verification]")
    print("Checking that ONLY model size changed from successful random test:\n")
    
    print("  SAME AS SUCCESSFUL TEST:")
    print(f"    vocab_size: {VOCAB_SIZE}")
    print(f"    max_seq_len: {MAX_SEQ_LEN}")
    print(f"    num_samples: {NUM_SAMPLES}")
    print(f"    num_steps: {NUM_STEPS}")
    print(f"    lr: {LR}")
    print(f"    beta1: {BETA1}")
    print(f"    beta2: {BETA2}")
    print(f"    weight_decay: {WEIGHT_DECAY}")
    print(f"    grad_clip: {GRAD_CLIP}")
    print(f"    seed: {SEED}")
    print(f"    Model: BitLinearGPT")
    print(f"    Objective: Next-token prediction")
    print(f"    Data: Random tokens")
    
    print("\n  ONLY CHANGES (Model Architecture):")
    print(f"    d_model: 64 → {D_MODEL}")
    print(f"    n_heads: 4 → {N_HEADS}")
    print(f"    n_layers: 2 → {N_LAYERS}")
    print(f"    d_ff: 256 → {D_FF}")
    
    # ==========================================================================
    # Create model - CHANGED architecture
    # ==========================================================================
    print("\n[Model] Creating BitLinearGPT with LARGE architecture...")
    torch.manual_seed(SEED)  # Reset seed for initialization
    model = BitLinearGPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {num_params:,}")
    
    # Verify expected size (~6.8M)
    expected_params = (
        VOCAB_SIZE * D_MODEL +  # token_embedding
        MAX_SEQ_LEN * D_MODEL +  # position_embedding
        N_LAYERS * (4 * D_MODEL * D_MODEL + 2 * D_MODEL * D_FF) +  # attention + ffn
        2 * D_MODEL +  # final layer norm
        D_MODEL * VOCAB_SIZE  # lm_head
    )
    print(f"[Model] Expected params: ~{expected_params:,}")
    
    # Get initial model statistics
    print("\n[Initial Model Statistics]")
    initial_stats = get_model_statistics(model)
    for name, stats in initial_stats.items():
        if stats:
            print(f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, rms={stats['rms']:.4f}")
    
    # ==========================================================================
    # Create dataset - SAME as successful test
    # ==========================================================================
    print("\n[Dataset] Creating random-token dataset...")
    inputs, targets = create_tiny_dataset(VOCAB_SIZE, MAX_SEQ_LEN, NUM_SAMPLES)
    
    print(f"[Dataset] Shape: inputs={inputs.shape}, targets={targets.shape}")
    print(f"[Dataset] Vocab range: [0, {VOCAB_SIZE})")
    
    # Verify next-token objective
    print("\n[Verification] Checking next-token objective:")
    print(f"  inputs[0][:5]: {inputs[0][:5].tolist()}")
    print(f"  targets[0][:5]: {targets[0][:5].tolist()}")
    for i in range(4):
        assert targets[0, i].item() == inputs[0, i+1].item(), \
            f"ASSERTION FAILED: targets[{i}] != inputs[{i+1}]"
    print("  ✓ Verified: targets[i] == inputs[i+1] (next-token prediction)")
    
    # ==========================================================================
    # Create optimizer - SAME as successful test
    # ==========================================================================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )
    
    print(f"\n[Optimizer] Adam")
    print(f"  LR: {LR}")
    print(f"  Beta1: {BETA1}")
    print(f"  Beta2: {BETA2}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    
    # ==========================================================================
    # Training loop
    # ==========================================================================
    print("\n" + "="*80)
    print("TRAINING START")
    print("="*80)
    
    results = {
        "config": {
            "vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "d_ff": D_FF,
            "max_seq_len": MAX_SEQ_LEN,
            "num_samples": NUM_SAMPLES,
            "num_steps": NUM_STEPS,
            "lr": LR,
            "beta1": BETA1,
            "beta2": BETA2,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "seed": SEED,
        },
        "metrics": {},
        "initial_stats": initial_stats,
    }
    
    model.train()
    losses = []
    
    for step in range(NUM_STEPS):
        optimizer.zero_grad()
        
        # Forward - SAME as successful test
        outputs = model(inputs, targets)
        loss = outputs["loss"]
        logits = outputs["logits"]
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\n[ERROR] NaN loss at step {step}")
            results["status"] = "failed"
            results["reason"] = "NaN loss"
            return results
        
        # Backward - SAME as successful test
        loss.backward()
        
        # Gradient clipping - SAME as successful test
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        # Report at key steps: 0, 50, 100, 250, 500
        if step in [0, 50, 100, 250, 500] or step == NUM_STEPS - 1:
            # Compute accuracy
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                correct = (predictions == targets).float()
                accuracy = correct.mean().item()
                
                # Logit statistics
                logit_stats = compute_stats(logits, "logits")
            
            perplexity = math.exp(loss.item()) if loss.item() < 10 else float('inf')
            
            print(f"\nStep {step:4d}:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  Accuracy: {accuracy*100:.2f}%")
            print(f"  Grad norm: {grad_norm:.4f}")
            print(f"  Logit RMS: {logit_stats['rms']:.4f}, max: {logit_stats['max']:.4f}")
            
            results["metrics"][step] = {
                "loss": loss.item(),
                "perplexity": perplexity,
                "accuracy": accuracy,
                "grad_norm": grad_norm.item(),
                "logit_rms": logit_stats['rms'],
                "logit_max": logit_stats['max'],
            }
    
    # ==========================================================================
    # Final metrics
    # ==========================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss reduction: {loss_reduction:.1f}%")
    
    # Compute final accuracy
    with torch.no_grad():
        outputs = model(inputs, targets)
        logits = outputs["logits"]
        predictions = logits.argmax(dim=-1)
        correct = (predictions == targets).float()
        final_accuracy = correct.mean().item()
        
        # Final logit statistics
        logit_stats = compute_stats(logits, "logits")
    
    final_perplexity = math.exp(final_loss) if final_loss < 10 else float('inf')
    
    print(f"  Final perplexity: {final_perplexity:.2f}")
    print(f"  Final accuracy: {final_accuracy*100:.2f}%")
    print(f"  Final logit RMS: {logit_stats['rms']:.4f}, max: {logit_stats['max']:.4f}")
    
    results["initial_loss"] = initial_loss
    results["final_loss"] = final_loss
    results["loss_reduction_pct"] = loss_reduction
    results["final_perplexity"] = final_perplexity
    results["final_accuracy"] = final_accuracy
    results["final_logit_stats"] = logit_stats
    
    # ==========================================================================
    # Final model statistics
    # ==========================================================================
    print("\n[Final Model Statistics]")
    final_stats = get_model_statistics(model)
    for name, stats in final_stats.items():
        if stats:
            print(f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, rms={stats['rms']:.4f}")
    
    results["final_stats"] = final_stats
    
    # ==========================================================================
    # Generate samples
    # ==========================================================================
    print("\n[Generated Samples]")
    model.eval()
    samples = []
    
    with torch.no_grad():
        for i in range(3):
            # Start with random token
            start_token = torch.randint(0, VOCAB_SIZE, (1, 1))
            
            # Generate
            generated = model.generate(
                start_token,
                max_new_tokens=20,
                temperature=1.0,
                top_k=50,
            )
            
            sample = generated[0].tolist()
            samples.append(sample)
            print(f"\nSample {i+1}: {sample}")
    
    results["samples"] = samples
    
    # ==========================================================================
    # Loss trajectory
    # ==========================================================================
    print("\n[Loss Trajectory]")
    for step in [0, 50, 100, 250, 500]:
        if step < len(losses):
            print(f"  Step {step:4d}: {losses[step]:.4f}")
    
    # ==========================================================================
    # Save results
    # ==========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "results.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Results] Saved to: {results_file}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model size: {num_params:,} params")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss reduction: {loss_reduction:.1f}%")
    print(f"Final accuracy: {final_accuracy*100:.2f}%")
    print(f"Final perplexity: {final_perplexity:.2f}")
    print("\nCOMPARISON:")
    print(f"  Successful test (115K, vocab=100): loss → 0.0001, 100% reduction")
    print(f"  Exp 1 (115K, vocab=4000): loss → 0.0004, 99.99% reduction")
    print(f"  This test (6.8M, vocab=100): loss → {final_loss:.4f}, {loss_reduction:.1f}% reduction")
    print("="*80)
    
    return results


if __name__ == "__main__":
    run_experiment()
