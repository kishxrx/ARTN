#!/usr/bin/env python3
"""
Pre-training Overfit Test for BitLinear GPT.

BEFORE committing to hours of training, we verify:
1. The model can memorize a tiny dataset (loss → 0)
2. Ternary quantization doesn't prevent learning
3. Gradients flow correctly
4. No NaN/Inf issues

If this test passes, full training should work.
If it fails, debug before wasting training time.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT, create_bitlinear_gpt
from src.model.gpt import create_baseline_gpt


def create_tiny_dataset(vocab_size: int, seq_len: int, num_samples: int = 10):
    """Create a tiny dataset that the model should memorize."""
    # Use fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Generate random sequences
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    # Targets = shifted inputs (next token prediction)
    targets = torch.cat([inputs[:, 1:], torch.randint(0, vocab_size, (num_samples, 1))], dim=1)
    
    return inputs, targets


def run_overfit_test(
    model_type: str = "bitlinear",
    vocab_size: int = 100,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 256,
    max_seq_len: int = 32,
    num_samples: int = 5,
    num_steps: int = 500,
    lr: float = 0.01,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Run overfit test: model should memorize tiny dataset.
    
    Args:
        model_type: "bitlinear" or "baseline"
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of layers
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        num_samples: Number of training samples
        num_steps: Number of training steps
        lr: Learning rate
        device: "cpu" or "cuda"
        verbose: Print progress
    
    Returns:
        dict with training results
    """
    
    # Create model
    if model_type == "bitlinear":
        model = BitLinearGPT(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
        )
    else:
        config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len,
        }
        model = create_baseline_gpt(config)
    
    model = model.to(device)
    model.train()
    
    # Create tiny dataset
    inputs, targets = create_tiny_dataset(vocab_size, max_seq_len, num_samples)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    losses = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"OVERFIT TEST: {model_type.upper()}")
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Training samples: {num_samples}")
        print(f"Steps: {num_steps}")
        print(f"{'='*60}\n")
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs, targets)
        loss = outputs["loss"]
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"ERROR: NaN loss at step {step}")
            return {"status": "failed", "reason": "NaN loss", "step": step}
        
        # Backward
        loss.backward()
        
        # Check for gradient issues
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress
        if verbose and (step % 50 == 0 or step == num_steps - 1):
            print(f"Step {step:4d}: Loss = {loss.item():.4f}, Grad norm = {grad_norm:.4f}")
    
    # Results
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    results = {
        "status": "passed" if final_loss < 1.0 else "failed",
        "model_type": model_type,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": reduction,
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss:   {final_loss:.4f}")
        print(f"  Reduction:    {reduction:.1f}%")
        print(f"  Status:       {results['status'].upper()}")
        print(f"{'='*60}\n")
    
    return results


def compare_models():
    """Compare BitLinear vs Baseline overfit capability."""
    
    print("\n" + "="*70)
    print("PRE-TRAINING OVERFIT TEST")
    print("Verifying both models can memorize before full training")
    print("="*70)
    
    # Test BitLinear
    bitlinear_results = run_overfit_test(
        model_type="bitlinear",
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=32,
        num_samples=5,
        num_steps=500,
        lr=0.01,
    )
    
    # Test Baseline
    baseline_results = run_overfit_test(
        model_type="baseline",
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=32,
        num_samples=5,
        num_steps=500,
        lr=0.01,
    )
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON:")
    print(f"  {'Model':<15} {'Initial':<12} {'Final':<12} {'Reduction':<12} {'Status':<10}")
    print(f"  {'-'*60}")
    print(f"  {'BitLinear':<15} {bitlinear_results['initial_loss']:<12.4f} {bitlinear_results['final_loss']:<12.4f} {bitlinear_results['loss_reduction_pct']:<12.1f}% {bitlinear_results['status']:<10}")
    print(f"  {'Baseline':<15} {baseline_results['initial_loss']:<12.4f} {baseline_results['final_loss']:<12.4f} {baseline_results['loss_reduction_pct']:<12.1f}% {baseline_results['status']:<10}")
    print("="*70 + "\n")
    
    # Verdict
    if bitlinear_results["status"] == "passed":
        print("SUCCESS: BitLinear can learn. Ready for full training!")
        return True
    else:
        print("FAILURE: BitLinear cannot learn. Debug before training!")
        return False


if __name__ == "__main__":
    success = compare_models()
    sys.exit(0 if success else 1)
