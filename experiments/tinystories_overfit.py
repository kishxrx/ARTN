#!/usr/bin/env python3
"""
TinyStories Overfit Test for Ternary BitLinear Model.

CRITICAL: Uses SAME configuration as real training to diagnose properly.

Purpose: Determine if ternary implementation can memorize structured 
TinyStories data under exact training configuration.

Date: Aug 16, 2026
"""

import sys
import torch
import random
import json
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer
from datasets import load_dataset


# =============================================================================
# Configuration - SAME AS REAL TRAINING
# =============================================================================

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

# Model architecture - SAME as train_optimized.py
MODEL_CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
}

# Optimizer - TESTING ADAM + LR=0.01 (SAME AS RANDOM TEST)
OPTIMIZER_CONFIG = {
    "optimizer": "Adam",  # Same as random test
    "lr": 0.01,  # Same as random test (NOT 0.001)
    "beta1": 0.5,  # NOT default 0.9
    "beta2": 0.98,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}

# Training
TRAINING_CONFIG = {
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 8,
    "total_steps": 1000,
    "max_seq_len": 256,
}

# Dataset
DATASET_CONFIG = {
    "num_stories": 20,
    "seed": SEED,
}

# Tokenizer path - SAME as real training
TOKENIZER_PATH = "/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json"

# Logging
LOG_INTERVAL = 50
SAMPLE_INTERVAL = 200


# =============================================================================
# Dataset
# =============================================================================

def select_tinystories_subset(num_stories=20, seed=42):
    """Select deterministic random subset of TinyStories."""
    
    print(f"\n{'='*60}")
    print("LOADING TINYSTORIES SUBSET")
    print(f"{'='*60}")
    
    # Load dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    total_stories = len(dataset)
    
    print(f"Total TinyStories: {total_stories:,}")
    
    # Deterministic random selection
    random.seed(seed)
    indices = sorted(random.sample(range(total_stories), num_stories))
    
    # Select stories
    subset = dataset.select(indices)
    stories = [item["text"] for item in subset]
    
    # Log selection details
    print(f"Seed: {seed}")
    print(f"Selected {num_stories} stories")
    print(f"Indices: {indices}")
    print(f"\nStory lengths (chars):")
    for i, story in enumerate(stories):
        print(f"  Story {i+1}: {len(story)} chars")
    
    # Save selection info
    selection_info = {
        "seed": seed,
        "num_stories": num_stories,
        "indices": indices,
        "story_lengths": [len(s) for s in stories],
        "timestamp": datetime.now().isoformat(),
    }
    
    return stories, selection_info


def tokenize_stories(stories, tokenizer, max_length=256):
    """Tokenize stories using SAME preprocessing as real training."""
    
    print(f"\n{'='*60}")
    print("TOKENIZING STORIES")
    print(f"{'='*60}")
    
    tokenized = []
    total_tokens = 0
    
    for i, story in enumerate(stories):
        # Tokenize - SAME as TinyStoriesDataset in train_optimized.py
        tokens = tokenizer.encode(story)[:max_length - 1]
        tokens = tokens + [tokenizer.eos_token_id]
        
        tokenized.append(tokens)
        total_tokens += len(tokens)
        
        if i < 3:  # Show first 3 examples
            print(f"Story {i+1}: {len(tokens)} tokens")
            print(f"  First 20 tokens: {tokens[:20]}")
            print(f"  Decoded first 50 chars: {tokenizer.decode(tokens[:30])}")
    
    print(f"\nTotal tokens: {total_tokens:,}")
    print(f"Avg tokens/story: {total_tokens / len(stories):.1f}")
    
    return tokenized


# =============================================================================
# Model
# =============================================================================

def create_model(device="cpu"):
    """Create BitLinearGPT with SAME architecture as real training."""
    
    print(f"\n{'='*60}")
    print("CREATING MODEL")
    print(f"{'='*60}")
    
    model = BitLinearGPT(
        vocab_size=MODEL_CONFIG["vocab_size"],
        d_model=MODEL_CONFIG["d_model"],
        n_heads=MODEL_CONFIG["n_heads"],
        n_layers=MODEL_CONFIG["n_layers"],
        d_ff=MODEL_CONFIG["d_ff"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
    )
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BitLinearGPT (Ternary)")
    print(f"Parameters: {num_params:,}")
    print(f"Config:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    
    return model


# =============================================================================
# Training
# =============================================================================

def train_overfit(
    model,
    tokenized_stories,
    tokenizer,
    device="cpu",
    output_dir=None,
):
    """Train to overfit on TinyStories subset."""
    
    print(f"\n{'='*60}")
    print("OVERFIT TRAINING")
    print(f"{'='*60}")
    print(f"Optimizer: {OPTIMIZER_CONFIG['optimizer']}")
    print(f"Learning Rate: {OPTIMIZER_CONFIG['lr']}")
    print(f"NOTE: Testing SAME config as random test (Adam + LR=0.01)")
    print(f"Beta1: {OPTIMIZER_CONFIG['beta1']}")
    print(f"Beta2: {OPTIMIZER_CONFIG['beta2']}")
    print(f"Weight Decay: {OPTIMIZER_CONFIG['weight_decay']}")
    print(f"Steps: {TRAINING_CONFIG['total_steps']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Create optimizer - TESTING ADAM
    import torch.optim as optim
    optimizer = optim.Adam(
        model.parameters(),
        lr=OPTIMIZER_CONFIG["lr"],
        betas=(OPTIMIZER_CONFIG["beta1"], OPTIMIZER_CONFIG["beta2"]),
        weight_decay=OPTIMIZER_CONFIG["weight_decay"],
    )
    
    # Prepare data
    # Pad all sequences to same length for batching
    max_len = max(len(t) for t in tokenized_stories)
    
    # Log results
    losses = []
    grad_norms = []
    samples_log = []
    
    # Fixed prompts for generation (first 5 stories, first 20 tokens)
    fixed_prompts = []
    for i in range(min(5, len(tokenized_stories))):
        prompt_tokens = tokenized_stories[i][:20]
        prompt_text = tokenizer.decode(prompt_tokens)
        fixed_prompts.append({
            "story_idx": i,
            "prompt_tokens": prompt_tokens,
            "prompt_text": prompt_text,
        })
    
    print(f"Fixed prompts for evaluation:")
    for i, p in enumerate(fixed_prompts):
        print(f"  Prompt {i+1}: {p['prompt_text'][:50]}...")
    
    print(f"\nStarting training...")
    print(f"{'='*60}\n")
    
    # Training loop
    model.train()
    
    for step in range(TRAINING_CONFIG["total_steps"]):
        optimizer.zero_grad()
        
        # Sample a batch of stories
        batch_indices = random.sample(
            range(len(tokenized_stories)), 
            min(TRAINING_CONFIG["effective_batch_size"], len(tokenized_stories))
        )
        
        accumulated_loss = 0.0
        
        for batch_idx in batch_indices[:TRAINING_CONFIG["gradient_accumulation_steps"]]:
            # Get tokens
            tokens = tokenized_stories[batch_idx]
            
            # Create input and target
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            labels = input_ids.clone()
            
            # Forward pass
            outputs = model(input_ids, targets=labels)
            loss = outputs["loss"]
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR: NaN/Inf loss at step {step}")
                return {"status": "failed", "reason": "NaN loss"}
            
            # Scale for accumulation
            scaled_loss = loss / TRAINING_CONFIG["gradient_accumulation_steps"]
            scaled_loss.backward()
            
            accumulated_loss += loss.item()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            OPTIMIZER_CONFIG["max_grad_norm"]
        )
        
        # Check for NaN gradients
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    has_nan_grad = True
                    break
        
        if has_nan_grad:
            print(f"ERROR: NaN gradients at step {step}")
            return {"status": "failed", "reason": "NaN gradients"}
        
        # Optimizer step
        optimizer.step()
        
        avg_loss = accumulated_loss / TRAINING_CONFIG["gradient_accumulation_steps"]
        losses.append(avg_loss)
        grad_norms.append(grad_norm.item())
        
        # Logging
        if step % LOG_INTERVAL == 0 or step == TRAINING_CONFIG["total_steps"] - 1:
            print(f"Step {step:4d}: Loss = {avg_loss:.4f}, Grad norm = {grad_norm:.4f}")
        
        # Sample generation
        if step % SAMPLE_INTERVAL == 0 or step == TRAINING_CONFIG["total_steps"] - 1:
            print(f"\n--- Generating samples at step {step} ---")
            
            model.eval()
            sample_results = []
            
            for prompt_info in fixed_prompts:
                # Generate
                input_ids = torch.tensor(
                    [prompt_info["prompt_tokens"]], 
                    dtype=torch.long, 
                    device=device
                )
                
                with torch.no_grad():
                    generated = model.generate(
                        input_ids,
                        max_new_tokens=50,
                        temperature=1.0,
                        top_k=50,
                    )
                
                generated_text = tokenizer.decode(generated[0].tolist())
                
                sample_results.append({
                    "story_idx": prompt_info["story_idx"],
                    "prompt": prompt_info["prompt_text"],
                    "generated": generated_text,
                })
                
                print(f"Story {prompt_info['story_idx'] + 1}:")
                print(f"  Generated: {generated_text[:100]}...")
            
            samples_log.append({
                "step": step,
                "samples": sample_results,
            })
            
            model.train()
            print(f"--- End samples ---\n")
    
    # Calculate results
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
    
    # Weight statistics
    weight_stats = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            data = param.data.float()
            weight_stats[name] = {
                "mean": data.mean().item(),
                "std": data.std().item(),
                "min": data.min().item(),
                "max": data.max().item(),
            }
    
    results = {
        "status": "completed",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": loss_reduction,
        "losses": losses,
        "grad_norms": grad_norms,
        "samples_log": samples_log,
        "weight_stats": weight_stats,
        "config": {
            "model": MODEL_CONFIG,
            "optimizer": OPTIMIZER_CONFIG,
            "training": TRAINING_CONFIG,
            "dataset": DATASET_CONFIG,
        },
    }
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss reduction: {loss_reduction:.1f}%")
    print(f"{'='*60}\n")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"\n{'='*60}")
    print("TINYSTORIES OVERFIT TEST")
    print("CRITICAL: Uses SAME config as real training")
    print(f"{'='*60}")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print(f"{'='*60}\n")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Select stories
    stories, selection_info = select_tinystories_subset(
        num_stories=DATASET_CONFIG["num_stories"],
        seed=DATASET_CONFIG["seed"],
    )
    
    # Tokenize
    tokenized_stories = tokenize_stories(
        stories, 
        tokenizer, 
        max_length=TRAINING_CONFIG["max_seq_len"]
    )
    
    # Create model
    model = create_model(device=device)
    
    # Output directory
    output_dir = Path(__file__).parent / "results"
    
    # Save selection info
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "selection_info.json", "w") as f:
        json.dump(selection_info, f, indent=2)
    
    # Train
    results = train_overfit(
        model,
        tokenized_stories,
        tokenizer,
        device=device,
        output_dir=output_dir,
    )
    
    # Print final interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    
    final_loss = results["final_loss"]
    loss_reduction = results["loss_reduction_pct"]
    
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss reduction: {loss_reduction:.1f}%")
    
    if isinstance(final_loss, (int, float)) and isinstance(loss_reduction, (int, float)):
        if final_loss < 2.0 and loss_reduction > 70:
            print("\nINTERPRETATION: Provides STRONG EVIDENCE that this ternary")
            print("implementation can memorize structured TinyStories data under")
            print("this configuration.")
            print("\nNEXT STEP: Investigate full training configuration")
            print("(LR schedule, duration, batch size, etc.)")
        elif final_loss < 4.0 and loss_reduction > 40:
            print("\nINTERPRETATION: SUGGESTS PARTIAL memorization.")
            print("May need more steps or different approach.")
        else:
            print("\nINTERPRETATION: Does NOT memorize with real training config.")
            print("\nNEXT STEP: Systematic debug needed:")
            print("- Tokenizer/preprocessing")
            print("- Target shifting")
            print("- Causal masking")
            print("- Loss calculation")
            print("- Optimizer settings")
            print("- Gradient flow")
            print("- Ternary quantization")
            print("- STE implementation")
    
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    main()
