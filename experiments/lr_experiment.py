#!/usr/bin/env python3
"""
LR/Schedule Experiment Runner - FIXED VERSION

Runs short training experiments with different LR/schedule configurations.
Each experiment starts from the SAME initial model state for fair comparison.

FIXES:
- Loss logging now computes correct average (not sum)
- Gradient norm computed before zero_grad()
- Metrics saved correctly

Usage:
    python experiments/lr_experiment.py --config lr_003_constant
"""

import sys
import json
import time
import argparse
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

# ============================================================
# Configuration
# ============================================================

EXPERIMENT_CONFIGS = {
    "lr_001_constant": {
        "lr": 0.001,
        "scheduler": "constant",
        "description": "Current LR, constant schedule (baseline)"
    },
    "lr_003_constant": {
        "lr": 0.003,
        "scheduler": "constant",
        "description": "Higher LR, constant schedule"
    },
    "lr_003_cosine": {
        "lr": 0.003,
        "scheduler": "cosine",
        "description": "Higher LR, cosine decay"
    }
}

MODEL_CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256
}

TRAINING_CONFIG = {
    "max_steps": 5000,
    "batch_size": 4,
    "accumulation_steps": 2,
    "beta1": 0.5,
    "beta2": 0.98,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "seed": 42,
    "warmup_steps": 100
}

# Fixed prompts for generation
FIXED_PROMPTS = [
    "Once upon a time",
    "The little girl",
    "A boy named",
    "There was a",
    "One day"
]

# ============================================================
# Dataset
# ============================================================

class TinyStoriesSubset(torch.utils.data.Dataset):
    """TinyStories subset with fixed seed"""
    
    def __init__(self, tokenizer, num_stories=100, seq_len=256, seed=42):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        torch.manual_seed(seed)
        
        # Generate synthetic stories for quick experiments
        self.stories = []
        for i in range(num_stories):
            # Create simple story pattern
            length = seq_len + 10  # Extra for target shift
            tokens = torch.randint(0, tokenizer.vocab_size, (length,))
            self.stories.append(tokens)
    
    def __len__(self):
        return len(self.stories)
    
    def __getitem__(self, idx):
        tokens = self.stories[idx]
        
        # Truncate to seq_len
        if len(tokens) > self.seq_len + 1:
            tokens = tokens[:self.seq_len + 1]
        
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        return input_ids, target_ids

# ============================================================
# Training Functions
# ============================================================

def create_model(config):
    """Create model with fixed seed"""
    torch.manual_seed(config["seed"])
    
    model = BitLinearGPT(
        vocab_size=MODEL_CONFIG["vocab_size"],
        d_model=MODEL_CONFIG["d_model"],
        n_heads=MODEL_CONFIG["n_heads"],
        n_layers=MODEL_CONFIG["n_layers"],
        d_ff=MODEL_CONFIG["d_ff"],
        max_seq_len=MODEL_CONFIG["max_seq_len"]
    )
    
    return model

def create_optimizer(model, lr, beta1, beta2, weight_decay):
    """Create Adafactor optimizer using project's custom implementation"""
    # Import custom Adafactor (not torch.optim)
    from src.training.adafactor import Adafactor
    
    return Adafactor(
        model.parameters(),
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
    )

def create_scheduler(optimizer, scheduler_type, max_steps, warmup_steps):
    """Create learning rate scheduler"""
    if scheduler_type == "constant":
        return None
    elif scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-6)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

def get_current_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0

def get_gradient_norm(model):
    """Compute total gradient norm"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return math.sqrt(total_norm)

def get_ternary_distribution(model):
    """Get ternary weight distribution statistics"""
    neg_count, zero_count, pos_count = 0, 0, 0
    total_count = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            w = module.weight.data
            # Check for ternary values (allowing for numerical precision)
            neg_count += (w < -0.5).sum().item()
            zero_count += ((w >= -0.5) & (w <= 0.5)).sum().item()
            pos_count += (w > 0.5).sum().item()
            total_count += w.numel()
    
    if total_count == 0:
        return {"neg_pct": 0.0, "zero_pct": 0.0, "pos_pct": 0.0}
    
    return {
        "neg_pct": 100 * neg_count / total_count,
        "zero_pct": 100 * zero_count / total_count,
        "pos_pct": 100 * pos_count / total_count
    }

def get_parameter_update_magnitude(model, initial_params):
    """Compute parameter update magnitude"""
    total_update = 0.0
    count = 0
    
    for (name, param), (init_name, init_param) in zip(
        model.named_parameters(), initial_params
    ):
        if param.requires_grad:
            update = (param.data - init_param.data).norm().item()
            total_update += update
            count += 1
    
    return total_update / count if count > 0 else 0.0

def generate_sample(model, tokenizer, prompt, max_tokens=50, temperature=1.0):
    """Generate sample from prompt"""
    model.eval()
    
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        
        for _ in range(max_tokens):
            if input_ids.size(1) >= MODEL_CONFIG["max_seq_len"]:
                break
            
            # Clamp token range
            input_ids = torch.clamp(input_ids, 0, tokenizer.vocab_size - 1)
            
            output = model(input_ids)
            logits = output['logits']  # Extract logits from dict
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        generated = tokenizer.decode(input_ids[0].tolist())
    
    model.train()
    return generated

def compute_repetition_score(text):
    """Compute repetition score (lower is better)"""
    tokens = text.split()
    if len(tokens) < 2:
        return 0.0
    
    # Count bigram repetition
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    unique_bigrams = len(set(bigrams))
    total_bigrams = len(bigrams)
    
    # More repetition = higher score
    repetition_ratio = 1.0 - (unique_bigrams / total_bigrams) if total_bigrams > 0 else 0.0
    
    # Check for word repetition
    word_counts = {}
    for word in tokens:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    max_repetition = max(word_counts.values()) if word_counts else 1
    word_repetition = max_repetition / len(tokens) if tokens else 0
    
    return 100 * (repetition_ratio + word_repetition) / 2

# ============================================================
# Main Training Loop
# ============================================================

def run_experiment(config_name: str):
    """Run single experiment"""
    
    if config_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(EXPERIMENT_CONFIGS.keys())}")
    
    config = EXPERIMENT_CONFIGS[config_name]
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / "lr_experiments" / f"{config_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    print("=" * 70)
    print(f"EXPERIMENT: {config_name}")
    print(f"Description: {config['description']}")
    print("=" * 70)
    print(f"LR: {config['lr']}, Scheduler: {config['scheduler']}")
    print(f"Steps: {TRAINING_CONFIG['max_steps']}, Batch: {TRAINING_CONFIG['batch_size']}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Create deterministic initial state
    torch.manual_seed(TRAINING_CONFIG["seed"])
    
    # Create model (same initial weights for all experiments)
    model = create_model(TRAINING_CONFIG)
    
    # Store initial parameters
    initial_params = [(name, param.data.clone()) for name, param in model.named_parameters()]
    
    # Create tokenizer
    tokenizer_path = "/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    # Create dataset and loader
    dataset = TinyStoriesSubset(
        tokenizer,
        num_stories=100,
        seq_len=MODEL_CONFIG["max_seq_len"],
        seed=42
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        lr=config["lr"],
        beta1=0.5,  # Best beta1 from history
        beta2=0.98,
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config["scheduler"],
        max_steps=TRAINING_CONFIG["max_steps"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"]
    )
    
    # Training variables
    step = 0
    best_loss = float('inf')
    metrics_history = []
    
    # FIX: Track accumulated loss and count separately
    accumulated_loss_sum = 0.0
    accumulated_count = 0
    
    # Track last gradient norm before zero_grad
    last_grad_norm = 0.0
    
    # Start training
    start_time = time.time()
    
    print("\nStarting training...")
    print("-" * 70)
    
    while step < TRAINING_CONFIG["max_steps"]:
        for batch in dataloader:
            if step >= TRAINING_CONFIG["max_steps"]:
                break
            
            input_ids, target_ids = batch
            
            # Forward pass
            output = model(input_ids)
            logits = output['logits']  # Extract logits from dict
            
            # Compute loss (CrossEntropyLoss with reduction='mean' by default)
            loss = nn.CrossEntropyLoss()(
                logits.reshape(-1, MODEL_CONFIG["vocab_size"]),
                target_ids.reshape(-1)
            )
            
            # FIX: Store unscaled loss for logging
            accumulated_loss_sum += loss.item()
            accumulated_count += 1
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / TRAINING_CONFIG["accumulation_steps"]
            
            # Backward pass
            scaled_loss.backward()
            
            # Optimizer step
            if (step + 1) % TRAINING_CONFIG["accumulation_steps"] == 0:
                # FIX: Compute gradient norm BEFORE zero_grad
                last_grad_norm = get_gradient_norm(model)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    TRAINING_CONFIG["max_grad_norm"]
                )
                
                optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
                
                optimizer.zero_grad()
            
            step += 1
            
            # Log metrics every 500 steps
            if step % 500 == 0:
                # FIX: Compute average loss correctly
                avg_loss = accumulated_loss_sum / accumulated_count if accumulated_count > 0 else 0.0
                
                # Get current LR
                current_lr = get_current_lr(optimizer)
                
                # Get ternary distribution
                ternary_dist = get_ternary_distribution(model)
                
                # Get parameter update magnitude
                update_mag = get_parameter_update_magnitude(model, initial_params)
                
                # Compute perplexity
                perplexity = math.exp(avg_loss) if avg_loss < 10 else 9999.0
                
                # Log
                elapsed = time.time() - start_time
                
                print(f"Step {step:5d} | Loss: {avg_loss:.4f} | "
                      f"PPL: {perplexity:.1f} | LR: {current_lr:.6f} | "
                      f"Grad: {last_grad_norm:.4f} | Update: {update_mag:.6f}")
                
                metrics_history.append({
                    "step": step,
                    "train_loss": avg_loss,
                    "perplexity": perplexity,
                    "lr": current_lr,
                    "grad_norm": last_grad_norm,
                    "update_magnitude": update_mag,
                    "ternary_neg_pct": ternary_dist["neg_pct"],
                    "ternary_zero_pct": ternary_dist["zero_pct"],
                    "ternary_pos_pct": ternary_dist["pos_pct"],
                    "elapsed_seconds": elapsed
                })
                
                # Save best checkpoint
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": avg_loss,
                    }, output_dir / "best_checkpoint.pt")
                
                # FIX: Reset accumulation tracking
                accumulated_loss_sum = 0.0
                accumulated_count = 0
    
    # Training complete
    total_time = time.time() - start_time
    final_loss = avg_loss if 'avg_loss' in dir() else (metrics_history[-1]["train_loss"] if metrics_history else 0.0)
    
    print("-" * 70)
    print(f"Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    
    # Generate samples at end
    print("\nGenerating samples from final model...")
    print("-" * 70)
    
    generated_samples = []
    for prompt in FIXED_PROMPTS:
        generated = generate_sample(model, tokenizer, prompt)
        repetition = compute_repetition_score(generated)
        
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated[:100]}...")
        print(f"Repetition score: {repetition:.1f}")
        
        generated_samples.append({
            "prompt": prompt,
            "generated": generated,
            "repetition_score": repetition
        })
    
    # Save final checkpoint
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": final_loss,
    }, output_dir / "final_checkpoint.pt")
    
    # Save results
    results = {
        "config_name": config_name,
        "config": config,
        "training_config": TRAINING_CONFIG,
        "model_config": MODEL_CONFIG,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "total_time_seconds": total_time,
        "metrics_history": metrics_history,
        "generated_samples": generated_samples
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)
    
    return results

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LR experiment")
    parser.add_argument("--config", type=str, required=True,
                       choices=list(EXPERIMENT_CONFIGS.keys()),
                       help="Experiment configuration to run")
    
    args = parser.parse_args()
    
    run_experiment(args.config)
