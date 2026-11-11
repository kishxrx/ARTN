#!/usr/bin/env python3
"""
CONTROLLED A/B DIAGNOSIS - Find why original 50k training never converged

Purpose: Test ONE variable at a time to identify the failing component.

Experiments:
  A: Original (Adafactor + BitLinear) - Reproduce failure
  B: AdamW + BitLinear - Test optimizer change
  C: Adafactor + FP32 Linear - Test architecture change
  D: AdamW + FP32 Linear - Baseline (should work)

Each experiment runs for 2000 steps with comprehensive diagnostics.
"""

import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Setup paths
BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIGURATION - EXACTLY FROM FAILED 50k RUN
# =============================================================================

CONFIG = {
    # Model architecture
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    "vocab_size": 4000,
    
    # Training parameters - FROM FAILED RUN
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "max_lr": 0.001,
    "min_lr": 0.00003,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "beta1": 0.5,
    "beta2": 0.98,
    
    # Experiment
    "total_steps": 2000,
    "warmup_steps": 200,  # Proportional to 2000 steps
    "seed": 42,  # FIXED SEED FOR REPRODUCIBILITY
    
    # Data
    "max_samples": 10000,  # Smaller for diagnostic
    "dataset": "roneneldan/TinyStories",
}


# =============================================================================
# FP32 GPT MODEL (Baseline)
# =============================================================================

class FP32GPT(nn.Module):
    """Standard FP32 GPT model for comparison."""
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        
        # Embeddings
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1
            )
        
        return {"logits": logits, "loss": loss}


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool(), float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


# =============================================================================
# BITLINEAR GPT MODEL
# =============================================================================

from src.model.bitlinear import BitLinear

class BitLinearGPT(nn.Module):
    """GPT model with BitLinear (ternary quantization)."""
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        
        # Embeddings (FP32)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks with BitLinear
        self.blocks = nn.ModuleList([
            BitLinearBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = BitLinear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        
        # Embeddings
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1
            )
        
        return {"logits": logits, "loss": loss}


class BitLinearBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BitLinearAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = BitLinearFF(d_model, d_ff)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class BitLinearAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = BitLinear(d_model, d_model)
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        self.o_proj = BitLinear(d_model, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool(), float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(y)


class BitLinearFF(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = BitLinear(d_model, d_ff)
        self.w2 = BitLinear(d_ff, d_model)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


# =============================================================================
# DATA LOADING
# =============================================================================

class TinyStoriesDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        tokens = self.tokenizer.encode(text)[:self.max_length]
        
        # Pad
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens)


def create_tokenizer():
    """Create a simple BPE tokenizer."""
    tokenizer_path = BASE_DIR / 'outputs' / 'artn-50k-best-config' / 'tokenizer.json'
    
    if tokenizer_path.exists():
        print(f"Loading tokenizer from: {tokenizer_path}")
        from src.tokenizer.bpe_tokenizer import BPETokenizer
        tokenizer = BPETokenizer.load(str(tokenizer_path))  # Classmethod load
        return tokenizer
    else:
        # Fallback to simple tokenizer
        print(f"Warning: Tokenizer not found at {tokenizer_path}")
        print("Using simple character-level tokenizer")
        return SimpleTokenizer()


class SimpleTokenizer:
    """Simple tokenizer for fallback."""
    def __init__(self, vocab_size=4000):
        self.vocab_size = vocab_size
    
    def encode(self, text):
        # Simple hash-based encoding
        tokens = []
        words = text.split()
        for word in words:
            token = hash(word) % (self.vocab_size - 1) + 1
            tokens.append(token)
        return tokens if tokens else [1]
    
    def decode(self, tokens):
        return " ".join([f"[{t}]" for t in tokens])


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

class CosineWarmupScheduler:
    def __init__(self, warmup_steps, total_steps, max_lr, min_lr=1e-5):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
    
    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def collect_diagnostics(model, batch, device, step, optimizer_type):
    """Collect comprehensive diagnostics."""
    diagnostics = {
        "step": step,
        "optimizer": optimizer_type,
    }
    
    model.eval()
    
    with torch.no_grad():
        input_ids = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        
        # Forward pass
        outputs = model(input_ids, targets=targets)
        logits = outputs["logits"]
        loss = outputs["loss"]
        
        # Loss
        diagnostics["loss"] = loss.item() if loss is not None else None
        
        # Logits statistics
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        
        diagnostics["logits_mean"] = float(logits.mean())
        diagnostics["logits_std"] = float(logits.std())
        diagnostics["logits_min"] = float(logits.min())
        diagnostics["logits_max"] = float(logits.max())
        
        # Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = math.log(logits.size(-1))
        diagnostics["entropy"] = entropy
        diagnostics["entropy_ratio"] = entropy / max_entropy
        
        # Top-k predictions
        top_probs, top_indices = probs.topk(5)
        diagnostics["top1_prob"] = float(top_probs[0])
        diagnostics["top5_prob_sum"] = float(top_probs.sum())
        
        # Weight statistics per layer
        weight_stats = {}
        for name, param in model.named_parameters():
            if param.dim() > 1 and param.is_floating_point():
                weight_stats[name] = {
                    "mean": float(param.mean()),
                    "std": float(param.std()),
                    "norm": float(param.norm()),
                    "min": float(param.min()),
                    "max": float(param.max()),
                }
        
        diagnostics["weight_stats"] = weight_stats
        
        # Overall weight statistics
        all_stds = [s["std"] for s in weight_stats.values()]
        diagnostics["avg_weight_std"] = sum(all_stds) / len(all_stds) if all_stds else 0
        
        # BitLinear statistics (if applicable)
        bitlinear_stats = {}
        for name, module in model.named_modules():
            if hasattr(module, 'running_sparsity'):
                bitlinear_stats[name] = {
                    "sparsity": float(module.running_sparsity),
                    "num_quantizations": int(module.num_quantizations),
                }
        
        if bitlinear_stats:
            diagnostics["bitlinear_stats"] = bitlinear_stats
            diagnostics["avg_sparsity"] = sum(s["sparsity"] for s in bitlinear_stats.values()) / len(bitlinear_stats)
    
    model.train()
    return diagnostics


def check_nan_inf(model, optimizer):
    """Check for NaN/Inf in model and optimizer."""
    issues = []
    
    # Check model parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            issues.append(f"NaN in parameter: {name}")
        if torch.isinf(param).any():
            issues.append(f"Inf in parameter: {name}")
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                issues.append(f"NaN in gradient: {name}")
            if torch.isinf(param.grad).any():
                issues.append(f"Inf in gradient: {name}")
    
    # Check optimizer state
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any():
                        issues.append(f"NaN in optimizer state: {key}")
                    if torch.isinf(value).any():
                        issues.append(f"Inf in optimizer state: {key}")
    
    return issues


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(
    exp_name: str,
    model_class: str,  # "fp32" or "bitlinear"
    optimizer_type: str,  # "adafactor" or "adamw"
    config: Dict,
    output_dir: Path,
    device: torch.device,
):
    """Run a single experiment."""
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Model: {model_class.upper()}")
    print(f"Optimizer: {optimizer_type.upper()}")
    print(f"{'='*80}\n")
    
    # Set seed
    torch.manual_seed(config["seed"])
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(config["dataset"], split="train")
    if config["max_samples"]:
        dataset = dataset.select(range(min(config["max_samples"], len(dataset))))
    
    train_dataset = TinyStoriesDataset(dataset, tokenizer, config["max_seq_len"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    
    # Create model
    print("Creating model...")
    if model_class == "fp32":
        model = FP32GPT(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_ff"],
            max_seq_len=config["max_seq_len"],
        )
    else:
        model = BitLinearGPT(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_ff"],
            max_seq_len=config["max_seq_len"],
        )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create optimizer
    print(f"Creating {optimizer_type} optimizer...")
    if optimizer_type == "adafactor":
        from src.training.adafactor import Adafactor
        optimizer = Adafactor(
            model.parameters(),
            lr=config["max_lr"],
            beta1=config["beta1"],
            beta2=config["beta2"],
            weight_decay=config["weight_decay"],
        )
    else:  # adamw
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["max_lr"],
            betas=(config["beta1"], config["beta2"]),
            weight_decay=config["weight_decay"],
        )
    
    # Scheduler
    scheduler = CosineWarmupScheduler(
        warmup_steps=config["warmup_steps"],
        total_steps=config["total_steps"],
        max_lr=config["max_lr"],
        min_lr=config["min_lr"],
    )
    
    # Training loop
    print("\nStarting training...")
    print(f"{'Step':<8} {'Loss':<12} {'LR':<12} {'GradNorm':<12} {'Status'}")
    print("-" * 70)
    
    all_diagnostics = []
    losses = []
    gradient_norms = []
    
    data_iter = iter(train_loader)
    
    for step in range(1, config["total_steps"] + 1):
        # Get learning rate
        lr = scheduler.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for _ in range(config["gradient_accumulation_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            input_ids = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            
            outputs = model(input_ids, targets=targets)
            loss = outputs["loss"]
            
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at step {step}")
                continue
            
            scaled_loss = loss / config["gradient_accumulation_steps"]
            scaled_loss.backward()
            accum_loss += loss.item()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        gradient_norms.append(grad_norm.item())
        
        # Check for NaN gradients
        has_nan = False
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan = True
                break
        
        if has_nan:
            print(f"  WARNING: NaN gradients at step {step}")
            optimizer.zero_grad()
            continue
        
        # Optimizer step
        optimizer.step()
        
        # Record loss
        avg_loss = accum_loss / config["gradient_accumulation_steps"]
        losses.append(avg_loss)
        
        # Status
        status = "OK"
        if avg_loss > 10:
            status = "HIGH_LOSS"
        elif grad_norm > 10:
            status = "HIGH_GRAD"
        
        # Progress
        if step % 100 == 0 or step <= 10:
            print(f"{step:<8} {avg_loss:<12.4f} {lr:<12.6f} {grad_norm.item():<12.4f} {status}")
        
        # Collect detailed diagnostics every 200 steps
        if step % 200 == 0:
            diagnostics = collect_diagnostics(model, batch, device, step, optimizer_type)
            diagnostics["grad_norm"] = grad_norm.item()
            diagnostics["lr"] = lr
            
            # Check for NaN/Inf
            nan_inf_issues = check_nan_inf(model, optimizer)
            diagnostics["nan_inf_issues"] = nan_inf_issues
            
            all_diagnostics.append(diagnostics)
            
            if nan_inf_issues:
                print(f"  ISSUES: {nan_inf_issues}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\nLoss trajectory:")
    print(f"  Start (step 100): {losses[99] if len(losses) > 99 else losses[0]:.4f}")
    print(f"  End (step {len(losses)}): {losses[-1]:.4f}")
    print(f"  Min: {min(losses):.4f}")
    print(f"  Max: {max(losses):.4f}")
    
    print(f"\nGradient norms:")
    print(f"  Mean: {sum(gradient_norms)/len(gradient_norms):.4f}")
    print(f"  Max: {max(gradient_norms):.4f}")
    
    if all_diagnostics:
        last_diag = all_diagnostics[-1]
        print(f"\nFinal diagnostics:")
        print(f"  Avg weight std: {last_diag.get('avg_weight_std', 'N/A'):.6f}")
        print(f"  Logits std: {last_diag.get('logits_std', 'N/A'):.4f}")
        print(f"  Entropy ratio: {last_diag.get('entropy_ratio', 'N/A'):.2%}")
        print(f"  Top-1 prob: {last_diag.get('top1_prob', 'N/A'):.2%}")
        
        if 'avg_sparsity' in last_diag:
            print(f"  BitLinear sparsity: {last_diag['avg_sparsity']:.4f}")
    
    # Save results
    results = {
        "experiment": exp_name,
        "model_class": model_class,
        "optimizer_type": optimizer_type,
        "config": config,
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "loss_trajectory": losses,
        "gradient_norms": gradient_norms,
        "diagnostics": all_diagnostics,
    }
    
    output_file = output_dir / f"{exp_name}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("CONTROLLED A/B DIAGNOSIS")
    print("Finding why original 50k training never converged")
    print("="*80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / "outputs" / f"ab_diagnosis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Experiments
    experiments = [
        ("A_original_adafactor_bitlinear", "bitlinear", "adafactor"),
        ("B_adamw_bitlinear", "bitlinear", "adamw"),
        ("C_adafactor_fp32", "fp32", "adafactor"),
        ("D_adamw_fp32", "fp32", "adamw"),
    ]
    
    all_results = {}
    
    for exp_name, model_class, optimizer_type in experiments:
        print(f"\n\n{'#'*80}")
        print(f"# {exp_name}")
        print(f"{'#'*80}")
        
        results = run_experiment(
            exp_name=exp_name,
            model_class=model_class,
            optimizer_type=optimizer_type,
            config=CONFIG,
            output_dir=output_dir,
            device=device,
        )
        
        all_results[exp_name] = {
            "final_loss": results["final_loss"],
            "min_loss": results["min_loss"],
            "model": model_class,
            "optimizer": optimizer_type,
        }
        
        # Brief pause between experiments
        time.sleep(2)
    
    # Final comparison
    print("\n\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    print(f"\n{'Experiment':<35} {'Model':<12} {'Optimizer':<12} {'Final Loss':<12} {'Min Loss':<12}")
    print("-" * 90)
    
    for exp_name, res in all_results.items():
        print(f"{exp_name:<35} {res['model']:<12} {res['optimizer']:<12} {res['final_loss']:<12.4f} {res['min_loss']:<12.4f}")
    
    # Save comparison
    comparison_file = output_dir / "comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComparison saved to: {comparison_file}")
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
