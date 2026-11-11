#!/usr/bin/env python3
"""
CONTROLLED BITLINEAR + ADAMW EXPERIMENT
Purpose: Test if BitLinear/ternary quantization converges with AdamW optimizer.

Setup: IDENTICAL to validated FP32 + AdamW baseline
- Same dataset (TinyStories, 50k train, 5k val)
- Same tokenizer
- Same model dimensions (d=256, L=6, heads=8)
- Same batch size (4), grad accum (2), seq length (256)
- Same LR schedule (max 0.001, warmup 1000, total 10000)
- Same evaluation and generation prompts

Control: FP32 + AdamW already validated (val loss 1.69, coherent generation)

Hypothesis: If BitLinear fails with AdamW, the issue is BitLinear/ternary STE.
            If BitLinear succeeds, optimizer was the root cause.
"""

import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

# =============================================================================
# CONFIG - IDENTICAL TO FP32 BASELINE
# =============================================================================

CONFIG = {
    # Model architecture - SAME as FP32 baseline
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    "vocab_size": 4000,
    
    # Training parameters - SAME as FP32 baseline
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "max_lr": 0.001,
    "min_lr": 0.00003,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "beta1": 0.5,
    "beta2": 0.98,
    
    # Training schedule - SAME as FP32 baseline
    "total_steps": 10000,
    "warmup_steps": 1000,
    
    # Data - SAME as FP32 baseline
    "train_samples": 50000,
    "val_samples": 5000,
    
    # Checkpointing - SAME as FP32 baseline
    "checkpoint_every": 500,
    "eval_every": 100,
    
    # Seed - SAME as FP32 baseline
    "seed": 42,
    
    # Generation - SAME as FP32 baseline
    "generation_prompt": "Once upon a time",
    "generation_max_tokens": 100,
    "generation_temperature": 0.9,
}

# =============================================================================
# BITLINEAR GPT MODEL
# =============================================================================

from src.model.bitlinear import BitLinear

class BitLinearGPT(nn.Module):
    """GPT model with BitLinear (ternary quantization) layers."""
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            BitLinearBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = BitLinear(d_model, vocab_size, bias=False)
        
        # Tie weights
        # Note: Can't directly tie BitLinear with Embedding, so we don't tie
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, BitLinear):
            # BitLinear has its own initialization
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1
            )
        
        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids[:, -self.token_embedding.num_embeddings:]
            
            logits = self(idx_cond)["logits"]
            logits = logits[:, -1, :] / temperature
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, idx_next], dim=1)
        
        return input_ids
    
    def get_ternary_stats(self):
        """Get statistics about ternary weight distribution."""
        stats = {
            "total_params": 0,
            "ternary_params": 0,
            "weight_stats": [],
        }
        
        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                if hasattr(module, 'weight') and module.weight is not None:
                    w = module.weight.data
                    stats["total_params"] += w.numel()
                    stats["ternary_params"] += w.numel()
                    
                    # Get quantized weights if available
                    if hasattr(module, 'get_ternary_weights'):
                        qw = module.get_ternary_weights()
                    else:
                        # Estimate ternary distribution
                        mean = w.mean().item()
                        std = w.std().item()
                        qw = w  # Fallback
                    
                    # Count ternary values
                    positive = (qw > 0.1).sum().item()
                    negative = (qw < -0.1).sum().item()
                    zero = qw.numel() - positive - negative
                    
                    stats["weight_stats"].append({
                        "name": name,
                        "shape": list(w.shape),
                        "mean": w.mean().item(),
                        "std": w.std().item(),
                        "positive": positive,
                        "negative": negative,
                        "zero": zero,
                        "sparsity": zero / qw.numel() if qw.numel() > 0 else 0,
                    })
        
        return stats


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
# DATA LOADING (SAME AS FP32 BASELINE)
# =============================================================================

class CachedDataset:
    """Dataset that loads pre-tokenized data from cache."""
    
    def __init__(self, tokenized_data, max_length=256):
        self.data = tokenized_data
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        
        # Pad
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens)


def load_data(config):
    """Load cached tokenized data - SAME as FP32 baseline."""
    from src.tokenizer.bpe_tokenizer import BPETokenizer
    
    # Load tokenizer
    tokenizer_path = BASE_DIR / 'outputs' / 'artn-50k-best-config' / 'tokenizer.json'
    if tokenizer_path.exists():
        print(f"Loading tokenizer from {tokenizer_path}", flush=True)
        tokenizer = BPETokenizer.load(str(tokenizer_path))
    else:
        raise RuntimeError(f"Tokenizer not found at {tokenizer_path}")
    
    # Check for cached tokenized data
    cache_dir = BASE_DIR / 'outputs' / 'tokenized_cache'
    train_cache = cache_dir / 'train_tokenized.pt'
    val_cache = cache_dir / 'val_tokenized.pt'
    
    if train_cache.exists() and val_cache.exists():
        print(f"\nLoading CACHED tokenized data...", flush=True)
        train_tokenized = torch.load(train_cache)
        val_tokenized = torch.load(val_cache)
        print(f"  Train: {len(train_tokenized)} samples", flush=True)
        print(f"  Val: {len(val_tokenized)} samples", flush=True)
    else:
        raise RuntimeError("Tokenized cache not found. Run save_tokenized_cache.py first!")
    
    # Create datasets
    train_dataset = CachedDataset(train_tokenized, config["max_seq_len"])
    val_dataset = CachedDataset(val_tokenized, config["max_seq_len"])
    
    return train_dataset, val_dataset, tokenizer


# =============================================================================
# LEARNING RATE SCHEDULER (SAME AS FP32 BASELINE)
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
# METRICS & GENERATION (SAME AS FP32 BASELINE)
# =============================================================================

def compute_perplexity(model, dataloader, device, max_batches=50):
    """Compute validation perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            
            input_ids = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            
            outputs = model(input_ids, targets=targets)
            loss = outputs["loss"]
            
            # Count non-padding tokens
            num_tokens = (targets != 0).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    model.train()
    return perplexity, avg_loss


def generate_sample(model, tokenizer, prompt, max_tokens=100, temperature=0.9, device='cpu'):
    """Generate text sample."""
    model.eval()
    
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        
        output_ids = model.generate(input_ids, max_tokens, temperature)
        
        generated_text = tokenizer.decode(output_ids[0].tolist())
    
    model.train()
    return generated_text


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(
    exp_name: str,
    config: Dict,
    output_dir: Path,
    device: torch.device,
    resume_step: int = 0,
):
    """Run BitLinear + AdamW experiment with comprehensive logging."""
    
    print(f"\n{'='*80}", flush=True)
    print(f"EXPERIMENT: {exp_name}", flush=True)
    print(f"Model: BitLinear (Ternary Quantization) | Optimizer: AdamW", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Set seed
    torch.manual_seed(config["seed"])
    
    # Load data
    train_dataset, val_dataset, tokenizer = load_data(config)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    # Create model
    print("Creating BitLinear model...", flush=True)
    model = BitLinearGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}", flush=True)
    
    # Get ternary stats
    ternary_stats = model.get_ternary_stats()
    print(f"BitLinear layers: {len(ternary_stats['weight_stats'])}", flush=True)
    
    # Create optimizer (ADAMW - SAME as FP32 baseline)
    print(f"Creating AdamW optimizer...", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["max_lr"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )
    
    # Scheduler (SAME as FP32 baseline)
    scheduler = CosineWarmupScheduler(
        warmup_steps=config["warmup_steps"],
        total_steps=config["total_steps"],
        max_lr=config["max_lr"],
        min_lr=config["min_lr"],
    )
    
    # RESUME: Load checkpoint if resuming
    start_step = 1
    if resume_step > 0:
        ckpt_path = output_dir / "checkpoints" / exp_name / f"step_{resume_step}.pt"
        if ckpt_path.exists():
            print(f"\nResuming from checkpoint: {ckpt_path}", flush=True)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = resume_step + 1
            print(f"Resuming from step {start_step}", flush=True)
        else:
            print(f"Warning: Checkpoint {ckpt_path} not found, starting from scratch", flush=True)
    
    # Training
    print("\nStarting training...", flush=True)
    print(f"{'Step':<8} {'Train Loss':<12} {'Val Loss':<12} {'PPL':<10} {'LR':<12} {'GradNorm':<12} {'Ternary%':<10} {'Status'}", flush=True)
    print("-" * 100, flush=True)
    
    all_metrics = []
    start_time = time.time()
    data_iter = iter(train_loader)
    
    for step in range(start_step, config["total_steps"] + 1):
        # Learning rate
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
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at step {step}", flush=True)
                continue
            
            scaled_loss = loss / config["gradient_accumulation_steps"]
            scaled_loss.backward()
            accum_loss += loss.item()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        
        # Optimizer step
        optimizer.step()
        
        train_loss = accum_loss / config["gradient_accumulation_steps"]
        
        # Metrics
        metrics = {
            "step": step,
            "train_loss": train_loss,
            "lr": lr,
            "grad_norm": grad_norm.item(),
        }
        
        # Get ternary statistics
        ternary_stats = model.get_ternary_stats()
        avg_sparsity = sum(s["sparsity"] for s in ternary_stats["weight_stats"]) / len(ternary_stats["weight_stats"]) if ternary_stats["weight_stats"] else 0
        metrics["ternary_sparsity"] = avg_sparsity
        
        # Validation
        if step % config["eval_every"] == 0 or step == 1:
            val_ppl, val_loss = compute_perplexity(model, val_loader, device)
            metrics["val_loss"] = val_loss
            metrics["val_perplexity"] = val_ppl
            
            # Generation
            if step % config["checkpoint_every"] == 0 or step == 1:
                sample = generate_sample(
                    model, tokenizer, config["generation_prompt"],
                    config["generation_max_tokens"], config["generation_temperature"], device
                )
                metrics["generation_sample"] = sample
                print(f"\n  Generated: {sample[:200]}...\n", flush=True)
        
        all_metrics.append(metrics)
        
        # Progress
        status = "OK"
        if train_loss > 9:
            status = "HIGH_LOSS"
        
        if step % config["eval_every"] == 0 or step == 1:
            print(f"{step:<8} {train_loss:<12.4f} {metrics.get('val_loss', 'N/A'):<12} {metrics.get('val_perplexity', 'N/A'):<10.2f} {lr:<12.6f} {grad_norm.item():<12.4f} {avg_sparsity:<10.2%} {status}", flush=True)
        elif step % 100 == 0:
            print(f"{step:<8} {train_loss:<12.4f} {'-':<12} {'-':<10} {lr:<12.6f} {grad_norm.item():<12.4f} {avg_sparsity:<10.2%} {status}", flush=True)
        
        # Checkpoint
        if step % config["checkpoint_every"] == 0:
            ckpt_path = output_dir / "checkpoints" / exp_name / f"step_{step}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": metrics.get("val_loss"),
                "ternary_stats": ternary_stats,
                "config": config,
            }, ckpt_path)
            
            print(f"  Checkpoint saved: {ckpt_path}", flush=True)
    
    # Final summary
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY: {exp_name}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total time: {elapsed/60:.1f} minutes", flush=True)
    print(f"Final train loss: {all_metrics[-1]['train_loss']:.4f}", flush=True)
    
    val_losses = [m.get("val_loss") for m in all_metrics if "val_loss" in m]
    if val_losses:
        print(f"Final val loss: {val_losses[-1]:.4f}", flush=True)
        print(f"Best val loss: {min(val_losses):.4f}", flush=True)
    
    # Save results
    results = {
        "experiment": exp_name,
        "model": "bitlinear",
        "optimizer": "adamw",
        "config": config,
        "total_time_seconds": elapsed,
        "metrics": all_metrics,
    }
    
    results_file = output_dir / f"{exp_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}", flush=True)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("CONTROLLED BITLINEAR + ADAMW EXPERIMENT")
    print("Testing if BitLinear converges with AdamW optimizer")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}", flush=True)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / "outputs" / f"bitlinear_adamw_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}", flush=True)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Run experiment
    results = run_experiment(
        exp_name="bitlinear_adamw",
        config=CONFIG,
        output_dir=output_dir,
        device=device,
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nFinal Results:", flush=True)
    print(f"  Train loss: {results['metrics'][-1]['train_loss']:.4f}", flush=True)
    print(f"  Val loss: {results['metrics'][-1].get('val_loss', 'N/A')}", flush=True)
    print(f"  Perplexity: {results['metrics'][-1].get('val_perplexity', 'N/A')}", flush=True)
    print(f"  Time: {results['total_time_seconds']/60:.1f} minutes", flush=True)
    
    # Compare to FP32 baseline
    print("\n" + "="*80)
    print("COMPARISON TO FP32 BASELINE")
    print("="*80)
    print(f"\nFP32 + AdamW baseline (validated):")
    print(f"  Final val loss: 1.69")
    print(f"  Perplexity: 5.44")
    print(f"  Generation: Coherent stories")
    print(f"\nBitLinear + AdamW (this run):")
    print(f"  Final val loss: {results['metrics'][-1].get('val_loss', 'N/A')}")
    print(f"  Perplexity: {results['metrics'][-1].get('val_perplexity', 'N/A')}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
