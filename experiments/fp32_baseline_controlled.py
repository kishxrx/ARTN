#!/usr/bin/env python3
"""
CONTROLLED LONG-RUN BASELINE EXPERIMENT
Purpose: Establish trustworthy FP32 baseline for ARTN on real TinyStories data.

Compares AdamW vs Adafactor with:
- Identical model initialization
- Identical data order
- Identical batch size, scheduler, warmup
- Real TinyStories subset
- Comprehensive metrics logging
- Generation quality tracking

Decision criterion: validation metrics + generation quality over time
"""

import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import gc
import tracemalloc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Setup
BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Model architecture (FP32 only)
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    "vocab_size": 4000,
    
    # Training parameters
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "max_lr": 0.001,
    "min_lr": 0.00003,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "beta1": 0.5,
    "beta2": 0.98,
    
    # Training schedule
    "total_steps": 10000,
    "warmup_steps": 1000,
    
    # Data
    "train_samples": 50000,
    "val_samples": 5000,
    "dataset": "roneneldan/TinyStories",
    
    # Checkpointing
    "checkpoint_every": 500,
    "eval_every": 100,
    
    # Fixed seed for reproducibility
    "seed": 42,
    
    # Generation
    "generation_prompt": "Once upon a time",
    "generation_max_tokens": 100,
    "generation_temperature": 0.9,
}

# =============================================================================
# FP32 GPT MODEL
# =============================================================================

class FP32GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
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
# DATA LOADING
# =============================================================================

class TinyStoriesDataset:
    def __init__(self, data, tokenizer, max_length=256, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        # Pre-tokenize all data
        print(f"Tokenizing {'train' if is_train else 'val'} data...", flush=True)
        self.tokenized = []
        for i, item in enumerate(data):
            if i % 1000 == 0:
                print(f"  Tokenized {i}/{len(data)}", flush=True)
            tokens = tokenizer.encode(item['text'])[:max_length]
            if len(tokens) > 10:  # Skip very short texts
                self.tokenized.append(tokens)
        
        print(f"  Done. Total samples: {len(self.tokenized)}", flush=True)
    
    def __len__(self):
        return len(self.tokenized)
    
    def __getitem__(self, idx):
        tokens = self.tokenized[idx]
        
        # Pad
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens)


def load_data(config):
    """Load and prepare TinyStories data."""
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
        # Need to tokenize
        print(f"\nCache not found. Tokenizing data...", flush=True)
        print(f"Run 'python experiments/save_tokenized_cache.py' first to create cache!", flush=True)
        raise RuntimeError("Tokenized cache not found. Run save_tokenized_cache.py first!")
    
    # Create datasets
    train_dataset = CachedDataset(train_tokenized, config["max_seq_len"])
    val_dataset = CachedDataset(val_tokenized, config["max_seq_len"])
    
    return train_dataset, val_dataset, tokenizer


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
# METRICS & GENERATION
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


def get_memory_usage():
    """Get current memory usage."""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        gpu_mem_cached = torch.cuda.max_memory_reserved() / (1024**2)
        return {
            "gpu_allocated_mb": gpu_mem,
            "gpu_cached_mb": gpu_mem_cached,
        }
    else:
        # Try psutil, if not available return empty dict
        try:
            import psutil
            process = psutil.Process()
            ram_mb = process.memory_info().rss / (1024**2)
            return {"ram_mb": ram_mb}
        except ImportError:
            return {}  # Skip memory reporting if psutil not available


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(
    exp_name: str,
    optimizer_type: str,
    model_init_state: Dict,
    train_dataset,
    val_dataset,
    tokenizer,
    config: Dict,
    output_dir: Path,
    device: torch.device,
    resume_step: int = 0,  # NEW: support resuming
):
    """Run single optimizer experiment with comprehensive logging."""
    
    print(f"\n{'='*80}", flush=True)
    print(f"EXPERIMENT: {exp_name}", flush=True)
    print(f"Optimizer: {optimizer_type.upper()}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Set seed for data shuffling
    torch.manual_seed(config["seed"] + hash(optimizer_type) % 1000)
    
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
    print("Creating model...", flush=True)
    model = FP32GPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
    )
    
    # Load identical initialization
    model.load_state_dict(model_init_state)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}", flush=True)
    
    # Create optimizer
    print(f"Creating {optimizer_type} optimizer...", flush=True)
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
    print(f"{'Step':<8} {'Train Loss':<12} {'Val Loss':<12} {'PPL':<10} {'LR':<12} {'GradNorm':<12} {'Status'}", flush=True)
    print("-" * 90, flush=True)
    
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
        
        # Logging
        metrics = {
            "step": step,
            "train_loss": train_loss,
            "lr": lr,
            "grad_norm": grad_norm.item(),
        }
        
        # Validation
        if step % config["eval_every"] == 0 or step == 1:
            val_ppl, val_loss = compute_perplexity(model, val_loader, device)
            metrics["val_loss"] = val_loss
            metrics["val_perplexity"] = val_ppl
            
            # Memory
            mem = get_memory_usage()
            metrics.update(mem)
            
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
            print(f"{step:<8} {train_loss:<12.4f} {metrics.get('val_loss', 'N/A'):<12} {metrics.get('val_perplexity', 'N/A'):<10.2f} {lr:<12.6f} {grad_norm.item():<12.4f} {status}", flush=True)
        elif step % 100 == 0:
            print(f"{step:<8} {train_loss:<12.4f} {'-':<12} {'-':<10} {lr:<12.6f} {grad_norm.item():<12.4f} {status}", flush=True)
        
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
        "optimizer": optimizer_type,
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
    print("CONTROLLED LONG-RUN BASELINE EXPERIMENT")
    print("FP32 Model on Real TinyStories Data")
    print("="*80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}", flush=True)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / "outputs" / f"fp32_baseline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}", flush=True)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80, flush=True)
    
    train_dataset, val_dataset, tokenizer = load_data(CONFIG)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Create IDENTICAL model initialization for both experiments
    print("\n" + "="*80)
    print("CREATING MODEL INITIALIZATION")
    print("="*80, flush=True)
    
    torch.manual_seed(CONFIG["seed"])
    init_model = FP32GPT(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        max_seq_len=CONFIG["max_seq_len"],
    )
    model_init_state = init_model.state_dict()
    
    # Save initialization
    torch.save(model_init_state, output_dir / "initial_model_state.pt")
    print(f"Initial model state saved", flush=True)
    
    # Run experiments
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80, flush=True)
    
    experiments = [
        ("adafactor", "adafactor"),
        ("adamw", "adamw"),
    ]
    
    all_results = {}
    
    for exp_name, optimizer_type in experiments:
        results = run_experiment(
            exp_name=exp_name,
            optimizer_type=optimizer_type,
            model_init_state=model_init_state,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            config=CONFIG,
            output_dir=output_dir,
            device=device,
        )
        
        all_results[exp_name] = {
            "final_train_loss": results["metrics"][-1]["train_loss"],
            "final_val_loss": results["metrics"][-1].get("val_loss"),
            "total_time_minutes": results["total_time_seconds"] / 60,
        }
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final comparison
    print("\n\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    print(f"\n{'Optimizer':<15} {'Final Train Loss':<18} {'Final Val Loss':<18} {'Time (min)':<15}")
    print("-" * 70)
    
    for exp_name, res in all_results.items():
        val_loss = res['final_val_loss']
        val_loss_str = f"{val_loss:.4f}" if val_loss else "N/A"
        print(f"{exp_name:<15} {res['final_train_loss']:<18.4f} {val_loss_str:<18} {res['total_time_minutes']:<15.1f}")
    
    # Save comparison
    with open(output_dir / "comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComparison saved to: {output_dir / 'comparison.json'}", flush=True)
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80, flush=True)


if __name__ == "__main__":
    main()
