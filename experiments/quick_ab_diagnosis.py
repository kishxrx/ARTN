#!/usr/bin/env python3
"""
QUICK A/B DIAGNOSIS - Find why original 50k training never converged
Fast version with synthetic data for rapid iteration.
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

# Setup
BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    "vocab_size": 4000,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "max_lr": 0.001,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "beta1": 0.5,
    "beta2": 0.98,
    "total_steps": 500,
    "warmup_steps": 50,
    "seed": 42,
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
            FP32Block(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
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


class FP32Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FP32Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FP32FF(d_model, d_ff)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class FP32Attention(nn.Module):
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


class FP32FF(nn.Module):
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
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
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
# SYNTHETIC DATA
# =============================================================================

def create_synthetic_batch(batch_size, seq_len, vocab_size, device):
    """Create synthetic batch for testing."""
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


# =============================================================================
# LR SCHEDULER
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
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(exp_name, model_class, optimizer_type, config, device):
    """Run single experiment with progress output."""
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Model: {model_class.upper()} | Optimizer: {optimizer_type.upper()}")
    print(f"{'='*80}\n")
    
    torch.manual_seed(config["seed"])
    
    # Create model
    print("Creating model...", flush=True)
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
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["max_lr"],
            betas=(config["beta1"], config["beta2"]),
            weight_decay=config["weight_decay"],
        )
    
    scheduler = CosineWarmupScheduler(
        warmup_steps=config["warmup_steps"],
        total_steps=config["total_steps"],
        max_lr=config["max_lr"],
        min_lr=1e-5,
    )
    
    # Training loop
    print("\nStarting training...", flush=True)
    print(f"{'Step':<6} {'Loss':<10} {'LR':<10} {'GradNorm':<10} {'Status'}", flush=True)
    print("-" * 50, flush=True)
    
    losses = []
    start_time = time.time()
    
    for step in range(1, config["total_steps"] + 1):
        lr = scheduler.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for _ in range(config["gradient_accumulation_steps"]):
            input_ids = create_synthetic_batch(
                config["batch_size"],
                config["max_seq_len"],
                config["vocab_size"],
                device
            )
            
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            
            outputs = model(input_ids, targets=targets)
            loss = outputs["loss"]
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at step {step}", flush=True)
                continue
            
            scaled_loss = loss / config["gradient_accumulation_steps"]
            scaled_loss.backward()
            accum_loss += loss.item()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        
        optimizer.step()
        
        avg_loss = accum_loss / config["gradient_accumulation_steps"]
        losses.append(avg_loss)
        
        # Progress every step for first 20, then every 50
        if step <= 20 or step % 50 == 0:
            status = "OK"
            if avg_loss > 10:
                status = "HIGH"
            print(f"{step:<6} {avg_loss:<10.4f} {lr:<10.6f} {grad_norm.item():<10.4f} {status}", flush=True)
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY: {exp_name}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Time: {elapsed:.1f}s", flush=True)
    print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
    print(f"Min loss: {min(losses):.4f}", flush=True)
    print(f"Max loss: {max(losses):.4f}", flush=True)
    
    return {
        "name": exp_name,
        "model": model_class,
        "optimizer": optimizer_type,
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "time": elapsed,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("QUICK A/B DIAGNOSIS")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}", flush=True)
    
    # Experiments
    experiments = [
        ("A_bitlinear_adafactor", "bitlinear", "adafactor"),
        ("B_bitlinear_adamw", "bitlinear", "adamw"),
        ("C_fp32_adafactor", "fp32", "adafactor"),
        ("D_fp32_adamw", "fp32", "adamw"),
    ]
    
    results = []
    
    for exp_name, model_class, optimizer_type in experiments:
        result = run_experiment(exp_name, model_class, optimizer_type, CONFIG, device)
        results.append(result)
    
    # Comparison
    print("\n\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    print(f"\n{'Experiment':<25} {'Model':<12} {'Optimizer':<12} {'Final Loss':<12} {'Min Loss':<12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<25} {r['model']:<12} {r['optimizer']:<12} {r['final_loss']:<12.4f} {r['min_loss']:<12.4f}")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
