#!/usr/bin/env python3
"""
READ-ONLY: Re-verify Gradient Magnitude and Optimizer State (Simplified)

Phase 19 reported:
- Ternary gradients 55x smaller than FP32
- Ternary momentum 10-30x smaller than FP32

Phase 20 found weight measurements were wrong by 463-5733x.
This script re-verifies gradient and optimizer state measurements.

CRITICAL: READ-ONLY. No model modification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.model.bitlinear import BitLinearGPT, BitLinear
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*100)
print("READ-ONLY: Re-verify Gradient Magnitude and Optimizer State")
print("="*100)

# ============================================================
# LOAD TOKENIZER AND DATA
# ============================================================

print("\n[Loading tokenizer]")

tokenizer_path = "outputs/artn-optimized/tokenizer.json"
tokenizer = BPETokenizer.load(tokenizer_path)
print(f"Tokenizer loaded: vocab_size = {tokenizer.vocab_size}")

# Create a simple input
test_text = "Once upon a time there was a little girl who lived in a small village."
tokens = tokenizer.encode(test_text)
input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
targets = torch.tensor([tokens[1:]], dtype=torch.long)
print(f"Test input shape: {input_ids.shape}")

# ============================================================
# LOAD TERNARY MODEL
# ============================================================

print("\n[Loading Ternary model]")

# Model config (from checkpoint)
vocab_size = 4000
d_model = 256
n_heads = 8
n_layers = 6
d_ff = 1024
max_seq_len = 256

ternary_ckpt_path = "outputs/artn-50k-best-config/checkpoints/checkpoint_step_5000.pt"
print(f"Loading from: {ternary_ckpt_path}")

ternary_model = BitLinearGPT(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    d_ff=d_ff,
    max_seq_len=max_seq_len,
)

ternary_ckpt = torch.load(ternary_ckpt_path, map_location='cpu')
ternary_model.load_state_dict(ternary_ckpt['model_state_dict'])
print("Ternary model loaded")

# ============================================================
# DEFINE FP32 MODEL (same architecture)
# ============================================================

print("\n[Creating FP32 model (StandardLinear equivalent)]")

class StandardLinear(nn.Linear):
    """Standard Linear layer for FP32 model"""
    pass

class StandardAttention(nn.Module):
    """Standard attention with regular Linear"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = StandardLinear(d_model, d_model)
        self.k_proj = StandardLinear(d_model, d_model)
        self.v_proj = StandardLinear(d_model, d_model)
        self.o_proj = StandardLinear(d_model, d_model)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

class StandardFF(nn.Module):
    """Standard FFN with GELU"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = StandardLinear(d_model, d_ff)
        self.w2 = StandardLinear(d_ff, d_model)
        
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class StandardBlock(nn.Module):
    """Standard transformer block"""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StandardAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = StandardFF(d_model, d_ff)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class StandardGPT(nn.Module):
    """Standard GPT with FP32 layers"""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            StandardBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = StandardLinear(d_model, vocab_size)
        
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        
        return {"logits": logits, "loss": loss}

fp32_model = StandardGPT(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len)

# Load FP32 checkpoint
fp32_ckpt_path = "outputs/fp32-5k-comparison/checkpoints/checkpoint_step_5000.pt"
print(f"Loading FP32 from: {fp32_ckpt_path}")

fp32_ckpt = torch.load(fp32_ckpt_path, map_location='cpu')
fp32_model.load_state_dict(fp32_ckpt['model_state_dict'])
print("FP32 model loaded")

# ============================================================
# GRADIENT COMPARISON
# ============================================================

print("\n" + "="*100)
print("GRADIENT MAGNITUDE COMPARISON")
print("="*100)

# Switch to training mode
ternary_model.train()
fp32_model.train()

# Zero gradients
ternary_model.zero_grad()
fp32_model.zero_grad()

# Forward + backward
print("\n[Computing gradients...]")

ternary_out = ternary_model(input_ids, targets)
ternary_loss = ternary_out['loss']
print(f"Ternary loss: {ternary_loss.item():.6f}")

fp32_out = fp32_model(input_ids, targets)
fp32_loss = fp32_out['loss']
print(f"FP32 loss: {fp32_loss.item():.6f}")

ternary_loss.backward()
fp32_loss.backward()

# Collect gradient stats
def collect_grad_stats(model, name):
    stats = []
    for n, p in model.named_parameters():
        if p.grad is not None:
            stats.append({
                'name': n,
                'mean': p.grad.abs().mean().item(),
                'std': p.grad.std().item(),
                'max': p.grad.abs().max().item(),
            })
    return stats

ternary_grads = collect_grad_stats(ternary_model, "Ternary")
fp32_grads = collect_grad_stats(fp32_model, "FP32")

# Overall stats
ternary_grad_mean = np.mean([g['mean'] for g in ternary_grads])
fp32_grad_mean = np.mean([g['mean'] for g in fp32_grads])

print(f"\n[Overall gradient statistics]")
print(f"Ternary mean gradient: {ternary_grad_mean:.6f}")
print(f"FP32 mean gradient:    {fp32_grad_mean:.6f}")
print(f"Ratio (FP32/Ternary):  {fp32_grad_mean / ternary_grad_mean if ternary_grad_mean > 0 else float('inf'):.2f}x")

# By layer type
print(f"\n[By layer type]")
for layer_type in ['attn', 'ff', 'embedding', 'ln']:
    ternary_type = [g for g in ternary_grads if layer_type in g['name'].lower()]
    fp32_type = [g for g in fp32_grads if layer_type in g['name'].lower()]
    
    if ternary_type and fp32_type:
        t_mean = np.mean([g['mean'] for g in ternary_type])
        f_mean = np.mean([g['mean'] for g in fp32_type])
        print(f"{layer_type:12} - Ternary: {t_mean:.6f}, FP32: {f_mean:.6f}, Ratio: {f_mean/t_mean if t_mean > 0 else float('inf'):.2f}x")

# ============================================================
# OPTIMIZER STATE COMPARISON
# ============================================================

print("\n" + "="*100)
print("OPTIMIZER STATE COMPARISON")
print("="*100)

ternary_opt = ternary_ckpt.get('optimizer_state_dict', {})
fp32_opt = fp32_ckpt.get('optimizer_state_dict', {})

def extract_opt_stats(opt_state):
    if 'state' not in opt_state:
        return None
    stats = []
    for pid, ps in opt_state['state'].items():
        if 'exp_avg' in ps:
            stats.append({
                'exp_avg_mean': ps['exp_avg'].abs().mean().item(),
                'exp_avg_std': ps['exp_avg'].std().item(),
            })
    return stats

ternary_opt_stats = extract_opt_stats(ternary_opt)
fp32_opt_stats = extract_opt_stats(fp32_opt)

if ternary_opt_stats and fp32_opt_stats:
    t_exp_mean = np.mean([s['exp_avg_mean'] for s in ternary_opt_stats])
    f_exp_mean = np.mean([s['exp_avg_mean'] for s in fp32_opt_stats])
    
    print(f"\n[Optimizer momentum (exp_avg)]")
    print(f"Ternary exp_avg mean: {t_exp_mean:.6f}")
    print(f"FP32 exp_avg mean:    {f_exp_mean:.6f}")
    print(f"Ratio (FP32/Ternary): {f_exp_mean / t_exp_mean if t_exp_mean > 0 else float('inf'):.2f}x")
else:
    print("\nOptimizer state not available in checkpoints")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*100)
print("SUMMARY")
print("="*100)

print(f"""
[GRADIENT MAGNITUDE]
- Ternary mean gradient: {ternary_grad_mean:.6f}
- FP32 mean gradient:    {fp32_grad_mean:.6f}
- Ratio (FP32/Ternary):  {fp32_grad_mean / ternary_grad_mean if ternary_grad_mean > 0 else float('inf'):.2f}x

Phase 19 reported: 55x smaller gradients for Ternary
Phase 20 re-verification: {fp32_grad_mean / ternary_grad_mean if ternary_grad_mean > 0 else float('inf'):.1f}x

[OPTIMIZER STATE]
""")

if ternary_opt_stats and fp32_opt_stats:
    print(f"""- Ternary exp_avg mean: {t_exp_mean:.6f}
- FP32 exp_avg mean:    {f_exp_mean:.6f}
- Ratio (FP32/Ternary): {f_exp_mean / t_exp_mean if t_exp_mean > 0 else float('inf'):.2f}x

Phase 19 reported: 10-30x smaller momentum for Ternary
Phase 20 re-verification: {f_exp_mean / t_exp_mean if t_exp_mean > 0 else float('inf'):.1f}x
""")

print("="*100)
print("AUDIT COMPLETE")
print("="*100)
