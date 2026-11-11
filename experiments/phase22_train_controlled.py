#!/usr/bin/env python3
"""
Phase 22: Controlled Ternarization Experiment - Training Script
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
import os
import math
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from src.model.bitlinear import BitLinear, BitLinearGPT, compute_abs_mean_scale
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*100)
print("PHASE 22: CONTROLLED TERNARIZATION - PRE-TRAINING VERIFICATION")
print("="*100)

# Configuration
CONFIG = {
    'vocab_size': 4000, 'd_model': 256, 'n_heads': 8, 'n_layers': 6,
    'd_ff': 1024, 'max_seq_len': 256, 'batch_size': 4,
    'gradient_accumulation_steps': 8, 'max_steps': 5000,
    'learning_rate': 3e-4, 'min_lr': 3e-5, 'warmup_steps': 500,
    'weight_decay': 0.01, 'max_grad_norm': 1.0,
    'checkpoint_steps': [1000, 2000, 5000],
    'seed': 42, 'max_samples': 50000, 'seq_length': 256,
    'eval_ratio': 0.05,
    'tokenizer_path': 'outputs/artn-optimized/tokenizer.json',
    'data_path': 'data/tiny_stories_tokenized.bin',
    'output_dir': 'outputs/phase22_controlled_experiment',
    'shared_init_path': 'outputs/phase22_controlled_experiment/shared_initialization.pt',
}

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Model classes
class StandardLinear(nn.Linear):
    pass

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
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
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

class StandardFF(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = StandardLinear(d_model, d_ff)
        self.w2 = StandardLinear(d_ff, d_model)
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class StandardBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StandardAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = StandardFF(d_model, d_ff)
    def forward(self, x, mask=None):
        return x + self.ff(self.ln2(x + self.attn(self.ln1(x), mask)))

class StandardGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([StandardBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = StandardLinear(d_model, vocab_size)
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        x = self.token_embedding(input_ids) + self.position_embedding(torch.arange(T, device=input_ids.device))
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100) if targets is not None else None
        return {"logits": logits, "loss": loss, "hidden_states": x}

# Load shared init
print("\n[Loading shared initialization]")
init_data = torch.load(CONFIG['shared_init_path'], map_location='cpu')
shared_weights = init_data['shared_weights']
weight_mapping = init_data['weight_mapping']
print(f"   Loaded {len(shared_weights)} shared tensors")

# Build FP32 model
fp32_model = StandardGPT(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'], CONFIG['d_ff'], CONFIG['max_seq_len'])
fp32_state = fp32_model.state_dict()
for k in shared_weights:
    if k in fp32_state:
        fp32_state[k] = shared_weights[k].clone()
fp32_model.load_state_dict(fp32_state)

# Build Ternary model
ternary_model = BitLinearGPT(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'], CONFIG['d_ff'], CONFIG['max_seq_len'])
ternary_state = ternary_model.state_dict()
for src, dst in weight_mapping.items():
    if src in shared_weights and dst in ternary_state:
        ternary_state[dst] = shared_weights[src].clone()
ternary_model.load_state_dict(ternary_state)

# Verify weights identical
print("\n[Verifying underlying weights are identical]")
identical, different = 0, 0
for src, dst in weight_mapping.items():
    if src in shared_weights and dst in ternary_state:
        if torch.allclose(shared_weights[src], ternary_state[dst], atol=1e-8):
            identical += 1
        else:
            different += 1
            print(f"   DIFFERENT: {dst}")

print(f"   Identical: {identical}, Different: {different}")
if different > 0:
    print("   ERROR: Weights not identical!")
    sys.exit(1)

# Quantization error
print("\n[Quantization error at initialization]")
quant_stats = []
for name, param in ternary_model.named_parameters():
    if 'weight' in name and 'embedding' not in name and 'ln' not in name and param.dim() == 2:
        scale = compute_abs_mean_scale(param)
        w_q = torch.zeros_like(param)
        w_q[param/scale > 0.5] = 1.0
        w_q[param/scale < -0.5] = -1.0
        w_eff = w_q * scale
        cos_sim = F.cosine_similarity(param.flatten().unsqueeze(0), w_eff.flatten().unsqueeze(0)).item()
        abs_err = (param - w_eff).abs().mean().item()
        quant_stats.append({'layer': name, 'cos_sim': cos_sim, 'abs_err': abs_err})

mean_cos = np.mean([s['cos_sim'] for s in quant_stats])
mean_abs = np.mean([s['abs_err'] for s in quant_stats])
print(f"   Layers: {len(quant_stats)}")
print(f"   Mean cosine similarity: {mean_cos:.4f}")
print(f"   Mean absolute error: {mean_abs:.6f}")

# Parameter counts
fp32_p = sum(p.numel() for p in fp32_model.parameters())
ternary_p = sum(p.numel() for p in ternary_model.parameters())
print(f"\n[Parameter counts]")
print(f"   FP32: {fp32_p:,}")
print(f"   Ternary: {ternary_p:,}")

print("\n" + "="*100)
print("VERIFICATION COMPLETE - READY FOR TRAINING")
print("="*100)
