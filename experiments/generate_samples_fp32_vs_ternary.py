#!/usr/bin/env python3
"""
Generate samples from FP32 and Ternary models for comparison
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.bitlinear import BitLinearGPT

# FP32 model definition
import torch.nn.functional as F
import math

class StandardLinear(torch.nn.Linear):
    pass

class StandardMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = StandardLinear(d_model, d_model)
        self.k_proj = StandardLinear(d_model, d_model)
        self.v_proj = StandardLinear(d_model, d_model)
        self.o_proj = StandardLinear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(context)

class StandardFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = StandardLinear(d_model, d_ff)
        self.w2 = StandardLinear(d_ff, d_model)
        self.activation = torch.nn.GELU()
    
    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))

class StandardTransformerBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.attn = StandardMultiHeadAttention(d_model, n_heads)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.ff = StandardFeedForward(d_model, d_ff)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class StandardGPT(torch.nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(max_seq_len, d_model)
        self.blocks = torch.nn.ModuleList([StandardTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_final = torch.nn.LayerNorm(d_model)
        self.lm_head = StandardLinear(d_model, vocab_size)
    
    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), ignore_index=-100, reduction='mean')
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

# Load tokenizer
tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')

# Prompts
PROMPTS = [
    "Once upon a time",
    "The little girl",
    "A boy named",
    "One day",
    "There was a",
]

print("="*100)
print("FP32 vs TERNARY TEXT GENERATION COMPARISON")
print("="*100)

# Load FP32 model
print("\n[Loading FP32 Model]")
fp32_ckpt = torch.load('outputs/fp32-5k-comparison/checkpoints/checkpoint_step_5000.pt', map_location='cpu')
fp32_model = StandardGPT(vocab_size=4000, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_seq_len=256)
fp32_model.load_state_dict(fp32_ckpt['model_state_dict'])
fp32_model.eval()
print(f"  Loaded FP32 checkpoint: step {fp32_ckpt['step']}")

# Load Ternary model
print("\n[Loading Ternary Model]")
ternary_ckpt = torch.load('outputs/artn-50k-best-config/checkpoints/checkpoint_step_5000.pt', map_location='cpu')
ternary_model = BitLinearGPT(vocab_size=4000, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_seq_len=256)
ternary_model.load_state_dict(ternary_ckpt['model_state_dict'])
ternary_model.eval()
print(f"  Loaded Ternary checkpoint: step {ternary_ckpt['step']}")

# Generate samples
print("\n" + "="*100)
print("GENERATED SAMPLES")
print("="*100)

for i, prompt in enumerate(PROMPTS, 1):
    print(f"\n{'='*100}")
    print(f"PROMPT {i}: \"{prompt}\"")
    print("="*100)
    
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
    
    # FP32 generation
    torch.manual_seed(42)
    fp32_output = fp32_model.generate(prompt_tensor.clone(), max_new_tokens=50, temperature=1.0)
    fp32_text = tokenizer.decode(fp32_output[0].tolist())
    
    print(f"\n[FP32 Output]")
    print(f"  {fp32_text}")
    
    # Ternary generation
    torch.manual_seed(42)
    ternary_output = ternary_model.generate(prompt_tensor.clone(), max_new_tokens=50, temperature=1.0)
    ternary_text = tokenizer.decode(ternary_output[0].tolist())
    
    print(f"\n[Ternary Output]")
    print(f"  {ternary_text}")

print("\n" + "="*100)
print("GENERATION COMPLETE")
print("="*100)
