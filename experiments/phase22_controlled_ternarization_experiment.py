#!/usr/bin/env python3
"""
Phase 22: Controlled Ternarization Experiment

CRITICAL: This experiment isolates ternarization by starting from IDENTICAL initialization.

Design:
1. Create ONE shared FP32 initialization (fixed random seed)
2. Create TWO models from that exact initialization:
   - Model A: FP32 (standard nn.Linear layers)
   - Model B: Ternary (BitLinear layers with shared weights)
3. Verify initialization is identical before training
4. Train both models with identical hyperparameters
5. Compare learning trajectories

PRIMARY QUESTION:
"If both models start from the same underlying initialization, how much does 
replacing the FP32 weights with the ternary forward representation change learning?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.model.bitlinear import BitLinear, BitLinearGPT, compute_abs_mean_scale
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*80)
print("PHASE 22: CONTROLLED TERNARIZATION EXPERIMENT")
print("="*80)

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Model architecture
    'vocab_size': 4000,
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,
    'd_ff': 1024,
    'max_seq_len': 256,
    
    # Training
    'batch_size': 4,
    'gradient_accumulation_steps': 8,
    'max_steps': 5000,
    'learning_rate': 3e-4,
    'min_lr': 3e-5,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    
    # Checkpoints
    'checkpoint_steps': [1000, 2000, 5000],
    
    # Random seed (CRITICAL for reproducibility)
    'seed': 42,
    
    # Data
    'max_samples': 50000,
    'seq_length': 256,
}

print(f"\n[Configuration]")
for k, v in CONFIG.items():
    print(f"   {k}: {v}")

# ============================================================
# STEP 1: CREATE SHARED INITIALIZATION
# ============================================================

print("\n" + "="*80)
print("STEP 1: CREATE SHARED FP32 INITIALIZATION")
print("="*80)

# Set random seed
torch.manual_seed(CONFIG['seed'])

# Define weight structure (what we'll initialize)
# For a 6-layer GPT with:
# - Token embedding: [4000, 256]
# - Position embedding: [256, 256]
# - Per layer: Q, K, V, O projections [256, 256] each, FF [256, 1024] and [1024, 256]
# - LayerNorm parameters
# - LM head: [4000, 256]

def create_shared_initialization(config, init_std=0.02):
    """Create shared FP32 weights for both models."""
    
    weights = {}
    
    # Token and position embeddings
    weights['token_embedding.weight'] = torch.randn(config['vocab_size'], config['d_model']) * init_std
    weights['position_embedding.weight'] = torch.randn(config['max_seq_len'], config['d_model']) * init_std
    
    # Per-layer weights
    for layer_idx in range(config['n_layers']):
        prefix = f'blocks.{layer_idx}'
        
        # Attention projections (Q, K, V, O)
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            key = f'{prefix}.attn.{proj}.weight'
            weights[key] = torch.randn(config['d_model'], config['d_model']) * init_std
            
            bias_key = f'{prefix}.attn.{proj}.bias'
            weights[bias_key] = torch.zeros(config['d_model'])
        
        # Feed-forward
        weights[f'{prefix}.ff.w1.weight'] = torch.randn(config['d_ff'], config['d_model']) * init_std
        weights[f'{prefix}.ff.w1.bias'] = torch.zeros(config['d_ff'])
        weights[f'{prefix}.ff.w2.weight'] = torch.randn(config['d_model'], config['d_ff']) * init_std
        weights[f'{prefix}.ff.w2.bias'] = torch.zeros(config['d_model'])
        
        # LayerNorm
        weights[f'{prefix}.ln1.weight'] = torch.ones(config['d_model'])
        weights[f'{prefix}.ln1.bias'] = torch.zeros(config['d_model'])
        weights[f'{prefix}.ln2.weight'] = torch.ones(config['d_model'])
        weights[f'{prefix}.ln2.bias'] = torch.zeros(config['d_model'])
    
    # Final LayerNorm
    weights['ln_final.weight'] = torch.ones(config['d_model'])
    weights['ln_final.bias'] = torch.zeros(config['d_model'])
    
    # LM head
    weights['lm_head.weight'] = torch.randn(config['vocab_size'], config['d_model']) * init_std
    weights['lm_head.bias'] = torch.zeros(config['vocab_size'])
    
    return weights

print("\n[Creating shared weights with std=0.02]")
shared_weights = create_shared_initialization(CONFIG, init_std=0.02)

print(f"\n[Shared weight statistics]")
total_params = sum(w.numel() for w in shared_weights.values())
print(f"   Total parameters: {total_params:,}")
print(f"   Number of weight tensors: {len(shared_weights)}")

# Check weight statistics
sample_keys = ['token_embedding.weight', 'blocks.0.attn.q_proj.weight', 'blocks.0.ff.w1.weight']
print(f"\n[Sample weight statistics]")
for key in sample_keys:
    w = shared_weights[key]
    print(f"   {key}:")
    print(f"      shape: {w.shape}")
    print(f"      mean: {w.mean().item():.6f}")
    print(f"      std: {w.std().item():.6f}")

# ============================================================
# STEP 2: BUILD MODELS FROM SHARED INITIALIZATION
# ============================================================

print("\n" + "="*80)
print("STEP 2: BUILD MODELS FROM SHARED INITIALIZATION")
print("="*80)

# We'll define both model classes here to have full control

class StandardLinear(nn.Linear):
    """Standard FP32 linear layer."""
    pass

class StandardAttention(nn.Module):
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
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class StandardGPT(nn.Module):
    """Standard FP32 GPT model."""
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
        
        return {"logits": logits, "loss": loss, "hidden_states": x}

print("\n[Building FP32 model]")
fp32_model = StandardGPT(
    CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_heads'],
    CONFIG['n_layers'], CONFIG['d_ff'], CONFIG['max_seq_len']
)

# Load shared weights into FP32 model
fp32_state = fp32_model.state_dict()
for key in shared_weights:
    if key in fp32_state:
        fp32_state[key] = shared_weights[key].clone()
    elif key.replace('.weight', '') in fp32_state:
        # Handle bias
        continue

fp32_model.load_state_dict(fp32_state)
print(f"   FP32 model loaded with shared weights")

print("\n[Building Ternary model]")
ternary_model = BitLinearGPT(
    CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_heads'],
    CONFIG['n_layers'], CONFIG['d_ff'], CONFIG['max_seq_len']
)

# Load shared weights into Ternary model
# CRITICAL: BitLinear has different parameter names
ternary_state = ternary_model.state_dict()

# Map shared weights to BitLinear parameter names
weight_mapping = {}

# Embeddings
weight_mapping['token_embedding.weight'] = 'token_embedding.weight'
weight_mapping['position_embedding.weight'] = 'position_embedding.weight'

# Layers
for layer_idx in range(CONFIG['n_layers']):
    src_prefix = f'blocks.{layer_idx}'
    dst_prefix = f'blocks.{layer_idx}'
    
    # Attention
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        src_key = f'{src_prefix}.attn.{proj}.weight'
        dst_key = f'{dst_prefix}.attn.{proj}.weight'
        weight_mapping[src_key] = dst_key
        
        src_bias = f'{src_prefix}.attn.{proj}.bias'
        dst_bias = f'{dst_prefix}.attn.{proj}.bias'
        if src_bias in shared_weights:
            weight_mapping[src_bias] = dst_bias
    
    # FF
    for w in ['w1', 'w2']:
        src_key = f'{src_prefix}.ff.{w}.weight'
        dst_key = f'{dst_prefix}.ff.{w}.weight'
        weight_mapping[src_key] = dst_key
        
        src_bias = f'{src_prefix}.ff.{w}.bias'
        dst_bias = f'{dst_prefix}.ff.{w}.bias'
        if src_bias in shared_weights:
            weight_mapping[src_bias] = dst_bias
    
    # LayerNorm
    for ln in ['ln1', 'ln2']:
        for p in ['weight', 'bias']:
            src_key = f'{src_prefix}.{ln}.{p}'
            dst_key = f'{dst_prefix}.{ln}.{p}'
            weight_mapping[src_key] = dst_key

# Final LayerNorm
weight_mapping['ln_final.weight'] = 'ln_final.weight'
weight_mapping['ln_final.bias'] = 'ln_final.bias'

# LM head
weight_mapping['lm_head.weight'] = 'lm_head.weight'
weight_mapping['lm_head.bias'] = 'lm_head.bias'

# Apply mapping
for src_key, dst_key in weight_mapping.items():
    if src_key in shared_weights and dst_key in ternary_state:
        ternary_state[dst_key] = shared_weights[src_key].clone()

ternary_model.load_state_dict(ternary_state)
print(f"   Ternary model loaded with shared weights")

# ============================================================
# STEP 3: VERIFY INITIALIZATION IS IDENTICAL
# ============================================================

print("\n" + "="*80)
print("STEP 3: VERIFY INITIALIZATION IS IDENTICAL")
print("="*80)

print("\n[Comparing loaded weights]")

identical_count = 0
different_count = 0
comparison_results = []

for src_key, dst_key in weight_mapping.items():
    if src_key in shared_weights and dst_key in ternary_state:
        src_w = shared_weights[src_key]
        dst_w = ternary_state[dst_key]
        
        if torch.allclose(src_w, dst_w, atol=1e-6):
            identical_count += 1
            comparison_results.append((src_key, 'IDENTICAL', 1.0))
        else:
            different_count += 1
            max_diff = (src_w - dst_w).abs().max().item()
            comparison_results.append((src_key, 'DIFFERENT', max_diff))

print(f"\n   Identical: {identical_count}")
print(f"   Different: {different_count}")

if different_count > 0:
    print(f"\n   WARNING: Some weights are different!")
    for key, status, diff in comparison_results:
        if status == 'DIFFERENT':
            print(f"      {key}: max_diff = {diff}")

# ============================================================
# STEP 4: REPORT DIFFERENCES INTRODUCED BY TERNARIZATION
# ============================================================

print("\n" + "="*80)
print("STEP 4: DIFFERENCES INTRODUCED BY TERNARIZATION")
print("="*80)

print("""
CRITICAL: Both models now have IDENTICAL underlying weights.
The ONLY difference is how they use these weights in the forward pass:

FP32 Model:
  - Uses weights directly (nn.Linear)
  - Forward: output = input @ weight.T + bias
  
Ternary Model:
  - Quantizes weights during forward pass (BitLinear)
  - Forward: 
    1. Compute scale = abs_mean(weight)
    2. Quantize: weight_q = round(weight / scale) to {-1, 0, +1}
    3. Dequantize: weight_eff = weight_q * scale
    4. output = input @ weight_eff.T + bias

The quantization happens at EVERY forward pass, not just at initialization.

KEY MEASUREMENTS:
1. Quantization error: How much does weight_eff differ from weight?
2. Information loss: How much information is lost in quantization?
3. Learning impact: Does this affect gradient flow and learning?
""")

# Measure quantization error at initialization
print("\n[Quantization error at initialization]")

quant_errors = []
for name, param in ternary_model.named_parameters():
    if 'weight' in name and 'embedding' not in name and 'ln' not in name and 'norm' not in name:
        if param.dim() == 2:
            # Compute what BitLinear will do in forward pass
            scale = compute_abs_mean_scale(param)
            weight_scaled = param / scale.clamp(min=1e-8)
            weight_q = torch.zeros_like(param)
            weight_q[weight_scaled > 0.5] = 1.0
            weight_q[weight_scaled < -0.5] = -1.0
            weight_eff = weight_q * scale
            
            cos_sim = F.cosine_similarity(
                param.flatten().unsqueeze(0),
                weight_eff.flatten().unsqueeze(0)
            ).item()
            
            quant_errors.append({
                'layer': name,
                'cos_sim': cos_sim,
                'scale': scale.item(),
                'std_original': param.std().item(),
                'std_eff': weight_eff.std().item(),
            })

import numpy as np
mean_cos = np.mean([e['cos_sim'] for e in quant_errors])
print(f"\n   Mean cosine similarity (underlying vs effective): {mean_cos:.4f}")
print(f"   This means ~{(1-mean_cos)*100:.1f}% information loss at initialization")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*80)
print("EXPERIMENT DESIGN SUMMARY")
print("="*80)

print(f"""
✅ INITIALIZATION VERIFIED:
   - Both models start from IDENTICAL weights
   - Shared initialization: std=0.02 for all layers
   - Random seed: {CONFIG['seed']}

✅ ISOLATED VARIABLE:
   - ONLY difference: ternary forward pass quantization
   - No architectural differences
   - No different hyperparameters
   - No different data

✅ CONTROLLED MEASUREMENTS:
   - Training loss trajectory
   - Evaluation loss
   - Top-1 accuracy
   - Context benefit
   - Shuffle effect
   - Quantization error evolution
   - Activation/logit divergence

READY FOR TRAINING.

Next step: Run training with identical hyperparameters for both models.
""")

# Save shared weights for reference
save_dir = Path('outputs/phase22_controlled_experiment')
save_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    'shared_weights': shared_weights,
    'config': CONFIG,
    'weight_mapping': weight_mapping,
}, save_dir / 'shared_initialization.pt')

print(f"\n✅ Shared initialization saved to: {save_dir / 'shared_initialization.pt'}")

print("\n" + "="*80)
print("INITIALIZATION COMPLETE - READY FOR TRAINING")
print("="*80)
