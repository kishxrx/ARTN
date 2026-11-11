#!/usr/bin/env python3
"""
ARCHITECTURAL ANALYSIS: Random Test (SUCCESS) vs TinyStories Test (FAILURE)

Purpose: Determine EXACT differences that could explain success/failure.
"""

import torch
from src.model.bitlinear import BitLinearGPT

print("="*80)
print("ARCHITECTURAL ANALYSIS")
print("="*80)

# =============================================================================
# SUCCESSFUL RANDOM TEST MODEL
# =============================================================================

print("\n" + "="*80)
print("SUCCESSFUL: Random-Token Test Model (vocab=100, model=115K)")
print("="*80)

model_random = BitLinearGPT(
    vocab_size=100,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    max_seq_len=32,
)

params_random = {}
total_random = 0

for name, param in model_random.named_parameters():
    params_random[name] = param.numel()
    total_random += param.numel()

print(f"\nParameter breakdown:")
for name, count in sorted(params_random.items()):
    print(f"  {name}: {count:,}")

print(f"\nTotal parameters: {total_random:,}")

# =============================================================================
# FAILED TINYSTORIES MODEL
# =============================================================================

print("\n" + "="*80)
print("FAILED: TinyStories Test Model (vocab=4000, model=6.8M)")
print("="*80)

model_tiny = BitLinearGPT(
    vocab_size=4000,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_seq_len=256,
)

params_tiny = {}
total_tiny = 0

for name, param in model_tiny.named_parameters():
    params_tiny[name] = param.numel()
    total_tiny += param.numel()

print(f"\nParameter breakdown:")
for name, count in sorted(params_tiny.items()):
    print(f"  {name}: {count:,}")

print(f"\nTotal parameters: {total_tiny:,}")

# =============================================================================
# KEY DIFFERENCES
# =============================================================================

print("\n" + "="*80)
print("KEY ARCHITECTURAL DIFFERENCES")
print("="*80)

print(f"""
| Component | Random Test (SUCCESS) | TinyStories (FAILED) | Ratio |
|-----------|----------------------|---------------------|-------|
| vocab_size | 100 | 4000 | 40x |
| d_model | 64 | 256 | 4x |
| n_heads | 4 | 8 | 2x |
| n_layers | 2 | 6 | 3x |
| d_ff | 256 | 1024 | 4x |
| max_seq_len | 32 | 256 | 8x |
| Total params | {total_random:,} | {total_tiny:,} | {total_tiny/total_random:.1f}x |
""")

# =============================================================================
# OUTPUT LAYER ANALYSIS (CRITICAL)
# =============================================================================

print("\n" + "="*80)
print("OUTPUT LAYER (lm_head) ANALYSIS")
print("="*80)

lm_head_random = params_random.get('lm_head.weight', 0)
lm_head_tiny = params_tiny.get('lm_head.weight', 0)

print(f"""
lm_head is BitLinear(d_model, vocab_size):

Random Test:
  - lm_head = BitLinear(64, 100)
  - Parameters: {lm_head_random:,}
  - Output dimension: 100 (vocab_size)
  - Ternary weights: 64 × 100 = 6,400 weights

TinyStories:
  - lm_head = BitLinear(256, 4000)
  - Parameters: {lm_head_tiny:,}
  - Output dimension: 4000 (vocab_size)
  - Ternary weights: 256 × 4000 = 1,024,000 weights

RATIO: {lm_head_tiny/lm_head_random:.1f}x more ternary weights in output layer

CRITICAL: Output layer grows LINEARLY with vocab_size.
  - vocab 100 → 6.4K ternary weights
  - vocab 4000 → 1.0M ternary weights
  - 160x more ternary weights computing logits
""")

# =============================================================================
# EMBEDDING LAYER ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("EMBEDDING LAYER ANALYSIS")
print("="*80)

emb_random = params_random.get('token_embedding.weight', 0)
emb_tiny = params_tiny.get('token_embedding.weight', 0)

print(f"""
token_embedding is nn.Embedding (FLOAT32, NOT ternary):

Random Test:
  - token_embedding = nn.Embedding(100, 64)
  - Parameters: {emb_random:,}
  - FP32 weights: 100 × 64 = 6,400 floats

TinyStories:
  - token_embedding = nn.Embedding(4000, 256)
  - Parameters: {emb_tiny:,}
  - FP32 weights: 4000 × 256 = 1,024,000 floats

RATIO: {emb_tiny/emb_random:.1f}x more embedding weights

NOTE: Embeddings are FP32 (NOT ternary), so this should NOT affect 
      ternary quantization behavior.
""")

# =============================================================================
# HIDDEN LAYERS ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("HIDDEN LAYERS (Transformer Blocks) ANALYSIS")
print("="*80)

# Count transformer block parameters
block_params_random = sum(v for k, v in params_random.items() if 'blocks' in k)
block_params_tiny = sum(v for k, v in params_tiny.items() if 'blocks' in k)

print(f"""
Random Test:
  - 2 transformer blocks
  - Total block params: {block_params_random:,}
  - Per block: {block_params_random//2:,}

TinyStories:
  - 6 transformer blocks
  - Total block params: {block_params_tiny:,}
  - Per block: {block_params_tiny//6:,}

RATIO: {block_params_tiny/block_params_random:.1f}x more in transformer blocks
       {block_params_tiny/block_params_random/3:.1f}x per block (accounting for layer count)

SCALING: Hidden layers scale with d_model² and d_ff.
  - Attention: 4 × d_model² per layer (Q,K,V,O projections)
  - FFN: 2 × d_model × d_ff per layer
  - These are ALL ternary (BitLinear)
""")

# =============================================================================
# COMPUTE EXACT PARAMETER COUNTS PER LAYER TYPE
# =============================================================================

print("\n" + "="*80)
print("EXACT PARAMETER BREAKDOWN BY LAYER TYPE")
print("="*80)

def count_by_layer_type(params_dict):
    embedding = 0
    position = 0
    attention = 0
    ffn = 0
    ln = 0
    lm_head = 0
    
    for name, count in params_dict.items():
        if 'token_embedding' in name:
            embedding += count
        elif 'position_embedding' in name:
            position += count
        elif 'attn' in name or any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'out_proj']):
            attention += count
        elif 'ff' in name or 'w1' in name or 'w2' in name:
            ffn += count
        elif 'ln' in name or 'layer_norm' in name:
            ln += count
        elif 'lm_head' in name:
            lm_head += count
    
    return {
        'embedding': embedding,
        'position': position,
        'attention': attention,
        'ffn': ffn,
        'layer_norm': ln,
        'lm_head': lm_head,
    }

breakdown_random = count_by_layer_type(params_random)
breakdown_tiny = count_by_layer_type(params_tiny)

print(f"""
| Layer Type | Random Test | TinyStories | Ratio | Quantized? |
|------------|-------------|-------------|-------|------------|
| Embedding | {breakdown_random['embedding']:,} | {breakdown_tiny['embedding']:,} | {breakdown_tiny['embedding']/breakdown_random['embedding']:.1f}x | FP32 |
| Position | {breakdown_random['position']:,} | {breakdown_tiny['position']:,} | {breakdown_tiny['position']/breakdown_random['position']:.1f}x | FP32 |
| Attention | {breakdown_random['attention']:,} | {breakdown_tiny['attention']:,} | {breakdown_tiny['attention']/breakdown_random['attention']:.1f}x | TERNARY |
| FFN | {breakdown_random['ffn']:,} | {breakdown_tiny['ffn']:,} | {breakdown_tiny['ffn']/breakdown_random['ffn']:.1f}x | TERNARY |
| LayerNorm | {breakdown_random['layer_norm']:,} | {breakdown_tiny['layer_norm']:,} | {breakdown_tiny['layer_norm']/breakdown_random['layer_norm']:.1f}x | FP32 |
| lm_head | {breakdown_random['lm_head']:,} | {breakdown_tiny['lm_head']:,} | {breakdown_tiny['lm_head']/breakdown_random['lm_head']:.1f}x | TERNARY |
""")

# =============================================================================
# THE KEY QUESTION
# =============================================================================

print("\n" + "="*80)
print("THE KEY QUESTION")
print("="*80)

print(f"""
Random test SUCCEEDED with:
  - lm_head ternary weights: {lm_head_random:,} (6.4K)
  - vocab_size: 100
  
TinyStories FAILED with:
  - lm_head ternary weights: {lm_head_tiny:,} (1.0M)
  - vocab_size: 4000

HYPOTHESIS: Ternary output layer may struggle with large vocab.

The lm_head must map from d_model (64 or 256) to vocab_size (100 or 4000).
With ternary weights limited to {-1, 0, +1}, a larger vocab means:
  - More output classes to distinguish
  - More ternary weights computing each logit
  - Potential capacity bottleneck

But is this the CAUSE or just CORRELATED?

CONFOUNDING FACTORS:
  1. Model size: 115K vs 6.8M params (60x)
  2. Vocab size: 100 vs 4000 (40x)
  3. Data: Random tokens vs Real language
  4. d_model: 64 vs 256 (4x)
  5. Depth: 2 vs 6 layers (3x)

To isolate, we need MINIMAL controlled experiments.
""")

print("="*80)
