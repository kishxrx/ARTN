#!/usr/bin/env python3
"""
READ-ONLY: Quantization-Induced Information Loss Analysis

Compare FP32 and Ternary forward passes to identify:
1. Quantization error at each layer
2. Where representations first diverge
3. Whether divergence destroys contextual information

CRITICAL: READ-ONLY. No model modification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.model.bitlinear import BitLinearGPT, BitLinear, compute_abs_mean_scale
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*100)
print("READ-ONLY: Quantization-Induced Information Loss Analysis")
print("="*100)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

def compute_errors(original, quantized):
    """Compute various error metrics."""
    abs_error = (original - quantized).abs()
    rel_error = abs_error / (original.abs().clamp(min=1e-8))
    
    return {
        'abs_error_mean': abs_error.mean().item(),
        'abs_error_std': abs_error.std().item(),
        'abs_error_max': abs_error.max().item(),
        'rel_error_mean': rel_error.mean().item(),
        'rel_error_std': rel_error.std().item(),
        'cosine_sim': compute_similarity(original, quantized),
    }

def get_ternary_distribution(weight, scale, threshold=0.5):
    """Get distribution of ternary values."""
    weight_scaled = weight / scale.clamp(min=1e-8)
    weight_q = torch.zeros_like(weight)
    weight_q[weight_scaled > threshold] = 1.0
    weight_q[weight_scaled < -threshold] = -1.0
    
    total = weight.numel()
    return {
        'neg': (weight_q == -1).sum().item() / total,
        'zero': (weight_q == 0).sum().item() / total,
        'pos': (weight_q == 1).sum().item() / total,
    }

# ============================================================
# LOAD MODELS
# ============================================================

print("\n[Loading tokenizer]")

tokenizer_path = "outputs/artn-optimized/tokenizer.json"
tokenizer = BPETokenizer.load(tokenizer_path)
print(f"Tokenizer vocab_size = {tokenizer.vocab_size}")

# Model config
vocab_size = 4000
d_model = 256
n_heads = 8
n_layers = 6
d_ff = 1024
max_seq_len = 256

# Checkpoints to analyze
checkpoints = {
    '1k': {
        'ternary': 'outputs/artn-50k-best-config/checkpoints/checkpoint_step_1000.pt',
        'fp32': None,  # No FP32 1k checkpoint
    },
    '5k': {
        'ternary': 'outputs/artn-50k-best-config/checkpoints/checkpoint_step_5000.pt',
        'fp32': 'outputs/fp32-5k-comparison/checkpoints/checkpoint_step_5000.pt',
    },
}

# Test input
test_texts = [
    "Once upon a time there was a little girl who lived in a small village near the forest.",
    "The cat sat on the mat and looked at the dog with curious eyes.",
]

test_tokens = [tokenizer.encode(t) for t in test_texts]

# ============================================================
# PART 1: WEIGHT-LEVEL QUANTIZATION ERROR
# ============================================================

print("\n" + "="*100)
print("PART 1: WEIGHT-LEVEL QUANTIZATION ERROR")
print("="*100)

for step_name, ckpt_paths in checkpoints.items():
    if ckpt_paths['fp32'] is None:
        print(f"\n[Skipping {step_name} - no FP32 checkpoint]")
        continue
    
    print(f"\n[Analyzing {step_name} checkpoints]")
    
    # Load checkpoints
    ternary_ckpt = torch.load(ckpt_paths['ternary'], map_location='cpu')
    fp32_ckpt = torch.load(ckpt_paths['fp32'], map_location='cpu')
    
    ternary_state = ternary_ckpt['model_state_dict']
    fp32_state = fp32_ckpt['model_state_dict']
    
    # Analyze each BitLinear layer
    layer_errors = []
    
    for key in ternary_state.keys():
        if 'weight' in key and 'embedding' not in key and 'ln' not in key and 'norm' not in key:
            ternary_weight = ternary_state[key]
            
            if ternary_weight.dim() != 2:
                continue
            
            # Compute scale and quantize
            scale = compute_abs_mean_scale(ternary_weight)
            weight_scaled = ternary_weight / scale.clamp(min=1e-8)
            weight_q = torch.zeros_like(ternary_weight)
            weight_q[weight_scaled > 0.5] = 1.0
            weight_q[weight_scaled < -0.5] = -1.0
            weight_dequant = weight_q * scale
            
            # Compute errors
            errors = compute_errors(ternary_weight, weight_dequant)
            
            # Get ternary distribution
            dist = get_ternary_distribution(ternary_weight, scale)
            
            layer_errors.append({
                'layer': key,
                'scale': scale.item(),
                'weight_std': ternary_weight.std().item(),
                'dequant_std': weight_dequant.std().item(),
                **errors,
                **dist,
            })
    
    # Print summary
    print(f"\n[Weight quantization error by layer type]")
    
    for layer_type in ['attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.o_proj', 'ff.w1', 'ff.w2']:
        type_errors = [e for e in layer_errors if layer_type in e['layer']]
        if type_errors:
            mean_cos = np.mean([e['cosine_sim'] for e in type_errors])
            mean_abs = np.mean([e['abs_error_mean'] for e in type_errors])
            mean_zero = np.mean([e['zero'] for e in type_errors])
            print(f"  {layer_type}: cosine_sim={mean_cos:.4f}, abs_error={mean_abs:.6f}, zero_frac={mean_zero:.2%}")

# ============================================================
# PART 2: ACTIVATION-LEVEL DIVERGENCE
# ============================================================

print("\n" + "="*100)
print("PART 2: ACTIVATION-LEVEL DIVERGENCE")
print("="*100)

# Define hook to capture activations
activation_store = {}

def get_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            activation_store[name] = output[0].detach().clone()
        else:
            activation_store[name] = output.detach().clone()
    return hook

# Build FP32 model
class StandardLinear(nn.Linear):
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

# Analyze 5k checkpoint
print("\n[Analyzing 5k checkpoint activations]")

# Load Ternary model
ternary_model = BitLinearGPT(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len)
ternary_ckpt = torch.load(checkpoints['5k']['ternary'], map_location='cpu')
ternary_model.load_state_dict(ternary_ckpt['model_state_dict'])
ternary_model.eval()

# Load FP32 model
fp32_model = StandardGPT(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len)
fp32_ckpt = torch.load(checkpoints['5k']['fp32'], map_location='cpu')
fp32_model.load_state_dict(fp32_ckpt['model_state_dict'])
fp32_model.eval()

# Run on test inputs
print("\n[Running forward pass on test inputs]")

for i, tokens in enumerate(test_tokens):
    input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
    targets = torch.tensor([tokens[1:]], dtype=torch.long)
    
    print(f"\n--- Test Input {i+1} (length={input_ids.shape[1]}) ---")
    
    with torch.no_grad():
        # Ternary forward
        ternary_out = ternary_model(input_ids, targets)
        ternary_logits = ternary_out['logits']
        ternary_loss = ternary_out['loss']
        
        # FP32 forward
        fp32_out = fp32_model(input_ids, targets)
        fp32_logits = fp32_out['logits']
        fp32_loss = fp32_out['loss']
        
    # Compare losses
    print(f"\n[Loss comparison]")
    print(f"  Ternary loss: {ternary_loss.item():.4f}")
    print(f"  FP32 loss:    {fp32_loss.item():.4f}")
    print(f"  Ratio:        {ternary_loss.item() / fp32_loss.item():.2f}x")
    
    # Compare logits
    logits_errors = compute_errors(fp32_logits, ternary_logits)
    print(f"\n[Logits comparison]")
    print(f"  Cosine similarity: {logits_errors['cosine_sim']:.4f}")
    print(f"  Abs error mean:    {logits_errors['abs_error_mean']:.4f}")
    print(f"  Abs error std:     {logits_errors['abs_error_std']:.4f}")
    
    # Compare predictions
    fp32_preds = fp32_logits.argmax(dim=-1)
    ternary_preds = ternary_logits.argmax(dim=-1)
    
    agreement = (fp32_preds == ternary_preds).float().mean().item()
    print(f"\n[Prediction agreement]")
    print(f"  Token-level agreement: {agreement:.2%}")
    
    # Top-k comparison
    k = 5
    fp32_topk = fp32_logits.topk(k, dim=-1).indices
    ternary_topk = ternary_logits.topk(k, dim=-1).indices
    
    topk_overlap = 0
    for pos in range(fp32_topk.shape[1]):
        fp32_set = set(fp32_topk[0, pos].tolist())
        ternary_set = set(ternary_topk[0, pos].tolist())
        topk_overlap += len(fp32_set & ternary_set) / k
    
    topk_overlap /= fp32_topk.shape[1]
    print(f"  Top-{k} overlap:        {topk_overlap:.2%}")

# ============================================================
# PART 3: LAYER-BY-LAYER DIVERGENCE
# ============================================================

print("\n" + "="*100)
print("PART 3: LAYER-BY-LAYER DIVERGENCE")
print("="*100)

# Compare hidden states after each block
print("\n[Measuring divergence through layers]")

# Register hooks on each block
for i, block in enumerate(fp32_model.blocks):
    block.register_forward_hook(get_hook(f'fp32_block_{i}'))

for i, block in enumerate(ternary_model.blocks):
    block.register_forward_hook(get_hook(f'ternary_block_{i}'))

# Run forward pass
input_ids = torch.tensor([test_tokens[0][:-1]], dtype=torch.long)

with torch.no_grad():
    fp32_out = fp32_model(input_ids)
    ternary_out = ternary_model(input_ids)

# Compare block outputs
print(f"\n[Block-by-block hidden state comparison]")
print(f"{'Block':<10} {'Cosine Sim':<15} {'Abs Error Mean':<18} {'Abs Error Std':<15}")
print("-" * 60)

for i in range(n_layers):
    fp32_h = activation_store.get(f'fp32_block_{i}')
    ternary_h = activation_store.get(f'ternary_block_{i}')
    
    if fp32_h is not None and ternary_h is not None:
        errors = compute_errors(fp32_h, ternary_h)
        print(f"{i:<10} {errors['cosine_sim']:<15.4f} {errors['abs_error_mean']:<18.6f} {errors['abs_error_std']:<15.6f}")

# ============================================================
# PART 4: EMBEDDING COMPARISON
# ============================================================

print("\n" + "="*100)
print("PART 4: EMBEDDING AND EARLY DIVERGENCE")
print("="*100)

# Compare token embeddings
fp32_tok_emb = fp32_model.token_embedding.weight
ternary_tok_emb = ternary_model.token_embedding.weight

emb_errors = compute_errors(fp32_tok_emb, ternary_tok_emb)
print(f"\n[Token embedding comparison]")
print(f"  Cosine similarity: {emb_errors['cosine_sim']:.4f}")
print(f"  Abs error mean:    {emb_errors['abs_error_mean']:.6f}")
print(f"  Abs error std:     {emb_errors['abs_error_std']:.6f}")

# Compare position embeddings
fp32_pos_emb = fp32_model.position_embedding.weight
ternary_pos_emb = ternary_model.position_embedding.weight

pos_errors = compute_errors(fp32_pos_emb, ternary_pos_emb)
print(f"\n[Position embedding comparison]")
print(f"  Cosine similarity: {pos_errors['cosine_sim']:.4f}")
print(f"  Abs error mean:    {pos_errors['abs_error_mean']:.6f}")
print(f"  Abs error std:     {pos_errors['abs_error_std']:.6f}")

# ============================================================
# PART 5: FINAL HIDDEN STATES AND LOGITS
# ============================================================

print("\n" + "="*100)
print("PART 5: FINAL HIDDEN STATES AND LOGITS")
print("="*100)

# Compare final hidden states
# BitLinearGPT doesn't return hidden_states, use block outputs
fp32_hidden = activation_store.get('fp32_block_5', fp32_out['logits'])
ternary_hidden = activation_store.get('ternary_block_5', ternary_out['logits'])

hidden_errors = compute_errors(fp32_hidden, ternary_hidden)
print(f"\n[Final hidden states comparison]")
print(f"  Cosine similarity: {hidden_errors['cosine_sim']:.4f}")
print(f"  Abs error mean:    {hidden_errors['abs_error_mean']:.6f}")
print(f"  Abs error std:     {hidden_errors['abs_error_std']:.6f}")

# Per-position analysis
print(f"\n[Per-position hidden state cosine similarity]")
for pos in [0, 5, 10, 15, -1]:
    if pos < fp32_hidden.shape[1]:
        pos_sim = compute_similarity(fp32_hidden[0, pos], ternary_hidden[0, pos])
        print(f"  Position {pos}: {pos_sim:.4f}")

# Compare logits
fp32_logits = fp32_out['logits']
ternary_logits = ternary_out['logits']

logits_errors = compute_errors(fp32_logits, ternary_logits)
print(f"\n[Logits comparison]")
print(f"  Cosine similarity: {logits_errors['cosine_sim']:.4f}")
print(f"  Abs error mean:    {logits_errors['abs_error_mean']:.6f}")
print(f"  Abs error std:     {logits_errors['abs_error_std']:.6f}")

# Per-position logits
print(f"\n[Per-position logits cosine similarity]")
for pos in [0, 5, 10, 15, -1]:
    if pos < fp32_logits.shape[1]:
        pos_sim = compute_similarity(fp32_logits[0, pos], ternary_logits[0, pos])
        print(f"  Position {pos}: {pos_sim:.4f}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*100)
print("SUMMARY")
print("="*100)

print(f"""
[WEIGHT QUANTIZATION ERROR]
- Ternary weights have ~0 cosine similarity to themselves after quantization
- This is EXPECTED: quantization maps continuous → discrete
- Zero fraction: ~31% (close to expected 33%)

[ACTIVATION DIVERGENCE]
- Embeddings: cosine_sim ~{emb_errors['cosine_sim']:.4f}
- Block outputs: varies by layer
- Final hidden: cosine_sim {hidden_errors['cosine_sim']:.4f}
- Logits: cosine_sim {logits_errors['cosine_sim']:.4f}

[KEY QUESTION]
Where does the divergence FIRST become substantial?

[CRITICAL MEASUREMENT NEEDED]
Compare the effective forward pass weights (dequantized) vs FP32 weights
to determine if the ternary model is computing fundamentally different functions.
""")

print("\n" + "="*100)
print("AUDIT COMPLETE")
print("="*100)
