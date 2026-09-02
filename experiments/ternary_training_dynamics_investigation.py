#!/usr/bin/env python3
"""
READ-ONLY: Ternary Training Dynamics Investigation

This diagnostic investigates WHY ternary training suppresses contextual learning.

POSSIBLE MECHANISMS:
1. Quantization/STE gradient limitation
2. Optimizer interaction with ternary weights  
3. Quantization scale/weight-resolution issue
4. Training-dynamics issue

CRITICAL: This is READ-ONLY. No model training or modification.

This diagnostic will:
1. Analyze gradient flow through STE
2. Compare weight distributions (FP32 vs Ternary)
3. Analyze scale evolution during training
4. Inspect optimizer states
5. Compare training dynamics metrics

GOAL: Identify the most likely mechanism causing ternary to learn less context.
"""

import torch
import torch.nn.functional as F
import sys
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.model.bitlinear import BitLinearGPT, BitLinear, TernarySTE
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*100)
print("READ-ONLY: TERNARY TRAINING DYNAMICS INVESTIGATION")
print("="*100)
print("""
This diagnostic investigates WHY ternary training suppresses contextual learning.

MECHANISMS TO TEST:
1. STE Gradient Limitation: Does gradient flow through STE correctly?
2. Weight Distribution: How do ternary weights differ from FP32?
3. Scale Evolution: How does quantization scale change during training?
4. Optimizer State: Does AdaFactor work correctly with ternary weights?
5. Gradient Magnitude: Are gradients smaller in ternary model?

READ-ONLY: Using existing checkpoints at steps 1000, 5000, 50000.
""")

# Load tokenizer
tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')

# Checkpoints to analyze
checkpoints = {
    "Ternary-1k": "outputs/artn-50k-best-config/checkpoints/checkpoint_step_1000.pt",
    "Ternary-5k": "outputs/artn-50k-best-config/checkpoints/checkpoint_step_5000.pt",
    "Ternary-50k": "outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt",
    "FP32-5k": "outputs/fp32-5k-comparison/checkpoints/checkpoint_step_5000.pt",
}

# ============================================================
# PART 1: WEIGHT DISTRIBUTION ANALYSIS
# ============================================================

print("\n" + "="*100)
print("PART 1: WEIGHT DISTRIBUTION ANALYSIS")
print("="*100)

def analyze_weight_distribution(ckpt_path, model_name="Model"):
    """Analyze weight distribution in a model"""
    print(f"\n[Analyzing {model_name}]")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    
    # Separate by layer type
    attention_weights = []
    ff_weights = []
    embedding_weights = []
    
    for name, param in state_dict.items():
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name:
            attention_weights.append(param)
        elif 'w1' in name or 'w2' in name:
            ff_weights.append(param)
        elif 'embedding' in name:
            embedding_weights.append(param)
    
    results = {}
    
    # Analyze attention weights
    if attention_weights:
        all_attn = torch.cat([w.flatten() for w in attention_weights])
        results['attention'] = {
            'mean': all_attn.mean().item(),
            'std': all_attn.std().item(),
            'min': all_attn.min().item(),
            'max': all_attn.max().item(),
            'sparsity': (all_attn.abs() < 0.01).float().mean().item(),
            'ternary_ratio': ((all_attn.abs() > 0.9) & (all_attn.abs() < 1.1)).float().mean().item(),
        }
        print(f"  Attention weights:")
        print(f"    Mean: {results['attention']['mean']:.6f}")
        print(f"    Std: {results['attention']['std']:.6f}")
        print(f"    Ternary ratio: {results['attention']['ternary_ratio']:.2%}")
    
    # Analyze FF weights
    if ff_weights:
        all_ff = torch.cat([w.flatten() for w in ff_weights])
        results['ff'] = {
            'mean': all_ff.mean().item(),
            'std': all_ff.std().item(),
            'min': all_ff.min().item(),
            'max': all_ff.max().item(),
            'sparsity': (all_ff.abs() < 0.01).float().mean().item(),
            'ternary_ratio': ((all_ff.abs() > 0.9) & (all_ff.abs() < 1.1)).float().mean().item(),
        }
        print(f"  Feed-forward weights:")
        print(f"    Mean: {results['ff']['mean']:.6f}")
        print(f"    Std: {results['ff']['std']:.6f}")
        print(f"    Ternary ratio: {results['ff']['ternary_ratio']:.2%}")
    
    return results

weight_results = {}
for name, path in checkpoints.items():
    if Path(path).exists():
        weight_results[name] = analyze_weight_distribution(path, name)

# ============================================================
# PART 2: QUANTIZATION SCALE ANALYSIS
# ============================================================

print("\n" + "="*100)
print("PART 2: QUANTIZATION SCALE ANALYSIS")
print("="*100)

def analyze_quantization_scale(ckpt_path, model_name="Model"):
    """Analyze quantization scale values in ternary model"""
    print(f"\n[Analyzing {model_name}]")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    
    # Find scale-related statistics
    scale_stats = {}
    
    for name, param in state_dict.items():
        if 'running_sparsity' in name:
            scale_stats['sparsity'] = param.item()
            print(f"  Running sparsity: {param.item():.4f}")
        if 'num_quantizations' in name:
            scale_stats['num_quantizations'] = param.item()
            print(f"  Number of quantizations: {param.item():.0f}")
    
    return scale_stats

for name, path in checkpoints.items():
    if 'Ternary' in name and Path(path).exists():
        analyze_quantization_scale(path, name)

# ============================================================
# PART 3: GRADIENT FLOW SIMULATION (FORWARD + BACKWARD)
# ============================================================

print("\n" + "="*100)
print("PART 3: GRADIENT FLOW SIMULATION")
print("="*100)

print("""
[Hypothesis: STE Gradient Limitation]

The Straight-Through Estimator passes gradients through unchanged.
This could cause issues if:
1. Gradient magnitude doesn't match weight magnitude
2. Gradient direction is not informative for weight updates
3. Discrete quantization creates optimization barriers

[Test: Compare gradient statistics in FP32 vs Ternary during a forward-backward pass]
""")

# Create test input
test_tokens = torch.randint(0, 4000, (1, 20))

# Load models for gradient analysis
def analyze_gradient_flow(ckpt_path, model_class, model_name="Model", is_ternary=False):
    """Analyze gradient flow through the model"""
    print(f"\n[Analyzing {model_name}]")
    
    model = model_class(vocab_size=4000, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_seq_len=256)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.train()
    
    # Forward pass
    input_ids = test_tokens
    targets = torch.roll(test_tokens, -1, dims=1)
    targets[:, -1] = -100
    
    outputs = model(input_ids, targets)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[1]
    
    # Backward pass
    loss.backward()
    
    # Collect gradient statistics
    grad_stats = {}
    
    # Attention gradients
    attn_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name):
            attn_grads.append(param.grad.abs())
    
    if attn_grads:
        all_attn_grads = torch.cat([g.flatten() for g in attn_grads])
        grad_stats['attention_grad'] = {
            'mean': all_attn_grads.mean().item(),
            'std': all_attn_grads.std().item(),
            'max': all_attn_grads.max().item(),
        }
        print(f"  Attention gradients:")
        print(f"    Mean: {grad_stats['attention_grad']['mean']:.6f}")
        print(f"    Std: {grad_stats['attention_grad']['std']:.6f}")
        print(f"    Max: {grad_stats['attention_grad']['max']:.6f}")
    
    # FF gradients
    ff_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None and ('w1' in name or 'w2' in name):
            ff_grads.append(param.grad.abs())
    
    if ff_grads:
        all_ff_grads = torch.cat([g.flatten() for g in ff_grads])
        grad_stats['ff_grad'] = {
            'mean': all_ff_grads.mean().item(),
            'std': all_ff_grads.std().item(),
            'max': all_ff_grads.max().item(),
        }
        print(f"  Feed-forward gradients:")
        print(f"    Mean: {grad_stats['ff_grad']['mean']:.6f}")
        print(f"    Std: {grad_stats['ff_grad']['std']:.6f}")
        print(f"    Max: {grad_stats['ff_grad']['max']:.6f}")
    
    # For ternary: Check actual quantization during forward
    if is_ternary:
        # Re-run forward to capture quantization statistics
        model.eval()
        with torch.no_grad():
            # Check sparsity in BitLinear layers
            for name, module in model.named_modules():
                if isinstance(module, BitLinear):
                    quant_w, scale = module.quantize_weights()
                    sparsity = (quant_w == 0).float().mean().item()
                    non_zero_mean = quant_w[quant_w != 0].abs().mean().item() if (quant_w != 0).any() else 0
                    print(f"  {name}: sparsity={sparsity:.2%}, non-zero mean={non_zero_mean:.4f}, scale={scale.item():.4f}")
    
    return grad_stats

# Analyze FP32
if Path(checkpoints['FP32-5k']).exists():
    # Import FP32 model definition
    import torch.nn as nn
    
    class StandardLinear(nn.Linear):
        pass
    
    class StandardMultiHeadAttention(nn.Module):
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
            import math
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
    
    class StandardFeedForward(nn.Module):
        def __init__(self, d_model, d_ff):
            super().__init__()
            self.w1 = StandardLinear(d_model, d_ff)
            self.w2 = StandardLinear(d_ff, d_model)
            self.activation = nn.GELU()
        
        def forward(self, x):
            return self.w2(self.activation(self.w1(x)))
    
    class StandardTransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = StandardMultiHeadAttention(d_model, n_heads)
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = StandardFeedForward(d_model, d_ff)
        
        def forward(self, x, mask=None):
            x = x + self.attn(self.ln1(x), mask)
            x = x + self.ff(self.ln2(x))
            return x
    
    class StandardGPT(nn.Module):
        def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
            super().__init__()
            self.vocab_size = vocab_size
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            self.blocks = nn.ModuleList([StandardTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
            self.ln_final = nn.LayerNorm(d_model)
            self.lm_head = StandardLinear(d_model, vocab_size)
        
        def forward(self, input_ids, targets=None):
            import math
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
                loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), ignore_index=-100)
            return logits, loss
    
    analyze_gradient_flow(checkpoints['FP32-5k'], StandardGPT, "FP32-5k", is_ternary=False)

# Analyze Ternary
if Path(checkpoints['Ternary-5k']).exists():
    analyze_gradient_flow(checkpoints['Ternary-5k'], BitLinearGPT, "Ternary-5k", is_ternary=True)

# ============================================================
# PART 4: OPTIMIZER STATE INSPECTION
# ============================================================

print("\n" + "="*100)
print("PART 4: OPTIMIZER STATE INSPECTION")
print("="*100)

def inspect_optimizer_state(ckpt_path, model_name="Model"):
    """Inspect optimizer state for anomalies"""
    print(f"\n[Inspecting {model_name}]")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    if 'optimizer_state_dict' not in ckpt:
        print("  No optimizer state found in checkpoint")
        return
    
    opt_state = ckpt['optimizer_state_dict']
    
    # Check state structure
    print(f"  Optimizer state keys: {list(opt_state.keys())}")
    
    if 'state' in opt_state:
        num_params = len(opt_state['state'])
        print(f"  Number of parameter states: {num_params}")
        
        # Sample a few parameter states
        for i, (param_id, state) in enumerate(opt_state['state'].items()):
            if i >= 3:  # Only show first 3
                break
            
            print(f"\n  Parameter {param_id}:")
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: shape={value.shape}, mean={value.mean().item():.6f}, std={value.std().item():.6f}")
                else:
                    print(f"    {key}: {value}")

for name, path in checkpoints.items():
    if Path(path).exists():
        inspect_optimizer_state(path, name)

# ============================================================
# PART 5: TRAINING CURVE COMPARISON
# ============================================================

print("\n" + "="*100)
print("PART 5: TRAINING CURVE COMPARISON")
print("="*100)

def analyze_training_curves(ckpt_path, model_name="Model"):
    """Analyze training curve metrics"""
    print(f"\n[Analyzing {model_name}]")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print(f"  Step: {ckpt['step']}")
    
    if 'train_loss' in ckpt:
        print(f"  Train loss: {ckpt['train_loss']:.4f}")
    
    if 'val_loss' in ckpt:
        print(f"  Val loss: {ckpt['val_loss']:.4f}")
    
    if 'learning_rate' in ckpt:
        print(f"  Learning rate: {ckpt['learning_rate']:.6f}")

for name, path in checkpoints.items():
    if Path(path).exists():
        analyze_training_curves(path, name)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*100)
print("INVESTIGATION SUMMARY")
print("="*100)

print("""
[Findings]

This diagnostic analyzed:
1. Weight distributions (FP32 vs Ternary)
2. Quantization scale evolution
3. Gradient flow through STE
4. Optimizer states
5. Training curves

[Key Questions to Answer]

Based on this analysis, we can determine:

1. STE GRADIENT LIMITATION:
   - Are gradients smaller in ternary model?
   - Do gradients flow through STE correctly?
   - Is gradient-to-weight ratio different?

2. WEIGHT DISTRIBUTION:
   - Are ternary weights properly quantized?
   - What is the sparsity ratio?
   - Are scale values reasonable?

3. OPTIMIZER INTERACTION:
   - Does AdaFactor work correctly with ternary weights?
   - Are optimizer states well-formed?
   - Is momentum accumulating correctly?

[NEXT STEP]

Based on these findings, design the SMALLEST experiment to test the most likely mechanism.
""")

print("\n" + "="*100)
print("DIAGNOSTIC COMPLETE")
print("="*100)
