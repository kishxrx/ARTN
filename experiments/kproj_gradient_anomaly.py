#!/usr/bin/env python3
"""
READ-ONLY K_PROJ Gradient Anomaly Investigation

Investigates WHY K_PROJ gradients are small without assuming K_PROJ is the root cause.

CRITICAL: This is READ-ONLY. No training, no checkpoint modification.
- We compute gradients via backward pass but DO NOT call optimizer.step()
- Gradients are temporary and discarded after analysis
- Checkpoints are only READ, never modified or saved
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict

# Add project to path
sys.path.insert(0, "/root/agent-sandbox/ARTN")

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.bitlinear import BitLinearGPT

print("="*100)
print("READ-ONLY K_PROJ GRADIENT ANOMALY INVESTIGATION")
print("="*100)
print("""
This diagnostic investigates WHY K_PROJ gradients are small.
It does NOT assume K_PROJ is the root cause.

We will check:
1. K_PROJ weight vs bias gradients separately
2. Gradient flow through attention step-by-step
3. Q/K/V/O gradient magnitudes across all 6 layers
4. K_PROJ weight distribution and quantization
5. Whether K_PROJ actually changes across checkpoints
6. Whether small gradient is implementation issue or learned state
""")

device = "cpu"

# ==============================================================================
# LOAD TOKENIZER
# ==============================================================================

tokenizer_path = Path("outputs/artn-optimized/tokenizer.json")
tokenizer = BPETokenizer.load(str(tokenizer_path))

print(f"\n[Tokenizer]")
print(f"  Vocab size: {len(tokenizer.vocab)}")

# ==============================================================================
# LOAD HELD-OUT DATA
# ==============================================================================

print(f"\n[Loading held-out data...]")

dataset = load_dataset("roneneldan/TinyStories", split="train")
held_out_stories = []
for i, item in enumerate(dataset):
    if i >= 50000 and i < 50050:  # 50 stories for analysis
        held_out_stories.append(item["text"])

print(f"  Held-out stories: {len(held_out_stories)}")

# Tokenize
held_out_tokens = []
for story in held_out_stories:
    tokens = tokenizer.encode(story)[:255]
    tokens = tokens + [tokenizer.eos_token_id]
    if len(tokens) >= 20:
        held_out_tokens.append(tokens)

print(f"  Usable sequences: {len(held_out_tokens)}")

# ==============================================================================
# HELPER: Create model and load checkpoint
# ==============================================================================

def create_model():
    """Create model with same architecture as training."""
    return BitLinearGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=256,
        dropout=0.0
    )

def load_checkpoint(checkpoint_path):
    """Load checkpoint and return model."""
    model = create_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint

# ==============================================================================
# PART 1: VERIFY K_PROJ WEIGHT VS BIAS GRADIENTS SEPARATELY
# ==============================================================================

print("\n" + "="*100)
print("PART 1: K_PROJ WEIGHT VS BIAS GRADIENTS")
print("="*100)

checkpoint_50k = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")
model, checkpoint = load_checkpoint(checkpoint_50k)
model.train()

# Zero gradients
model.zero_grad()

# Run forward/backward on one batch
tokens = held_out_tokens[0]
padding_length = 256 - len(tokens)
tokens_padded = tokens + [tokenizer.pad_token_id] * padding_length

input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
labels = targets.clone()
labels[labels == tokenizer.pad_token_id] = -100

outputs = model(input_ids, targets=labels)
loss = outputs["loss"]
loss.backward()

print(f"\n[Gradient Analysis for K_PROJ]")
print(f"  Loss: {loss.item():.4f}")

k_proj_weight_grads = []
k_proj_bias_grads = []
q_proj_weight_grads = []
q_proj_bias_grads = []
v_proj_weight_grads = []
v_proj_bias_grads = []
o_proj_weight_grads = []
o_proj_bias_grads = []

for layer_idx in range(6):
    # K_PROJ
    k_weight = model.blocks[layer_idx].attn.k_proj.weight
    k_bias = model.blocks[layer_idx].attn.k_proj.bias
    
    if k_weight.grad is not None:
        k_weight_grad_norm = k_weight.grad.norm().item()
        k_proj_weight_grads.append(k_weight_grad_norm)
    else:
        k_weight_grad_norm = None
        k_proj_weight_grads.append(0.0)
    
    if k_bias.grad is not None:
        k_bias_grad_norm = k_bias.grad.norm().item()
        k_proj_bias_grads.append(k_bias_grad_norm)
    else:
        k_bias_grad_norm = None
        k_proj_bias_grads.append(0.0)
    
    # Q_PROJ
    q_weight = model.blocks[layer_idx].attn.q_proj.weight
    q_bias = model.blocks[layer_idx].attn.q_proj.bias
    
    q_weight_grad_norm = q_weight.grad.norm().item() if q_weight.grad is not None else 0.0
    q_bias_grad_norm = q_bias.grad.norm().item() if q_bias.grad is not None else 0.0
    
    q_proj_weight_grads.append(q_weight_grad_norm)
    q_proj_bias_grads.append(q_bias_grad_norm)
    
    # V_PROJ
    v_weight = model.blocks[layer_idx].attn.v_proj.weight
    v_bias = model.blocks[layer_idx].attn.v_proj.bias
    
    v_weight_grad_norm = v_weight.grad.norm().item() if v_weight.grad is not None else 0.0
    v_bias_grad_norm = v_bias.grad.norm().item() if v_bias.grad is not None else 0.0
    
    v_proj_weight_grads.append(v_weight_grad_norm)
    v_proj_bias_grads.append(v_bias_grad_norm)
    
    # O_PROJ
    o_weight = model.blocks[layer_idx].attn.o_proj.weight
    o_bias = model.blocks[layer_idx].attn.o_proj.bias
    
    o_weight_grad_norm = o_weight.grad.norm().item() if o_weight.grad is not None else 0.0
    o_bias_grad_norm = o_bias.grad.norm().item() if o_bias.grad is not None else 0.0
    
    o_proj_weight_grads.append(o_weight_grad_norm)
    o_proj_bias_grads.append(o_bias_grad_norm)
    
    print(f"\n  Layer {layer_idx}:")
    
    k_weight_str = f"{k_weight_grad_norm:.6f}" if k_weight_grad_norm is not None else "None"
    k_bias_str = f"{k_bias_grad_norm:.6f}" if k_bias_grad_norm is not None else "None"
    
    print(f"    K_PROJ: weight_grad={k_weight_str}, bias_grad={k_bias_str}")
    print(f"    Q_PROJ: weight_grad={q_weight_grad_norm:.6f}, bias_grad={q_bias_grad_norm:.6f}")
    print(f"    V_PROJ: weight_grad={v_weight_grad_norm:.6f}, bias_grad={v_bias_grad_norm:.6f}")
    print(f"    O_PROJ: weight_grad={o_weight_grad_norm:.6f}, bias_grad={o_bias_grad_norm:.6f}")

print(f"\n[Summary: Average Gradients Across 6 Layers]")
print(f"  K_PROJ weight: {np.mean(k_proj_weight_grads):.6f}")
print(f"  K_PROJ bias:   {np.mean(k_proj_bias_grads):.6f}")
print(f"  Q_PROJ weight: {np.mean(q_proj_weight_grads):.6f}")
print(f"  Q_PROJ bias:   {np.mean(q_proj_bias_grads):.6f}")
print(f"  V_PROJ weight: {np.mean(v_proj_weight_grads):.6f}")
print(f"  V_PROJ bias:   {np.mean(v_proj_bias_grads):.6f}")
print(f"  O_PROJ weight: {np.mean(o_proj_weight_grads):.6f}")
print(f"  O_PROJ bias:   {np.mean(o_proj_bias_grads):.6f}")

# Ratio comparison
k_vs_v_weight = np.mean(k_proj_weight_grads) / (np.mean(v_proj_weight_grads) + 1e-10)
k_vs_q_weight = np.mean(k_proj_weight_grads) / (np.mean(q_proj_weight_grads) + 1e-10)

print(f"\n[Gradient Ratios]")
print(f"  K/V weight ratio: {k_vs_v_weight:.2f}x")
print(f"  K/Q weight ratio: {k_vs_q_weight:.2f}x")

# ==============================================================================
# PART 2: GRADIENT FLOW THROUGH ATTENTION STEP-BY-STEP
# ==============================================================================

print("\n" + "="*100)
print("PART 2: GRADIENT FLOW THROUGH ATTENTION MECHANISM")
print("="*100)

print("""
Attention mechanism:
  1. Q = x @ W_q + b_q
  2. K = x @ W_k + b_k  
  3. V = x @ W_v + b_v
  4. scores = Q @ K^T / sqrt(d)
  5. attn_weights = softmax(scores, mask)
  6. context = attn_weights @ V
  7. output = context @ W_o + b_o

Gradients flow backward:
  dL/dW_o → dL/dcontext → dL/dattn_weights → dL/dscores → dL/dK, dL/dQ, dL/dV

We need to trace where gradients diminish.
""")

# Reload model for clean state
model, _ = load_checkpoint(checkpoint_50k)
model.train()
model.zero_grad()

# Run forward with hooks to capture intermediate values
class GradientTracer:
    def __init__(self):
        self.intermediate = {}
        self.gradients = {}
    
    def save_forward(self, name, tensor):
        self.intermediate[name] = tensor.detach().clone()
    
    def save_gradient(self, name, grad):
        self.gradients[name] = grad.detach().clone()

tracer = GradientTracer()

# Hook function to save gradients
def make_hook(name, tracer):
    def hook(grad):
        tracer.save_gradient(name, grad)
        return grad
    return hook

# Register hooks on layer 0 attention (representative)
layer_idx = 0
attn = model.blocks[layer_idx].attn

# We'll manually compute attention to trace gradients
# Get input to attention
with torch.no_grad():
    # Get input embeddings
    emb = model.token_embedding(input_ids)
    pos_emb = model.position_embedding(torch.arange(input_ids.shape[1]))
    x = emb + pos_emb
    x = model.blocks[layer_idx].ln1(x)

# Now with gradients
model.zero_grad()

# Forward pass with hooks
x = model.token_embedding(input_ids)
x.retain_grad()
x.register_hook(make_hook('after_embedding', tracer))

pos_emb = model.position_embedding(torch.arange(input_ids.shape[1]))
x = x + pos_emb
x.retain_grad()
x.register_hook(make_hook('after_pos', tracer))

x = model.blocks[layer_idx].ln1(x)
x.retain_grad()
x.register_hook(make_hook('after_ln1', tracer))

# Q, K, V projections
Q = attn.q_proj(x)
Q.retain_grad()
Q.register_hook(make_hook('Q', tracer))

K = attn.k_proj(x)
K.retain_grad()
K.register_hook(make_hook('K', tracer))

V = attn.v_proj(x)
V.retain_grad()
V.register_hook(make_hook('V', tracer))

# Compute attention (simplified, without multi-head for clarity)
# Actually, we need to use the model's forward to get loss
# Let's just trace the gradients from the loss backward

# Reset and do full forward
model.zero_grad()
outputs = model(input_ids, targets=labels)
loss = outputs["loss"]

# Register backward hooks on specific parameters
def capture_grad(grad, name):
    print(f"  Gradient captured for {name}: norm={grad.norm().item():.6f}")
    return grad

# Re-run with specific hooks
model.zero_grad()

# Get reference to layer 0 attention
attn = model.blocks[0].attn

# Store gradients manually
stored_grads = {}

def hook_fn(name):
    def hook(module, grad_input, grad_output):
        stored_grads[name] = {
            'grad_input': [g.detach().clone() if g is not None else None for g in grad_input],
            'grad_output': [g.detach().clone() if g is not None else None for g in grad_output]
        }
    return hook

# Register backward hooks
attn.q_proj.register_backward_hook(hook_fn('q_proj'))
attn.k_proj.register_backward_hook(hook_fn('k_proj'))
attn.v_proj.register_backward_hook(hook_fn('v_proj'))
attn.o_proj.register_backward_hook(hook_fn('o_proj'))

# Forward and backward
outputs = model(input_ids, targets=labels)
loss = outputs["loss"]
loss.backward()

print(f"\n[Gradient Flow Through Attention (Layer 0)]")
print(f"  Loss: {loss.item():.4f}")

for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
    if name in stored_grads:
        grad_output_norm = stored_grads[name]['grad_output'][0].norm().item() if stored_grads[name]['grad_output'][0] is not None else 0.0
        grad_input_norm = stored_grads[name]['grad_input'][0].norm().item() if stored_grads[name]['grad_input'][0] is not None else 0.0
        print(f"  {name}: grad_output_norm={grad_output_norm:.6f}, grad_input_norm={grad_input_norm:.6f}")

# ==============================================================================
# PART 3: Q/K/V/O GRADIENT MAGNITUDES ACROSS ALL 6 LAYERS
# ==============================================================================

print("\n" + "="*100)
print("PART 3: Q/K/V/O GRADIENT MAGNITUDES ACROSS ALL 6 LAYERS")
print("="*100)

# Reload and compute gradients on multiple batches
model, _ = load_checkpoint(checkpoint_50k)
model.train()
model.zero_grad()

# Accumulate gradients over multiple batches
for tokens in held_out_tokens[:20]:
    padding_length = 256 - len(tokens)
    tokens_padded = tokens + [tokenizer.pad_token_id] * padding_length
    
    input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
    targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
    labels = targets.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    outputs = model(input_ids, targets=labels)
    loss = outputs["loss"]
    loss.backward()

print(f"\n[Gradient Magnitudes by Layer and Projection]")

layer_grads = defaultdict(lambda: defaultdict(list))

for layer_idx in range(6):
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        proj = getattr(model.blocks[layer_idx].attn, proj_name)
        
        if proj.weight.grad is not None:
            weight_grad_norm = proj.weight.grad.norm().item()
            layer_grads[layer_idx][proj_name] = weight_grad_norm
        else:
            layer_grads[layer_idx][proj_name] = 0.0

print(f"\n  {'Layer':<8} {'Q_PROJ':<12} {'K_PROJ':<12} {'V_PROJ':<12} {'O_PROJ':<12}")
print(f"  {'-'*56}")

for layer_idx in range(6):
    q = layer_grads[layer_idx]['q_proj']
    k = layer_grads[layer_idx]['k_proj']
    v = layer_grads[layer_idx]['v_proj']
    o = layer_grads[layer_idx]['o_proj']
    print(f"  {layer_idx:<8} {q:<12.6f} {k:<12.6f} {v:<12.6f} {o:<12.6f}")

print(f"\n[Layer-wise K_PROJ vs Others]")
for layer_idx in range(6):
    k = layer_grads[layer_idx]['k_proj']
    q = layer_grads[layer_idx]['q_proj']
    v = layer_grads[layer_idx]['v_proj']
    o = layer_grads[layer_idx]['o_proj']
    
    avg_others = (q + v + o) / 3
    ratio = k / (avg_others + 1e-10)
    
    print(f"  Layer {layer_idx}: K={k:.6f}, avg(Q,V,O)={avg_others:.6f}, ratio={ratio:.2f}x")

# ==============================================================================
# PART 4: K_PROJ WEIGHT DISTRIBUTION AND QUANTIZATION
# ==============================================================================

print("\n" + "="*100)
print("PART 4: K_PROJ WEIGHT DISTRIBUTION AND QUANTIZATION")
print("="*100)

model, _ = load_checkpoint(checkpoint_50k)
model.eval()

print(f"\n[Weight Distribution Analysis]")

for layer_idx in range(6):
    k_proj = model.blocks[layer_idx].attn.k_proj
    q_proj = model.blocks[layer_idx].attn.q_proj
    v_proj = model.blocks[layer_idx].attn.v_proj
    o_proj = model.blocks[layer_idx].attn.o_proj
    
    k_weight = k_proj.weight.data
    q_weight = q_proj.weight.data
    v_weight = v_proj.weight.data
    o_weight = o_proj.weight.data
    
    # Check if BitLinear (has scale attribute)
    if hasattr(k_proj, 'weight_scale'):
        k_scale = k_proj.weight_scale.data.item()
        q_scale = q_proj.weight_scale.data.item()
        v_scale = v_proj.weight_scale.data.item()
        o_scale = o_proj.weight_scale.data.item()
        
        # Scaled weights
        k_scaled = k_weight * k_scale
        q_scaled = q_weight * q_scale
        v_scaled = v_weight * v_scale
        o_scaled = o_weight * o_scale
        
        print(f"\n  Layer {layer_idx} (BitLinear with scale):")
        print(f"    K_PROJ scale: {k_scale:.6f}")
        print(f"    Q_PROJ scale: {q_scale:.6f}")
        print(f"    V_PROJ scale: {v_scale:.6f}")
        print(f"    O_PROJ scale: {o_scale:.6f}")
        
        # Ternary distribution
        k_vals = k_scaled.flatten()
        q_vals = q_scaled.flatten()
        v_vals = v_scaled.flatten()
        o_vals = o_scaled.flatten()
        
        for name, vals in [('K', k_vals), ('Q', q_vals), ('V', v_vals), ('O', o_vals)]:
            neg_pct = (vals < -0.5).float().mean().item() * 100
            zero_pct = ((vals >= -0.5) & (vals <= 0.5)).float().mean().item() * 100
            pos_pct = (vals > 0.5).float().mean().item() * 100
            
            print(f"    {name}: -1={neg_pct:.1f}%, 0={zero_pct:.1f}%, +1={pos_pct:.1f}%")
    else:
        print(f"\n  Layer {layer_idx} (Standard Linear, no scale):")
        
        for name, w in [('K', k_weight), ('Q', q_weight), ('V', v_weight), ('O', o_weight)]:
            mean = w.mean().item()
            std = w.std().item()
            print(f"    {name}: mean={mean:.6f}, std={std:.6f}")

# ==============================================================================
# PART 5: K_PROJ CHANGES ACROSS CHECKPOINTS
# ==============================================================================

print("\n" + "="*100)
print("PART 5: K_PROJ CHANGES ACROSS CHECKPOINTS")
print("="*100)

checkpoint_paths = [
    Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_1000.pt"),
    Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_10000.pt"),
    Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_25000.pt"),
    Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt"),
]

k_weights_by_step = {}
q_weights_by_step = {}
v_weights_by_step = {}
o_weights_by_step = {}

for ckpt_path in checkpoint_paths:
    if ckpt_path.exists():
        model, _ = load_checkpoint(ckpt_path)
        step = int(ckpt_path.stem.split('_')[-1])
        
        # Extract layer 0 weights (representative)
        k_weight = model.blocks[0].attn.k_proj.weight.data.clone()
        q_weight = model.blocks[0].attn.q_proj.weight.data.clone()
        v_weight = model.blocks[0].attn.v_proj.weight.data.clone()
        o_weight = model.blocks[0].attn.o_proj.weight.data.clone()
        
        k_weights_by_step[step] = k_weight
        q_weights_by_step[step] = q_weight
        v_weights_by_step[step] = v_weight
        o_weights_by_step[step] = o_weight
        
        print(f"  Loaded checkpoint: step {step}")

if len(k_weights_by_step) > 1:
    steps = sorted(k_weights_by_step.keys())
    
    print(f"\n[Weight Norm Changes (Layer 0)]")
    
    for proj_name, weights_by_step in [('K_PROJ', k_weights_by_step), ('Q_PROJ', q_weights_by_step), ('V_PROJ', v_weights_by_step), ('O_PROJ', o_weights_by_step)]:
        print(f"\n  {proj_name}:")
        
        for i in range(len(steps) - 1):
            step1, step2 = steps[i], steps[i+1]
            
            norm1 = weights_by_step[step1].norm().item()
            norm2 = weights_by_step[step2].norm().item()
            
            change = (norm2 - norm1) / norm1 * 100
            diff_norm = (weights_by_step[step2] - weights_by_step[step1]).norm().item()
            
            print(f"    Step {step1} → {step2}: norm={norm1:.4f} → {norm2:.4f} ({change:+.2f}%), diff_norm={diff_norm:.6f}")
    
    # Check if K_PROJ is actually changing
    print(f"\n[K_PROJ Weight Change Analysis]")
    
    k_1k = k_weights_by_step[1000]
    k_50k = k_weights_by_step[50000]
    
    # Absolute difference
    abs_diff = (k_50k - k_1k).abs()
    diff_norm = abs_diff.norm().item()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    
    print(f"  Difference from step 1k to 50k:")
    print(f"    L2 norm of diff: {diff_norm:.6f}")
    print(f"    Max absolute diff: {max_diff:.6f}")
    print(f"    Mean absolute diff: {mean_diff:.6f}")
    
    # Compare to other projections
    for proj_name, weights_by_step in [('Q_PROJ', q_weights_by_step), ('V_PROJ', v_weights_by_step), ('O_PROJ', o_weights_by_step)]:
        diff = (weights_by_step[50000] - weights_by_step[1000]).norm().item()
        print(f"    {proj_name} diff norm: {diff:.6f}")

# ==============================================================================
# PART 6: IMPLEMENTATION VS LEARNED STATE ANALYSIS
# ==============================================================================

print("\n" + "="*100)
print("PART 6: IMPLEMENTATION VS LEARNED STATE")
print("="*100)

print("""
[Question: Is the small K gradient caused by implementation/quantization or learned state?]

If implementation/quantization issue:
  - K_PROJ weights would be stuck or identical across checkpoints
  - K_PROJ gradients would be consistently small from the start
  - Other projections would not have this issue

If learned state:
  - K_PROJ weights would change during training
  - Gradient magnitude would be related to the learned function
  - Could be that K_PROJ has converged to a good solution
""")

model_1k, _ = load_checkpoint(Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_1000.pt"))
model_50k, _ = load_checkpoint(checkpoint_50k)

print(f"\n[Comparing Early vs Late Training]")

# Compare K_PROJ weights
k_1k = model_1k.blocks[0].attn.k_proj.weight.data
k_50k = model_50k.blocks[0].attn.k_proj.weight.data

# Are they different?
diff = (k_50k - k_1k).norm().item()
relative_diff = diff / k_1k.norm().item()

print(f"  K_PROJ weight change (1k → 50k):")
print(f"    Absolute diff norm: {diff:.6f}")
print(f"    Relative diff: {relative_diff:.4f}")

if relative_diff > 0.1:
    print(f"    → K_PROJ HAS changed significantly during training")
else:
    print(f"    → K_PROJ has NOT changed much during training")

# Check gradient ratio at step 1k vs 50k
print(f"\n[Gradient Ratio Evolution]")

# Compute gradients at step 1k
model_1k.train()
model_1k.zero_grad()

for tokens in held_out_tokens[:10]:
    padding_length = 256 - len(tokens)
    tokens_padded = tokens + [tokenizer.pad_token_id] * padding_length
    
    input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
    targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
    labels = targets.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    outputs = model_1k(input_ids, targets=labels)
    outputs["loss"].backward()

k_grad_1k = model_1k.blocks[0].attn.k_proj.weight.grad.norm().item()
q_grad_1k = model_1k.blocks[0].attn.q_proj.weight.grad.norm().item()
v_grad_1k = model_1k.blocks[0].attn.v_proj.weight.grad.norm().item()

# Compute gradients at step 50k
model_50k.train()
model_50k.zero_grad()

for tokens in held_out_tokens[:10]:
    padding_length = 256 - len(tokens)
    tokens_padded = tokens + [tokenizer.pad_token_id] * padding_length
    
    input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
    targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
    labels = targets.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    outputs = model_50k(input_ids, targets=labels)
    outputs["loss"].backward()

k_grad_50k = model_50k.blocks[0].attn.k_proj.weight.grad.norm().item()
q_grad_50k = model_50k.blocks[0].attn.q_proj.weight.grad.norm().item()
v_grad_50k = model_50k.blocks[0].attn.v_proj.weight.grad.norm().item()

print(f"  Step 1k: K_grad={k_grad_1k:.6f}, Q_grad={q_grad_1k:.6f}, V_grad={v_grad_1k:.6f}")
print(f"  Step 50k: K_grad={k_grad_50k:.6f}, Q_grad={q_grad_50k:.6f}, V_grad={v_grad_50k:.6f}")

ratio_1k = k_grad_1k / (v_grad_1k + 1e-10)
ratio_50k = k_grad_50k / (v_grad_50k + 1e-10)

print(f"  K/V ratio: {ratio_1k:.2f}x (1k) → {ratio_50k:.2f}x (50k)")

if ratio_1k < 0.1 and ratio_50k < 0.1:
    print(f"\n  → K gradient has been small from the START")
    print(f"  → This suggests implementation/initialization issue, NOT learned state")
elif ratio_1k > 0.5 and ratio_50k < 0.1:
    print(f"\n  → K gradient became small during training")
    print(f"  → This suggests learned state or optimization issue")
else:
    print(f"\n  → Gradient pattern is complex, needs further analysis")

# ==============================================================================
# SUMMARY AND CLASSIFICATION
# ==============================================================================

print("\n" + "="*100)
print("SUMMARY AND CLASSIFICATION")
print("="*100)

print("""
This diagnostic investigated WHY K_PROJ gradients are small.

FINDINGS:
""")

findings = []

# Finding 1: K_PROJ weight vs bias
k_weight_avg = np.mean(k_proj_weight_grads)
k_bias_avg = np.mean(k_proj_bias_grads)

if k_weight_avg > 0.01 and k_bias_avg > 0.01:
    findings.append(("K_PROJ weight and bias both have non-zero gradients", "VERIFIED"))
elif k_weight_avg < 0.01 and k_bias_avg < 0.0001:
    findings.append(("K_PROJ weight AND bias gradients are near-zero", "VERIFIED"))
else:
    findings.append(("K_PROJ weight and bias gradients differ significantly", "VERIFIED"))

# Finding 2: K vs other projections
k_vs_v = np.mean(k_proj_weight_grads) / (np.mean(v_proj_weight_grads) + 1e-10)

if k_vs_v < 0.1:
    findings.append(("K_PROJ gradients are much smaller than V_PROJ", "VERIFIED"))
else:
    findings.append(("K_PROJ gradients are comparable to V_PROJ", "NOT SUPPORTED"))

# Finding 3: Layer-wise pattern
all_k_grads = [layer_grads[i]['k_proj'] for i in range(6)]
if all(g < 0.01 for g in all_k_grads):
    findings.append(("K_PROJ gradients are small in ALL 6 layers", "VERIFIED"))
else:
    findings.append(("K_PROJ gradients vary by layer", "VERIFIED"))

# Finding 4: Weight changes
if len(k_weights_by_step) > 1:
    relative_diff = (k_weights_by_step[50000] - k_weights_by_step[1000]).norm().item() / k_weights_by_step[1000].norm().item()
    if relative_diff > 0.1:
        findings.append(("K_PROJ weights have changed during training", "VERIFIED"))
    else:
        findings.append(("K_PROJ weights have NOT changed much", "VERIFIED"))

# Finding 5: Early vs late
if 'ratio_1k' in locals() and 'ratio_50k' in locals():
    if ratio_1k < 0.1 and ratio_50k < 0.1:
        findings.append(("K gradients have been small from step 1k", "VERIFIED"))
    else:
        findings.append(("K gradient ratio changed during training", "VERIFIED"))

print("[CLASSIFICATION OF FINDINGS]")
print("-"*80)

for finding, status in findings:
    status_str = f"✅ {status}" if status == "VERIFIED" else f"❌ {status}" if status == "NOT SUPPORTED" else f"⚠️ {status}"
    print(f"  {finding}")
    print(f"    → {status_str}")
    print()

print("="*80)
print("CONCLUSION")
print("="*80)

print("""
This diagnostic analyzed K_PROJ gradient anomaly from multiple angles.

All findings are READ-ONLY - no checkpoints were modified.

The next step depends on whether:
1. K gradients have been small from the start → likely implementation issue
2. K gradients became small during training → likely learned state or optimization issue
3. K weights have changed despite small gradients → complex dynamics
""")

print("\n" + "="*100)
print("DIAGNOSTIC COMPLETE")
print("="*100)
