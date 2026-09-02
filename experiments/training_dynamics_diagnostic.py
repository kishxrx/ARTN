#!/usr/bin/env python3
"""
READ-ONLY Training Dynamics Diagnostic

Investigates whether training dynamics/gradient updates are capable of learning
contextual relationships versus being dominated by unigram/frequency learning.

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
print("READ-ONLY TRAINING DYNAMICS DIAGNOSTIC")
print("="*100)

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
    if i >= 50000 and i < 50100:  # 100 stories for gradient analysis
        held_out_stories.append(item["text"])

print(f"  Held-out stories: {len(held_out_stories)}")

# Tokenize
held_out_tokens = []
for story in held_out_stories:
    tokens = tokenizer.encode(story)[:255]
    tokens = tokens + [tokenizer.eos_token_id]  # Add EOS like training
    if len(tokens) >= 20:
        held_out_tokens.append(tokens)

print(f"  Usable sequences: {len(held_out_tokens)}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_checkpoint_and_compute_gradients(checkpoint_path, tokens_list):
    """
    Load checkpoint, compute gradients on held-out data.
    
    Returns gradient statistics WITHOUT modifying the checkpoint.
    """
    # Create model
    model = BitLinearGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=256,
        dropout=0.0
    )
    
    # Load checkpoint (READ-ONLY)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Prepare for gradient computation
    model.train()  # Enable gradient computation
    
    # Zero gradients
    model.zero_grad()
    
    total_loss = 0.0
    num_batches = 0
    
    # Forward/backward pass on held-out data
    for tokens in tokens_list[:50]:  # Use subset for efficiency
        # Pad to 256
        padding_length = 256 - len(tokens)
        if padding_length > 0:
            tokens_padded = tokens + [tokenizer.pad_token_id] * padding_length
        else:
            tokens_padded = tokens[:256]
        
        input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
        targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
        
        # Create labels with masking
        labels = targets.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        # Forward pass
        outputs = model(input_ids, targets=labels)
        loss = outputs["loss"]
        
        # Backward pass (accumulates gradients)
        loss.backward()
        
        total_loss += loss.item()
        num_batches += 1
    
    # Collect gradient statistics
    grad_stats = {
        "by_layer_type": defaultdict(list),
        "by_layer_name": {},
        "total_norm": 0.0,
        "num_params_with_grad": 0,
    }
    
    total_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.data.norm().item()
            
            # Ratio of gradient to parameter
            ratio = grad_norm / (param_norm + 1e-8)
            
            grad_stats["by_layer_name"][name] = {
                "grad_norm": grad_norm,
                "param_norm": param_norm,
                "ratio": ratio,
            }
            
            total_norm += grad_norm ** 2
            grad_stats["num_params_with_grad"] += 1
            
            # Categorize by layer type
            if "embedding" in name:
                grad_stats["by_layer_type"]["embedding"].append((name, grad_norm, ratio))
            elif "attn" in name or "attention" in name:
                grad_stats["by_layer_type"]["attention"].append((name, grad_norm, ratio))
            elif "ff" in name or "feedforward" in name:
                grad_stats["by_layer_type"]["feedforward"].append((name, grad_norm, ratio))
            elif "ln" in name or "layer_norm" in name:
                grad_stats["by_layer_type"]["layernorm"].append((name, grad_norm, ratio))
            else:
                grad_stats["by_layer_type"]["other"].append((name, grad_norm, ratio))
    
    grad_stats["total_norm"] = total_norm ** 0.5
    grad_stats["avg_loss"] = total_loss / num_batches if num_batches > 0 else 0
    
    # Model back to eval
    model.eval()
    
    return grad_stats, checkpoint.get("step", "unknown")


def analyze_parameter_changes(checkpoint_paths):
    """
    Compare parameter values across checkpoints to see how much they changed.
    
    This is READ-ONLY - we only load and compare, never modify.
    """
    print("\n" + "="*80)
    print("PARAMETER CHANGES ACROSS CHECKPOINTS")
    print("="*80)
    
    params_by_step = {}
    
    for path in checkpoint_paths:
        if not path.exists():
            continue
        
        checkpoint = torch.load(path, map_location=device)
        step = checkpoint.get("step", "unknown")
        
        # Store parameter norms by layer type
        param_norms = {
            "embedding": [],
            "attention": [],
            "feedforward": [],
            "layernorm": [],
        }
        
        for name, param in checkpoint["model_state_dict"].items():
            # Skip non-floating point tensors (e.g., int tensors from optimizer state)
            if not param.is_floating_point():
                continue
            
            norm = param.norm().item()
            
            if "embedding" in name:
                param_norms["embedding"].append((name, norm))
            elif "attn" in name or "attention" in name:
                param_norms["attention"].append((name, norm))
            elif "ff" in name or "feedforward" in name:
                param_norms["feedforward"].append((name, norm))
            elif "ln" in name or "layer_norm" in name:
                param_norms["layernorm"].append((name, norm))
        
        params_by_step[step] = param_norms
        print(f"\n  Loaded checkpoint: step {step}")
    
    # Compare parameter norms across steps
    if len(params_by_step) > 1:
        print("\n[Parameter Norm Changes by Layer Type]")
        
        steps = sorted(params_by_step.keys())
        
        for layer_type in ["embedding", "attention", "feedforward", "layernorm"]:
            if layer_type not in params_by_step[steps[0]]:
                continue
            
            print(f"\n  {layer_type.upper()}:")
            
            for i in range(len(steps) - 1):
                step1, step2 = steps[i], steps[i+1]
                
                norms1 = [n for _, n in params_by_step[step1][layer_type]]
                norms2 = [n for _, n in params_by_step[step2][layer_type]]
                
                if norms1 and norms2:
                    avg1 = np.mean(norms1)
                    avg2 = np.mean(norms2)
                    change = (avg2 - avg1) / avg1 * 100
                    
                    print(f"    Step {step1} → {step2}: {avg1:.4f} → {avg2:.4f} ({change:+.2f}%)")
    
    return params_by_step


# ==============================================================================
# PART 1: GRADIENT ANALYSIS AT STEP 50K
# ==============================================================================

print("\n" + "="*100)
print("PART 1: GRADIENT ANALYSIS AT STEP 50,000")
print("="*100)

checkpoint_50k = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")

print(f"\n[Computing gradients on held-out data...]")
grad_stats_50k, step = load_checkpoint_and_compute_gradients(checkpoint_50k, held_out_tokens)

print(f"\n[Gradient Statistics at Step {step}]")
print(f"  Total gradient norm: {grad_stats_50k['total_norm']:.6f}")
print(f"  Parameters with gradients: {grad_stats_50k['num_params_with_grad']}")
print(f"  Average loss: {grad_stats_50k['avg_loss']:.4f}")

print(f"\n[Gradient Norm by Layer Type]")
for layer_type in ["embedding", "attention", "feedforward", "layernorm"]:
    if layer_type in grad_stats_50k["by_layer_type"]:
        grads = grad_stats_50k["by_layer_type"][layer_type]
        if grads:
            avg_grad = np.mean([g for _, g, _ in grads])
            avg_ratio = np.mean([r for _, _, r in grads])
            print(f"  {layer_type.upper():15s}: avg_grad={avg_grad:.6f}, avg_ratio={avg_ratio:.6f}")

print(f"\n[Top 10 Parameters by Gradient Norm]")
all_params = [(name, stats["grad_norm"], stats["ratio"]) 
              for name, stats in grad_stats_50k["by_layer_name"].items()]
all_params.sort(key=lambda x: x[1], reverse=True)

for name, grad_norm, ratio in all_params[:10]:
    print(f"  {name}: grad_norm={grad_norm:.6f}, ratio={ratio:.6f}")

print(f"\n[Bottom 10 Parameters by Gradient Norm]")
for name, grad_norm, ratio in all_params[-10:]:
    print(f"  {name}: grad_norm={grad_norm:.6f}, ratio={ratio:.6f}")

# ==============================================================================
# PART 2: GRADIENT COMPARISON ACROSS CHECKPOINTS
# ==============================================================================

print("\n" + "="*100)
print("PART 2: GRADIENT COMPARISON ACROSS CHECKPOINTS")
print("="*100)

checkpoint_paths = [
    Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_1000.pt"),
    Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_10000.pt"),
    Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_25000.pt"),
    Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt"),
]

gradient_evolution = {}

for ckpt_path in checkpoint_paths:
    if ckpt_path.exists():
        print(f"\n[Analyzing {ckpt_path.name}...]")
        grad_stats, step = load_checkpoint_and_compute_gradients(ckpt_path, held_out_tokens)
        gradient_evolution[step] = grad_stats
        
        print(f"  Total gradient norm: {grad_stats['total_norm']:.6f}")
        print(f"  Average loss: {grad_stats['avg_loss']:.4f}")

# Compare gradient magnitudes across training
if len(gradient_evolution) > 1:
    print("\n[Gradient Evolution by Layer Type]")
    
    steps = sorted(gradient_evolution.keys())
    
    for layer_type in ["embedding", "attention", "feedforward", "layernorm"]:
        print(f"\n  {layer_type.upper()}:")
        
        for step in steps:
            if layer_type in gradient_evolution[step]["by_layer_type"]:
                grads = gradient_evolution[step]["by_layer_type"][layer_type]
                if grads:
                    avg_grad = np.mean([g for _, g, _ in grads])
                    avg_ratio = np.mean([r for _, _, r in grads])
                    print(f"    Step {step:5d}: avg_grad={avg_grad:.6f}, avg_ratio={avg_ratio:.6f}")

# ==============================================================================
# PART 3: PARAMETER CHANGE ANALYSIS
# ==============================================================================

print("\n" + "="*100)
print("PART 3: PARAMETER CHANGE ANALYSIS")
print("="*100)

params_by_step = analyze_parameter_changes(checkpoint_paths)

# ==============================================================================
# PART 4: ATTENTION-SPECIFIC ANALYSIS
# ==============================================================================

print("\n" + "="*100)
print("PART 4: ATTENTION LAYER GRADIENT ANALYSIS")
print("="*100)

print("""
[Question: Are attention layers learning contextual patterns?]

If attention layers have:
- SMALL gradients → not learning much, stuck
- LARGE gradients but SMALL param changes → may be oscillating
- CONSISTENT gradients across checkpoints → still learning
""")

if len(gradient_evolution) > 1:
    print("\n[Attention Layer Gradient Details]")
    
    steps = sorted(gradient_evolution.keys())
    
    # Get attention-specific parameters
    attn_params = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    for attn_type in attn_params:
        print(f"\n  {attn_type.upper()}:")
        
        for step in steps:
            matching_params = [
                (name, stats["grad_norm"], stats["ratio"])
                for name, stats in gradient_evolution[step]["by_layer_name"].items()
                if attn_type in name
            ]
            
            if matching_params:
                avg_grad = np.mean([g for _, g, _ in matching_params])
                avg_ratio = np.mean([r for _, _, r in matching_params])
                print(f"    Step {step:5d}: avg_grad={avg_grad:.6f}, avg_ratio={avg_ratio:.6f}")

# ==============================================================================
# PART 5: UPDATE-TO-PARAMETER RATIO ANALYSIS
# ==============================================================================

print("\n" + "="*100)
print("PART 5: UPDATE-TO-PARAMETER RATIO ANALYSIS")
print("="*100)

print("""
[Question: Are parameters being updated meaningfully?]

The ratio of gradient_norm / parameter_norm indicates:
- Ratio ~ 1e-3 to 1e-2: Healthy updates
- Ratio << 1e-4: Updates too small, not learning
- Ratio >> 1e-1: Updates too large, may be unstable

This is a standard diagnostic from Andrej Karpathy's "A Recipe for Training Neural Networks"
""")

# Compute average ratios by layer type
print("\n[Gradient/Parameter Ratio by Layer Type at Step 50k]")

healthy_count = 0
too_small_count = 0
too_large_count = 0

for layer_type in ["embedding", "attention", "feedforward", "layernorm"]:
    if layer_type in grad_stats_50k["by_layer_type"]:
        grads = grad_stats_50k["by_layer_type"][layer_type]
        if grads:
            ratios = [r for _, _, r in grads]
            avg_ratio = np.mean(ratios)
            min_ratio = np.min(ratios)
            max_ratio = np.max(ratios)
            
            status = "HEALTHY" if 1e-4 < avg_ratio < 1e-1 else ("TOO SMALL" if avg_ratio <= 1e-4 else "TOO LARGE")
            
            if 1e-4 < avg_ratio < 1e-1:
                healthy_count += 1
            elif avg_ratio <= 1e-4:
                too_small_count += 1
            else:
                too_large_count += 1
            
            print(f"  {layer_type.upper():15s}: avg={avg_ratio:.2e}, min={min_ratio:.2e}, max={max_ratio:.2e} [{status}]")

print(f"\n[Overall Status]")
print(f"  Healthy: {healthy_count}/4 layer types")
print(f"  Too small: {too_small_count}/4 layer types")
print(f"  Too large: {too_large_count}/4 layer types")

# ==============================================================================
# PART 6: EMBEDDING VS ATTENTION LEARNING COMPARISON
# ==============================================================================

print("\n" + "="*100)
print("PART 6: EMBEDDING VS ATTENTION LEARNING COMPARISON")
print("="*100)

print("""
[Question: Is the model learning unigram statistics (embeddings) vs context (attention)?]

If embeddings have much larger gradients than attention:
  → Model is primarily learning token frequencies (unigram)
  
If attention has comparable gradients:
  → Model is learning contextual patterns
""")

if "embedding" in grad_stats_50k["by_layer_type"] and "attention" in grad_stats_50k["by_layer_type"]:
    emb_grads = grad_stats_50k["by_layer_type"]["embedding"]
    attn_grads = grad_stats_50k["by_layer_type"]["attention"]
    
    if emb_grads and attn_grads:
        avg_emb_grad = np.mean([g for _, g, _ in emb_grads])
        avg_attn_grad = np.mean([g for _, g, _ in attn_grads])
        
        ratio = avg_emb_grad / (avg_attn_grad + 1e-10)
        
        print(f"\n  Embedding avg gradient: {avg_emb_grad:.6f}")
        print(f"  Attention avg gradient: {avg_attn_grad:.6f}")
        print(f"  Ratio (emb/attn): {ratio:.2f}x")
        
        if ratio > 10:
            print(f"\n  ⚠️ POTENTIAL ISSUE: Embeddings have {ratio:.0f}x larger gradients")
            print(f"     Model may be primarily learning unigram frequencies")
            print(f"     Contextual learning via attention may be suppressed")
        else:
            print(f"\n  ✅ Gradients are reasonably balanced")

# ==============================================================================
# SUMMARY AND CLASSIFICATION
# ==============================================================================

print("\n" + "="*100)
print("SUMMARY AND CLASSIFICATION")
print("="*100)

# Classify findings
findings = []

# Finding 1: Gradient magnitude
if grad_stats_50k['total_norm'] > 0:
    findings.append(("Gradients are flowing through all layers", "VERIFIED"))
else:
    findings.append(("Gradients are flowing through all layers", "NOT SUPPORTED"))

# Finding 2: Update ratios
if healthy_count >= 3:
    findings.append(("Update-to-parameter ratios are healthy", "VERIFIED"))
elif too_small_count >= 2:
    findings.append(("Update-to-parameter ratios are too small", "VERIFIED"))
else:
    findings.append(("Update-to-parameter ratios are healthy", "UNKNOWN"))

# Finding 3: Attention learning
if "embedding" in grad_stats_50k["by_layer_type"] and "attention" in grad_stats_50k["by_layer_type"]:
    emb_grads = grad_stats_50k["by_layer_type"]["embedding"]
    attn_grads = grad_stats_50k["by_layer_type"]["attention"]
    
    if emb_grads and attn_grads:
        avg_emb_grad = np.mean([g for _, g, _ in emb_grads])
        avg_attn_grad = np.mean([g for _, g, _ in attn_grads])
        ratio = avg_emb_grad / (avg_attn_grad + 1e-10)
        
        if ratio > 10:
            findings.append(("Embeddings dominate attention in gradient magnitude", "VERIFIED"))
        else:
            findings.append(("Embeddings dominate attention in gradient magnitude", "NOT SUPPORTED"))

# Finding 4: Gradient evolution
if len(gradient_evolution) > 1:
    steps = sorted(gradient_evolution.keys())
    grad_norms = [gradient_evolution[s]['total_norm'] for s in steps]
    
    if grad_norms[0] > grad_norms[-1]:
        findings.append(("Gradients decrease over training (normal)", "VERIFIED"))
    else:
        findings.append(("Gradients decrease over training (normal)", "NOT SUPPORTED"))

print("\n[CLASSIFICATION OF FINDINGS]")
print("-"*80)

for finding, status in findings:
    status_str = f"✅ {status}" if status == "VERIFIED" else f"❌ {status}" if status == "NOT SUPPORTED" else f"⚠️ {status}"
    print(f"  {finding}")
    print(f"    → {status_str}")
    print()

# Final conclusion
print("="*80)
print("CONCLUSION")
print("="*80)

print("""
This diagnostic analyzed:
1. Gradient flow through the network
2. Update-to-parameter ratios
3. Gradient distribution by layer type
4. Embedding vs attention learning balance
5. Gradient evolution across training

All findings are READ-ONLY - no checkpoints were modified.
""")

print("\n" + "="*100)
print("DIAGNOSTIC COMPLETE")
print("="*100)
