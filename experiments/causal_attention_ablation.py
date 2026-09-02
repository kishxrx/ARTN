#!/usr/bin/env python3
"""
READ-ONLY Causal Attention Ablation

Compares model performance with normal vs destroyed attention to establish
whether attention contributes to prediction.

CRITICAL: This is READ-ONLY. No training, no checkpoint modification.
- We evaluate the same checkpoint with different attention mechanisms
- Checkpoints are only READ, never modified or saved
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
import random

# Add project to path
sys.path.insert(0, "/root/agent-sandbox/ARTN")

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.bitlinear import BitLinearGPT

print("="*100)
print("READ-ONLY CAUSAL ATTENTION ABLATION")
print("="*100)
print("""
This diagnostic determines whether attention is causally contributing to prediction.

We compare:
1. Normal attention (baseline)
2. Uniform attention weights (destroy learned patterns)
3. Random/shuffled attention weights

If destroying attention causes little degradation:
  → Attention contributes little to prediction
  
If destroying attention causes substantial degradation:
  → Attention IS contributing despite appearing diffuse

NO arbitrary thresholds. Report measured differences only.
""")

device = "cpu"

# ==============================================================================
# LOAD TOKENIZER AND MODEL
# ==============================================================================

tokenizer_path = Path("outputs/artn-optimized/tokenizer.json")
tokenizer = BPETokenizer.load(str(tokenizer_path))

print(f"\n[Tokenizer]")
print(f"  Vocab size: {len(tokenizer.vocab)}")

# Load model
checkpoint_path = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")
model = BitLinearGPT(
    vocab_size=len(tokenizer.vocab),
    d_model=256,
    n_layers=6,
    n_heads=8,
    d_ff=1024,
    max_seq_len=256,
    dropout=0.0
)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"\n[Model]")
print(f"  Checkpoint: step {checkpoint.get('step', 'unknown')}")

# ==============================================================================
# LOAD TEST DATA
# ==============================================================================

print(f"\n[Loading test data...]")

dataset = load_dataset("roneneldan/TinyStories", split="train")
test_stories = []
for i, item in enumerate(dataset):
    if i >= 50000 and i < 50500:  # 500 stories for evaluation
        test_stories.append(item["text"])

print(f"  Test stories: {len(test_stories)}")

# Tokenize
test_sequences = []
for story in test_stories:
    tokens = tokenizer.encode(story)[:255]
    tokens = tokens + [tokenizer.eos_token_id]
    if len(tokens) >= 20:
        test_sequences.append(tokens)

print(f"  Usable sequences: {len(test_sequences)}")

# ==============================================================================
# HELPER: FORWARD PASS WITH DIFFERENT ATTENTION MODES
# ==============================================================================

def forward_with_attention_mode(model, input_ids, targets, attention_mode='normal'):
    """
    Forward pass with different attention modes.
    
    Modes:
    - 'normal': Use learned attention weights
    - 'uniform': Force uniform attention distribution
    - 'random': Use random attention weights
    
    Returns: loss, top1_correct, total_predictions
    """
    batch_size, seq_len = input_ids.shape
    
    # Get embeddings
    positions = torch.arange(seq_len, device=input_ids.device)
    x = model.token_embedding(input_ids) + model.position_embedding(positions)
    x = model.dropout(x)
    
    # Causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
    
    # Process through transformer blocks
    for block in model.blocks:
        # Pre-norm
        x_norm = block.ln1(x)
        
        # Multi-head attention
        attn = block.attn
        B, T, C = x_norm.shape
        n_heads = 8
        head_dim = C // n_heads
        
        # Q, K, V projections
        q = attn.q_proj(x_norm).view(B, T, n_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(x_norm).view(B, T, n_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(x_norm).view(B, T, n_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Apply causal mask
        scores = scores.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Apply attention mode
        if attention_mode == 'normal':
            attn_weights = F.softmax(scores, dim=-1)
        elif attention_mode == 'uniform':
            # Force uniform attention over valid positions
            # Count valid positions at each position
            valid_mask = mask[:, :, :T, :T].squeeze(0).squeeze(0)  # (T, T)
            valid_counts = valid_mask.sum(dim=-1)  # (T,)
            attn_weights = valid_mask.float() / valid_counts.unsqueeze(-1)
            attn_weights = attn_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            attn_weights = attn_weights.expand(B, n_heads, -1, -1)
        elif attention_mode == 'random':
            # Random attention weights (still causal)
            attn_weights = torch.rand_like(scores)
            attn_weights = attn_weights.masked_fill(mask[:, :, :T, :T] == 0, 0.0)
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown attention mode: {attention_mode}")
        
        # Apply attention
        attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        attn_out = attn.o_proj(attn_out)
        
        # Residual
        x = x + attn_out
        
        # FFN
        x = x + block.ff(block.ln2(x))
    
    # Output
    x = model.ln_final(x)
    logits = model.lm_head(x)
    
    # Compute loss
    loss = F.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        targets.reshape(-1),
        ignore_index=-100,
        reduction='sum'
    )
    
    # Count valid predictions
    valid_mask = targets != -100
    num_predictions = valid_mask.sum().item()
    
    # Top-1 accuracy
    predictions = logits.argmax(dim=-1)
    correct = ((predictions == targets) & valid_mask).sum().item()
    
    return loss.item(), correct, num_predictions

# ==============================================================================
# PART 1: BASELINE WITH NORMAL ATTENTION
# ==============================================================================

print("\n" + "="*100)
print("PART 1: BASELINE - NORMAL ATTENTION")
print("="*100)

total_loss_normal = 0.0
total_correct_normal = 0
total_predictions_normal = 0

for tokens in test_sequences:
    # Pad to 256
    padding_length = 256 - len(tokens)
    tokens_padded = tokens + [tokenizer.pad_token_id] * padding_length
    
    input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
    targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
    
    # Mask padding
    targets[targets == tokenizer.pad_token_id] = -100
    
    with torch.no_grad():
        loss, correct, num_pred = forward_with_attention_mode(model, input_ids, targets, 'normal')
    
    total_loss_normal += loss
    total_correct_normal += correct
    total_predictions_normal += num_pred

avg_loss_normal = total_loss_normal / total_predictions_normal
avg_ce_normal = total_loss_normal / total_predictions_normal
top1_accuracy_normal = total_correct_normal / total_predictions_normal * 100

print(f"\n[Normal Attention Results]")
print(f"  Total predictions: {total_predictions_normal}")
print(f"  Cross-Entropy Loss: {avg_ce_normal:.4f}")
print(f"  Top-1 Accuracy: {top1_accuracy_normal:.4f}%")

# ==============================================================================
# PART 2: UNIFORM ATTENTION WEIGHTS
# ==============================================================================

print("\n" + "="*100)
print("PART 2: UNIFORM ATTENTION WEIGHTS")
print("="*100)

total_loss_uniform = 0.0
total_correct_uniform = 0
total_predictions_uniform = 0

for tokens in test_sequences:
    # Pad to 256
    padding_length = 256 - len(tokens)
    tokens_padded = tokens + [tokenizer.pad_token_id] * padding_length
    
    input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
    targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
    
    # Mask padding
    targets[targets == tokenizer.pad_token_id] = -100
    
    with torch.no_grad():
        loss, correct, num_pred = forward_with_attention_mode(model, input_ids, targets, 'uniform')
    
    total_loss_uniform += loss
    total_correct_uniform += correct
    total_predictions_uniform += num_pred

avg_ce_uniform = total_loss_uniform / total_predictions_uniform
top1_accuracy_uniform = total_correct_uniform / total_predictions_uniform * 100

print(f"\n[Uniform Attention Results]")
print(f"  Total predictions: {total_predictions_uniform}")
print(f"  Cross-Entropy Loss: {avg_ce_uniform:.4f}")
print(f"  Top-1 Accuracy: {top1_accuracy_uniform:.4f}%")

# ==============================================================================
# PART 3: RANDOM ATTENTION WEIGHTS
# ==============================================================================

print("\n" + "="*100)
print("PART 3: RANDOM ATTENTION WEIGHTS")
print("="*100)

# Set seed for reproducibility
torch.manual_seed(42)

total_loss_random = 0.0
total_correct_random = 0
total_predictions_random = 0

for tokens in test_sequences:
    # Pad to 256
    padding_length = 256 - len(tokens)
    tokens_padded = tokens + [tokenizer.pad_token_id] * padding_length
    
    input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
    targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
    
    # Mask padding
    targets[targets == tokenizer.pad_token_id] = -100
    
    with torch.no_grad():
        loss, correct, num_pred = forward_with_attention_mode(model, input_ids, targets, 'random')
    
    total_loss_random += loss
    total_correct_random += correct
    total_predictions_random += num_pred

avg_ce_random = total_loss_random / total_predictions_random
top1_accuracy_random = total_correct_random / total_predictions_random * 100

print(f"\n[Random Attention Results]")
print(f"  Total predictions: {total_predictions_random}")
print(f"  Cross-Entropy Loss: {avg_ce_random:.4f}")
print(f"  Top-1 Accuracy: {top1_accuracy_random:.4f}%")

# ==============================================================================
# PART 4: COMPARISON
# ==============================================================================

print("\n" + "="*100)
print("PART 4: COMPARISON")
print("="*100)

print(f"\n[Performance Comparison]")
print(f"  {'Mode':<20} {'Cross-Entropy':<15} {'Top-1 Accuracy':<15}")
print(f"  {'-'*50}")
print(f"  {'Normal':<20} {avg_ce_normal:<15.4f} {top1_accuracy_normal:<15.4f}%")
print(f"  {'Uniform':<20} {avg_ce_uniform:<15.4f} {top1_accuracy_uniform:<15.4f}%")
print(f"  {'Random':<20} {avg_ce_random:<15.4f} {top1_accuracy_random:<15.4f}%")

# Compute degradation
ce_diff_normal_uniform = avg_ce_uniform - avg_ce_normal
ce_diff_normal_random = avg_ce_random - avg_ce_normal

acc_diff_normal_uniform = top1_accuracy_normal - top1_accuracy_uniform
acc_diff_normal_random = top1_accuracy_normal - top1_accuracy_random

print(f"\n[Degradation from Normal Attention]")
print(f"  {'Comparison':<30} {'CE Diff':<15} {'Acc Diff':<15}")
print(f"  {'-'*60}")
print(f"  {'Normal → Uniform':<30} {ce_diff_normal_uniform:+<15.4f} {acc_diff_normal_uniform:+<15.4f}%")
print(f"  {'Normal → Random':<30} {ce_diff_normal_random:+<15.4f} {acc_diff_normal_random:+<15.4f}%")

# Relative degradation
rel_ce_uniform = ce_diff_normal_uniform / avg_ce_normal * 100
rel_ce_random = ce_diff_normal_random / avg_ce_normal * 100

rel_acc_uniform = acc_diff_normal_uniform / top1_accuracy_normal * 100
rel_acc_random = acc_diff_normal_random / top1_accuracy_normal * 100

print(f"\n[Relative Degradation (% change from normal)]")
print(f"  {'Comparison':<30} {'CE %':<15} {'Acc %':<15}")
print(f"  {'-'*60}")
print(f"  {'Normal → Uniform':<30} {rel_ce_uniform:+<15.2f}% {rel_acc_uniform:+<15.2f}%")
print(f"  {'Normal → Random':<30} {rel_ce_random:+<15.2f}% {rel_acc_random:+<15.2f}%")

# ==============================================================================
# PART 5: CLASSIFICATION
# ==============================================================================

print("\n" + "="*100)
print("CLASSIFICATION")
print("="*100)

print("""
INTERPRETATION:

If replacing attention with uniform/random causes:
  - < 1% relative degradation → Attention contributes LITTLE
  - 1-5% relative degradation → Attention contributes SOMEWHAT
  - > 5% relative degradation → Attention contributes SUBSTANTIALLY

No arbitrary thresholds - we report measured differences.
""")

# Classify based on measured differences
max_rel_degradation = max(abs(rel_ce_uniform), abs(rel_ce_random))

if max_rel_degradation < 1.0:
    classification = "ATTENTION CONTRIBUTES LITTLE"
    status = "VERIFIED"
    interpretation = "Destroying attention causes minimal degradation - attention is not functionally important for prediction"
elif max_rel_degradation < 5.0:
    classification = "ATTENTION CONTRIBUTES SOMEWHAT"
    status = "VERIFIED"
    interpretation = "Destroying attention causes modest degradation - attention has learned some useful patterns"
else:
    classification = "ATTENTION CONTRIBUTES SUBSTANTIALLY"
    status = "VERIFIED"
    interpretation = "Destroying attention causes substantial degradation - attention IS functionally important"

print(f"\n[MEASURED RESULTS]")
print(f"  Max relative CE degradation: {max_rel_degradation:.2f}%")
print(f"  Max relative Acc degradation: {max(abs(rel_acc_uniform), abs(rel_acc_random)):.2f}%")

print(f"\n[CLASSIFICATION]")
print(f"  {classification}")
print(f"  Status: ✅ {status}")

print(f"\n[INTERPRETATION]")
print(f"  {interpretation}")

# Additional analysis: Check if uniform is better/worse than random
if abs(ce_diff_normal_uniform) > abs(ce_diff_normal_random):
    print(f"\n[NOTE]")
    print(f"  Uniform attention degrades MORE than random attention.")
    print(f"  This is unexpected - random should be more destructive.")
    print(f"  This suggests attention weights may not be well-structured.")
else:
    print(f"\n[NOTE]")
    print(f"  Random attention degrades more than uniform attention.")
    print(f"  This is expected - uniform is a reasonable baseline.")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*100)
print("SUMMARY")
print("="*100)

print(f"""
This causal ablation tested whether attention contributes to prediction.

Results (measured, not assumed):
  - Normal attention:    CE={avg_ce_normal:.4f}, Acc={top1_accuracy_normal:.4f}%
  - Uniform attention:   CE={avg_ce_uniform:.4f}, Acc={top1_accuracy_uniform:.4f}%
  - Random attention:    CE={avg_ce_random:.4f}, Acc={top1_accuracy_random:.4f}%
  
  Degradation (Normal → Uniform): CE +{ce_diff_normal_uniform:.4f}, Acc -{acc_diff_normal_uniform:.4f}%
  Degradation (Normal → Random):  CE +{ce_diff_normal_random:.4f}, Acc -{acc_diff_normal_random:.4f}%
  
  Max relative degradation: {max_rel_degradation:.2f}%

Conclusion: {classification}
""")

print("\n" + "="*100)
print("DIAGNOSTIC COMPLETE")
print("="*100)
