#!/usr/bin/env python3
"""
READ-ONLY EOS Isolation Diagnostic

Compares evaluation performance with and without EOS token to determine
whether the EOS mismatch materially affects next-token prediction.

CRITICAL: This is READ-ONLY. No training, no checkpoint modification.
"""

import sys
import torch
import torch.nn.functional as F
import math
from pathlib import Path
from datasets import load_dataset

# Add project to path
sys.path.insert(0, "/root/agent-sandbox/ARTN")

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.bitlinear import BitLinearGPT

print("="*100)
print("READ-ONLY EOS ISOLATION DIAGNOSTIC")
print("="*100)

device = "cpu"

# ==============================================================================
# LOAD TOKENIZER AND MODEL
# ==============================================================================

tokenizer_path = Path("outputs/artn-optimized/tokenizer.json")
checkpoint_path = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")

print(f"\n[Loading tokenizer and model...]")
tokenizer = BPETokenizer.load(str(tokenizer_path))

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

print(f"  Tokenizer vocab: {len(tokenizer.vocab)}")
print(f"  PAD token ID: {tokenizer.pad_token_id}")
print(f"  EOS token ID: {tokenizer.eos_token_id}")
print(f"  Checkpoint: step {checkpoint.get('step', 'unknown')}")

# ==============================================================================
# LOAD HELD-OUT DATA (SAME AS EVALUATION)
# ==============================================================================

print(f"\n[Loading held-out data...]")

dataset = load_dataset("roneneldan/TinyStories", split="train")

# Use same held-out stories as trajectory analysis (50000-50500)
held_out_stories = []
for i, item in enumerate(dataset):
    if i >= 50000 and i < 50500:
        held_out_stories.append(item["text"])

print(f"  Held-out stories: {len(held_out_stories)}")

# Tokenize held-out data
held_out_tokens = []
for story in held_out_stories:
    tokens = tokenizer.encode(story)[:255]  # Leave room for EOS
    if len(tokens) >= 10:
        held_out_tokens.append(tokens)

print(f"  Usable sequences: {len(held_out_tokens)}")

# ==============================================================================
# EVALUATION METHOD A: CURRENT STYLE (WITHOUT EOS)
# ==============================================================================

print("\n" + "="*100)
print("EVALUATION A: CURRENT STYLE (WITHOUT EOS)")
print("="*100)

print("""
Construction:
  tokens = tokenizer.encode(story)[:256]
  input_ids = tokens[:-1]
  targets = tokens[1:]
  No EOS token added
  
Loss: CrossEntropyLoss on all positions (no masking)
""")

model.eval()
total_loss_a = 0.0
total_predictions_a = 0
correct_top1_a = 0
correct_top5_a = 0

with torch.no_grad():
    for tokens in held_out_tokens:
        if len(tokens) < 2:
            continue
        
        # Current evaluation style: NO EOS
        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
        targets = torch.tensor([tokens[1:]], dtype=torch.long)
        
        # Forward pass
        output = model(input_ids)
        logits = output["logits"]
        
        # Calculate loss (on all positions)
        loss = F.cross_entropy(logits.view(-1, len(tokenizer.vocab)), targets.view(-1))
        total_loss_a += loss.item() * targets.size(1)
        
        # Accuracy (exclude padding token 0 from accuracy)
        probs = F.softmax(logits, dim=-1)
        top1_pred = torch.argmax(probs, dim=-1)
        top5_pred = torch.topk(probs, k=5, dim=-1).indices
        
        mask = targets != 0  # Exclude padding
        correct_top1_a += ((top1_pred == targets) & mask).sum().item()
        correct_top5_a += ((top5_pred == targets.unsqueeze(-1)).any(dim=-1) & mask).sum().item()
        total_predictions_a += mask.sum().item()

avg_loss_a = total_loss_a / total_predictions_a if total_predictions_a > 0 else float('inf')
perplexity_a = math.exp(avg_loss_a) if avg_loss_a < 20 else float('inf')
top1_acc_a = correct_top1_a / total_predictions_a if total_predictions_a > 0 else 0
top5_acc_a = correct_top5_a / total_predictions_a if total_predictions_a > 0 else 0

print(f"\nResults (WITHOUT EOS):")
print(f"  Total predictions: {total_predictions_a:,}")
print(f"  Cross-Entropy Loss: {avg_loss_a:.4f}")
print(f"  Perplexity: {perplexity_a:.2f}")
print(f"  Top-1 Accuracy: {top1_acc_a*100:.2f}%")
print(f"  Top-5 Accuracy: {top5_acc_a*100:.2f}%")

# ==============================================================================
# EVALUATION METHOD B: TRAINING STYLE (WITH EOS)
# ==============================================================================

print("\n" + "="*100)
print("EVALUATION B: TRAINING STYLE (WITH EOS)")
print("="*100)

print("""
Construction:
  tokens = tokenizer.encode(story)[:255]
  tokens = tokens + [EOS]  # Add EOS token
  input_ids = tokens[:-1]
  targets = tokens[1:]
  EOS token included as target
  
Loss: CrossEntropyLoss with ignore_index=-100 for padding
""")

model.eval()
total_loss_b = 0.0
total_predictions_b = 0
correct_top1_b = 0
correct_top5_b = 0

# Also track EOS-specific predictions
eos_predictions = 0
eos_correct = 0

with torch.no_grad():
    for tokens in held_out_tokens:
        if len(tokens) < 2:
            continue
        
        # Training style: ADD EOS
        tokens_with_eos = tokens + [tokenizer.eos_token_id]
        
        # Pad to 256 for fair comparison (same as training)
        padding_length = 256 - len(tokens_with_eos)
        if padding_length > 0:
            tokens_padded = tokens_with_eos + [tokenizer.pad_token_id] * padding_length
        else:
            tokens_padded = tokens_with_eos[:256]
        
        input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
        targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
        
        # Create loss mask (same as training)
        loss_mask = targets != tokenizer.pad_token_id
        
        # Forward pass
        output = model(input_ids)
        logits = output["logits"]
        
        # Calculate loss with masking (ignore padding)
        loss = F.cross_entropy(
            logits.view(-1, len(tokenizer.vocab)),
            targets.view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        
        # Only count non-padding positions
        valid_positions = loss_mask.sum().item()
        total_loss_b += loss.item() * valid_positions
        
        # Accuracy (exclude padding)
        probs = F.softmax(logits, dim=-1)
        top1_pred = torch.argmax(probs, dim=-1)
        top5_pred = torch.topk(probs, k=5, dim=-1).indices
        
        correct_top1_b += ((top1_pred == targets) & loss_mask).sum().item()
        correct_top5_b += ((top5_pred == targets.unsqueeze(-1)).any(dim=-1) & loss_mask).sum().item()
        total_predictions_b += valid_positions
        
        # Track EOS prediction accuracy
        # Find positions where target is EOS
        eos_positions = (targets == tokenizer.eos_token_id).nonzero(as_tuple=True)
        if len(eos_positions[0]) > 0:
            for i in range(len(eos_positions[0])):
                batch_idx = eos_positions[0][i]
                pos_idx = eos_positions[1][i]
                eos_predictions += 1
                if top1_pred[batch_idx, pos_idx] == tokenizer.eos_token_id:
                    eos_correct += 1

avg_loss_b = total_loss_b / total_predictions_b if total_predictions_b > 0 else float('inf')
perplexity_b = math.exp(avg_loss_b) if avg_loss_b < 20 else float('inf')
top1_acc_b = correct_top1_b / total_predictions_b if total_predictions_b > 0 else 0
top5_acc_b = correct_top5_b / total_predictions_b if total_predictions_b > 0 else 0
eos_acc = eos_correct / eos_predictions if eos_predictions > 0 else 0

print(f"\nResults (WITH EOS):")
print(f"  Total predictions: {total_predictions_b:,}")
print(f"  Cross-Entropy Loss: {avg_loss_b:.4f}")
print(f"  Perplexity: {perplexity_b:.2f}")
print(f"  Top-1 Accuracy: {top1_acc_b*100:.2f}%")
print(f"  Top-5 Accuracy: {top5_acc_b*100:.2f}%")
print(f"\n  EOS Prediction:")
print(f"    EOS positions: {eos_predictions:,}")
print(f"    EOS correctly predicted: {eos_correct:,}")
print(f"    EOS accuracy: {eos_acc*100:.2f}%")

# ==============================================================================
# FAIR COMPARISON: IDENTICAL PREDICTION POSITIONS
# ==============================================================================

print("\n" + "="*100)
print("FAIR COMPARISON: IDENTICAL PREDICTION POSITIONS")
print("="*100)

print("""
Comparing ONLY the positions that exist in BOTH evaluations:
- Positions 0 to len(original_tokens)-2 (before EOS)
- Excluding the EOS position itself
- Excluding any padding positions
""")

model.eval()
total_loss_shared = 0.0
total_predictions_shared = 0
correct_top1_a_shared = 0
correct_top1_b_shared = 0

with torch.no_grad():
    for tokens in held_out_tokens:
        if len(tokens) < 3:
            continue
        
        # Compare ONLY positions that exist in both:
        # - Without EOS: positions 0 to len(tokens)-2
        # - With EOS: positions 0 to len(tokens)-2 (same positions, before EOS)
        
        shared_len = len(tokens) - 1  # Number of prediction positions shared
        
        # Method A: WITHOUT EOS
        input_ids_a = torch.tensor([tokens[:-1]], dtype=torch.long)
        targets_a = torch.tensor([tokens[1:]], dtype=torch.long)
        
        output_a = model(input_ids_a)
        logits_a = output_a["logits"][0, :shared_len, :]
        targets_shared = targets_a[0, :shared_len]
        
        loss_a = F.cross_entropy(logits_a, targets_shared, reduction='sum')
        top1_a = torch.argmax(F.softmax(logits_a, dim=-1), dim=-1)
        
        # Method B: WITH EOS
        tokens_with_eos = tokens + [tokenizer.eos_token_id]
        input_ids_b = torch.tensor([tokens_with_eos[:-1]], dtype=torch.long)
        targets_b = torch.tensor([tokens_with_eos[1:]], dtype=torch.long)
        
        output_b = model(input_ids_b)
        logits_b = output_b["logits"][0, :shared_len, :]
        
        loss_b = F.cross_entropy(logits_b, targets_shared, reduction='sum')
        top1_b = torch.argmax(F.softmax(logits_b, dim=-1), dim=-1)
        
        # Accumulate (exclude padding from accuracy)
        mask = targets_shared != 0
        total_loss_shared += loss_a.item()
        total_predictions_shared += shared_len
        
        correct_top1_a_shared += ((top1_a == targets_shared) & mask).sum().item()
        correct_top1_b_shared += ((top1_b == targets_shared) & mask).sum().item()

avg_loss_shared = total_loss_shared / total_predictions_shared if total_predictions_shared > 0 else float('inf')
top1_acc_a_shared = correct_top1_a_shared / total_predictions_shared if total_predictions_shared > 0 else 0
top1_acc_b_shared = correct_top1_b_shared / total_predictions_shared if total_predictions_shared > 0 else 0

print(f"\nShared prediction positions: {total_predictions_shared:,}")
print(f"\n[Method A: WITHOUT EOS]")
print(f"  Top-1 Accuracy (shared positions): {top1_acc_a_shared*100:.4f}%")
print(f"\n[Method B: WITH EOS]")
print(f"  Top-1 Accuracy (shared positions): {top1_acc_b_shared*100:.4f}%")
print(f"\nDifference: {(top1_acc_b_shared - top1_acc_a_shared)*100:.4f}%")

# ==============================================================================
# CONTEXTUAL METRICS COMPARISON
# ==============================================================================

print("\n" + "="*100)
print("CONTEXTUAL METRICS COMPARISON")
print("="*100)

# Test context benefit with and without EOS
print("\n[Context Benefit Test]")

# Sample a few sequences for detailed analysis
sample_tokens = held_out_tokens[:10]

context_benefits_a = []
context_benefits_b = []

with torch.no_grad():
    for tokens in sample_tokens:
        if len(tokens) < 20:
            continue
        
        # Method A: WITHOUT EOS
        # Full context
        input_full_a = torch.tensor([tokens[:-1]], dtype=torch.long)
        output_full_a = model(input_full_a)
        loss_full_a = F.cross_entropy(
            output_full_a["logits"][0], 
            torch.tensor(tokens[1:])
        ).item()
        
        # Limited context (last 8 tokens)
        input_limited_a = torch.tensor([tokens[-9:-1]], dtype=torch.long)
        output_limited_a = model(input_limited_a)
        loss_limited_a = F.cross_entropy(
            output_limited_a["logits"][0],
            torch.tensor(tokens[-8:])
        ).item()
        
        context_benefit_a = loss_limited_a - loss_full_a
        context_benefits_a.append(context_benefit_a)
        
        # Method B: WITH EOS
        tokens_with_eos = tokens + [tokenizer.eos_token_id]
        
        # Full context (input has EOS, but we compare on same positions as Method A)
        input_full_b = torch.tensor([tokens_with_eos[:-1]], dtype=torch.long)
        output_full_b = model(input_full_b)
        # Compare on positions 0 to len(tokens)-2 (same as Method A)
        loss_full_b = F.cross_entropy(
            output_full_b["logits"][0, :len(tokens)-1],
            torch.tensor(tokens[1:])
        ).item()
        
        # Limited context (last 8 tokens before EOS)
        input_limited_b = torch.tensor([tokens[-9:-1]], dtype=torch.long)
        output_limited_b = model(input_limited_b)
        loss_limited_b = F.cross_entropy(
            output_limited_b["logits"][0],
            torch.tensor(tokens[-8:])
        ).item()
        
        context_benefit_b = loss_limited_b - loss_full_b
        context_benefits_b.append(context_benefit_b)

avg_context_benefit_a = sum(context_benefits_a) / len(context_benefits_a) if context_benefits_a else 0
avg_context_benefit_b = sum(context_benefits_b) / len(context_benefits_b) if context_benefits_b else 0

print(f"  Method A (WITHOUT EOS): {avg_context_benefit_a:.4f}")
print(f"  Method B (WITH EOS): {avg_context_benefit_b:.4f}")
print(f"  Difference: {avg_context_benefit_b - avg_context_benefit_a:.4f}")

# ==============================================================================
# SUMMARY AND CLASSIFICATION
# ==============================================================================

print("\n" + "="*100)
print("SUMMARY AND CLASSIFICATION")
print("="*100)

print(f"""
COMPARISON RESULTS:

| Metric | WITHOUT EOS (A) | WITH EOS (B) | Difference |
|--------|-----------------|--------------|------------|
| Cross-Entropy | {avg_loss_a:.4f} | {avg_loss_b:.4f} | {avg_loss_b - avg_loss_a:+.4f} |
| Perplexity | {perplexity_a:.2f} | {perplexity_b:.2f} | {perplexity_b - perplexity_a:+.2f} |
| Top-1 Accuracy | {top1_acc_a*100:.2f}% | {top1_acc_b*100:.2f}% | {(top1_acc_b - top1_acc_a)*100:+.2f}% |
| Top-5 Accuracy | {top5_acc_a*100:.2f}% | {top5_acc_b*100:.2f}% | {(top5_acc_b - top5_acc_a)*100:+.2f}% |

SHARED POSITION COMPARISON (FAIR):
| Method | Top-1 Accuracy | Difference |
|--------|----------------|------------|
| WITHOUT EOS | {top1_acc_a_shared*100:.4f}% | - |
| WITH EOS | {top1_acc_b_shared*100:.4f}% | {(top1_acc_b_shared - top1_acc_a_shared)*100:+.4f}% |

CONTEXT BENEFIT:
| Method | Benefit | Difference |
|--------|---------|------------|
| WITHOUT EOS | {avg_context_benefit_a:.4f} | - |
| WITH EOS | {avg_context_benefit_b:.4f} | {avg_context_benefit_b - avg_context_benefit_a:+.4f} |

EOS-SPECIFIC:
  EOS predictions: {eos_predictions:,}
  EOS accuracy: {eos_acc*100:.2f}%
""")

# Classification
print("="*80)
print("CLASSIFICATION")
print("="*80)

diff_pct = abs((top1_acc_b_shared - top1_acc_a_shared) * 100)

if diff_pct < 0.1:
    print(f"""
❌ NOT SUPPORTED: EOS mismatch does NOT materially affect next-token prediction

Evidence:
  - Accuracy difference on shared positions: {diff_pct:.4f}%
  - This is negligible (< 0.1%)
  - Adding EOS does not improve performance

Conclusion:
  The EOS token mismatch is NOT the cause of weak contextual learning.
  The model performs identically with and without EOS on shared prediction positions.
""")
elif diff_pct < 1.0:
    print(f"""
⚠️ UNKNOWN: EOS mismatch has a SMALL effect on next-token prediction

Evidence:
  - Accuracy difference on shared positions: {diff_pct:.4f}%
  - This is small but non-negligible
  - More investigation needed

Conclusion:
  The EOS token mismatch may have a minor effect, but is unlikely to be the
  primary cause of weak contextual learning.
""")
else:
    print(f"""
✅ VERIFIED: EOS mismatch DOES materially affect next-token prediction

Evidence:
  - Accuracy difference on shared positions: {diff_pct:.4f}%
  - This is significant (> 1.0%)
  - Adding EOS improves/degrades performance

Conclusion:
  The EOS token mismatch affects model performance and should be investigated
  as a potential contributor to weak contextual learning.
""")

print("\n" + "="*100)
print("DIAGNOSTIC COMPLETE")
print("="*100)
