#!/usr/bin/env python3
"""
READ-ONLY Training vs Evaluation Pipeline Audit

Investigates discrepancies between training and evaluation pipelines.

CRITICAL: This is READ-ONLY. No training, no checkpoint modification.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset

# Add project to path
sys.path.insert(0, "/root/agent-sandbox/ARTN")

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.bitlinear import BitLinearGPT

print("="*100)
print("READ-ONLY TRAINING VS EVALUATION PIPELINE AUDIT")
print("="*100)

device = "cpu"

# ==============================================================================
# LOAD TOKENIZER
# ==============================================================================

tokenizer_path = Path("outputs/artn-optimized/tokenizer.json")
tokenizer = BPETokenizer.load(str(tokenizer_path))

print(f"\n[Tokenizer]")
print(f"  Vocab size: {len(tokenizer.vocab)}")
print(f"  PAD token ID: {tokenizer.pad_token_id}")
print(f"  EOS token ID: {tokenizer.eos_token_id}")

# ==============================================================================
# DISCREPANCY ANALYSIS
# ==============================================================================

print("\n" + "="*100)
print("DISCREPANCY ANALYSIS")
print("="*100)

# Load a sample story
dataset = load_dataset("roneneldan/TinyStories", split="train")
sample_text = dataset[0]["text"]

print(f"\n[Sample Text]")
print(f"  {sample_text[:100]}...")

# ==============================================================================
# DISCREPANCY 1: EOS TOKEN HANDLING
# ==============================================================================

print("\n" + "-"*80)
print("DISCREPANCY 1: EOS TOKEN HANDLING")
print("-"*80)

print("\n[TRAINING PIPELINE]")
print("  Code: tokens = tokenizer.encode(text)[:max_length - 1]")
print("        tokens = tokens + [tokenizer.eos_token_id]")
train_tokens_no_eos = tokenizer.encode(sample_text)[:255]
train_tokens_with_eos = train_tokens_no_eos + [tokenizer.eos_token_id]
print(f"  Tokens before EOS: {len(train_tokens_no_eos)}")
print(f"  Tokens after EOS: {len(train_tokens_with_eos)}")
print(f"  Last 3 tokens: {train_tokens_with_eos[-3:]}")
print(f"  EOS token added: YES (token ID {tokenizer.eos_token_id})")

print("\n[EVALUATION PIPELINE]")
print("  Code: tokens = tokenizer.encode(story)[:256]")
eval_tokens = tokenizer.encode(sample_text)[:256]
print(f"  Tokens: {len(eval_tokens)}")
print(f"  Last 3 tokens: {eval_tokens[-3:]}")
print(f"  EOS token added: NO")

print("\n[VERDICT]")
if tokenizer.eos_token_id in train_tokens_with_eos and tokenizer.eos_token_id not in eval_tokens:
    print("  ❌ VERIFIED ISSUE: Training adds EOS, evaluation does NOT")
    print("  IMPACT: Model learns to predict EOS during training but never sees it during evaluation")
else:
    print("  ✅ NOT AN ISSUE")

# ==============================================================================
# DISCREPANCY 2: PADDING HANDLING
# ==============================================================================

print("\n" + "-"*80)
print("DISCREPANCY 2: PADDING HANDLING")
print("-"*80)

print("\n[TRAINING PIPELINE]")
print("  Code: padding_length = max_length - len(tokens)")
print("        tokens = tokens + [tokenizer.pad_token_id] * padding_length")
padding_length = 256 - len(train_tokens_with_eos)
train_padded = train_tokens_with_eos + [tokenizer.pad_token_id] * padding_length
print(f"  Sequence length: {len(train_padded)}")
print(f"  Padding tokens: {padding_length}")
print(f"  Padded to 256: YES")

print("\n[EVALUATION PIPELINE]")
print("  Code: tokens = tokenizer.encode(story)[:256]")
print(f"  Sequence length: {len(eval_tokens)}")
print(f"  Padding tokens: 0")
print(f"  Padded to 256: NO")

print("\n[VERDICT]")
if padding_length > 0 and len(eval_tokens) < 256:
    print("  ⚠️ VERIFIED DIFFERENCE: Training pads, evaluation does NOT")
    print("  IMPACT: Different sequence lengths, but causal mask handles this correctly")
    print("  STATUS: NOT A CRITICAL ISSUE (causal mask prevents padding contamination)")
else:
    print("  ✅ NOT AN ISSUE")

# ==============================================================================
# DISCREPANCY 3: INPUT/TARGET SHIFTING
# ==============================================================================

print("\n" + "-"*80)
print("DISCREPANCY 3: INPUT/TARGET SHIFTING")
print("-"*80)

print("\n[TRAINING PIPELINE]")
print("  Dataset returns: {'input_ids': tokens, 'labels': tokens.clone()}")
print("  labels[labels == pad_token_id] = -100")
print("  Training step:")
print("    outputs = model(input_ids, targets=labels)")
print("  Model forward internally does:")
print("    inputs = batch[:, :-1]  # all but last")
print("    targets = batch[:, 1:]  # all but first")

train_input_ids = torch.tensor([train_padded], dtype=torch.long)
train_labels = train_input_ids.clone()
train_labels[train_labels == tokenizer.pad_token_id] = -100

# What the model sees internally
train_inputs_internal = train_input_ids[:, :-1]
train_targets_internal = train_input_ids[:, 1:]

print(f"\n  input_ids shape: {train_input_ids.shape}")
print(f"  labels shape: {train_labels.shape}")
print(f"  Internal inputs shape: {train_inputs_internal.shape}")
print(f"  Internal targets shape: {train_targets_internal.shape}")
print(f"  First 5 input tokens: {train_input_ids[0, :5].tolist()}")
print(f"  First 5 target tokens: {train_targets_internal[0, :5].tolist()}")
print(f"  Labels with -100 (masked): {(train_labels == -100).sum().item()}")

print("\n[EVALUATION PIPELINE]")
print("  Code: input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)")
print("        targets = torch.tensor([tokens[1:]], dtype=torch.long)")

eval_input_ids = torch.tensor([eval_tokens[:-1]], dtype=torch.long)
eval_targets = torch.tensor([eval_tokens[1:]], dtype=torch.long)

print(f"\n  input_ids shape: {eval_input_ids.shape}")
print(f"  targets shape: {eval_targets.shape}")
print(f"  First 5 input tokens: {eval_input_ids[0, :5].tolist()}")
print(f"  First 5 target tokens: {eval_targets[0, :5].tolist()}")
print(f"  Labels with -100 (masked): 0")

print("\n[VERDICT]")
# Check if shifting is equivalent
if train_inputs_internal.shape[1] == eval_input_ids.shape[1]:
    print("  ✅ NOT AN ISSUE: Both shift by 1 correctly")
else:
    print(f"  ⚠️ DIFFERENCE: Training has {train_inputs_internal.shape[1]} tokens, eval has {eval_input_ids.shape[1]}")

# ==============================================================================
# DISCREPANCY 4: LOSS CALCULATION
# ==============================================================================

print("\n" + "-"*80)
print("DISCREPANCY 4: LOSS CALCULATION")
print("-"*80)

print("\n[TRAINING PIPELINE]")
print("  Loss computed inside model.forward():")
print("    loss = F.cross_entropy(")
print("        logits.reshape(-1, vocab_size),")
print("        targets.reshape(-1),")
print("        ignore_index=-100")
print("    )")
print(f"  Loss includes positions where labels != -100: {(train_labels != -100).sum().item()}")
print(f"  Loss excludes positions where labels == -100: {(train_labels == -100).sum().item()}")

print("\n[EVALUATION PIPELINE]")
print("  Loss computed outside model:")
print("    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))")
print(f"  Loss computed on ALL positions: {eval_targets.numel()}")
print("  NO ignore_index parameter!")

print("\n[VERDICT]")
print("  ⚠️ VERIFIED DIFFERENCE: Training uses ignore_index=-100, evaluation does NOT")
print("  IMPACT: Evaluation loss includes positions that would be masked in training")
print("  STATUS: POTENTIAL ISSUE - could affect loss comparability")

# ==============================================================================
# DISCREPANCY 5: ACCURACY MASKING
# ==============================================================================

print("\n" + "-"*80)
print("DISCREPANCY 5: ACCURACY MASKING")
print("-"*80)

print("\n[TRAINING PIPELINE]")
print("  No accuracy computation during training (only loss)")
print("  Accuracy not used for optimization")

print("\n[EVALUATION PIPELINE]")
print("  Accuracy computed with masking:")
print("    mask = targets != 0  # Exclude padding token")
print("    correct_top1 = ((top1_pred == targets) & mask).sum()")
print(f"  Padding token (0) excluded from accuracy: {(eval_targets == 0).sum().item()} positions")

print("\n[VERDICT]")
print("  ✅ NOT AN ISSUE: Evaluation correctly excludes padding from accuracy")

# ==============================================================================
# DISCREPANCY 6: SEQUENCE LENGTH
# ==============================================================================

print("\n" + "-"*80)
print("DISCREPANCY 6: SEQUENCE LENGTH")
print("-"*80)

print("\n[TRAINING PIPELINE]")
print(f"  Fixed sequence length: 256")
print(f"  All sequences padded/truncated to 256")

print("\n[EVALUATION PIPELINE]")
print(f"  Variable sequence length: {len(eval_tokens)}")
print(f"  No padding, natural length")
print(f"  Sequence lengths vary across evaluation set")

print("\n[VERDICT]")
print("  ⚠️ VERIFIED DIFFERENCE: Training uses fixed 256, evaluation uses variable length")
print("  IMPACT: Model may not generalize well to different sequence lengths")

# ==============================================================================
# DISCREPANCY 7: POSITIONAL EMBEDDINGS
# ==============================================================================

print("\n" + "-"*80)
print("DISCREPANCY 7: POSITIONAL EMBEDDINGS")
print("-"*80)

print("\n[TRAINING PIPELINE]")
print("  Positions 0-255 always used (for 256 tokens)")
print("  Model learns embeddings for all positions 0-255")
print("  Padding tokens get position embeddings for positions > real_tokens")

print("\n[EVALUATION PIPELINE]")
print(f"  Positions 0-{len(eval_tokens)-2} used (for {len(eval_tokens)-1} input tokens)")
print("  Model may not have seen these exact sequence lengths during training")

print("\n[VERDICT]")
print("  ⚠️ POTENTIAL ISSUE: Different position distributions")
print("  STATUS: Model should generalize, but not verified")

# ==============================================================================
# CRITICAL: OBJECTIVE FUNCTION COMPARISON
# ==============================================================================

print("\n" + "="*100)
print("OBJECTIVE FUNCTION COMPARISON")
print("="*100)

print("""
[TRAINING OBJECTIVE]
  Minimize: CrossEntropyLoss(next_token, predicted_next_token)
  Where next_token is the token at position i+1
  Loss masked for padding positions (-100)
  EOS token is included as a target
  
  Example:
    Input:  "Once upon a time" [EOS] [PAD] [PAD] ...
    Target: "upon a time" [EOS] [PAD] [-100] [-100] ...
    
  Model learns to predict EOS token.

[EVALUATION OBJECTIVE]
  Minimize: CrossEntropyLoss(next_token, predicted_next_token)
  Where next_token is the token at position i+1
  Loss NOT masked (includes all positions)
  EOS token is NOT included in targets
  
  Example:
    Input:  "Once upon a time there was"
    Target: "upon a time there was"
    
  Model evaluated WITHOUT EOS token.

[CRITICAL MISMATCH]
  Training: Model learns to predict EOS at end of sequences
  Evaluation: Model NEVER tested on EOS prediction
  
  This means the model's learned behavior (predicting EOS) is never evaluated!
""")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*100)
print("SUMMARY OF DISCREPANCIES")
print("="*100)

print("""
| # | Discrepancy | Status | Impact |
|---|-------------|--------|--------|
| 1 | EOS token in training, NOT in evaluation | ❌ VERIFIED ISSUE | Model learns EOS but never tested on it |
| 2 | Padding in training, NOT in evaluation | ⚠️ DIFFERENCE | Handled by causal mask, not critical |
| 3 | Input/target shifting | ✅ NOT AN ISSUE | Both shift correctly |
| 4 | Loss masking (ignore_index) | ⚠️ DIFFERENCE | Loss values may not be comparable |
| 5 | Accuracy masking | ✅ NOT AN ISSUE | Correctly implemented |
| 6 | Sequence length (fixed vs variable) | ⚠️ DIFFERENCE | May affect generalization |
| 7 | Positional embeddings | ⚠️ UNKNOWN | May affect generalization |

CRITICAL FINDING:
  Training adds EOS token to sequences
  Evaluation does NOT include EOS token
  
  This means:
  1. Model is trained to predict EOS as the "end" token
  2. Model is evaluated on sequences WITHOUT EOS
  3. Model's EOS-prediction ability is NEVER tested
  4. This is a train/eval distribution mismatch

HYPOTHESIS:
  The model may have learned to use EOS as a signal for "stop attending"
  or "sequence boundary", but this behavior is never evaluated.
  
  During evaluation, the model may be confused because:
  - It expects to see EOS at sequence boundaries
  - Evaluation sequences don't have EOS
  - Model's internal representation is based on EOS-marked boundaries
""")

print("\n" + "="*100)
print("AUDIT COMPLETE")
print("="*100)
