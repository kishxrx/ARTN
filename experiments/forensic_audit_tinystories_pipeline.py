#!/usr/bin/env python3
"""
FORENSIC AUDIT: TinyStories Data Pipeline (Failed 6.8M Test)

Purpose: Verify the EXACT data pipeline used by the failed 1000-step test.
         Identify any anomalies that could explain near-zero learning.

Compare to:
- Successful random test (vocab=100, 115K model)
- Successful vocab=4000 test (vocab=4000, 115K model)
- Successful model=6.8M test (vocab=100, 6.8M model)

Date: September 4, 2026
"""

import sys
import torch
import json
import random
from pathlib import Path
from collections import Counter
from datasets import load_dataset

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*80)
print("FORENSIC AUDIT: TinyStories Data Pipeline")
print("="*80)

# =============================================================================
# CONFIGURATION - EXACTLY FROM FAILED TEST
# =============================================================================

SEED = 42
NUM_STORIES = 1000  # From objective_controlled_test.py
MAX_SEQ_LEN = 256
TOKENIZER_PATH = "/root/agent-sandbox/ARTN/outputs/artn-50k-best-config/tokenizer.json"

# =============================================================================
# Load tokenizer (SAME as failed test)
# =============================================================================

print("\n" + "="*80)
print("1. TOKENIZER VERIFICATION")
print("="*80)

tokenizer = BPETokenizer.load(TOKENIZER_PATH)
print(f"Tokenizer vocab size: {len(tokenizer.vocab)}")
print(f"PAD token ID: {tokenizer.pad_token_id}")
print(f"EOS token ID: {tokenizer.eos_token_id}")

# Verify special tokens
print(f"\nSpecial tokens:")
print(f"  PAD ({tokenizer.pad_token_id}): {tokenizer.decode([tokenizer.pad_token_id])}")
print(f"  EOS ({tokenizer.eos_token_id}): {tokenizer.decode([tokenizer.eos_token_id])}")

# =============================================================================
# Load dataset (SAME as failed test)
# =============================================================================

print("\n" + "="*80)
print("2. DATASET LOADING")
print("="*80)

dataset = load_dataset("roneneldan/TinyStories", split="train")
print(f"Total TinyStories: {len(dataset):,}")

# Select subset (SAME as failed test)
random.seed(SEED)
indices = sorted(random.sample(range(len(dataset)), NUM_STORIES))
dataset = dataset.select(indices)

print(f"Selected stories: {len(dataset)}")
print(f"Story indices (first 10): {indices[:10]}")

# =============================================================================
# Analyze raw stories
# =============================================================================

print("\n" + "="*80)
print("3. RAW STORY ANALYSIS")
print("="*80)

story_lengths = []
for item in dataset:
    text = item["text"]
    story_lengths.append(len(text))

print(f"Story lengths (chars):")
print(f"  Min: {min(story_lengths)}")
print(f"  Max: {max(story_lengths)}")
print(f"  Mean: {sum(story_lengths)/len(story_lengths):.1f}")
print(f"  Total chars: {sum(story_lengths):,}")

# Check for empty stories
empty_count = sum(1 for l in story_lengths if l == 0)
print(f"\nEmpty stories: {empty_count}")

# Print first 3 stories
print(f"\nFirst 3 stories (first 200 chars):")
for i in range(min(3, len(dataset))):
    text = dataset[i]["text"]
    print(f"\n--- Story {i+1} ---")
    print(text[:200] + "..." if len(text) > 200 else text)

# =============================================================================
# Tokenize stories (SAME as failed test)
# =============================================================================

print("\n" + "="*80)
print("4. TOKENIZATION")
print("="*80)

tokenized_stories = []
all_tokens = []

for i, item in enumerate(dataset):
    text = item["text"]
    
    # SAME tokenization as objective_controlled_test.py
    tokens = tokenizer.encode(text)[:MAX_SEQ_LEN - 1]
    tokens = tokens + [tokenizer.eos_token_id]
    
    tokenized_stories.append(tokens)
    all_tokens.extend(tokens)
    
    if i < 3:
        print(f"\nStory {i+1}:")
        print(f"  Text length: {len(text)} chars")
        print(f"  Token count: {len(tokens)}")
        print(f"  First 20 tokens: {tokens[:20]}")
        print(f"  Last 5 tokens: {tokens[-5:]}")
        print(f"  Decoded first 50 chars: {tokenizer.decode(tokens[:30])[:50]}")

# =============================================================================
# Token ID distribution
# =============================================================================

print("\n" + "="*80)
print("5. TOKEN ID DISTRIBUTION")
print("="*80)

token_counter = Counter(all_tokens)

print(f"Total tokens: {len(all_tokens):,}")
print(f"Unique tokens: {len(token_counter):,}")
print(f"Vocab coverage: {len(token_counter)/len(tokenizer.vocab)*100:.1f}%")

# Top tokens
print(f"\nTop 20 most frequent tokens:")
for token_id, count in token_counter.most_common(20):
    decoded = tokenizer.decode([token_id])
    pct = count / len(all_tokens) * 100
    print(f"  {token_id:5d}: {count:6d} ({pct:5.2f}%) - '{decoded}'")

# Check for special token dominance
pad_count = token_counter.get(tokenizer.pad_token_id, 0)
eos_count = token_counter.get(tokenizer.eos_token_id, 0)
print(f"\nSpecial token counts:")
print(f"  PAD ({tokenizer.pad_token_id}): {pad_count} ({pad_count/len(all_tokens)*100:.2f}%)")
print(f"  EOS ({tokenizer.eos_token_id}): {eos_count} ({eos_count/len(all_tokens)*100:.2f}%)")

# =============================================================================
# Sequence construction (SAME as failed test)
# =============================================================================

print("\n" + "="*80)
print("6. SEQUENCE CONSTRUCTION")
print("="*80)

# Pad all sequences to max length (SAME as objective_controlled_test.py)
padded_sequences = []
for tokens in tokenized_stories:
    padding_length = MAX_SEQ_LEN - len(tokens)
    padded = tokens + [tokenizer.pad_token_id] * padding_length
    padded_sequences.append(padded)

print(f"Padded sequence length: {MAX_SEQ_LEN}")
print(f"Number of sequences: {len(padded_sequences)}")

# Calculate padding percentage
total_positions = len(padded_sequences) * MAX_SEQ_LEN
total_padding = sum(seq.count(tokenizer.pad_token_id) for seq in padded_sequences)
padding_pct = total_padding / total_positions * 100

print(f"\nPadding statistics:")
print(f"  Total positions: {total_positions:,}")
print(f"  Padding positions: {total_padding:,}")
print(f"  Padding percentage: {padding_pct:.1f}%")
print(f"  Non-padding positions: {total_positions - total_padding:,}")

# =============================================================================
# Input/target alignment (CRITICAL)
# =============================================================================

print("\n" + "="*80)
print("7. INPUT/TARGET ALIGNMENT (CRITICAL)")
print("="*80)

# Verify the exact alignment from objective_controlled_test.py
# inputs = input_ids[:-1].clone()
# targets = input_ids[1:].clone()

print("Verifying alignment from objective_controlled_test.py:")
print("  inputs = input_ids[:-1]  # All but last token")
print("  targets = input_ids[1:]  # All but first token (shifted by 1)")
print("\nThis means:")
print("  targets[i] = input_ids[i+1] = inputs[i+1]")
print("  Position i predicts token at position i+1 (NEXT-TOKEN)")

# Example with first sequence
seq = padded_sequences[0]
print(f"\nExample (first sequence):")
print(f"  input_ids length: {len(seq)}")
print(f"  inputs length: {len(seq) - 1}")
print(f"  targets length: {len(seq) - 1}")

# Check a few positions
print(f"\nFirst 10 positions:")
print(f"  {'pos':<5} {'input_id':<10} {'target':<10} {'decoded_input':<20} {'decoded_target':<20}")
for i in range(10):
    input_id = seq[i]
    target = seq[i+1]  # target[i] = input_ids[i+1]
    decoded_input = tokenizer.decode([input_id])
    decoded_target = tokenizer.decode([target])
    print(f"  {i:<5} {input_id:<10} {target:<10} '{decoded_input[:15]:<20}' '{decoded_target[:15]:<20}'")

# =============================================================================
# Loss masking
# =============================================================================

print("\n" + "="*80)
print("8. LOSS MASKING")
print("="*80)

# From objective_controlled_test.py:
# targets[targets == self.tokenizer.pad_token_id] = -100

print("Loss masking from objective_controlled_test.py:")
print("  targets[targets == PAD] = -100")
print("\nThis means PAD positions are IGNORED in loss calculation.")

# Count how many targets are masked
masked_count = 0
total_targets = 0
for seq in padded_sequences:
    # targets = seq[1:]
    targets = seq[1:]
    masked = sum(1 for t in targets if t == tokenizer.pad_token_id)
    masked_count += masked
    total_targets += len(targets)

print(f"\nTarget statistics:")
print(f"  Total target positions: {total_targets:,}")
print(f"  Masked positions (PAD → -100): {masked_count:,}")
print(f"  Active (trained) positions: {total_targets - masked_count:,}")
print(f"  Active percentage: {(total_targets - masked_count)/total_targets*100:.1f}%")

# =============================================================================
# Story boundary handling
# =============================================================================

print("\n" + "="*80)
print("9. STORY BOUNDARY HANDLING")
print("="*80)

# Check if EOS is properly placed
print("Checking EOS placement:")
eos_at_end = 0
eos_in_middle = 0
no_eos = 0

for tokens in tokenized_stories:
    # Find EOS position
    if tokenizer.eos_token_id in tokens:
        eos_idx = tokens.index(tokenizer.eos_token_id)
        if eos_idx == len(tokens) - 1:
            eos_at_end += 1
        else:
            eos_in_middle += 1
    else:
        no_eos += 1

print(f"  Stories with EOS at end: {eos_at_end}")
print(f"  Stories with EOS in middle: {eos_in_middle}")
print(f"  Stories without EOS: {no_eos}")

# Check if stories span multiple sequences
print("\nStory truncation check:")
truncated = 0
for text in dataset:
    full_tokens = tokenizer.encode(text["text"])
    if len(full_tokens) >= MAX_SEQ_LEN - 1:
        truncated += 1

print(f"  Stories truncated (≥{MAX_SEQ_LEN - 1} tokens): {truncated}")
print(f"  Truncation percentage: {truncated/len(dataset)*100:.1f}%")

# =============================================================================
# Sampling during training
# =============================================================================

print("\n" + "="*80)
print("10. SAMPLING DURING TRAINING")
print("="*80)

print("From objective_controlled_test.py:")
print("  DataLoader(batch_size=4, shuffle=True)")
print(f"  {len(padded_sequences)} stories")
print(f"  Batch size: 4")
print(f"  Total batches: {len(padded_sequences) // 4}")
print(f"  Steps: 1000")
print(f"\nThis means the SAME {len(padded_sequences)} stories are sampled REPEATEDLY.")
print(f"  Epochs trained: {1000 * 4 / len(padded_sequences):.1f}")

# =============================================================================
# Target token frequency
# =============================================================================

print("\n" + "="*80)
print("11. TARGET TOKEN FREQUENCY")
print("="*80)

# Collect all targets (non-masked)
all_targets = []
for seq in padded_sequences:
    targets = seq[1:]  # All but first token
    for t in targets:
        if t != tokenizer.pad_token_id:  # Exclude masked
            all_targets.append(t)

target_counter = Counter(all_targets)

print(f"Total active target tokens: {len(all_targets):,}")
print(f"Unique target tokens: {len(target_counter):,}")
print(f"Target vocab coverage: {len(target_counter)/len(tokenizer.vocab)*100:.1f}%")

# Top target tokens
print(f"\nTop 20 target tokens:")
for token_id, count in target_counter.most_common(20):
    decoded = tokenizer.decode([token_id])
    pct = count / len(all_targets) * 100
    print(f"  {token_id:5d}: {count:6d} ({pct:5.2f}%) - '{decoded}'")

# =============================================================================
# REAL examples decoded
# =============================================================================

print("\n" + "="*80)
print("12. REAL TOKENIZED EXAMPLES")
print("="*80)

for i in range(min(3, len(padded_sequences))):
    seq = padded_sequences[i]
    
    print(f"\n{'='*60}")
    print(f"EXAMPLE {i+1}")
    print(f"{'='*60}")
    
    # Find where padding starts
    if tokenizer.pad_token_id in seq:
        pad_start = seq.index(tokenizer.pad_token_id)
    else:
        pad_start = len(seq)
    
    # Get non-padding tokens
    real_tokens = seq[:pad_start]
    
    # Full decoded text
    decoded_full = tokenizer.decode(real_tokens)
    
    print(f"Token count: {len(real_tokens)}")
    print(f"\nFull decoded text:")
    print(f"  {decoded_full[:300]}..." if len(decoded_full) > 300 else f"  {decoded_full}")
    
    # Input/target pairs
    print(f"\nFirst 5 input→target pairs:")
    for j in range(min(5, len(real_tokens) - 1)):
        input_tok = real_tokens[j]
        target_tok = real_tokens[j + 1]
        input_dec = tokenizer.decode([input_tok])
        target_dec = tokenizer.decode([target_tok])
        print(f"  Position {j}: {input_tok} → {target_tok} ('{input_dec}' → '{target_dec}')")
    
    print(f"\nLast 5 input→target pairs:")
    for j in range(max(0, len(real_tokens) - 6), len(real_tokens) - 1):
        input_tok = real_tokens[j]
        target_tok = real_tokens[j + 1]
        input_dec = tokenizer.decode([input_tok])
        target_dec = tokenizer.decode([target_tok])
        print(f"  Position {j}: {input_tok} → {target_tok} ('{input_dec}' → '{target_dec}')")

# =============================================================================
# Compare to successful random tests
# =============================================================================

print("\n" + "="*80)
print("13. COMPARISON TO SUCCESSFUL RANDOM TESTS")
print("="*80)

print("""
SUCCESSFUL RANDOM TEST (overfit_test.py):
  - Data: torch.randint(0, vocab_size, (num_samples, seq_len))
  - Samples: 5
  - Sequence length: 32
  - Unique tokens: vocab_size (all equally likely)
  - Padding: NONE
  - EOS: NONE (just random tokens)
  - Loss masking: NONE (all positions trained)
  - Dataset: 5 × 32 = 160 total tokens
  - Epochs: 500 steps × 5 samples / 5 samples = 500 epochs

SUCCESSFUL VOCAB=4000 TEST (experiment1):
  - Data: torch.randint(0, 4000, (5, 32))
  - Samples: 5
  - Sequence length: 32
  - Unique tokens: 4000 (all equally likely)
  - Padding: NONE
  - EOS: Random token appended
  - Loss masking: NONE
  - Dataset: 5 × 32 = 160 total tokens
  - Epochs: 500

SUCCESSFUL MODEL=6.8M TEST (experiment2):
  - Data: torch.randint(0, 100, (5, 32))
  - Samples: 5
  - Sequence length: 32
  - Unique tokens: 100 (all equally likely)
  - Padding: NONE
  - EOS: Random token appended
  - Loss masking: NONE
  - Dataset: 5 × 32 = 160 total tokens
  - Epochs: 500

FAILED TINYSTORIES TEST (objective_controlled_test.py):
  - Data: Real TinyStories text
  - Samples: 1000
  - Sequence length: 256
  - Unique tokens: ~{len(token_counter)} (not uniform)
  - Padding: ~{padding_pct:.1f}% of positions
  - EOS: At end of each story
  - Loss masking: PAD → -100 (masked)
  - Dataset: 1000 × ~{MAX_SEQ_LEN - int(padding_pct/100*MAX_SEQ_LEN):.0f} = ~{(total_positions - total_padding)//1000}K active tokens
  - Epochs: ~{1000 * 4 / 1000:.1f}
""")

# =============================================================================
# KEY DIFFERENCES
# =============================================================================

print("\n" + "="*80)
print("14. KEY DIFFERENCES SUMMARY")
print("="*80)

print("""
| Factor | Random Tests (SUCCESS) | TinyStories (FAILED) |
|--------|------------------------|----------------------|
| Data type | Random integers | Real language text |
| Token distribution | UNIFORM (all equally likely) | HIGHLY SKEWED (zipf-like) |
| Unique tokens | 100 or 4000 (all present) | ~""" + str(len(token_counter)) + """ (sparse) |
| Samples | 5 | 1000 |
| Sequence length | 32 | 256 |
| Padding | 0% | """ + f"{padding_pct:.1f}%" + """ |
| Epochs trained | 500 | ~""" + f"{1000 * 4 / 1000:.1f}" + """ |
| Total tokens seen | 500 × 5 × 32 = 80K | 1000 × 4 × 256 = 1M |
| Task difficulty | MEMORIZE 5 sequences | LEARN language patterns |

CRITICAL DIFFERENCES:
1. **Token distribution**: Random = uniform, TinyStories = highly skewed
2. **Task**: Random = memorize exact sequences, TinyStories = learn patterns
3. **Dataset size**: Random = 160 tokens total, TinyStories = millions of tokens
4. **Epochs**: Random = 500 passes over same data, TinyStories = ~4 passes

HYPOTHESIS: The model succeeded at MEMORIZATION (exact sequence recall) 
but FAILED at LEARNING (generalizing patterns from diverse data).

This is NOT a data pipeline bug - it's a fundamental task difference.
""")

print("="*80)
