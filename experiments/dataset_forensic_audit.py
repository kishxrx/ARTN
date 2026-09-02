#!/usr/bin/env python3
"""
Forensic Audit of Training Dataset and Preprocessing Pipeline

Investigates the shared factors that could explain why both FP32 and Ternary
learn/memorize training data but generalize poorly.
"""

import torch
import sys
import json
from pathlib import Path
from collections import Counter
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*80)
print("FORENSIC AUDIT: Training Dataset and Preprocessing Pipeline")
print("="*80)

# ============================================================
# STEP 1: Load Training and Held-Out Sentences
# ============================================================

print(f"\n{'='*80}")
print("1. LOADING DATASET")
print("="*80)

# The exact 20 stories used in training
TRAINING_SENTENCES = [
    "Once upon a time there was a little girl named Sarah.",
    "The little girl loved to play in the garden every day.",
    "One day she found a magic flower in the garden.",
    "The flower glowed with beautiful colors.",
    "Sarah picked the flower and made a wish.",
    "Suddenly a fairy appeared before her eyes.",
    "The fairy granted Sarah three special wishes.",
    "Sarah wished for happiness and peace.",
    "The fairy smiled and disappeared into the light.",
    "From that day on Sarah was always happy.",
    "There was once a brave little boy named Tom.",
    "Tom loved to explore the forest behind his house.",
    "One day he found a secret path in the woods.",
    "The path led to a hidden treasure chest.",
    "Tom opened the chest and found golden coins.",
    "He shared the treasure with his family.",
    "Everyone was proud of the brave boy.",
    "Tom learned that adventures are exciting.",
    "He promised to have more adventures.",
    "The end of the story made everyone smile.",
]

# The 30 held-out stories used in evaluation
HELD_OUT_SENTENCES = [
    "Once there was a little boy named Tim who loved animals.",
    "Tim had a pet dog named Max who was very playful.",
    "Every morning Tim and Max would go for a walk in the park.",
    "One sunny day they saw a beautiful butterfly in the garden.",
    "Tim tried to catch the butterfly but it flew away quickly.",
    "Max barked happily and chased the butterfly around the flowers.",
    "Tim laughed and clapped his hands with joy.",
    "The butterfly landed on a red rose and stayed very still.",
    "Tim watched quietly and smiled at the beautiful sight.",
    "He learned that watching animals was better than catching them.",
    "There was a little girl named Emma who loved to paint.",
    "Emma had many colorful paints and brushes in her room.",
    "She painted pictures of flowers and trees and birds.",
    "Her mother hung the paintings on the wall in the kitchen.",
    "Everyone said Emma was a very talented artist.",
    "Emma felt proud and wanted to paint more pictures.",
    "She painted a big picture of her family for her birthday.",
    "Her parents were very happy with the special gift.",
    "Emma knew that practice would make her even better.",
    "She promised to paint something new every single day.",
    "A little cat named Whiskers lived in a cozy house.",
    "Whiskers liked to sleep in the warm sunshine by the window.",
    "Sometimes she chased mice in the garden behind the house.",
    "Her owner gave her fresh milk and fish every evening.",
    "Whiskers purred happily when she got her favorite treats.",
    "She was a very happy cat with a loving home.",
    "All the neighbors knew Whiskers and loved to pet her.",
    "She was the friendliest cat in the whole neighborhood.",
    "Whiskers taught everyone that cats can be very loving pets.",
    "She lived a long and happy life with her family.",
]

print(f"\nNumber of training stories: {len(TRAINING_SENTENCES)}")
print(f"Number of held-out stories: {len(HELD_OUT_SENTENCES)}")

# ============================================================
# STEP 2: Load Tokenizer
# ============================================================

print(f"\n{'='*80}")
print("2. LOADING TOKENIZER")
print("="*80)

tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
vocab_size = tokenizer.vocab_size
print(f"Tokenizer vocabulary size: {vocab_size}")

# ============================================================
# STEP 3: Dataset Statistics
# ============================================================

print(f"\n{'='*80}")
print("3. DATASET STATISTICS")
print("="*80)

# Analyze training data
print("\n--- TRAINING DATA ---")

# 1. Number of stories
num_training_stories = len(TRAINING_SENTENCES)
print(f"\n1. Number of stories: {num_training_stories}")

# 2. Total raw characters
total_training_chars = sum(len(s) for s in TRAINING_SENTENCES)
print(f"2. Total raw characters: {total_training_chars}")

# Tokenize all training sentences
training_token_sequences = []
for sentence in TRAINING_SENTENCES:
    tokens = tokenizer.encode(sentence)
    training_token_sequences.append(tokens)

# 3. Total token count before padding
total_training_tokens = sum(len(seq) for seq in training_token_sequences)
print(f"3. Total token count (before padding): {total_training_tokens}")

# 4. Will be calculated after we understand the training windowing

# 5. Average/min/max story token length
story_lengths = [len(seq) for seq in training_token_sequences]
avg_story_len = sum(story_lengths) / len(story_lengths)
min_story_len = min(story_lengths)
max_story_len = max(story_lengths)
print(f"5. Story token lengths:")
print(f"   Average: {avg_story_len:.1f} tokens")
print(f"   Min: {min_story_len} tokens")
print(f"   Max: {max_story_len} tokens")
print(f"   Individual lengths: {story_lengths}")

# 6. Number of training sequences (depends on batch creation)
# For now, assume each story is one sequence (no windowing)
num_training_sequences = len(training_token_sequences)
print(f"6. Number of training sequences: {num_training_sequences}")

# 7. Number of unique tokens
all_training_tokens = []
for seq in training_token_sequences:
    all_training_tokens.extend(seq)

unique_training_tokens = set(all_training_tokens)
num_unique_training_tokens = len(unique_training_tokens)
print(f"7. Number of unique tokens used: {num_unique_training_tokens}")

# 8. Percentage of vocabulary used
vocab_usage_pct = 100 * num_unique_training_tokens / vocab_size
print(f"8. Vocabulary usage: {num_unique_training_tokens}/{vocab_size} ({vocab_usage_pct:.1f}%)")

# 9. Token frequency distribution
token_counter = Counter(all_training_tokens)
print(f"9. Token frequency distribution:")
print(f"   Total token occurrences: {len(all_training_tokens)}")
print(f"   Unique tokens: {len(token_counter)}")
print(f"   Average frequency: {len(all_training_tokens)/len(token_counter):.2f}")

# 10. Most frequent and rarest tokens
most_common = token_counter.most_common(20)
rarest = token_counter.most_common()[:-21:-1]

print(f"\n10. Most frequent tokens (top 20):")
for rank, (token_id, count) in enumerate(most_common, 1):
    token_str = tokenizer.decode([token_id])
    pct = 100 * count / len(all_training_tokens)
    print(f"   {rank:2}. Token {token_id:4} '{token_str:15}' - {count:3} times ({pct:.1f}%)")

print(f"\n   Rarest tokens (bottom 20):")
for rank, (token_id, count) in enumerate(reversed(rarest), 1):
    token_str = tokenizer.decode([token_id])
    print(f"   {rank:2}. Token {token_id:4} '{token_str:15}' - {count} time(s)")

# 11. Percentage accounted for by top tokens
top_10_count = sum(count for _, count in most_common[:10])
top_20_count = sum(count for _, count in most_common[:20])
top_50_count = sum(count for _, count in token_counter.most_common(50))

print(f"\n11. Token concentration:")
print(f"   Top 10 tokens: {100*top_10_count/len(all_training_tokens):.1f}% of all tokens")
print(f"   Top 20 tokens: {100*top_20_count/len(all_training_tokens):.1f}% of all tokens")
print(f"   Top 50 tokens: {100*top_50_count/len(all_training_tokens):.1f}% of all tokens")

# ============================================================
# STEP 4: Analyze Held-Out Data
# ============================================================

print(f"\n{'='*80}")
print("4. HELD-OUT DATA ANALYSIS")
print("="*80)

# Tokenize held-out
heldout_token_sequences = []
for sentence in HELD_OUT_SENTENCES:
    tokens = tokenizer.encode(sentence)
    heldout_token_sequences.append(tokens)

total_heldout_tokens = sum(len(seq) for seq in heldout_token_sequences)
all_heldout_tokens = []
for seq in heldout_token_sequences:
    all_heldout_tokens.extend(seq)

unique_heldout_tokens = set(all_heldout_tokens)

print(f"Total held-out tokens: {total_heldout_tokens}")
print(f"Unique held-out tokens: {len(unique_heldout_tokens)}")
print(f"Held-out vocab usage: {len(unique_heldout_tokens)}/{vocab_size} ({100*len(unique_heldout_tokens)/vocab_size:.1f}%)")

# Vocabulary overlap
vocab_overlap = unique_training_tokens & unique_heldout_tokens
training_only = unique_training_tokens - unique_heldout_tokens
heldout_only = unique_heldout_tokens - unique_training_tokens

print(f"\nVocabulary overlap:")
print(f"   Shared tokens: {len(vocab_overlap)}")
print(f"   Training-only tokens: {len(training_only)}")
print(f"   Held-out-only tokens: {len(heldout_only)}")
print(f"   Overlap percentage: {100*len(vocab_overlap)/len(unique_training_tokens):.1f}% of training tokens")

# ============================================================
# STEP 5: Training Sequence Reconstruction
# ============================================================

print(f"\n{'='*80}")
print("5. TRAINING SEQUENCE RECONSTRUCTION")
print("="*80)

print("\n--- Example Training Sequences (First 5 stories) ---")

for i in range(min(5, len(TRAINING_SENTENCES))):
    sentence = TRAINING_SENTENCES[i]
    tokens = training_token_sequences[i]
    
    print(f"\nStory {i+1}:")
    print(f"  Raw text: \"{sentence}\"")
    print(f"  Token count: {len(tokens)}")
    print(f"  Token IDs: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
    print(f"  Decoded: \"{tokenizer.decode(tokens[:20])}{'...' if len(tokens) > 20 else ''}\"")
    
    # Show input → target construction
    print(f"  Input → Target pairs (first 8):")
    for j in range(min(8, len(tokens)-1)):
        input_token = tokens[j]
        target_token = tokens[j+1]
        input_word = tokenizer.decode([input_token])
        target_word = tokenizer.decode([target_token])
        print(f"    Position {j}: {input_token:4} '{input_word:10}' → {target_token:4} '{target_word:10}'")

# ============================================================
# STEP 6: Padding Analysis
# ============================================================

print(f"\n{'='*80}")
print("6. PADDING ANALYSIS")
print("="*80)

# Check how training sequences were padded
# From the training code: batch size 8, so sequences padded to max length in batch
# For 20 stories, they would be padded

max_story_len_all = max(len(seq) for seq in training_token_sequences)
print(f"Max story length (all stories): {max_story_len_all} tokens")

# Calculate padding if all stories padded to max length
total_padded_tokens = num_training_stories * max_story_len_all
padding_tokens = total_padded_tokens - total_training_tokens
padding_pct = 100 * padding_tokens / total_padded_tokens

print(f"\nIf padded to max length ({max_story_len_all}):")
print(f"  Total tokens with padding: {total_padded_tokens}")
print(f"  Real tokens: {total_training_tokens} ({100-padding_pct:.1f}%)")
print(f"  Padding tokens: {padding_tokens} ({padding_pct:.1f}%)")

# Held-out padding
max_heldout_len = max(len(seq) for seq in heldout_token_sequences)
total_heldout_padded = len(HELD_OUT_SENTENCES) * max_heldout_len
heldout_padding = total_heldout_padded - total_heldout_tokens
heldout_padding_pct = 100 * heldout_padding / total_heldout_padded

print(f"\nHeld-out padding:")
print(f"  Max held-out length: {max_heldout_len} tokens")
print(f"  Total padded: {total_heldout_padded}")
print(f"  Real tokens: {total_heldout_tokens} ({100-heldout_padding_pct:.1f}%)")
print(f"  Padding tokens: {heldout_padding} ({heldout_padding_pct:.1f}%)")

# ============================================================
# STEP 7: Training Exposure Analysis
# ============================================================

print(f"\n{'='*80}")
print("7. TRAINING EXPOSURE ANALYSIS")
print("="*80)

# Calculate training exposure for different checkpoint steps
print("\n--- Training Exposure ---")

# Known training configurations
training_configs = [
    ("5k steps", 5000),
    ("50k steps", 50000),
]

batch_size = 8  # From training config

for config_name, steps in training_configs:
    total_samples = steps * batch_size
    total_tokens_seen = total_samples * max_story_len_all  # Including padding
    
    # Epochs
    if num_training_sequences > 0:
        epochs = total_samples / num_training_sequences
    else:
        epochs = 0
    
    # Unique tokens seen
    unique_tokens_seen = total_samples * avg_story_len  # Approximate
    
    print(f"\n{config_name}:")
    print(f"  Total optimization samples: {total_samples}")
    print(f"  Total tokens seen (with padding): {total_tokens_seen}")
    print(f"  Real tokens seen (estimated): {int(total_samples * avg_story_len)}")
    print(f"  Epochs over dataset: {epochs:.1f}")
    print(f"  Times each story seen: {epochs:.1f}")

# ============================================================
# STEP 8: Duplicate/Near-Duplicate Analysis
# ============================================================

print(f"\n{'='*80}")
print("8. DUPLICATE SEQUENCE ANALYSIS")
print("="*80)

# Check for exact duplicates
sentence_set = set()
duplicates = []
for i, sentence in enumerate(TRAINING_SENTENCES):
    if sentence in sentence_set:
        duplicates.append((i, sentence))
    sentence_set.add(sentence)

if duplicates:
    print(f"✗ Found {len(duplicates)} duplicate sentences")
    for idx, sent in duplicates:
        print(f"   Index {idx}: \"{sent[:40]}...\"")
else:
    print(f"✓ No exact duplicate sentences found")

# Check for similar starting phrases
starts = []
for i, sentence in enumerate(TRAINING_SENTENCES):
    start = sentence.split()[:3]  # First 3 words
    starts.append((' '.join(start), i))

from collections import Counter
start_counter = Counter(s[0] for s in starts)
common_starts = [(start, count) for start, count in start_counter.items() if count > 1]

if common_starts:
    print(f"\n⚠ Found {len(common_starts)} repeated starting phrases:")
    for start, count in common_starts:
        indices = [i for s, i in starts if s == start]
        print(f"   \"{start}...\" appears {count} times at indices {indices}")

# Check for token-level subsequence duplicates
print(f"\nChecking for repeated token n-grams...")

# Extract all 5-grams
ngrams_5 = []
for seq in training_token_sequences:
    for i in range(len(seq) - 4):
        ngram = tuple(seq[i:i+5])
        ngrams_5.append(ngram)

ngram_counter = Counter(ngrams_5)
repeated_ngrams = [(ng, count) for ng, count in ngram_counter.items() if count > 1]

print(f"  Total 5-grams: {len(ngrams_5)}")
print(f"  Unique 5-grams: {len(ngram_counter)}")
print(f"  Repeated 5-grams: {len(repeated_ngrams)}")

if repeated_ngrams:
    print(f"\n  Most repeated 5-grams (top 5):")
    for ngram, count in sorted(repeated_ngrams, key=lambda x: x[1], reverse=True)[:5]:
        decoded = tokenizer.decode(list(ngram))
        print(f"    '{decoded}' - {count} times")

# ============================================================
# STEP 9: Contiguity Check
# ============================================================

print(f"\n{'='*80}")
print("9. TEXT CONTIGUITY CHECK")
print("="*80)

print("\nChecking if training preserves contiguous text...")

# For each story, verify that tokens decode back to original
contiguity_ok = True
for i, (sentence, tokens) in enumerate(zip(TRAINING_SENTENCES, training_token_sequences)):
    decoded = tokenizer.decode(tokens)
    
    # Check if decoded matches original (allowing for tokenization artifacts)
    # The decoded version should contain the original sentence
    if sentence not in decoded and decoded not in sentence:
        # Try a more lenient check
        sentence_clean = sentence.replace('.', '').replace(',', '').lower()
        decoded_clean = decoded.replace('.', '').replace(',', '').lower()
        if sentence_clean not in decoded_clean:
            print(f"  ⚠ Story {i}: Contiguity issue")
            print(f"     Original: \"{sentence[:50]}...\"")
            print(f"     Decoded:  \"{decoded[:50]}...\"")
            contiguity_ok = False

if contiguity_ok:
    print("  ✓ All stories decode correctly - text is contiguous")

# ============================================================
# STEP 10: Data Representativeness Check
# ============================================================

print(f"\n{'='*80}")
print("10. DATA REPRESENTATIVENESS")
print("="*80)

print("\nAnalyzing whether 20 stories are representative...")

# Total tokens
print(f"\nDataset size:")
print(f"  Total training tokens: {total_training_tokens}")
print(f"  Total held-out tokens: {total_heldout_tokens}")
print(f"  Ratio (train/heldout): {total_training_tokens/total_heldout_tokens:.2f}x")

# Vocabulary coverage
print(f"\nVocabulary coverage:")
print(f"  Training uses {num_unique_training_tokens} unique tokens")
print(f"  Held-out uses {len(unique_heldout_tokens)} unique tokens")
print(f"  Vocab size: {vocab_size}")
print(f"  Coverage: {100*num_unique_training_tokens/vocab_size:.1f}% of vocabulary")

# Token type analysis
print(f"\nToken type distribution:")

# Identify common words, names, etc.
common_words = ['the', 'a', 'was', 'to', 'and', 'of', 'in', 'is', 'it']
name_tokens = []
for token_id in unique_training_tokens:
    decoded = tokenizer.decode([token_id]).strip().lower()
    if decoded and decoded[0].isupper():
        name_tokens.append(decoded)

print(f"  Names found: {name_tokens[:10]}")

# Calculate vocabulary diversity
type_token_ratio = num_unique_training_tokens / total_training_tokens
print(f"\n  Type-Token Ratio: {type_token_ratio:.3f}")
print(f"  (Higher = more diverse vocabulary)")

if type_token_ratio > 0.5:
    print(f"  ⚠ Very high TTR - dataset may be too diverse for its size")
elif type_token_ratio < 0.2:
    print(f"  ⚠ Low TTR - vocabulary may be too limited")
else:
    print(f"  ✓ Reasonable TTR")

# ============================================================
# STEP 11: Effective Dataset Size
# ============================================================

print(f"\n{'='*80}")
print("11. EFFECTIVE DATASET SIZE")
print("="*80)

print("\nComparing to typical language model training...")

# Typical small LM training
typical_tokens = {
    "GPT-2 small": 1024 * 1000 * 1000,  # ~1B tokens
    "GPT-2 tiny": 100 * 1000 * 1000,    # ~100M tokens
    "Minimal LM": 1 * 1000 * 1000,      # ~1M tokens
    "Toy dataset": 100 * 1000,           # ~100K tokens
}

print(f"\nOur dataset: {total_training_tokens} tokens")
print(f"\nComparison:")
for name, tokens in typical_tokens.items():
    ratio = tokens / total_training_tokens
    print(f"  {name}: {tokens:,} tokens ({ratio:.0f}x larger)")

print(f"\n⚠ Our dataset is {total_training_tokens} tokens")
print(f"   This is {1000*1000/total_training_tokens:.0f}x smaller than 'minimal' LM training")
print(f"   This is {100*1000/total_training_tokens:.0f}x smaller than 'toy' dataset")

# ============================================================
# STEP 12: Classification Summary
# ============================================================

print(f"\n{'='*80}")
print("12. CLASSIFICATION SUMMARY")
print("="*80)

classifications = []

# Issue 1: Dataset size
if total_training_tokens < 1000:
    classifications.append(("Dataset size", "VERIFIED PROBLEM", f"Only {total_training_tokens} tokens"))
else:
    classifications.append(("Dataset size", "POSSIBLE", f"{total_training_tokens} tokens may be insufficient"))

# Issue 2: Vocabulary usage
if vocab_usage_pct < 5:
    classifications.append(("Vocabulary usage", "VERIFIED PROBLEM", f"Only {vocab_usage_pct:.1f}% of vocab used"))
else:
    classifications.append(("Vocabulary usage", "VERIFIED CORRECT", f"{vocab_usage_pct:.1f}% of vocab used"))

# Issue 3: Padding in training
if padding_pct > 30:
    classifications.append(("Padding in training", "VERIFIED PROBLEM", f"{padding_pct:.1f}% padding"))
elif padding_pct > 10:
    classifications.append(("Padding in training", "POSSIBLE", f"{padding_pct:.1f}% padding"))
else:
    classifications.append(("Padding in training", "UNLIKELY", f"Only {padding_pct:.1f}% padding"))

# Issue 4: Data contiguity
if contiguity_ok:
    classifications.append(("Text contiguity", "VERIFIED CORRECT", "Stories decode correctly"))
else:
    classifications.append(("Text contiguity", "VERIFIED PROBLEM", "Stories not contiguous"))

# Issue 5: Duplicate data
if len(duplicates) > 0:
    classifications.append(("Duplicate stories", "VERIFIED PROBLEM", f"{len(duplicates)} duplicates"))
elif len(repeated_ngrams) > 10:
    classifications.append(("Repeated sequences", "POSSIBLE", f"{len(repeated_ngrams)} repeated 5-grams"))
else:
    classifications.append(("Duplicate data", "VERIFIED CORRECT", "No duplicates"))

# Issue 6: Training/held-out overlap
overlap_pct = 100*len(vocab_overlap)/len(unique_training_tokens)
if overlap_pct > 90:
    classifications.append(("Train/heldout overlap", "VERIFIED PROBLEM", f"{overlap_pct:.1f}% vocab overlap"))
elif overlap_pct > 50:
    classifications.append(("Train/heldout overlap", "POSSIBLE", f"{overlap_pct:.1f}% vocab overlap"))
else:
    classifications.append(("Train/heldout overlap", "UNLIKELY", f"{overlap_pct:.1f}% vocab overlap"))

# Issue 7: Epoch count
if epochs > 100:
    classifications.append(("Over-training", "VERIFIED PROBLEM", f"{epochs:.0f} epochs - severe overfitting risk"))
elif epochs > 10:
    classifications.append(("Over-training", "POSSIBLE", f"{epochs:.0f} epochs"))
else:
    classifications.append(("Over-training", "UNLIKELY", f"{epochs:.0f} epochs"))

print("\nIssue Classification:")
for issue, status, evidence in classifications:
    print(f"  {issue:25} [{status:20}] {evidence}")

# ============================================================
# STEP 13: Save Results
# ============================================================

print(f"\n{'='*80}")
print("13. SAVING RESULTS")
print("="*80)

results = {
    "dataset_stats": {
        "num_stories": num_training_stories,
        "total_chars": total_training_chars,
        "total_tokens": total_training_tokens,
        "avg_story_len": avg_story_len,
        "min_story_len": min_story_len,
        "max_story_len": max_story_len,
        "unique_tokens": num_unique_training_tokens,
        "vocab_size": vocab_size,
        "vocab_usage_pct": vocab_usage_pct,
        "type_token_ratio": type_token_ratio,
    },
    "training_exposure": {
        "5k": {"samples": 5000*8, "epochs": 5000*8/num_training_stories},
        "50k": {"samples": 50000*8, "epochs": 50000*8/num_training_stories},
    },
    "padding": {
        "training_padding_pct": padding_pct,
        "heldout_padding_pct": heldout_padding_pct,
    },
    "overlap": {
        "shared_tokens": len(vocab_overlap),
        "training_only": len(training_only),
        "heldout_only": len(heldout_only),
        "overlap_pct": overlap_pct,
    },
    "classifications": {issue: status for issue, status, _ in classifications},
}

output_path = Path("experiments/results/dataset_forensic_audit.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {output_path}")

print(f"\n{'='*80}")
print("FORENSIC AUDIT COMPLETE")
print("="*80)
