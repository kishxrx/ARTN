#!/usr/bin/env python3
"""
FORENSIC AUDIT: Actual 50k Training Dataset
Uses LOCAL cached TinyStories (no download)
"""

import sys
import json
import torch
from pathlib import Path
from collections import Counter
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*80)
print("FORENSIC AUDIT: 50k TRAINING DATASET (LOCAL CACHE)")
print("="*80)

# ============================================================
# LOAD TOKENIZER
# ============================================================

print(f"\nLoading tokenizer...")
tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
print(f"Vocab size: {tokenizer.vocab_size}")

# ============================================================
# LOAD DATASET FROM LOCAL CACHE
# ============================================================

print(f"\nLoading TinyStories from local cache...")
dataset = load_dataset("roneneldan/TinyStories", split="train")
print(f"Total stories: {len(dataset):,}")

# Select first 50,000 (same as training)
dataset = dataset.select(range(50000))
print(f"Selected: {len(dataset):,} stories")

# ============================================================
# TOKENIZE ALL STORIES
# ============================================================

print(f"\nTokenizing all 50,000 stories...")

total_chars = 0
total_tokens = 0
all_token_ids = []
story_lengths = []
truncated = 0
max_seq_len = 256

for i, item in enumerate(dataset):
    text = item['text']
    tokens = tokenizer.encode(text)
    
    total_chars += len(text)
    
    # Check truncation
    if len(tokens) >= max_seq_len - 1:
        truncated += 1
    
    # Same as training
    tokens = tokens[:max_seq_len - 1] + [tokenizer.eos_token_id]
    
    story_lengths.append(len(tokens))
    all_token_ids.extend(tokens)
    total_tokens += len(tokens)
    
    if (i + 1) % 10000 == 0:
        print(f"  Progress: {i+1:,} / 50,000")

print(f"\n✓ Tokenization complete!")

# ============================================================
# REPORT ALL MEASUREMENTS
# ============================================================

print(f"\n{'='*80}")
print("COMPLETE DATASET STATISTICS")
print("="*80)

unique_tokens = set(all_token_ids)
token_counter = Counter(all_token_ids)

print(f"\n1. Stories loaded: {len(dataset):,}")
print(f"2. Total raw characters: {total_chars:,}")
print(f"3. Total tokens: {total_tokens:,}")
print(f"4. Unique tokens: {len(unique_tokens):,}")
print(f"5. Vocab coverage: {len(unique_tokens)}/{tokenizer.vocab_size} ({100*len(unique_tokens)/tokenizer.vocab_size:.1f}%)")
print(f"6. Avg tokens/story: {total_tokens/len(dataset):.1f}")
print(f"7. Min tokens: {min(story_lengths)}")
print(f"8. Max tokens: {max(story_lengths)}")
print(f"9. Stories truncated: {truncated:,} ({100*truncated/len(dataset):.1f}%)")

# Token frequency
print(f"\n{'='*80}")
print("TOKEN FREQUENCY")
print("="*80)

most_common = token_counter.most_common(20)
print(f"\nTop 20 tokens:")
for rank, (tid, count) in enumerate(most_common, 1):
    tok_str = tokenizer.decode([tid])
    pct = 100 * count / total_tokens
    print(f"  {rank:2}. {tid:4} '{tok_str:15}' - {count:7,} ({pct:.2f}%)")

top10_pct = 100 * sum(c for _, c in most_common[:10]) / total_tokens
print(f"\nTop 10 tokens: {top10_pct:.1f}% of all tokens")

# Duplicates
print(f"\n{'='*80}")
print("DUPLICATE CHECK")
print("="*80)

texts_seen = set()
dups = 0
for item in dataset:
    if item['text'] in texts_seen:
        dups += 1
    texts_seen.add(item['text'])

print(f"Exact duplicates: {dups:,}")

# Training exposure
print(f"\n{'='*80}")
print("TRAINING EXPOSURE")
print("="*80)

steps = 50000
batch = 4
accum = 2
eff_batch = batch * accum

total_samples = steps * eff_batch
epochs = total_samples / len(dataset)

print(f"Steps: {steps:,}")
print(f"Effective batch: {eff_batch}")
print(f"Total samples: {total_samples:,}")
print(f"Epochs: {epochs:.1f}")

# Tokens per param
print(f"\n{'='*80}")
print("TOKENS PER PARAMETER")
print("="*80)

# Approx params for 6-layer 256-dim GPT
params = 4000*256 + 6*(4*256*256 + 2*256*1024)

print(f"Model params: {params:,}")
print(f"Training tokens: {total_tokens:,}")
print(f"Tokens/param: {total_tokens/params:.2f}")

# Save results
results = {
    "stories": len(dataset),
    "total_chars": total_chars,
    "total_tokens": total_tokens,
    "unique_tokens": len(unique_tokens),
    "vocab_coverage_pct": 100*len(unique_tokens)/tokenizer.vocab_size,
    "avg_tokens_per_story": total_tokens/len(dataset),
    "min_tokens": min(story_lengths),
    "max_tokens": max(story_lengths),
    "truncated_stories": truncated,
    "duplicates": dups,
    "epochs": epochs,
    "tokens_per_param": total_tokens/params,
}

with open("experiments/results/50k_audit_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*80}")
print("RESULTS SAVED: experiments/results/50k_audit_results.json")
print("="*80)
