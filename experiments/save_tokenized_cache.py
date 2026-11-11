#!/usr/bin/env python3
"""
SAVE TOKENIZED DATA - Run once to cache tokenized TinyStories
After this, experiments load directly from cache (instant startup)
"""

import sys
import torch
import json
from pathlib import Path

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from datasets import load_dataset
from src.tokenizer.bpe_tokenizer import BPETokenizer

# Config
MAX_TRAIN_SAMPLES = 50000
MAX_VAL_SAMPLES = 5000
MAX_SEQ_LEN = 256

# Output
CACHE_DIR = BASE_DIR / 'outputs' / 'tokenized_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("TOKENIZING TINYSTORIES (ONE-TIME SETUP)")
print("="*80)

# Load tokenizer
tokenizer_path = BASE_DIR / 'outputs' / 'artn-50k-best-config' / 'tokenizer.json'
print(f"\nLoading tokenizer from {tokenizer_path}")
tokenizer = BPETokenizer.load(str(tokenizer_path))

# Load dataset
print(f"Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

# Split
total_samples = min(len(dataset), MAX_TRAIN_SAMPLES + MAX_VAL_SAMPLES)
dataset = dataset.select(range(total_samples))

train_data = dataset.select(range(MAX_TRAIN_SAMPLES))
val_data = dataset.select(range(MAX_TRAIN_SAMPLES, MAX_TRAIN_SAMPLES + MAX_VAL_SAMPLES))

print(f"Train samples: {len(train_data)}")
print(f"Val samples: {len(val_data)}")

# Tokenize train
print(f"\nTokenizing TRAIN data...")
train_tokenized = []
for i, item in enumerate(train_data):
    if i % 5000 == 0:
        print(f"  {i}/{len(train_data)}")
    
    tokens = tokenizer.encode(item['text'])[:MAX_SEQ_LEN]
    if len(tokens) > 10:
        train_tokenized.append(tokens)

print(f"  Done. Total: {len(train_tokenized)}")

# Tokenize val
print(f"\nTokenizing VAL data...")
val_tokenized = []
for i, item in enumerate(val_data):
    if i % 1000 == 0:
        print(f"  {i}/{len(val_data)}")
    
    tokens = tokenizer.encode(item['text'])[:MAX_SEQ_LEN]
    if len(tokens) > 10:
        val_tokenized.append(tokens)

print(f"  Done. Total: {len(val_tokenized)}")

# Save
print(f"\nSaving to cache...")
torch.save(train_tokenized, CACHE_DIR / 'train_tokenized.pt')
torch.save(val_tokenized, CACHE_DIR / 'val_tokenized.pt')

print(f"\nCache saved to: {CACHE_DIR}")
print(f"  train_tokenized.pt: {len(train_tokenized)} samples")
print(f"  val_tokenized.pt: {len(val_tokenized)} samples")

print("\n" + "="*80)
print("TOKENIZATION COMPLETE - Future runs will load instantly!")
print("="*80)
