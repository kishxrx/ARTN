#!/usr/bin/env python3
"""
Profile each operation separately to find the ~11-second delay.

Operations to profile:
1. self.data[idx]["text"] - TinyStories dataset access
2. tokenizer.encode(text) - BPE tokenization
3. batch construction/collation
4. DataLoader iterator creation/reset
5. actual forward/backward

Do NOT modify training pipeline. Establish exactly where delay comes from.
"""

import sys
import time
import torch
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.bitlinear import BitLinearGPT

print("=" * 70)
print("OPERATION-BY-OPERATION PROFILING")
print("=" * 70)

# Load dataset
print("\n[SETUP] Loading TinyStories dataset...")
start = time.time()
full_dataset = load_dataset("roneneldan/TinyStories", split="train")
print(f"  Dataset loaded in {time.time() - start:.1f}s")
print(f"  Total stories: {len(full_dataset):,}")

# Load tokenizer
print("\n[SETUP] Loading tokenizer...")
TOKENIZER_PATH = Path("/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json")
tokenizer = BPETokenizer.load(TOKENIZER_PATH)
print(f"  Tokenizer loaded, vocab size: {len(tokenizer.vocab)}")

# Select 20 stories
print("\n[SETUP] Selecting 20 stories...")
random.seed(42)
indices = random.sample(range(len(full_dataset)), 20)
print(f"  Selected indices: {indices[:5]}...")

# ============================================================
# PROFILE 1: self.data[idx]["text"]
# ============================================================
print("\n" + "=" * 70)
print("[PROFILE 1] self.data[idx]['text'] - TinyStories access")
print("=" * 70)

access_times = []
for i in range(100):
    idx = indices[i % 20]
    start = time.time()
    text = full_dataset[idx]["text"]
    elapsed = time.time() - start
    access_times.append(elapsed)
    if i < 10:
        print(f"  Access {i+1}: {elapsed:.6f}s, text_len={len(text)}")

print(f"\n  TOTAL (100 accesses): {sum(access_times):.3f}s")
print(f"  Average: {sum(access_times)/len(access_times):.6f}s per access")
print(f"  Min: {min(access_times):.6f}s, Max: {max(access_times):.6f}s")

# ============================================================
# PROFILE 2: tokenizer.encode(text)
# ============================================================
print("\n" + "=" * 70)
print("[PROFILE 2] tokenizer.encode(text) - BPE tokenization")
print("=" * 70)

# Get sample texts
sample_texts = [full_dataset[idx]["text"] for idx in indices[:10]]

encode_times = []
for i in range(100):
    text = sample_texts[i % 10]
    start = time.time()
    tokens = tokenizer.encode(text)
    elapsed = time.time() - start
    encode_times.append(elapsed)
    if i < 10:
        print(f"  Encode {i+1}: {elapsed:.6f}s, tokens={len(tokens)}")

print(f"\n  TOTAL (100 encodes): {sum(encode_times):.3f}s")
print(f"  Average: {sum(encode_times)/len(encode_times):.6f}s per encode")
print(f"  Min: {min(encode_times):.6f}s, Max: {max(encode_times):.6f}s")

# ============================================================
# PROFILE 3: Batch construction/collation
# ============================================================
print("\n" + "=" * 70)
print("[PROFILE 3] Batch construction/collation")
print("=" * 70)

# Pre-tokenize some samples
sample_tokens = []
for idx in indices[:20]:
    text = full_dataset[idx]["text"]
    tokens = tokenizer.encode(text)[:255]
    tokens = tokens + [tokenizer.eos_token_id]
    sample_tokens.append(tokens)

def collate_fn(batch):
    """Pad sequences."""
    max_len = max(len(seq) for seq in batch)
    padded = []
    for seq in batch:
        if len(seq) < max_len:
            padding = [tokenizer.pad_token_id] * (max_len - len(seq))
            seq = seq + padding
        padded.append(torch.tensor(seq, dtype=torch.long))
    return torch.stack(padded)

collate_times = []
for i in range(100):
    batch_idx = random.sample(range(20), 4)
    batch = [sample_tokens[idx] for idx in batch_idx]
    start = time.time()
    result = collate_fn(batch)
    elapsed = time.time() - start
    collate_times.append(elapsed)
    if i < 10:
        print(f"  Collate {i+1}: {elapsed:.6f}s, shape={result.shape}")

print(f"\n  TOTAL (100 collates): {sum(collate_times):.3f}s")
print(f"  Average: {sum(collate_times)/len(collate_times):.6f}s per collate")
print(f"  Min: {min(collate_times):.6f}s, Max: {max(collate_times):.6f}s")

# ============================================================
# PROFILE 4: DataLoader iterator creation/reset
# ============================================================
print("\n" + "=" * 70)
print("[PROFILE 4] DataLoader iterator creation/reset")
print("=" * 70)

class TestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

test_dataset = TestDataset(sample_tokens)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

# 4a: Create iterator
print("\n  [4a] Creating iterator (iter(dataloader)):")
iter_times = []
for i in range(20):
    start = time.time()
    data_iter = iter(test_dataloader)
    elapsed = time.time() - start
    iter_times.append(elapsed)
    if i < 5:
        print(f"    Create {i+1}: {elapsed:.6f}s")

print(f"  Average: {sum(iter_times)/len(iter_times):.6f}s")

# 4b: Complete epoch iteration
print("\n  [4b] Complete epoch (5 batches):")
epoch_times = []
for epoch in range(20):
    start = time.time()
    for batch in test_dataloader:
        pass
    elapsed = time.time() - start
    epoch_times.append(elapsed)
    if epoch < 5:
        print(f"    Epoch {epoch+1}: {elapsed:.6f}s")

print(f"  Average: {sum(epoch_times)/len(epoch_times):.6f}s per epoch")

# ============================================================
# PROFILE 5: Forward/backward pass
# ============================================================
print("\n" + "=" * 70)
print("[PROFILE 5] Forward/backward pass")
print("=" * 70)

model = BitLinearGPT(
    vocab_size=4000,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_seq_len=256,
)
model.train()

# Create sample batch
batch_tensor = torch.randint(0, 4000, (4, 256), dtype=torch.long)
inputs = batch_tensor[:, :-1]
targets = batch_tensor[:, 1:]

forward_times = []
backward_times = []

for i in range(20):
    # Forward
    start = time.time()
    outputs = model(inputs, targets)
    loss = outputs["loss"]
    forward_time = time.time() - start
    forward_times.append(forward_time)
    
    # Backward
    start = time.time()
    loss.backward()
    backward_time = time.time() - start
    backward_times.append(backward_time)
    
    if i < 5:
        print(f"  Pass {i+1}: Forward={forward_time:.3f}s, Backward={backward_time:.3f}s")

print(f"\n  Forward average: {sum(forward_times)/len(forward_times):.3f}s")
print(f"  Backward average: {sum(backward_times)/len(backward_times):.3f}s")
print(f"  Total pass: {sum(forward_times)/len(forward_times) + sum(backward_times)/len(backward_times):.3f}s")

# ============================================================
# PROFILE 6: Full TinyStoriesDataset __getitem__
# ============================================================
print("\n" + "=" * 70)
print("[PROFILE 6] Full TinyStoriesDataset __getitem__")
print("=" * 70)

class TinyStoriesDataset(Dataset):
    """Same as train_optimized.py."""
    
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        tokens = self.tokenizer.encode(text)[:self.max_length - 1]
        tokens = tokens + [self.tokenizer.eos_token_id]
        padding_length = self.max_length - len(tokens)
        tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}

# Create dataset with subset
subset_data = full_dataset.select(indices)
ts_dataset = TinyStoriesDataset(subset_data, tokenizer, max_length=256)

getitem_times = []
for i in range(100):
    idx = i % 20
    start = time.time()
    sample = ts_dataset[idx]
    elapsed = time.time() - start
    getitem_times.append(elapsed)
    if i < 10:
        print(f"  __getitem__ {i+1}: {elapsed:.6f}s")

print(f"\n  TOTAL (100 calls): {sum(getitem_times):.3f}s")
print(f"  Average: {sum(getitem_times)/len(getitem_times):.6f}s per call")
print(f"  Min: {min(getitem_times):.6f}s, Max: {max(getitem_times):.6f}s")

# ============================================================
# PROFILE 7: DataLoader with TinyStoriesDataset
# ============================================================
print("\n" + "=" * 70)
print("[PROFILE 7] DataLoader with TinyStoriesDataset (full pipeline)")
print("=" * 70)

ts_dataloader = DataLoader(ts_dataset, batch_size=4, shuffle=True, num_workers=0)

# Single epoch
print("\n  Single epoch (5 batches):")
start = time.time()
batch_times = []
for i, batch in enumerate(ts_dataloader):
    batch_time = time.time() - start
    batch_times.append(batch_time)
    print(f"    Batch {i+1}: {batch_time:.3f}s total, shape={batch['input_ids'].shape}")
    start = time.time()

print(f"\n  Total epoch time: {sum(batch_times):.3f}s")
print(f"  Average per batch: {sum(batch_times)/len(batch_times):.3f}s")

# Multiple epochs
print("\n  Multiple epochs (10 epochs):")
epoch_times = []
for epoch in range(10):
    start = time.time()
    for batch in ts_dataloader:
        pass
    elapsed = time.time() - start
    epoch_times.append(elapsed)
    if epoch < 5:
        print(f"    Epoch {epoch+1}: {elapsed:.3f}s")

print(f"\n  Average epoch: {sum(epoch_times)/len(epoch_times):.3f}s")
print(f"  Min: {min(epoch_times):.3f}s, Max: {max(epoch_times):.3f}s")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n  1. TinyStories access (self.data[idx]['text']):")
print(f"     Average: {sum(access_times)/len(access_times):.6f}s per access")
print(f"     100 accesses: {sum(access_times):.3f}s")

print(f"\n  2. Tokenizer encode (tokenizer.encode):")
print(f"     Average: {sum(encode_times)/len(encode_times):.6f}s per encode")
print(f"     100 encodes: {sum(encode_times):.3f}s")

print(f"\n  3. Batch collation:")
print(f"     Average: {sum(collate_times)/len(collate_times):.6f}s per collate")
print(f"     100 collates: {sum(collate_times):.3f}s")

print(f"\n  4. Forward pass:")
print(f"     Average: {sum(forward_times)/len(forward_times):.3f}s")

print(f"\n  5. Backward pass:")
print(f"     Average: {sum(backward_times)/len(backward_times):.3f}s")

print(f"\n  6. Full __getitem__ (access + encode + tensor):")
print(f"     Average: {sum(getitem_times)/len(getitem_times):.6f}s per call")
print(f"     100 calls: {sum(getitem_times):.3f}s")

print(f"\n  7. DataLoader with TinyStoriesDataset epoch:")
print(f"     Average: {sum(epoch_times)/len(epoch_times):.3f}s per epoch (5 batches)")

print("\n" + "=" * 70)
