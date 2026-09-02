#!/usr/bin/env python3
"""
Investigate PyTorch DataLoader delay issue.
"""

import sys
import time
import random
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import Dataset, DataLoader

print("=" * 70)
print("PYTORCH DATALOADER INVESTIGATION")
print("=" * 70)

# Simulate TinyStories dataset that caches data
class CachedDataset(Dataset):
    """Dataset that caches all samples in memory."""
    
    def __init__(self, num_samples=20, seq_len=256):
        # Pre-create all samples
        self.samples = []
        for i in range(num_samples):
            tokens = torch.randint(0, 4000, (seq_len,), dtype=torch.long)
            self.samples.append(tokens)
        print(f"CachedDataset: {num_samples} samples cached")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """Pad sequences."""
    max_len = max(len(seq) for seq in batch)
    padded = []
    for seq in batch:
        if len(seq) < max_len:
            padding = torch.zeros(max_len - len(seq), dtype=torch.long)
            seq = torch.cat([seq, padding])
        padded.append(seq)
    return torch.stack(padded)

# Test 1: Create dataset
print("\n[TEST 1] Creating dataset...")
dataset = CachedDataset(num_samples=20, seq_len=256)

# Test 2: Create DataLoader
print("\n[TEST 2] Creating DataLoader...")
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)
print(f"  Batches per epoch: {len(dataloader)}")

# Test 3: First epoch iteration
print("\n[TEST 3] First epoch (5 batches)...")
epoch_times = []
start_epoch = time.time()
for i, batch in enumerate(dataloader):
    batch_time = time.time()
    print(f"  Batch {i+1}: shape={batch.shape}, time={time.time() - start_epoch:.3f}s")
epoch_time = time.time() - start_epoch
print(f"  Epoch 1 total: {epoch_time:.3f}s")
epoch_times.append(epoch_time)

# Test 4: Multiple epochs
print("\n[TEST 4] Multiple epochs (10 epochs)...")
for epoch in range(10):
    start = time.time()
    for batch in dataloader:
        pass
    elapsed = time.time() - start
    epoch_times.append(elapsed)
    print(f"  Epoch {epoch+2}: {elapsed:.3f}s")

print(f"\n  Average epoch time: {sum(epoch_times)/len(epoch_times):.3f}s")
print(f"  Min: {min(epoch_times):.3f}s, Max: {max(epoch_times):.3f}s")

# Test 5: Iterator reset vs recreate
print("\n[TEST 5] Iterator reset vs recreate...")
print("  Method 1: Reset iterator (iter(dataloader))")
for i in range(5):
    start = time.time()
    data_iter = iter(dataloader)
    batch = next(data_iter)
    elapsed = time.time() - start
    print(f"    Reset {i+1}: {elapsed:.6f}s")

print("\n  Method 2: Recreate DataLoader")
for i in range(5):
    start = time.time()
    new_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    data_iter = iter(new_loader)
    batch = next(data_iter)
    elapsed = time.time() - start
    print(f"    Recreate {i+1}: {elapsed:.6f}s")

# Test 6: Shuffle impact
print("\n[TEST 6] Shuffle impact...")
print("  With shuffle=True:")
dataloader_shuffle = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
start = time.time()
for _ in dataloader_shuffle:
    pass
print(f"  Time: {time.time() - start:.3f}s")

print("\n  With shuffle=False:")
dataloader_no_shuffle = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
start = time.time()
for _ in dataloader_no_shuffle:
    pass
print(f"  Time: {time.time() - start:.3f}s")

# Test 7: Measure time per batch fetch
print("\n[TEST 7] Batch fetch timing (detailed)...")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
for epoch in range(3):
    print(f"\n  Epoch {epoch+1}:")
    batch_times = []
    data_iter = iter(dataloader)
    for i in range(5):
        start = time.time()
        batch = next(data_iter)
        elapsed = time.time() - start
        batch_times.append(elapsed)
        print(f"    Batch {i+1}: {elapsed:.6f}s")
    print(f"    Total: {sum(batch_times):.3f}s")

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)
