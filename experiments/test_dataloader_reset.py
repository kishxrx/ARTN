#!/usr/bin/env python3
"""
Test dataloader reset performance.
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("DATALOADER RESET PERFORMANCE TEST")
print("=" * 70)

# Create a small test dataset
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, size=20, seq_len=256):
        # Pre-create all samples in memory
        self.samples = []
        for i in range(size):
            tokens = torch.randint(0, 4000, (seq_len,), dtype=torch.long)
            self.samples.append(tokens)
        print(f"CachedDataset: {size} samples, {seq_len} tokens each")
        print(f"Memory used: ~{sum(s.numel() * s.element_size() for s in self.samples) / 1024:.1f} KB")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """Pad sequences (same as training script)."""
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
start = time.time()
dataset = CachedDataset(size=20, seq_len=256)
elapsed = time.time() - start
print(f"  Dataset created in {elapsed:.3f}s")

# Test 2: Create dataloader
print("\n[TEST 2] Creating dataloader...")
start = time.time()
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)
elapsed = time.time() - start
print(f"  Dataloader created in {elapsed:.3f}s")
print(f"  Batches per epoch: {len(dataloader)}")

# Test 3: First iteration
print("\n[TEST 3] First iteration (5 batches)...")
data_iter = iter(dataloader)
for i in range(5):
    start = time.time()
    batch = next(data_iter)
    elapsed = time.time() - start
    print(f"  Batch {i+1}: {elapsed:.6f}s, shape={batch.shape}")

# Test 4: Reset and iterate
print("\n[TEST 4] Resetting iterator...")
start = time.time()
data_iter = iter(dataloader)
elapsed = time.time() - start
print(f"  Iterator reset in {elapsed:.6f}s")

print("\n[TEST 5] Second iteration (5 batches)...")
for i in range(5):
    start = time.time()
    batch = next(data_iter)
    elapsed = time.time() - start
    print(f"  Batch {i+1}: {elapsed:.6f}s, shape={batch.shape}")

# Test 6: Multiple resets
print("\n[TEST 6] Multiple iterator resets (10 times)...")
for i in range(10):
    start = time.time()
    data_iter = iter(dataloader)
    elapsed = time.time() - start
    print(f"  Reset {i+1}: {elapsed:.6f}s")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
