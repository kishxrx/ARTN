#!/usr/bin/env python3
"""
Reconcile timing: measure complete epoch end-to-end and sum all components.
Verify components add up to measured epoch time.
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
from src.training.adafactor import Adafactor

print("=" * 70)
print("TIMING RECONCILIATION")
print("=" * 70)

# Load dataset
print("\n[SETUP] Loading TinyStories dataset...")
start = time.time()
full_dataset = load_dataset("roneneldan/TinyStories", split="train")
print(f"  Dataset loaded in {time.time() - start:.1f}s")

# Load tokenizer
TOKENIZER_PATH = Path("/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json")
tokenizer = BPETokenizer.load(TOKENIZER_PATH)
print(f"  Tokenizer loaded, vocab size: {len(tokenizer.vocab)}")

# Select 20 stories
random.seed(42)
indices = random.sample(range(len(full_dataset)), 20)
print(f"  Selected 20 stories")

# Create dataset class (same as train_optimized.py)
class TinyStoriesDataset(Dataset):
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

# Create dataset subset
dataset_subset = full_dataset.select(indices)
dataset = TinyStoriesDataset(dataset_subset, tokenizer, max_length=256)

def collate_fn(batch):
    """Collate batch."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}

# Create model and optimizer
model = BitLinearGPT(
    vocab_size=4000,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_seq_len=256,
)
optimizer = Adafactor(model.parameters(), lr=0.001, beta1=0.5, beta2=0.98)
model.train()

print("\n" + "=" * 70)
print("MEASUREMENT 1: Complete epoch end-to-end")
print("=" * 70)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

# Measure complete epoch
epoch_start = time.time()
data_iter = iter(dataloader)

step = 0
optimizer.zero_grad()

# Track all batches
batch_times = []

while True:
    try:
        batch_start = time.time()
        batch = next(data_iter)
        
        # Forward
        input_ids = batch["input_ids"][:, :-1]
        labels = batch["labels"][:, 1:]
        
        fwd_start = time.time()
        outputs = model(input_ids, labels)
        loss = outputs["loss"]
        fwd_time = time.time() - fwd_start
        
        # Backward
        bwd_start = time.time()
        loss.backward()
        bwd_time = time.time() - bwd_start
        
        # Optimizer step
        opt_start = time.time()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        opt_time = time.time() - opt_start
        
        batch_time = time.time() - batch_start
        batch_times.append({
            'total': batch_time,
            'forward': fwd_time,
            'backward': bwd_time,
            'optimizer': opt_time,
        })
        
        step += 1
        
    except StopIteration:
        break

epoch_time_measured = time.time() - epoch_start

print(f"\n  Epoch complete in {epoch_time_measured:.3f}s")
print(f"  Batches processed: {step}")
print(f"  Average per batch: {epoch_time_measured / step:.3f}s")

print("\n" + "=" * 70)
print("MEASUREMENT 2: Component breakdown (sum parts)")
print("=" * 70)

# Re-create dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)

# Track components
components = {
    'data_access': 0.0,
    'tokenization': 0.0,
    'collation': 0.0,
    'forward': 0.0,
    'backward': 0.0,
    'optimizer': 0.0,
    'iterator_overhead': 0.0,
}

data_iter = iter(dataloader)
step = 0
optimizer.zero_grad()

while True:
    try:
        # Iterator overhead
        iter_start = time.time()
        batch = next(data_iter)
        components['iterator_overhead'] += time.time() - iter_start
        
        # Forward
        input_ids = batch["input_ids"][:, :-1]
        labels = batch["labels"][:, 1:]
        
        fwd_start = time.time()
        outputs = model(input_ids, labels)
        loss = outputs["loss"]
        components['forward'] += time.time() - fwd_start
        
        # Backward
        bwd_start = time.time()
        loss.backward()
        components['backward'] += time.time() - bwd_start
        
        # Optimizer
        opt_start = time.time()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        components['optimizer'] += time.time() - opt_start
        
        step += 1
        
    except StopIteration:
        break

components_sum = sum(components.values())

print(f"\n  Component breakdown:")
for name, t in components.items():
    pct = (t / components_sum * 100) if components_sum > 0 else 0
    print(f"    {name:20s}: {t:8.3f}s ({pct:5.1f}%)")

print(f"\n  SUM of components: {components_sum:.3f}s")
print(f"  Measured epoch:    {epoch_time_measured:.3f}s")
print(f"  Difference:         {abs(epoch_time_measured - components_sum):.3f}s")

print("\n" + "=" * 70)
print("MEASUREMENT 3: Tokenization isolated (per sample)")
print("=" * 70)

# Measure tokenization for each sample
tokenize_times = []
for idx in indices:
    text = full_dataset[idx]["text"]
    start = time.time()
    tokens = tokenizer.encode(text)
    elapsed = time.time() - start
    tokenize_times.append(elapsed)

total_tokenize_per_epoch = sum(tokenize_times)

print(f"\n  Average tokenization per story: {sum(tokenize_times)/len(tokenize_times):.6f}s")
print(f"  Total tokenization time (20 stories): {total_tokenize_per_epoch:.3f}s")
print(f"  Note: Each epoch processes 20 stories once")

print("\n" + "=" * 70)
print("MEASUREMENT 4: Forward/backward isolated")
print("=" * 70)

# Measure forward/backward without data loading
input_ids = torch.randint(0, 4000, (4, 255))
labels = torch.randint(0, 4000, (4, 255))

fwd_bwd_times = []
for i in range(5):
    optimizer.zero_grad()
    
    fwd_start = time.time()
    outputs = model(input_ids, labels)
    loss = outputs["loss"]
    fwd_time = time.time() - fwd_start
    
    bwd_start = time.time()
    loss.backward()
    bwd_time = time.time() - bwd_start
    
    opt_start = time.time()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    opt_time = time.time() - opt_start
    
    fwd_bwd_times.append({
        'forward': fwd_time,
        'backward': bwd_time,
        'optimizer': opt_time,
    })

print(f"\n  Isolated forward/backward/optimizer:")
print(f"    Forward avg:  {sum(t['forward'] for t in fwd_bwd_times)/len(fwd_bwd_times):.3f}s")
print(f"    Backward avg: {sum(t['backward'] for t in fwd_bwd_times)/len(fwd_bwd_times):.3f}s")
print(f"    Optimizer avg: {sum(t['optimizer'] for t in fwd_bwd_times)/len(fwd_bwd_times):.3f}s")

print("\n" + "=" * 70)
print("RECONCILIATION")
print("=" * 70)

# Calculate expected epoch time
expected_components = {
    'Tokenization (20×)': total_tokenize_per_epoch,
    'Forward (5×)': sum(t['forward'] for t in fwd_bwd_times) / len(fwd_bwd_times) * 5,
    'Backward (5×)': sum(t['backward'] for t in fwd_bwd_times) / len(fwd_bwd_times) * 5,
    'Optimizer (5×)': sum(t['optimizer'] for t in fwd_bwd_times) / len(fwd_bwd_times) * 5,
}

expected_total = sum(expected_components.values())

print(f"\n  Expected components:")
for name, t in expected_components.items():
    print(f"    {name:25s}: {t:8.3f}s")

print(f"\n  Expected total: {expected_total:.3f}s")
print(f"  Measured epoch: {epoch_time_measured:.3f}s")
print(f"  Difference:     {abs(epoch_time_measured - expected_total):.3f}s")

# Calculate per-second breakdown
print(f"\n  Per-second breakdown of epoch:")
for name, t in expected_components.items():
    print(f"    {name:25s}: {t:6.3f}s ({t/expected_total*100:5.1f}%)")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

tokenize_pct = total_tokenize_per_epoch / expected_total * 100
fwd_pct = expected_components['Forward (5×)'] / expected_total * 100
bwd_pct = expected_components['Backward (5×)'] / expected_total * 100
opt_pct = expected_components['Optimizer (5×)'] / expected_total * 100

print(f"\n  Tokenization: {tokenize_pct:.1f}% of epoch time")
print(f"  Forward:      {fwd_pct:.1f}% of epoch time")
print(f"  Backward:     {bwd_pct:.1f}% of epoch time")
print(f"  Optimizer:    {opt_pct:.1f}% of epoch time")

if total_tokenize_per_epoch > 1.0:
    print(f"\n  TOKENIZATION IS SIGNIFICANT: {total_tokenize_per_epoch:.1f}s per epoch")
else:
    print(f"\n  Tokenization is minor: {total_tokenize_per_epoch:.3f}s per epoch")

print("\n" + "=" * 70)
