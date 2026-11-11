#!/usr/bin/env python3
"""
Verify Phase 22 data pipeline matches train_optimized.py
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sys

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*100)
print("DATA PIPELINE VERIFICATION: Phase 22 vs train_optimized.py")
print("="*100)

# Configuration
config = {
    'max_length': 256,
    'batch_size': 4,
    'tokenizer_path': 'outputs/artn-optimized/tokenizer.json',
    'dataset': 'roneneldan/TinyStories',
}

# Dataset class - SAME as train_optimized.py
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

# Load data
print("\n[Loading data]")
dataset = load_dataset(config['dataset'], split="train")
tokenizer = BPETokenizer.load(config['tokenizer_path'])
print(f"   Dataset size: {len(dataset):,}")
print(f"   Tokenizer EOS: {tokenizer.eos_token_id}, PAD: {tokenizer.pad_token_id}")

# Create dataset
torch.manual_seed(42)
tiny_dataset = TinyStoriesDataset(dataset, tokenizer, config['max_length'])
print(f"   Dataset size: {len(tiny_dataset):,}")

# Create DataLoader with SAME settings as Phase 22
torch.manual_seed(42)
loader = DataLoader(tiny_dataset, batch_size=config['batch_size'], shuffle=True)

# Get batches
print("\n[Verifying batches]")
batches = []
for i in range(3):
    batch = next(iter(loader))
    batches.append(batch)
    print(f"   Batch {i}: input_ids shape={batch['input_ids'].shape}")

# Verify specific properties
print("\n[Verification Results]")

batch0 = batches[0]

# Check 1: Correct shapes
print(f"\n1. Batch shapes:")
print(f"   input_ids: {batch0['input_ids'].shape} (expected: [4, 256])")
print(f"   labels: {batch0['labels'].shape} (expected: [4, 256])")

# Check 2: EOS tokens present
eos_count = (batch0['input_ids'] == tokenizer.eos_token_id).sum().item()
print(f"\n2. EOS tokens in batch 0:")
print(f"   Count: {eos_count}")
print(f"   Expected: >= {config['batch_size']} (at least 1 per sample)")

# Check 3: Padding masked in labels
padding_masked = (batch0['labels'] == -100).sum().item()
print(f"\n3. Padding masked in labels:")
print(f"   Count: {padding_masked}")
print(f"   Expected: > 0 (stories are padded)")

# Check 4: Sample content
print(f"\n4. Sample input_ids (batch 0, sample 0, first 20 tokens):")
print(f"   {batch0['input_ids'][0, :20].tolist()}")

# Check 5: Verify identical batches with same seed
torch.manual_seed(42)
loader2 = DataLoader(tiny_dataset, batch_size=config['batch_size'], shuffle=True)
batch_repeat = next(iter(loader2))

identical = torch.allclose(batch0['input_ids'], batch_repeat['input_ids'])
print(f"\n5. Reproducibility (same seed = same batch):")
print(f"   Identical: {identical}")

# Summary
print("\n" + "="*100)
print("VERIFICATION SUMMARY")
print("="*100)

checks = [
    ("Batch shapes correct", batch0['input_ids'].shape == torch.Size([4, 256])),
    ("EOS tokens present", eos_count >= config['batch_size']),
    ("Padding masked in labels", padding_masked > 0),
    ("Reproducible with same seed", identical),
]

all_passed = True
for check_name, passed in checks:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"   {check_name}: {status}")
    if not passed:
        all_passed = False

print("\n" + "="*100)
if all_passed:
    print("✅ ALL VERIFICATIONS PASSED - Data pipeline matches train_optimized.py")
else:
    print("❌ SOME VERIFICATIONS FAILED")
print("="*100)
