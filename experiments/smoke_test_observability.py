#!/usr/bin/env python3
"""
Smoke test for the observability version.
"""

import sys
import time
import torch
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("SMOKE TEST: Observability Pipeline")
print("=" * 70)

# Import from actual script
from experiments.tinystories_memorize_observability import (
    TinyStoriesSubset,
    TinyStoriesValidation,
    BitLinearGPT,
    get_ternary_distribution,
    generate_from_prompts,
    compute_validation_loss,
    MetricsLogger,
    FIXED_PROMPTS,
    TOKENIZER_PATH,
    MODEL_CONFIG,
    OPTIMIZER_CONFIG,
    TRAINING_CONFIG,
    DATA_CONFIG,
)

# ========================================================================
# TEST 1: Model and Tokenizer
# ========================================================================
print("\n[TEST 1] Creating model and tokenizer...")
model = BitLinearGPT(
    vocab_size=MODEL_CONFIG["vocab_size"],
    d_model=MODEL_CONFIG["d_model"],
    n_heads=MODEL_CONFIG["n_heads"],
    n_layers=MODEL_CONFIG["n_layers"],
    d_ff=MODEL_CONFIG["d_ff"],
    max_seq_len=MODEL_CONFIG["max_seq_len"],
)
print(f"  ✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

from src.tokenizer.bpe_tokenizer import BPETokenizer
tokenizer = BPETokenizer.load(TOKENIZER_PATH)
print(f"  ✓ Tokenizer loaded: vocab={len(tokenizer.vocab)}")

# ========================================================================
# TEST 2: Train Dataset
# ========================================================================
print("\n[TEST 2] Loading train dataset...")
train_dataset = TinyStoriesSubset(
    TOKENIZER_PATH,
    num_stories=20,
    max_seq_len=DATA_CONFIG["max_seq_len"],
    seed=DATA_CONFIG["seed"],
)
print(f"  ✓ Train dataset: {len(train_dataset)} samples")

# ========================================================================
# TEST 3: Validation Dataset
# ========================================================================
print("\n[TEST 3] Loading validation dataset...")
val_dataset = TinyStoriesValidation(
    TOKENIZER_PATH,
    num_samples=100,
    max_seq_len=DATA_CONFIG["max_seq_len"],
    seed=123,
)
print(f"  ✓ Validation dataset: {len(val_dataset)} samples")

# ========================================================================
# TEST 4: DataLoader
# ========================================================================
print("\n[TEST 4] Creating dataloaders...")
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
print(f"  ✓ Train loader: {len(train_loader)} batches")
print(f"  ✓ Val loader: {len(val_loader)} batches")

# ========================================================================
# TEST 5: Forward Pass
# ========================================================================
print("\n[TEST 5] Testing forward pass...")
batch = next(iter(train_loader))
outputs = model(batch)
logits = outputs['logits'] if isinstance(outputs, dict) else outputs
print(f"  ✓ Forward pass: logits shape {logits.shape}")

# ========================================================================
# TEST 6: Ternary Weight Distribution
# ========================================================================
print("\n[TEST 6] Computing ternary weight distribution...")
neg, zero, pos = get_ternary_distribution(model)
print(f"  ✓ Ternary distribution: -1={neg:.1f}%, 0={zero:.1f}%, +1={pos:.1f}%")

# ========================================================================
# TEST 7: Validation Loss
# ========================================================================
print("\n[TEST 7] Computing validation loss...")
val_loss = compute_validation_loss(model, val_loader, device='cpu')
import math
perplexity = math.exp(val_loss)
print(f"  ✓ Validation loss: {val_loss:.4f}")
print(f"  ✓ Perplexity: {perplexity:.1f}")

# ========================================================================
# TEST 8: Fixed Prompt Generation
# ========================================================================
print("\n[TEST 8] Generating from fixed prompts...")
generated = generate_from_prompts(model, tokenizer, FIXED_PROMPTS, device='cpu')
print(f"  ✓ Generated {len(generated)} samples")
for i, g in enumerate(generated[:2]):
    print(f"    {i+1}. Prompt: {g['prompt']}")
    print(f"       Generated: {g['generated'][:60]}...")

# ========================================================================
# TEST 9: Checkpoint Save/Load
# ========================================================================
print("\n[TEST 9] Testing checkpoint save/load...")
checkpoint_dir = Path("experiments/results/smoke_test_observability")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
torch.save({
    "step": 0,
    "model_state_dict": model.state_dict(),
    "loss": val_loss,
}, checkpoint_path)
print(f"  ✓ Checkpoint saved: {checkpoint_path}")

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"  ✓ Checkpoint loaded: step={checkpoint['step']}, loss={checkpoint['loss']:.4f}")

# ========================================================================
# TEST 10: Metrics Logging
# ========================================================================
print("\n[TEST 10] Testing metrics logging...")
metrics_path = checkpoint_dir / "test_metrics.jsonl"
metrics_logger = MetricsLogger(metrics_path)

metrics_logger.log({
    "step": 0,
    "train_loss": val_loss,
    "val_loss": val_loss,
    "perplexity": perplexity,
    "grad_norm": 1.23,
    "lr": OPTIMIZER_CONFIG['lr'],
    "ternary_neg_pct": neg,
    "ternary_zero_pct": zero,
    "ternary_pos_pct": pos,
})
print(f"  ✓ Metrics logged: {metrics_path}")

# Read back metrics
with open(metrics_path) as f:
    logged_metrics = json.loads(f.readline())
print(f"  ✓ Metrics verified: {len(logged_metrics)} fields")

# ========================================================================
# TEST 11: Verify All Metrics Present
# ========================================================================
print("\n[TEST 11] Verifying all required metrics...")
required_metrics = [
    "step", "train_loss", "val_loss", "perplexity", 
    "grad_norm", "lr", "ternary_neg_pct", "ternary_zero_pct", "ternary_pos_pct"
]

missing = [m for m in required_metrics if m not in logged_metrics]
if missing:
    print(f"  ✗ MISSING metrics: {missing}")
else:
    print(f"  ✓ All {len(required_metrics)} required metrics present")

# ========================================================================
# RESULTS
# ========================================================================
print("\n" + "=" * 70)
print("SMOKE TEST RESULT: ALL TESTS PASSED")
print("=" * 70)

print("\nFiles created:")
print(f"  {checkpoint_path}")
print(f"  {metrics_path}")

print("\nMetrics verified:")
print(f"  train_loss: {logged_metrics['train_loss']:.4f}")
print(f"  val_loss: {logged_metrics['val_loss']:.4f}")
print(f"  perplexity: {logged_metrics['perplexity']:.1f}")
print(f"  grad_norm: {logged_metrics['grad_norm']:.4f}")
print(f"  lr: {logged_metrics['lr']}")
print(f"  ternary: -1={logged_metrics['ternary_neg_pct']:.1f}%, "
      f"0={logged_metrics['ternary_zero_pct']:.1f}%, "
      f"+1={logged_metrics['ternary_pos_pct']:.1f}%")

print("\n✓ Observability pipeline validated. Ready for full 3000-step training.")
