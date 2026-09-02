#!/usr/bin/env python3
"""
Test full 6-layer BitLinearGPT with actual training batch size and sequence length.
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT

print("=" * 70)
print("FULL BITLINEAR GPT FORWARD TEST")
print("=" * 70)

# EXACT config from training
config = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
}

print("\n[MODEL CONFIG]")
for k, v in config.items():
    print(f"  {k}: {v}")

# Create model
print("\n[CREATING MODEL]")
model = BitLinearGPT(**config)
num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params:,}")

# Test configurations
test_configs = [
    {"name": "Minimal", "batch": 1, "seq": 10},
    {"name": "Small", "batch": 2, "seq": 50},
    {"name": "Medium", "batch": 4, "seq": 100},
    {"name": "Training", "batch": 4, "seq": 256},
]

print("\n[FWD PASS TESTS]")
print("=" * 70)

for test in test_configs:
    name = test["name"]
    batch = test["batch"]
    seq = test["seq"]
    
    print(f"\n[{name}] batch={batch}, seq={seq}")
    print("-" * 70)
    
    # Create inputs
    inputs = torch.randint(0, 4000, (batch, seq))
    targets = torch.randint(0, 4000, (batch, seq))
    
    print(f"  Input shape: {inputs.shape}")
    print(f"  Input dtype: {inputs.dtype}")
    print(f"  Input memory: {inputs.element_size() * inputs.numel():,} bytes")
    
    # Forward pass
    print(f"  Running forward...")
    start = time.time()
    
    try:
        outputs = model(inputs, targets)
        elapsed = time.time() - start
        
        print(f"  ✓ SUCCESS in {elapsed:.4f}s")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Loss: {outputs['loss']:.4f}")
        
        # Memory estimate
        logits_mem = outputs['logits'].element_size() * outputs['logits'].numel()
        print(f"  Output memory: {logits_mem:,} bytes ({logits_mem/1024:.2f} KB)")
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ FAILED after {elapsed:.4f}s")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        break

print("\n" + "=" * 70)
print("MULTIPLE FORWARD PASSES (simulating training)")
print("=" * 70)

# Test 10 forward passes with training batch size
batch = 4
seq = 256
num_passes = 10

print(f"\n[{num_passes} passes] batch={batch}, seq={seq}")
print("-" * 70)

inputs = torch.randint(0, 4000, (batch, seq))
targets = torch.randint(0, 4000, (batch, seq))

total_time = 0
for i in range(num_passes):
    start = time.time()
    outputs = model(inputs, targets)
    elapsed = time.time() - start
    total_time += elapsed
    print(f"  Pass {i+1:2d}: {elapsed:.4f}s, loss={outputs['loss']:.4f}")

avg_time = total_time / num_passes
print(f"\n  Total: {total_time:.4f}s")
print(f"  Average: {avg_time:.4f}s per forward pass")

# Estimate training time
print("\n[TRAINING TIME ESTIMATE]")
print("-" * 70)
steps = 3000
accum_steps = 2
fwd_per_step = accum_steps  # 2 forward passes per step (gradient accumulation)
total_fwd = steps * fwd_per_step
est_time = total_fwd * avg_time

print(f"  Steps: {steps}")
print(f"  Gradient accumulation: {accum_steps}")
print(f"  Total forward passes: {total_fwd}")
print(f"  Avg time per forward: {avg_time:.4f}s")
print(f"  Estimated training time: {est_time:.1f}s ({est_time/60:.1f} min)")
print(f"  Estimated training time: {est_time/3600:.2f} hours")

print("\n" + "=" * 70)
print("TEST COMPLETE - NO HANG")
print("=" * 70)
