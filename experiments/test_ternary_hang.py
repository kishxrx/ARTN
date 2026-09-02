#!/usr/bin/env python3
"""
Minimal TernarySTE Test - Isolate the hang.
"""

import sys
import time
import signal
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("MINIMAL TERNARY STE TEST")
print("=" * 60)

# Test 1: Basic tensor creation
print("\n[TEST 1] Basic tensor operations")
print("-" * 60)

try:
    print(f"Creating tensor shape (256, 1024)...")
    t = torch.randn(256, 1024)
    print(f"  Shape: {t.shape}")
    print(f"  Dtype: {t.dtype}")
    print(f"  Device: {t.device}")
    print(f"  Elements: {t.numel():,}")
    print(f"  Memory: {t.element_size() * t.numel():,} bytes")
    print("  ✓ SUCCESS")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test 2: zeros_like
print("\n[TEST 2] torch.zeros_like()")
print("-" * 60)

try:
    weight = torch.randn(256, 1024)
    print(f"Input tensor:")
    print(f"  Shape: {weight.shape}")
    print(f"  Dtype: {weight.dtype}")
    print(f"  Device: {weight.device}")
    print(f"  Elements: {weight.numel():,}")
    
    start = time.time()
    zeros = torch.zeros_like(weight)
    elapsed = time.time() - start
    
    print(f"Output tensor:")
    print(f"  Shape: {zeros.shape}")
    print(f"  Dtype: {zeros.dtype}")
    print(f"  Elapsed: {elapsed:.6f}s")
    print("  ✓ SUCCESS")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test 3: TernarySTE forward (isolated)
print("\n[TEST 3] TernarySTE forward (isolated)")
print("-" * 60)

from src.model.bitlinear import TernarySTE

try:
    weight = torch.randn(256, 1024)
    print(f"Input:")
    print(f"  Shape: {weight.shape}")
    print(f"  Dtype: {weight.dtype}")
    print(f"  Device: {weight.device}")
    print(f"  Elements: {weight.numel():,}")
    print(f"  Memory: {weight.element_size() * weight.numel():,} bytes")
    print(f"  Mean: {weight.mean():.6f}")
    print(f"  Std: {weight.std():.6f}")
    
    print(f"\nCalling TernarySTE.apply()...")
    start = time.time()
    
    # Add timeout wrapper
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out after 10 seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10 second timeout
    
    try:
        result = TernarySTE.apply(weight)
        signal.alarm(0)  # Cancel timeout
        elapsed = time.time() - start
        print(f"  Elapsed: {elapsed:.6f}s")
        print(f"  Result shape: {result[0].shape}")
        print(f"  Result dtype: {result[0].dtype}")
        print("  ✓ SUCCESS")
    except TimeoutError as e:
        signal.alarm(0)
        print(f"  ✗ TIMEOUT: {e}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: BitLinear forward (isolated)
print("\n[TEST 4] BitLinear forward (isolated)")
print("-" * 60)

from src.model.bitlinear import BitLinear

try:
    layer = BitLinear(in_features=256, out_features=1024)
    x = torch.randn(2, 50, 256)  # batch=2, seq=50, features=256
    
    print(f"Input:")
    print(f"  Shape: {x.shape}")
    print(f"  Dtype: {x.dtype}")
    print(f"  Device: {x.device}")
    print(f"  Elements: {x.numel():,}")
    
    print(f"\nCalling BitLinear.forward()...")
    start = time.time()
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    
    try:
        out = layer(x)
        signal.alarm(0)
        elapsed = time.time() - start
        print(f"  Elapsed: {elapsed:.6f}s")
        print(f"  Output shape: {out.shape}")
        print("  ✓ SUCCESS")
    except TimeoutError as e:
        signal.alarm(0)
        print(f"  ✗ TIMEOUT: {e}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: BitLinearGPT forward (isolated)
print("\n[TEST 5] BitLinearGPT forward (isolated)")
print("-" * 60)

from src.model.bitlinear import BitLinearGPT

try:
    model = BitLinearGPT(
        vocab_size=4000,
        d_model=256,
        n_heads=8,
        n_layers=1,  # Just 1 layer for test
        d_ff=1024,
        max_seq_len=256,
    )
    
    batch_size = 2
    seq_len = 50
    inputs = torch.randint(0, 4000, (batch_size, seq_len))
    targets = torch.randint(0, 4000, (batch_size, seq_len))
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input:")
    print(f"  Shape: {inputs.shape}")
    print(f"  Dtype: {inputs.dtype}")
    print(f"  Device: {inputs.device}")
    
    print(f"\nCalling BitLinearGPT.forward()...")
    start = time.time()
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout
    
    try:
        outputs = model(inputs, targets)
        signal.alarm(0)
        elapsed = time.time() - start
        print(f"  Elapsed: {elapsed:.6f}s")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Loss: {outputs['loss']:.4f}")
        print("  ✓ SUCCESS")
    except TimeoutError as e:
        signal.alarm(0)
        print(f"  ✗ TIMEOUT: {e}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
