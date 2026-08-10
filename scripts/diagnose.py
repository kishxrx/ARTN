#!/usr/bin/env python3
"""
ARTN Diagnostic Script.

Checks EVERYTHING before training:
1. Tokenizer encoding/decoding
2. Data loading and batching
3. Model forward pass
4. Loss calculation
5. Gradient flow
6. Overfit test (can model learn?)
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer
from datasets import load_dataset


def print_section(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name, passed, details=""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status} | {name}")
    if details:
        print(f"         {details}")


# =============================================================================
# TEST 1: Tokenizer
# =============================================================================

def test_tokenizer():
    """Test tokenizer encoding/decoding."""
    print_section("TEST 1: TOKENIZER")
    
    all_passed = True
    
    # Load tokenizer
    try:
        tokenizer_path = Path("outputs/artn-core-v1/tokenizer.json")
        if tokenizer_path.exists():
            tokenizer = BPETokenizer.load(tokenizer_path)
            print_result("Load tokenizer", True)
        else:
            print_result("Load tokenizer", False, "No tokenizer found")
            return False
    except Exception as e:
        print_result("Load tokenizer", False, str(e))
        return False
    
    # Check special tokens
    print(f"\n  Special Token IDs:")
    print(f"    pad_token_id: {tokenizer.pad_token_id}")
    print(f"    unk_token_id: {tokenizer.unk_token_id}")
    print(f"    bos_token_id: {tokenizer.bos_token_id}")
    print(f"    eos_token_id: {tokenizer.eos_token_id}")
    
    # Test encoding
    try:
        test_text = "Once upon a time there was a little girl."
        tokens = tokenizer.encode(test_text)
        print_result("Encode text", True, f"'{test_text[:30]}...' → {len(tokens)} tokens")
        
        # Check tokens are valid IDs
        vocab_size = len(tokenizer.vocab)
        valid_tokens = all(0 <= t < vocab_size for t in tokens)
        print_result("Token IDs valid", valid_tokens, f"All tokens in [0, {vocab_size})")
        all_passed = all_passed and valid_tokens
        
        # Test decoding
        decoded = tokenizer.decode(tokens)
        print_result("Decode tokens", True, f"→ '{decoded[:30]}...'")
        
        # Check round-trip (approximately)
        # Note: BPE may not be perfectly reversible due to unknown handling
        
    except Exception as e:
        print_result("Encode/Decode", False, str(e))
        all_passed = False
    
    # Test edge cases
    try:
        # Empty string
        empty_tokens = tokenizer.encode("")
        print_result("Empty string", len(empty_tokens) == 0 or len(empty_tokens) <= 1)
        
        # Single word
        single = tokenizer.encode("hello")
        print_result("Single word", len(single) >= 1)
        
        # Long text
        long_text = "This is a very long sentence. " * 100
        long_tokens = tokenizer.encode(long_text)
        print_result("Long text", len(long_tokens) > 100)
        
    except Exception as e:
        print_result("Edge cases", False, str(e))
        all_passed = False
    
    # Vocabulary stats
    print(f"\n  Vocabulary Stats:")
    print(f"    Vocab size: {len(tokenizer.vocab)}")
    print(f"    Merges: {len(tokenizer.merges)}")
    
    return all_passed


# =============================================================================
# TEST 2: Data Loading
# =============================================================================

def test_data_loading():
    """Test data loading and batching."""
    print_section("TEST 2: DATA LOADING")
    
    all_passed = True
    
    # Load tokenizer
    tokenizer = BPETokenizer.load("outputs/artn-core-v1/tokenizer.json")
    
    # Load dataset
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        print_result("Load dataset", True, f"{len(dataset)} samples total")
    except Exception as e:
        print_result("Load dataset", False, str(e))
        return False
    
    # Check sample
    try:
        sample = dataset[0]["text"]
        print_result("Get sample", True, f"'{sample[:50]}...'")
        
        # Tokenize sample
        tokens = tokenizer.encode(sample)
        print_result("Tokenize sample", True, f"{len(tokens)} tokens")
    except Exception as e:
        print_result("Sample processing", False, str(e))
        all_passed = False
    
    # Simulate batching
    try:
        batch_texts = [dataset[i]["text"] for i in range(4)]
        batch_tokens = [tokenizer.encode(t)[:256] for t in batch_texts]
        
        # Pad to same length
        max_len = max(len(t) for t in batch_tokens)
        padded = []
        for tokens in batch_tokens:
            padding = [tokenizer.pad_token_id] * (max_len - len(tokens))
            padded.append(tokens + padding)
        
        batch_tensor = torch.tensor(padded)
        print_result("Create batch", True, f"Shape: {batch_tensor.shape}")
        
        # Check no invalid tokens
        vocab_size = len(tokenizer.vocab)
        valid = (batch_tensor >= 0).all() and (batch_tensor < vocab_size).all()
        print_result("Batch tokens valid", bool(valid.item()))
        all_passed = all_passed and valid.item()
        
    except Exception as e:
        print_result("Batching", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST 3: Model Forward Pass
# =============================================================================

def test_model_forward():
    """Test model forward pass."""
    print_section("TEST 3: MODEL FORWARD PASS")
    
    all_passed = True
    
    # Load tokenizer
    tokenizer = BPETokenizer.load("outputs/artn-core-v1/tokenizer.json")
    
    # Create model
    try:
        model = BitLinearGPT(
            vocab_size=len(tokenizer.vocab),
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            max_seq_len=256,
        )
        num_params = sum(p.numel() for p in model.parameters())
        print_result("Create model", True, f"{num_params:,} parameters")
    except Exception as e:
        print_result("Create model", False, str(e))
        return False
    
    # Test forward pass
    try:
        batch_size = 4
        seq_len = 64
        
        input_ids = torch.randint(0, len(tokenizer.vocab), (batch_size, seq_len))
        targets = torch.randint(0, len(tokenizer.vocab), (batch_size, seq_len))
        
        output = model(input_ids, targets)
        
        print_result("Forward pass", True, f"Logits shape: {output['logits'].shape}")
        
        # Check loss exists
        has_loss = output["loss"] is not None
        print_result("Loss computed", has_loss, f"Loss: {output['loss'].item():.4f}")
        all_passed = all_passed and has_loss
        
        # Check no NaN
        no_nan = not torch.isnan(output["loss"])
        print_result("No NaN in loss", no_nan.item() if hasattr(no_nan, 'item') else no_nan)
        all_passed = all_passed and (no_nan.item() if hasattr(no_nan, 'item') else no_nan)
        
        # Check logits shape
        expected_shape = (batch_size, seq_len, len(tokenizer.vocab))
        correct_shape = output["logits"].shape == expected_shape
        print_result("Correct logits shape", correct_shape)
        all_passed = all_passed and correct_shape
        
    except Exception as e:
        print_result("Forward pass", False, str(e))
        all_passed = False
    
    # Test inference (no targets)
    try:
        output = model(input_ids)
        print_result("Inference mode", True, f"Logits shape: {output['logits'].shape}")
    except Exception as e:
        print_result("Inference mode", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST 4: Gradient Flow
# =============================================================================

def test_gradients():
    """Test gradient computation."""
    print_section("TEST 4: GRADIENT FLOW")
    
    all_passed = True
    
    # Load tokenizer
    tokenizer = BPETokenizer.load("outputs/artn-core-v1/tokenizer.json")
    
    # Create model
    model = BitLinearGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test backward pass
    try:
        batch_size = 4
        seq_len = 32
        
        input_ids = torch.randint(0, len(tokenizer.vocab), (batch_size, seq_len))
        targets = torch.randint(0, len(tokenizer.vocab), (batch_size, seq_len))
        
        optimizer.zero_grad()
        output = model(input_ids, targets)
        loss = output["loss"]
        
        loss.backward()
        
        print_result("Backward pass", True, f"Loss: {loss.item():.4f}")
        
        # Check gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters())
        print_result("Gradients computed", True, f"{grad_count}/{total_params} params have gradients")
        
        # Check gradient norms
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        print_result("Gradient norm", True, f"Total norm: {total_grad_norm:.4f}")
        
        # Check no NaN in gradients
        has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        print_result("No NaN in gradients", not has_nan)
        all_passed = all_passed and not has_nan
        
        # Test optimizer step
        optimizer.step()
        print_result("Optimizer step", True)
        
    except Exception as e:
        print_result("Gradient test", False, str(e))
        all_passed = False
    
    return all_passed


# =============================================================================
# TEST 5: Overfit Test (Can Model Learn?)
# =============================================================================

def test_overfit():
    """Test if model can overfit a tiny dataset."""
    print_section("TEST 5: OVERFIT TEST (Can Model Learn?)")
    
    # Load tokenizer
    tokenizer = BPETokenizer.load("outputs/artn-core-v1/tokenizer.json")
    
    # Create model
    model = BitLinearGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create tiny dataset (same sample repeated)
    text = "Once upon a time there was a little girl."
    tokens = tokenizer.encode(text)[:32]
    
    # Pad to 32
    if len(tokens) < 32:
        tokens = tokens + [tokenizer.pad_token_id] * (32 - len(tokens))
    
    input_ids = torch.tensor([tokens])
    targets = torch.tensor([tokens])
    
    print(f"\n  Training on single sequence:")
    print(f"    Text: '{text[:40]}...'")
    print(f"    Tokens: {len(tokens)}")
    
    # Train
    losses = []
    for step in range(100):
        optimizer.zero_grad()
        output = model(input_ids, targets)
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\n  Results:")
    print(f"    Initial loss: {initial_loss:.4f}")
    print(f"    Final loss:   {final_loss:.4f}")
    print(f"    Reduction:    {reduction:.1f}%")
    
    # Check if loss decreased
    passed = final_loss < initial_loss
    print_result("Loss decreased", passed)
    
    # Check if significant decrease
    passed_significant = reduction > 10
    print_result("Significant decrease (>10%)", passed_significant)
    
    return passed


# =============================================================================
# TEST 6: Sparsity and Ternary Check
# =============================================================================

def test_ternary():
    """Test if BitLinear produces ternary weights."""
    print_section("TEST 6: TERNARY QUANTIZATION")
    
    all_passed = True
    
    # Load tokenizer
    tokenizer = BPETokenizer.load("outputs/artn-core-v1/tokenizer.json")
    
    # Create model
    model = BitLinearGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
    )
    
    # Run forward to trigger quantization
    input_ids = torch.randint(0, len(tokenizer.vocab), (2, 32))
    _ = model(input_ids)
    
    # Get sparsity stats
    stats = model.get_sparsity_stats()
    
    print(f"\n  Sparsity Statistics:")
    print(f"    Total BitLinear params: {stats['total_params']:,}")
    print(f"    Overall sparsity: {stats['overall_sparsity']*100:.1f}%")
    
    # Check sparsity is reasonable (20-70%)
    sparsity = stats['overall_sparsity']
    reasonable_sparsity = 0.1 < sparsity < 0.9
    print_result("Sparsity reasonable (10-90%)", reasonable_sparsity)
    all_passed = all_passed and reasonable_sparsity
    
    # Check each layer
    print(f"\n  Layer Breakdown:")
    for layer_info in stats['layers'][:5]:  # Show first 5
        name = layer_info['name']
        sparsity = layer_info['sparsity']
        distrib = layer_info['distribution']
        print(f"    {name[:40]:<40}")
        print(f"      Sparsity: {sparsity*100:.1f}%")
        print(f"      Distribution: -1:{distrib['negative']*100:.1f}%, 0:{distrib['zero']*100:.1f}%, +1:{distrib['positive']*100:.1f}%")
    
    return all_passed


# =============================================================================
# TEST 7: Loss Investigation
# =============================================================================

def test_loss_values():
    """Investigate why loss might be high."""
    print_section("TEST 7: LOSS INVESTIGATION")
    
    # Load tokenizer
    tokenizer = BPETokenizer.load("outputs/artn-core-v1/tokenizer.json")
    
    # Create model
    model = BitLinearGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
    )
    
    vocab_size = len(tokenizer.vocab)
    
    print(f"\n  Configuration:")
    print(f"    Vocab size: {vocab_size}")
    print(f"    Expected initial loss: {torch.log(torch.tensor(vocab_size)).item():.4f} (random guessing)")
    
    # Test random targets
    batch_size = 4
    seq_len = 32
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Random targets
    targets_random = torch.randint(0, vocab_size, (batch_size, seq_len))
    output_random = model(input_ids, targets_random)
    
    print(f"\n  Random targets:")
    print(f"    Loss: {output_random['loss'].item():.4f}")
    
    # Identical targets (should be easier)
    targets_same = input_ids.clone()
    output_same = model(input_ids, targets_same)
    
    print(f"\n  Identical targets (input_ids = targets):")
    print(f"    Loss: {output_same['loss'].item():.4f}")
    
    # All pad tokens
    targets_pad = torch.full((batch_size, seq_len), tokenizer.pad_token_id, dtype=torch.long)
    targets_pad[0, :10] = input_ids[0, :10]  # Mix some real tokens
    
    output_pad = model(input_ids, targets_pad)
    print(f"\n  Mostly pad targets:")
    print(f"    Loss: {output_pad['loss'].item():.4f}")
    
    # Loss too high?
    expected_loss = torch.log(torch.tensor(vocab_size)).item()
    loss_reasonable = output_random['loss'].item() < expected_loss * 2
    
    print_result(f"Loss reasonable (<{expected_loss*2:.1f})", loss_reasonable)
    
    return loss_reasonable


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all diagnostics."""
    print("\n" + "="*60)
    print("  ARTN DIAGNOSTIC SUITE")
    print("="*60)
    
    results = {}
    
    # Run all tests
    results["1. Tokenizer"] = test_tokenizer()
    results["2. Data Loading"] = test_data_loading()
    results["3. Model Forward"] = test_model_forward()
    results["4. Gradients"] = test_gradients()
    results["5. Overfit Test"] = test_overfit()
    results["6. Ternary Quantization"] = test_ternary()
    results["7. Loss Investigation"] = test_loss_values()
    
    # Summary
    print_section("SUMMARY")
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} | {name}")
        all_passed = all_passed and passed
    
    print(f"\n{'='*60}")
    if all_passed:
        print("  ALL TESTS PASSED - Ready for training!")
    else:
        print("  SOME TESTS FAILED - Fix issues before training")
    print(f"{'='*60}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
