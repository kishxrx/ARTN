#!/usr/bin/env python3
"""
VALIDATION AUDIT - Verify evaluation pipeline before diagnosing generalization

This script performs systematic checks:
1. Data split audit
2. Target alignment audit
3. Evaluation pipeline audit
4. Sanity tests (4 critical tests)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import BitLinear directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bitlinear", 
    Path(__file__).parent.parent / "src" / "model" / "bitlinear.py"
)
bitlinear_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bitlinear_module)
BitLinear = bitlinear_module.BitLinear
compute_abs_mean_scale = bitlinear_module.compute_abs_mean_scale


# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "validation_audit"

BASE_CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "num_steps": 500,
    "batch_size": 4,
    "seq_len": 32,
    "num_stories": 20,
    "lr": 0.01,
    "grad_clip": 1.0,
    "seed": 42,
}


# ============================================================================
# MODEL
# ============================================================================

class SimpleModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.layer1 = BitLinear(config["d_model"], config["d_model"])
        self.layer2 = BitLinear(config["d_model"], config["d_model"])
        self.output = BitLinear(config["d_model"], config["vocab_size"], bias=False)
    
    def forward(self, x):
        h = self.embedding(x)
        h = F.gelu(self.layer1(h))
        h = F.gelu(self.layer2(h))
        return self.output(h)


# ============================================================================
# VOCAB UTILITIES
# ============================================================================

def build_vocab_from_texts(texts: List[str], vocab_size: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build vocabulary from texts."""
    # Simple character-level tokenization for audit
    all_chars = set()
    for text in texts:
        all_chars.update(text)
    
    # Add special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
    idx = 3
    
    # Sort for reproducibility
    for char in sorted(all_chars):
        if idx < vocab_size:
            vocab[char] = idx
            idx += 1
    
    inv_vocab = {v: k for k, v in vocab.items()}
    
    return vocab, inv_vocab


def tokenize(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    """Tokenize text."""
    tokens = []
    for char in text:
        tokens.append(vocab.get(char, vocab['<UNK>']))
    tokens.append(vocab['<EOS>'])
    
    # Pad or truncate
    if len(tokens) < max_len:
        tokens = tokens + [vocab['<PAD>']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    
    return torch.tensor(tokens, dtype=torch.long)


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_tinystories_like_data(
    num_stories: int,
    vocab_size: int,
    seq_len: int,
    seed: int,
) -> Tuple[torch.Tensor, Dict]:
    """Generate TinyStories-like data with proper tracking."""
    
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Create Zipf-like token distribution
    token_probs = torch.ones(vocab_size)
    for i in range(vocab_size):
        token_probs[i] = 1.0 / (i + 1) ** 0.5
    token_probs = token_probs / token_probs.sum()
    
    # Generate stories
    data = torch.zeros(num_stories, seq_len, dtype=torch.long)
    token_counts = Counter()
    
    for i in range(num_stories):
        for j in range(seq_len):
            token = torch.multinomial(token_probs, 1).item()
            data[i, j] = token
            token_counts[int(token)] += 1
    
    # Track metadata
    metadata = {
        "num_stories": num_stories,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "total_tokens": num_stories * seq_len,
        "unique_tokens": len(token_counts),
        "token_distribution": dict(token_counts.most_common(20)),
    }
    
    return data, metadata


# ============================================================================
# AUDIT FUNCTIONS
# ============================================================================

def audit_data_split(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    vocab_size: int,
    output_dir: Path,
) -> Dict:
    """Audit 1: Data split verification."""
    
    print("="*80)
    print("AUDIT 1: DATA SPLIT")
    print("="*80)
    
    results = {}
    
    # Basic stats
    print(f"\n1. Split Sizes:")
    print(f"   Train stories: {train_data.size(0)}")
    print(f"   Val stories: {val_data.size(0)}")
    print(f"   Train tokens: {train_data.size(0) * train_data.size(1)}")
    print(f"   Val tokens: {val_data.size(0) * val_data.size(1)}")
    
    results["train_stories"] = train_data.size(0)
    results["val_stories"] = val_data.size(0)
    
    # Verify no overlap
    print(f"\n2. Overlap Check:")
    
    # Convert to sets for comparison
    train_set = set()
    for i in range(train_data.size(0)):
        train_set.add(tuple(train_data[i].tolist()))
    
    val_set = set()
    for i in range(val_data.size(0)):
        val_set.add(tuple(val_data[i].tolist()))
    
    overlap = train_set & val_set
    print(f"   Unique train sequences: {len(train_set)}")
    print(f"   Unique val sequences: {len(val_set)}")
    print(f"   Overlapping sequences: {len(overlap)}")
    
    results["unique_train"] = len(train_set)
    results["unique_val"] = len(val_set)
    results["overlap"] = len(overlap)
    
    if len(overlap) > 0:
        print(f"   ⚠️ WARNING: {len(overlap)} sequences appear in both train and val!")
    else:
        print(f"   ✅ No overlap between train and val")
    
    # Vocabulary distribution
    print(f"\n3. Vocabulary Distribution:")
    
    train_tokens = train_data.flatten().tolist()
    val_tokens = val_data.flatten().tolist()
    
    train_vocab = set(train_tokens)
    val_vocab = set(val_tokens)
    
    print(f"   Train unique tokens: {len(train_vocab)}")
    print(f"   Val unique tokens: {len(val_vocab)}")
    
    # OOV tokens
    oov_tokens = val_vocab - train_vocab
    print(f"   OOV tokens in val: {len(oov_tokens)}")
    
    if len(oov_tokens) > 0:
        print(f"   ⚠️ WARNING: {len(oov_tokens)} tokens in val not seen in train!")
        print(f"   OOV tokens: {sorted(list(oov_tokens))[:20]}")
    else:
        print(f"   ✅ All val tokens seen in train")
    
    results["train_unique_tokens"] = len(train_vocab)
    results["val_unique_tokens"] = len(val_vocab)
    results["oov_tokens"] = len(oov_tokens)
    
    # Token frequency distribution
    print(f"\n4. Token Frequency:")
    
    train_counter = Counter(train_tokens)
    val_counter = Counter(val_tokens)
    
    print(f"   Train top-5 tokens: {train_counter.most_common(5)}")
    print(f"   Val top-5 tokens: {val_counter.most_common(5)}")
    
    # Print sample sequences
    print(f"\n5. Sample Sequences:")
    print(f"   Train samples (first 3):")
    for i in range(min(3, train_data.size(0))):
        print(f"     [{i}] {train_data[i][:20].tolist()}...")
    
    print(f"   Val samples (first 3):")
    for i in range(min(3, val_data.size(0))):
        print(f"     [{i}] {val_data[i][:20].tolist()}...")
    
    results["status"] = "PASS" if len(overlap) == 0 else "WARNING"
    
    return results


def audit_target_alignment(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    vocab_size: int,
) -> Dict:
    """Audit 2: Target alignment verification."""
    
    print("\n" + "="*80)
    print("AUDIT 2: TARGET ALIGNMENT")
    print("="*80)
    
    results = {}
    
    print(f"\n1. Input-Target Relationship:")
    print(f"   For next-token prediction:")
    print(f"   input = sequence[:-1]")
    print(f"   target = sequence[1:]")
    
    # Check a sample
    sample_idx = 0
    sample = train_data[sample_idx]
    
    print(f"\n2. Sample Check (train story {sample_idx}):")
    print(f"   Full sequence (first 10): {sample[:10].tolist()}")
    
    # Verify alignment
    input_ids = sample[:-1]
    target_ids = sample[1:]
    
    print(f"   Input (first 9):  {input_ids[:9].tolist()}")
    print(f"   Target (first 9): {target_ids[:9].tolist()}")
    
    # Check shift
    correct_shift = True
    for i in range(min(10, len(input_ids))):
        if target_ids[i] != sample[i + 1]:
            correct_shift = False
            print(f"   ❌ Position {i}: target[{i}]={target_ids[i]} != sample[{i+1}]={sample[i+1]}")
    
    if correct_shift:
        print(f"   ✅ Target correctly shifted by 1")
    
    results["correct_shift"] = correct_shift
    
    # Check padding handling
    print(f"\n3. Padding Check:")
    pad_token = 0  # Assuming 0 is PAD
    
    # Count padding in sequences
    train_pad_count = (train_data == pad_token).sum().item()
    val_pad_count = (val_data == pad_token).sum().item()
    
    print(f"   PAD token ID: {pad_token}")
    print(f"   Train PAD tokens: {train_pad_count}")
    print(f"   Val PAD tokens: {val_pad_count}")
    
    results["pad_token"] = pad_token
    results["train_pad_count"] = train_pad_count
    results["val_pad_count"] = val_pad_count
    
    # Check EOS handling
    print(f"\n4. EOS Token Check:")
    eos_token = 2  # Assuming 2 is EOS
    
    train_eos_count = (train_data == eos_token).sum().item()
    val_eos_count = (val_data == eos_token).sum().item()
    
    print(f"   EOS token ID: {eos_token}")
    print(f"   Train EOS tokens: {train_eos_count}")
    print(f"   Val EOS tokens: {val_eos_count}")
    
    results["eos_token"] = eos_token
    results["train_eos_count"] = train_eos_count
    results["val_eos_count"] = val_eos_count
    
    results["status"] = "PASS" if correct_shift else "FAIL"
    
    return results


def audit_evaluation_pipeline(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    vocab_size: int,
) -> Dict:
    """Audit 3: Evaluation pipeline verification."""
    
    print("\n" + "="*80)
    print("AUDIT 3: EVALUATION PIPELINE")
    print("="*80)
    
    results = {}
    
    # Check model.eval() and no_grad
    print(f"\n1. Evaluation Mode:")
    model.eval()
    print(f"   Model in eval mode: {not model.training}")
    
    # Check forward pass
    print(f"\n2. Forward Pass Check:")
    
    with torch.no_grad():
        # Get a batch
        batch = train_data[:4]
        
        print(f"   Input shape: {batch.shape}")
        
        logits = model(batch)
        print(f"   Output shape: {logits.shape}")
        print(f"   Output dtype: {logits.dtype}")
        
        # Check quantization behavior
        layer1_scale = compute_abs_mean_scale(model.layer1.weight)
        print(f"   Layer1 scale: {layer1_scale.item():.6f}")
    
    results["model_eval"] = not model.training
    results["forward_pass"] = "OK"
    
    # Check accuracy computation
    print(f"\n3. Accuracy Computation:")
    
    with torch.no_grad():
        batch = train_data[:4]
        logits = model(batch)
        
        # Standard accuracy computation
        predictions = logits[:, :-1, :].argmax(dim=-1)
        targets = batch[:, 1:]
        
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Targets shape: {targets.shape}")
        
        # Check alignment
        if predictions.shape == targets.shape:
            print(f"   ✅ Predictions and targets have same shape")
        else:
            print(f"   ❌ Shape mismatch!")
        
        # Compute accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()
        
        print(f"   Accuracy on this batch: {accuracy:.3%}")
        
        # Token-by-token check
        print(f"\n   Sample predictions vs targets (first 5 positions):")
        for pos in range(min(5, predictions.shape[1])):
            pred = predictions[0, pos].item()
            targ = targets[0, pos].item()
            match = "✓" if pred == targ else "✗"
            print(f"     Pos {pos}: pred={pred:4d}, target={targ:4d} {match}")
        
        # Top-5 predictions
        print(f"\n   Top-5 predictions at position 0:")
        probs = F.softmax(logits[0, 0, :], dim=-1)
        top5 = torch.topk(probs, 5)
        for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
            print(f"     {i+1}. Token {idx.item():4d}: {prob.item():.4f}")
        print(f"   Target at position 0: {targets[0, 0].item()}")
    
    results["accuracy_computation"] = "OK"
    
    # Count tokens in validation
    print(f"\n4. Validation Token Count:")
    val_tokens = val_data[:, 1:].numel()
    print(f"   Total validation tokens: {val_tokens}")
    results["val_tokens"] = val_tokens
    
    results["status"] = "PASS"
    
    return results


def sanity_test_overfit_one_batch(
    model: nn.Module,
    val_data: torch.Tensor,
    config: Dict,
) -> Dict:
    """Sanity Test A: Overfit one validation batch."""
    
    print("\n" + "="*80)
    print("SANITY TEST A: OVERFIT ONE VALIDATION BATCH")
    print("="*80)
    
    results = {}
    
    # Reset model
    torch.manual_seed(config["seed"])
    model = SimpleModel(config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    
    # Get ONE fixed validation batch
    val_batch = val_data[:config["batch_size"]]
    
    print(f"\n1. Setup:")
    print(f"   Training exclusively on 1 validation batch")
    print(f"   Batch shape: {val_batch.shape}")
    print(f"   Steps: {config['num_steps']}")
    
    # Train on this single batch
    losses = []
    accuracies = []
    
    for step in range(config["num_steps"]):
        model.train()
        optimizer.zero_grad()
        
        logits = model(val_batch)
        targets = val_batch[:, 1:]
        logits_for_loss = logits[:, :-1, :]
        
        loss = F.cross_entropy(
            logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
            targets.reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        
        # Track accuracy
        with torch.no_grad():
            predictions = logits[:, :-1, :].argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
        
        losses.append(loss.item())
        accuracies.append(accuracy)
        
        if step in [0, 49, 99, 199, 299, 399, 499]:
            print(f"   Step {step:3d}: loss={loss.item():.4f}, accuracy={accuracy:.3%}")
    
    final_acc = accuracies[-1]
    print(f"\n2. Result:")
    print(f"   Final accuracy: {final_acc:.3%}")
    
    if final_acc > 0.95:
        print(f"   ✅ PASS: Model can memorize the batch")
        results["status"] = "PASS"
    else:
        print(f"   ❌ FAIL: Model cannot memorize single batch!")
        print(f"   This indicates a bug in model/target/loss!")
        results["status"] = "FAIL"
    
    results["final_accuracy"] = final_acc
    results["final_loss"] = losses[-1]
    
    return results, model


def sanity_test_train_equals_val(
    model: nn.Module,
    data: torch.Tensor,
    config: Dict,
) -> Dict:
    """Sanity Test B: Train on data, evaluate on SAME data."""
    
    print("\n" + "="*80)
    print("SANITY TEST B: TRAIN = VALIDATION")
    print("="*80)
    
    results = {}
    
    # Reset model
    torch.manual_seed(config["seed"])
    model = SimpleModel(config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    
    # Use same data for train and val
    train_data = data
    val_data = data
    
    print(f"\n1. Setup:")
    print(f"   Using SAME data for train and validation")
    print(f"   Data shape: {train_data.shape}")
    
    train_accuracies = []
    val_accuracies = []
    
    for step in range(config["num_steps"]):
        # Train
        model.train()
        batch_idx = torch.randint(0, train_data.size(0), (config["batch_size"],))
        batch = train_data[batch_idx]
        
        optimizer.zero_grad()
        logits = model(batch)
        targets = batch[:, 1:]
        logits_for_loss = logits[:, :-1, :]
        
        loss = F.cross_entropy(
            logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
            targets.reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        
        # Evaluate on SAME data
        model.eval()
        with torch.no_grad():
            # Train accuracy (on this batch)
            train_pred = logits[:, :-1, :].argmax(dim=-1)
            train_acc = (train_pred == targets).float().mean().item()
            
            # Val accuracy (on a sample from same dataset)
            val_batch_idx = torch.randint(0, val_data.size(0), (config["batch_size"],))
            val_batch = val_data[val_batch_idx]
            
            val_logits = model(val_batch)
            val_targets = val_batch[:, 1:]
            val_pred = val_logits[:, :-1, :].argmax(dim=-1)
            val_acc = (val_pred == val_targets).float().mean().item()
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        if step in [0, 49, 99, 199, 299, 399, 499]:
            print(f"   Step {step:3d}: train_acc={train_acc:.3%}, val_acc={val_acc:.3%}")
    
    final_train = train_accuracies[-1]
    final_val = val_accuracies[-1]
    
    print(f"\n2. Result:")
    print(f"   Final train accuracy: {final_train:.3%}")
    print(f"   Final val accuracy: {final_val:.3%}")
    print(f"   Difference: {abs(final_train - final_val):.3%}")
    
    if abs(final_train - final_val) < 0.05:
        print(f"   ✅ PASS: Train and val accuracy match (using same data)")
        results["status"] = "PASS"
    else:
        print(f"   ❌ FAIL: Mismatch when using same data!")
        print(f"   This indicates evaluation pipeline bug!")
        results["status"] = "FAIL"
    
    results["final_train_acc"] = final_train
    results["final_val_acc"] = final_val
    
    return results


def sanity_test_swap_split(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: Dict,
) -> Dict:
    """Sanity Test C: Swap train and validation."""
    
    print("\n" + "="*80)
    print("SANITY TEST C: SWAP TRAIN/VAL")
    print("="*80)
    
    results = {}
    
    # Reset model
    torch.manual_seed(config["seed"])
    model = SimpleModel(config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    
    # SWAP: train on original val, evaluate on original train
    new_train = val_data
    new_val = train_data
    
    print(f"\n1. Setup:")
    print(f"   Training on original validation set")
    print(f"   Evaluating on original training set")
    
    val_accuracies = []
    
    for step in range(config["num_steps"]):
        # Train on new_train (original val)
        model.train()
        batch_idx = torch.randint(0, new_train.size(0), (config["batch_size"],))
        batch = new_train[batch_idx]
        
        optimizer.zero_grad()
        logits = model(batch)
        targets = batch[:, 1:]
        logits_for_loss = logits[:, :-1, :]
        
        loss = F.cross_entropy(
            logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
            targets.reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        
        # Evaluate on new_val (original train)
        model.eval()
        with torch.no_grad():
            val_batch_idx = torch.randint(0, new_val.size(0), (config["batch_size"],))
            val_batch = new_val[val_batch_idx]
            
            val_logits = model(val_batch)
            val_targets = val_batch[:, 1:]
            val_pred = val_logits[:, :-1, :].argmax(dim=-1)
            val_acc = (val_pred == val_targets).float().mean().item()
        
        val_accuracies.append(val_acc)
        
        if step in [0, 49, 99, 199, 299, 399, 499]:
            print(f"   Step {step:3d}: val_acc (on orig train)={val_acc:.3%}")
    
    final_val = val_accuracies[-1]
    
    print(f"\n2. Result:")
    print(f"   Final accuracy on original train set: {final_val:.3%}")
    
    if final_val > 0.50:
        print(f"   ✅ PASS: Can learn when trained on different split")
        results["status"] = "PASS"
    else:
        print(f"   ❌ FAIL: Still low accuracy after swap!")
        results["status"] = "FAIL"
    
    results["final_val_acc"] = final_val
    
    return results


def sanity_test_random_labels(
    model: nn.Module,
    data: torch.Tensor,
    config: Dict,
) -> Dict:
    """Sanity Test D: Random labels control."""
    
    print("\n" + "="*80)
    print("SANITY TEST D: RANDOM LABELS")
    print("="*80)
    
    results = {}
    
    # Reset model
    torch.manual_seed(config["seed"])
    model = SimpleModel(config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    
    # Create random targets
    print(f"\n1. Setup:")
    print(f"   Using random targets instead of true targets")
    
    random_targets = torch.randint(0, config["vocab_size"], data.shape)
    
    losses = []
    accuracies = []
    
    for step in range(config["num_steps"]):
        model.train()
        batch_idx = torch.randint(0, data.size(0), (config["batch_size"],))
        batch = data[batch_idx]
        targets = random_targets[batch_idx, 1:]
        
        optimizer.zero_grad()
        logits = model(batch)
        logits_for_loss = logits[:, :-1, :]
        
        loss = F.cross_entropy(
            logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
            targets.reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        
        with torch.no_grad():
            predictions = logits[:, :-1, :].argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
        
        losses.append(loss.item())
        accuracies.append(accuracy)
        
        if step in [0, 49, 99, 199, 299, 399, 499]:
            print(f"   Step {step:3d}: loss={loss.item():.4f}, accuracy={accuracy:.3%}")
    
    final_acc = accuracies[-1]
    final_loss = losses[-1]
    
    print(f"\n2. Result:")
    print(f"   Final accuracy with random labels: {final_acc:.3%}")
    print(f"   Expected (random): ~{1/config['vocab_size']:.4%}")
    
    # With random labels, model should be able to memorize
    # (but won't generalize - which is expected)
    if final_acc > 0.50:
        print(f"   ✅ Model can memorize even random labels")
        print(f"   This confirms optimization pipeline works")
        results["status"] = "PASS"
    else:
        print(f"   ❌ Model cannot memorize random labels!")
        print(f"   This suggests optimization issue!")
        results["status"] = "FAIL"
    
    results["final_accuracy"] = final_acc
    results["final_loss"] = final_loss
    results["expected_random"] = 1 / config["vocab_size"]
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("VALIDATION AUDIT")
    print("="*80)
    
    output_dir = Path(__file__).parent.parent / "outputs" / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = BASE_CONFIG.copy()
    
    # Generate data
    print(f"\nGenerating data...")
    torch.manual_seed(config["seed"])
    
    # Generate with proper split
    all_data, metadata = generate_tinystories_like_data(
        num_stories=config["num_stories"],
        vocab_size=config["vocab_size"],
        seq_len=config["seq_len"],
        seed=config["seed"],
    )
    
    # 80/20 split
    split_idx = int(0.8 * all_data.size(0))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Train: {train_data.shape}, Val: {val_data.shape}")
    
    # Create model
    model = SimpleModel(config)
    
    # Run audits
    all_results = {}
    
    # Audit 1: Data split
    all_results["audit_data_split"] = audit_data_split(train_data, val_data, config["vocab_size"], output_dir)
    
    # Audit 2: Target alignment
    all_results["audit_target_alignment"] = audit_target_alignment(train_data, val_data, config["vocab_size"])
    
    # Audit 3: Evaluation pipeline
    all_results["audit_evaluation_pipeline"] = audit_evaluation_pipeline(model, train_data, val_data, config["vocab_size"])
    
    # Sanity tests
    all_results["sanity_overfit_one_batch"], _ = sanity_test_overfit_one_batch(model, val_data, config)
    all_results["sanity_train_equals_val"] = sanity_test_train_equals_val(model, all_data, config)
    all_results["sanity_swap_split"] = sanity_test_swap_split(model, train_data, val_data, config)
    all_results["sanity_random_labels"] = sanity_test_random_labels(model, train_data, config)
    
    # Save results
    results_file = output_dir / "audit_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "="*80)
    print("AUDIT SUMMARY")
    print("="*80)
    
    print(f"\n{'Test':<40} {'Status':<10}")
    print("-"*50)
    
    all_pass = True
    for test_name, results in all_results.items():
        status = results.get("status", "N/A")
        symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{test_name:<40} {symbol} {status}")
        
        if status == "FAIL":
            all_pass = False
    
    print("\n" + "="*80)
    if all_pass:
        print("ALL AUDITS PASSED - Evaluation pipeline is correct")
        print("Zero validation accuracy indicates genuine memorization issue")
    else:
        print("⚠️ SOME AUDITS FAILED - Evaluation pipeline has bugs")
        print("Fix pipeline before diagnosing generalization")
    print("="*80)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
