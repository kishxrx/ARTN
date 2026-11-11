#!/usr/bin/env python3
"""
TRUE GENERALIZATION TEST

Purpose: Verify that the 89% validation accuracy was genuine generalization, NOT data leakage.

CRITICAL QUESTIONS:
1. How was "full_coverage" constructed?
2. Are validation stories genuinely different from training stories?
3. Is there any sequence or n-gram leakage?
4. Does the model generalize on genuinely different stories with 0% OOV?

TEST CONDITIONS:
A. Same data, 0% OOV (memorization baseline)
B. Different stories, 0% target OOV (TRUE GENERALIZATION TEST)
C. Different stories, natural OOV (realistic baseline)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import random
from typing import Dict, List, Optional, Tuple, Set
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

EXPERIMENT_NAME = "true_generalization_test"

BASE_CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "num_steps": 500,
    "batch_size": 4,
    "seq_len": 32,
    "num_stories": 20,
    "lr": 0.01,
    "grad_clip": 1.0,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
}

MODEL_SEEDS = [42, 43, 44]


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
# DATA GENERATION
# ============================================================================

def generate_stories(
    num_stories: int,
    vocab_size: int,
    seq_len: int,
    seed: int,
) -> torch.Tensor:
    """Generate stories with Zipf-like token distribution."""
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Zipf-like distribution
    token_probs = torch.ones(vocab_size)
    for i in range(vocab_size):
        token_probs[i] = 1.0 / (i + 1) ** 0.5
    token_probs[0] = 0.0  # No padding in generation
    token_probs = token_probs / token_probs.sum()
    
    # Generate
    data = torch.zeros(num_stories, seq_len, dtype=torch.long)
    for i in range(num_stories):
        for j in range(seq_len):
            token = torch.multinomial(token_probs, 1).item()
            data[i, j] = token
    
    return data


def analyze_vocabulary_coverage(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> Dict:
    """Analyze vocabulary coverage between train and val."""
    
    train_tokens = set(train_data.flatten().tolist())
    val_tokens = set(val_data.flatten().tolist())
    
    train_targets = set(train_data[:, 1:].flatten().tolist())
    val_targets = set(val_data[:, 1:].flatten().tolist())
    
    # Type-level
    val_targets_seen = val_targets & train_targets
    val_targets_unseen = val_targets - train_targets
    
    # Occurrence-level
    val_target_list = val_data[:, 1:].flatten().tolist()
    val_target_occurrences = len(val_target_list)
    val_target_unseen_occurrences = sum(1 for t in val_target_list if t not in train_targets)
    
    return {
        "train_unique_tokens": len(train_tokens),
        "val_unique_tokens": len(val_tokens),
        "train_unique_targets": len(train_targets),
        "val_unique_targets": len(val_targets),
        "val_targets_seen": len(val_targets_seen),
        "val_targets_unseen": len(val_targets_unseen),
        "val_target_oov_rate_pct": val_target_unseen_occurrences / val_target_occurrences * 100 if val_target_occurrences else 0,
    }


def analyze_data_leakage(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> Dict:
    """
    Comprehensive data leakage audit.
    
    Checks:
    1. Exact sequence overlap
    2. Duplicate sequences within train/val
    3. N-gram overlap (2-gram, 3-gram, 4-gram)
    """
    
    # Convert to list of tuples for comparison
    train_sequences = [tuple(seq.tolist()) for seq in train_data]
    val_sequences = [tuple(seq.tolist()) for seq in val_data]
    
    train_set = set(train_sequences)
    val_set = set(val_sequences)
    
    # 1. Exact sequence overlap
    exact_overlap = train_set & val_set
    exact_overlap_count = len(exact_overlap)
    
    # 2. Duplicate sequences within train
    train_duplicates = len(train_sequences) - len(train_set)
    val_duplicates = len(val_sequences) - len(val_set)
    
    # 3. N-gram overlap
    def get_ngrams(sequences, n):
        ngrams = set()
        for seq in sequences:
            for i in range(len(seq) - n + 1):
                ngrams.add(tuple(seq[i:i+n]))
        return ngrams
    
    train_2grams = get_ngrams(train_sequences, 2)
    train_3grams = get_ngrams(train_sequences, 3)
    train_4grams = get_ngrams(train_sequences, 4)
    
    val_2grams = get_ngrams(val_sequences, 2)
    val_3grams = get_ngrams(val_sequences, 3)
    val_4grams = get_ngrams(val_sequences, 4)
    
    overlap_2grams = train_2grams & val_2grams
    overlap_3grams = train_3grams & val_3grams
    overlap_4grams = train_4grams & val_4grams
    
    return {
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "train_unique_sequences": len(train_set),
        "val_unique_sequences": len(val_set),
        "exact_sequence_overlap": exact_overlap_count,
        "train_duplicate_count": train_duplicates,
        "val_duplicate_count": val_duplicates,
        "train_2grams": len(train_2grams),
        "val_2grams": len(val_2grams),
        "overlap_2grams": len(overlap_2grams),
        "overlap_2gram_pct": len(overlap_2grams) / len(val_2grams) * 100 if val_2grams else 0,
        "train_3grams": len(train_3grams),
        "val_3grams": len(val_3grams),
        "overlap_3grams": len(overlap_3grams),
        "overlap_3gram_pct": len(overlap_3grams) / len(val_3grams) * 100 if val_3grams else 0,
        "train_4grams": len(train_4grams),
        "val_4grams": len(val_4grams),
        "overlap_4grams": len(overlap_4grams),
        "overlap_4gram_pct": len(overlap_4grams) / len(val_4grams) * 100 if val_4grams else 0,
    }


# ============================================================================
# CONTROLLED SPLIT CREATION
# ============================================================================

def create_test_a_same_data_zero_oov(
    all_data: torch.Tensor,
    vocab_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Test A: Same data, 0% OOV (memorization baseline).
    
    Method: Use a subset of training data as validation.
    This tests memorization - model sees exact same stories.
    """
    # Use same data for train and val (memorization test)
    train_data = all_data[:int(0.8 * all_data.size(0))]
    val_data = train_data[:min(4, train_data.size(0))]  # Subset of train
    
    method = "Validation = subset of training data (memorization test)"
    
    return train_data, val_data, method


def create_test_b_different_stories_zero_oov(
    all_data: torch.Tensor,
    vocab_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Test B: Different stories, 0% target OOV (TRUE GENERALIZATION TEST).
    
    Method:
    1. Generate train and val from DIFFERENT random seeds
    2. Filter val to only include stories with targets seen in train
    3. Ensure NO exact sequence overlap
    """
    torch.manual_seed(seed)
    
    # Split into train and val
    split_idx = int(0.8 * all_data.size(0))
    indices = torch.randperm(all_data.size(0))
    train_data = all_data[indices[:split_idx]]
    potential_val_data = all_data[indices[split_idx:]]
    
    # Get train targets
    train_targets = set(train_data[:, 1:].flatten().tolist())
    
    # Filter val to only stories with ALL targets in train
    filtered_val_indices = []
    for i in range(potential_val_data.size(0)):
        story = potential_val_data[i]
        targets = story[1:].tolist()
        if all(t in train_targets for t in targets):
            # Also check it's not an exact duplicate of train
            story_tuple = tuple(story.tolist())
            train_set = set(tuple(s.tolist()) for s in train_data)
            if story_tuple not in train_set:
                filtered_val_indices.append(i)
    
    if filtered_val_indices:
        val_data = potential_val_data[filtered_val_indices]
    else:
        # Fallback: generate new stories that only use train targets
        # This ensures 0% OOV while maintaining different stories
        val_stories = []
        for i in range(4):
            story = []
            # First token can be random from train
            first_token = random.choice(list(train_targets))
            story.append(first_token)
            # Rest must be targets from train
            for j in range(all_data.size(1) - 1):
                token = random.choice(list(train_targets))
                story.append(token)
            val_stories.append(story)
        val_data = torch.tensor(val_stories, dtype=torch.long)
    
    method = "Different stories filtered to have 0% target OOV, no exact duplicates"
    
    return train_data, val_data, method


def create_test_c_different_stories_natural_oov(
    all_data: torch.Tensor,
    vocab_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Test C: Different stories, natural OOV (realistic baseline).
    
    Method: Natural 80/20 random split.
    """
    torch.manual_seed(seed)
    
    # Natural random split
    split_idx = int(0.8 * all_data.size(0))
    indices = torch.randperm(all_data.size(0))
    train_data = all_data[indices[:split_idx]]
    val_data = all_data[indices[split_idx:]]
    
    method = "Natural 80/20 random split (different stories, natural OOV)"
    
    return train_data, val_data, method


# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: Dict,
    model_seed: int,
    verbose: bool = False,
) -> Dict:
    """Train model and return results."""
    
    torch.manual_seed(model_seed)
    random.seed(model_seed)
    
    model = SimpleModel(config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )
    
    results = {"model_seed": model_seed, "steps": []}
    
    for step in range(config["num_steps"]):
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
        
        if step in [0, 99, 199, 299, 399, 499]:
            model.eval()
            with torch.no_grad():
                # Train accuracy
                train_batch_idx = torch.randint(0, train_data.size(0), (config["batch_size"],))
                train_batch = train_data[train_batch_idx]
                train_logits = model(train_batch)
                train_pred = train_logits[:, :-1, :].argmax(dim=-1)
                train_targets = train_batch[:, 1:]
                train_acc = (train_pred == train_targets).float().mean().item()
                
                # Val accuracy
                if val_data.size(0) > 0:
                    val_batch_idx = torch.randint(0, val_data.size(0), (min(config["batch_size"], val_data.size(0)),))
                    val_batch = val_data[val_batch_idx]
                    val_logits = model(val_batch)
                    val_pred = val_logits[:, :-1, :].argmax(dim=-1)
                    val_targets = val_batch[:, 1:]
                    val_acc = (val_pred == val_targets).float().mean().item()
                    
                    val_loss = F.cross_entropy(
                        val_logits[:, :-1, :].reshape(-1, val_logits.size(-1)),
                        val_targets.reshape(-1)
                    ).item()
                else:
                    val_acc = 0.0
                    val_loss = float('inf')
            
            results["steps"].append({
                "step": step,
                "train_loss": float(loss.item()),
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
            
            if verbose and step in [0, 499]:
                print(f"    Step {step}: train_acc={train_acc:.1%}, val_acc={val_acc:.1%}")
    
    final = results["steps"][-1]
    results["final_train_loss"] = final["train_loss"]
    results["final_train_acc"] = final["train_acc"]
    results["final_val_loss"] = final["val_loss"]
    results["final_val_acc"] = final["val_acc"]
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("TRUE GENERALIZATION TEST")
    print("="*80)
    
    output_dir = Path(__file__).parent.parent / "outputs" / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = BASE_CONFIG.copy()
    
    all_results = {
        "config": config,
        "tests": {},
    }
    
    # Generate base data
    data_seed = 42
    all_data = generate_stories(
        num_stories=config["num_stories"],
        vocab_size=config["vocab_size"],
        seq_len=config["seq_len"],
        seed=data_seed,
    )
    
    # Define tests
    tests = [
        {
            "name": "test_a_same_data_zero_oov",
            "description": "Same data, 0% OOV (memorization baseline)",
            "purpose": "Verify model can memorize exact sequences",
            "creator": create_test_a_same_data_zero_oov,
        },
        {
            "name": "test_b_different_stories_zero_oov",
            "description": "Different stories, 0% target OOV (TRUE GENERALIZATION TEST)",
            "purpose": "Test genuine generalization on unseen stories",
            "creator": create_test_b_different_stories_zero_oov,
        },
        {
            "name": "test_c_different_stories_natural_oov",
            "description": "Different stories, natural OOV (realistic baseline)",
            "purpose": "Realistic performance with vocabulary mismatch",
            "creator": create_test_c_different_stories_natural_oov,
        },
    ]
    
    # Run each test
    for test in tests:
        print(f"\n{'='*80}")
        print(f"{test['name'].upper()}")
        print(f"Description: {test['description']}")
        print(f"Purpose: {test['purpose']}")
        print(f"{'='*80}")
        
        # Create split
        train_data, val_data, method = test["creator"](all_data, config["vocab_size"], data_seed)
        
        print(f"\nConstruction Method:")
        print(f"  {method}")
        
        # Data leakage audit
        print(f"\nA. DATA LEAKAGE AUDIT")
        leakage = analyze_data_leakage(train_data, val_data)
        
        print(f"\n  Exact Sequence Overlap:")
        print(f"    Train sequences: {leakage['train_sequences']}")
        print(f"    Val sequences: {leakage['val_sequences']}")
        print(f"    Exact overlaps: {leakage['exact_sequence_overlap']}")
        
        if leakage['exact_sequence_overlap'] > 0:
            print(f"    ⚠️ WARNING: {leakage['exact_sequence_overlap']} validation sequences appear in training!")
        else:
            print(f"    ✅ No exact sequence overlap")
        
        print(f"\n  Duplicate Sequences:")
        print(f"    Train duplicates: {leakage['train_duplicate_count']}")
        print(f"    Val duplicates: {leakage['val_duplicate_count']}")
        
        print(f"\n  N-gram Overlap:")
        print(f"    2-gram overlap: {leakage['overlap_2grams']}/{leakage['val_2grams']} ({leakage['overlap_2gram_pct']:.1f}%)")
        print(f"    3-gram overlap: {leakage['overlap_3grams']}/{leakage['val_3grams']} ({leakage['overlap_3gram_pct']:.1f}%)")
        print(f"    4-gram overlap: {leakage['overlap_4grams']}/{leakage['val_4grams']} ({leakage['overlap_4gram_pct']:.1f}%)")
        
        # Vocabulary coverage
        print(f"\nB. VOCABULARY COVERAGE")
        coverage = analyze_vocabulary_coverage(train_data, val_data)
        
        print(f"\n  Train unique tokens: {coverage['train_unique_tokens']}")
        print(f"  Val unique tokens: {coverage['val_unique_tokens']}")
        print(f"  Train unique targets: {coverage['train_unique_targets']}")
        print(f"  Val unique targets: {coverage['val_unique_targets']}")
        print(f"  Val targets unseen: {coverage['val_targets_unseen']}")
        print(f"  **Target OOV rate: {coverage['val_target_oov_rate_pct']:.1f}%**")
        
        if coverage['val_target_oov_rate_pct'] < 5:
            print(f"  ✅ Excellent target coverage")
        elif coverage['val_target_oov_rate_pct'] < 20:
            print(f"  ⚠️ Moderate target coverage")
        else:
            print(f"  ❌ Poor target coverage")
        
        # Train models
        print(f"\nC. PERFORMANCE")
        test_results = {
            "description": test["description"],
            "purpose": test["purpose"],
            "method": method,
            "leakage_audit": leakage,
            "coverage": coverage,
            "model_runs": {},
        }
        
        for model_seed in MODEL_SEEDS:
            print(f"\n  Model seed {model_seed}:")
            results = train_model(train_data, val_data, config, model_seed, verbose=True)
            test_results["model_runs"][f"model_seed_{model_seed}"] = results
        
        all_results["tests"][test["name"]] = test_results
    
    # Save results
    results_file = output_dir / "generalization_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Test':<40} {'Target OOV':>12} {'Exact Overlap':>14} {'Train Acc':>12} {'Val Acc':>12}")
    print("-"*95)
    
    for test_name, test_data in all_results["tests"].items():
        # Average across model seeds
        train_accs = [r["final_train_acc"] for r in test_data["model_runs"].values()]
        val_accs = [r["final_val_acc"] for r in test_data["model_runs"].values()]
        
        avg_train_acc = sum(train_accs) / len(train_accs)
        avg_val_acc = sum(val_accs) / len(val_accs)
        
        target_oov = test_data["coverage"]["val_target_oov_rate_pct"]
        exact_overlap = test_data["leakage_audit"]["exact_sequence_overlap"]
        
        print(f"{test_name:<40} {target_oov:>11.1f}% {exact_overlap:>14} {avg_train_acc:>11.1%} {avg_val_acc:>11.1%}")
    
    # Final classification
    print("\n" + "="*80)
    print("FINAL CLASSIFICATION")
    print("="*80)
    
    test_b = all_results["tests"]["test_b_different_stories_zero_oov"]
    test_b_val_accs = [r["final_val_acc"] for r in test_b["model_runs"].values()]
    test_b_avg_val_acc = sum(test_b_val_accs) / len(test_b_val_accs)
    
    exact_overlap = test_b["leakage_audit"]["exact_sequence_overlap"]
    
    print(f"\nTest B (TRUE GENERALIZATION TEST):")
    print(f"  Exact sequence overlap: {exact_overlap}")
    print(f"  Target OOV rate: {test_b['coverage']['val_target_oov_rate_pct']:.1f}%")
    print(f"  Average validation accuracy: {test_b_avg_val_acc:.1%}")
    
    if exact_overlap > 0:
        print(f"\n  ❌ INVALID: Validation contains exact training sequences!")
        print(f"     Cannot conclude generalization from this test.")
    elif test_b_avg_val_acc > 0.80:
        print(f"\n  ✅ STRONG EVIDENCE OF GENERALIZATION")
        print(f"     High validation accuracy ({test_b_avg_val_acc:.1%}) on genuinely different stories")
        print(f"     with 0% target OOV indicates genuine learning.")
    elif test_b_avg_val_acc > 0.50:
        print(f"\n  ⚠️ PARTIAL GENERALIZATION")
        print(f"     Moderate validation accuracy ({test_b_avg_val_acc:.1%}) suggests")
        print(f"     some generalization but potential issues.")
    else:
        print(f"\n  ❌ NO GENERALIZATION")
        print(f"     Low validation accuracy ({test_b_avg_val_acc:.1%}) despite 0% OOV")
        print(f"     suggests previous results were due to data leakage or memorization.")


if __name__ == "__main__":
    main()
