#!/usr/bin/env python3
"""
DIAGNOSTIC: 5 TinyStories × 500 Steps with Adam + LR=0.01

Purpose: Test if Adam + LR=0.01 (same as successful random tests)
         allows memorization of 5 TinyStories.

ONLY CHANGES from previous test:
  - Optimizer: Adafactor → Adam
  - LR: 0.001 → 0.01

EVERYTHING else identical.

Date: September 4, 2026
"""

import sys
import torch
import json
import math
import random
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer


# =============================================================================
# CONFIGURATION - IDENTICAL TO PREVIOUS TEST EXCEPT optimizer + LR
# =============================================================================

SEED = 42

# Model - IDENTICAL
MODEL_CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    "threshold": 0.5,
    "dropout": 0.0,
}

# Optimizer - ONLY CHANGES
OPTIMIZER_CONFIG = {
    "optimizer": "Adam",  # CHANGED: Adafactor → Adam
    "lr": 0.01,           # CHANGED: 0.001 → 0.01
    "beta1": 0.9,         # Adam default (same as successful random tests)
    "beta2": 0.999,       # Adam default
    "weight_decay": 0.01, # SAME as previous test
    "max_grad_norm": 1.0, # SAME as previous test
}

# Training - IDENTICAL
TRAINING_CONFIG = {
    "num_stories": 5,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "total_steps": 500,
    "max_seq_len": 256,
}

# Tokenizer - IDENTICAL
TOKENIZER_PATH = "/root/agent-sandbox/ARTN/outputs/artn-50k-best-config/tokenizer.json"

# Output
OUTPUT_DIR = Path("/root/agent-sandbox/ARTN/outputs/diagnostic_5stories_adam")


# =============================================================================
# DATASET - IDENTICAL TO PREVIOUS TEST
# =============================================================================

class TinyStoriesMemorizationDataset(Dataset):
    """Dataset for memorizing 5 TinyStories - IDENTICAL to previous test."""
    
    def __init__(self, stories, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []
        
        for story in stories:
            text = story["text"]
            tokens = tokenizer.encode(text)[:max_length - 1]
            tokens = tokens + [tokenizer.eos_token_id]
            
            padding_length = max_length - len(tokens)
            tokens = tokens + [tokenizer.pad_token_id] * padding_length
            
            self.sequences.append(tokens)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.sequences[idx], dtype=torch.long)
        
        inputs = input_ids[:-1].clone()
        targets = input_ids[1:].clone()
        
        targets[targets == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": inputs, "targets": targets}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("DIAGNOSTIC: 5 TinyStories × 500 Steps with Adam + LR=0.01")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print("="*80)
    
    torch.manual_seed(SEED)
    random.seed(SEED)
    
    # ==========================================================================
    # Load tokenizer
    # ==========================================================================
    print("\n[Tokenizer] Loading...")
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    print(f"  Vocab size: {len(tokenizer.vocab)}")
    print(f"  PAD: {tokenizer.pad_token_id}, EOS: {tokenizer.eos_token_id}")
    
    # ==========================================================================
    # Load 5 real TinyStories - IDENTICAL TO PREVIOUS TEST
    # ==========================================================================
    print("\n[Dataset] Loading 5 TinyStories (same as previous test)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    random.seed(SEED)
    indices = sorted(random.sample(range(len(dataset)), TRAINING_CONFIG["num_stories"]))
    stories = [dataset[i] for i in indices]
    
    print(f"  Selected {len(stories)} stories")
    
    for i, story in enumerate(stories):
        print(f"\n  Story {i+1} (first 100 chars): {story['text'][:100]}...")
    
    train_dataset = TinyStoriesMemorizationDataset(
        stories, tokenizer, TRAINING_CONFIG["max_seq_len"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    
    print(f"\n  DataLoader: {len(train_dataset)} sequences, batch_size={TRAINING_CONFIG['batch_size']}")
    
    # ==========================================================================
    # Create model - IDENTICAL TO PREVIOUS TEST
    # ==========================================================================
    print("\n[Model] Creating 6.8M ternary BitLinearGPT (IDENTICAL to previous test)...")
    torch.manual_seed(SEED)
    model = BitLinearGPT(
        vocab_size=MODEL_CONFIG["vocab_size"],
        d_model=MODEL_CONFIG["d_model"],
        n_heads=MODEL_CONFIG["n_heads"],
        n_layers=MODEL_CONFIG["n_layers"],
        d_ff=MODEL_CONFIG["d_ff"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
        threshold=MODEL_CONFIG["threshold"],
        dropout=MODEL_CONFIG["dropout"],
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # ==========================================================================
    # Create optimizer - ONLY CHANGES: Adam + LR=0.01
    # ==========================================================================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=OPTIMIZER_CONFIG["lr"],
        betas=(OPTIMIZER_CONFIG["beta1"], OPTIMIZER_CONFIG["beta2"]),
        weight_decay=OPTIMIZER_CONFIG["weight_decay"],
    )
    
    print(f"\n[Optimizer] Adam")
    print(f"  LR: {OPTIMIZER_CONFIG['lr']}")  # CHANGED from 0.001
    print(f"  Beta1: {OPTIMIZER_CONFIG['beta1']}, Beta2: {OPTIMIZER_CONFIG['beta2']}")
    print(f"  Weight decay: {OPTIMIZER_CONFIG['weight_decay']}")
    print(f"  Grad clip: {OPTIMIZER_CONFIG['max_grad_norm']}")
    
    print("\n[VERIFICATION] ONLY CHANGES from previous test:")
    print("  Previous: Adafactor, LR=0.001")
    print("  This:     Adam, LR=0.01")
    print("  Everything else: IDENTICAL")
    
    # ==========================================================================
    # Training - IDENTICAL TO PREVIOUS TEST
    # ==========================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    results = {
        "config": {
            "model": MODEL_CONFIG,
            "optimizer": OPTIMIZER_CONFIG,
            "training": TRAINING_CONFIG,
        },
        "metrics": {},
    }
    
    model.train()
    data_iter = iter(train_loader)
    
    for step in range(1, TRAINING_CONFIG["total_steps"] + 1):
        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_active_loss = 0.0
        accumulated_active_tokens = 0
        
        for _ in range(TRAINING_CONFIG["gradient_accumulation_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            input_ids = batch["input_ids"]
            targets = batch["targets"]
            
            outputs = model(input_ids, targets=targets)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            if torch.isnan(loss):
                print(f"\n[ERROR] NaN loss at step {step}")
                continue
            
            with torch.no_grad():
                active_mask = targets != -100
                active_logits = logits[active_mask]
                active_targets = targets[active_mask]
                if len(active_targets) > 0:
                    active_loss = torch.nn.functional.cross_entropy(
                        active_logits, active_targets
                    )
                    accumulated_active_loss += active_loss.item()
                    accumulated_active_tokens += len(active_targets)
            
            scaled_loss = loss / TRAINING_CONFIG["gradient_accumulation_steps"]
            scaled_loss.backward()
            accumulated_loss += loss.item()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), OPTIMIZER_CONFIG["max_grad_norm"]
        )
        
        optimizer.step()
        
        # Report at key steps: 1, 50, 100, 250, 500
        if step in [1, 50, 100, 250, 500]:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                correct = (predictions == targets) & (targets != -100)
                accuracy = correct.sum().item() / (targets != -100).sum().item()
                
                perplexity = math.exp(accumulated_loss / TRAINING_CONFIG["gradient_accumulation_steps"])
                active_loss_avg = accumulated_active_loss / TRAINING_CONFIG["gradient_accumulation_steps"]
            
            print(f"\nStep {step:4d}:")
            print(f"  Loss: {accumulated_loss / TRAINING_CONFIG['gradient_accumulation_steps']:.4f}")
            print(f"  Active-token loss: {active_loss_avg:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  Accuracy: {accuracy*100:.2f}%")
            print(f"  Grad norm: {grad_norm:.4f}")
            
            results["metrics"][step] = {
                "loss": accumulated_loss / TRAINING_CONFIG["gradient_accumulation_steps"],
                "active_loss": active_loss_avg,
                "perplexity": perplexity,
                "accuracy": accuracy,
                "grad_norm": grad_norm.item(),
            }
    
    # ==========================================================================
    # Final evaluation
    # ==========================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    model.eval()
    total_correct = 0
    total_active = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for i in range(len(train_dataset)):
            batch = train_dataset[i]
            input_ids = batch["input_ids"].unsqueeze(0)
            targets = batch["targets"].unsqueeze(0)
            
            outputs = model(input_ids, targets=targets)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            predictions = logits.argmax(dim=-1)
            correct = (predictions == targets) & (targets != -100)
            
            total_correct += correct.sum().item()
            total_active += (targets != -100).sum().item()
            total_loss += loss.item()
    
    final_loss = total_loss / len(train_dataset)
    final_accuracy = total_correct / total_active if total_active > 0 else 0
    final_perplexity = math.exp(final_loss) if final_loss < 10 else float('inf')
    
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Final perplexity: {final_perplexity:.2f}")
    print(f"  Final accuracy: {final_accuracy*100:.2f}%")
    
    results["final_loss"] = final_loss
    results["final_accuracy"] = final_accuracy
    results["final_perplexity"] = final_perplexity
    
    # ==========================================================================
    # Generation - SAME PROMPT AS PREVIOUS TEST
    # ==========================================================================
    print("\n[Generation] From first training story prompt (SAME as previous test)...")
    
    first_story_tokens = train_dataset.sequences[0]
    if tokenizer.pad_token_id in first_story_tokens:
        pad_start = first_story_tokens.index(tokenizer.pad_token_id)
    else:
        pad_start = len(first_story_tokens)
    
    real_tokens = first_story_tokens[:pad_start]
    prompt_tokens = real_tokens[:20]
    prompt_text = tokenizer.decode(prompt_tokens)
    print(f"  Prompt: {prompt_text}")
    
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        generated = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=1.0,
            top_k=50,
        )
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"  Generated: {generated_text}")
    
    results["generation"] = {
        "prompt": prompt_text,
        "generated": generated_text,
    }
    
    # ==========================================================================
    # Save results
    # ==========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "results.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Results] Saved to: {results_file}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    initial_loss = results["metrics"][1]["loss"]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss reduction: {loss_reduction:.1f}%")
    print(f"Final accuracy: {final_accuracy*100:.2f}%")
    
    print("\nCOMPARISON:")
    print(f"  Random tests (Adam, LR=0.01): 99.9-100% reduction, 100% accuracy")
    print(f"  Previous test (Adafactor, LR=0.001): 1.5% reduction, 0.11% accuracy")
    print(f"  THIS test (Adam, LR=0.01): {loss_reduction:.1f}% reduction, {final_accuracy*100:.2f}% accuracy")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
