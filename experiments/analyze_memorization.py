#!/usr/bin/env python3
"""
Analyze memorization by generating samples from the trained model.

We need to:
1. Load the final checkpoint
2. Generate text from same prompts used in training data
3. Compare generated outputs to original stories
4. Check per-example loss on training data
"""

import sys
import json
import torch
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("=" * 70)
print("MEMORIZATION ANALYSIS")
print("=" * 70)

# Load config
TOKENIZER_PATH = Path("/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json")
RESULTS_PATH = Path("/root/agent-sandbox/ARTN/experiments/results/tinystories_memorize_results.json")

# Load results to get config
with open(RESULTS_PATH) as f:
    results = json.load(f)

config = results["config"]
print(f"\n[CONFIG]")
print(f"  Model: {config['model']['n_layers']} layers, {config['model']['d_model']} dim")
print(f"  Training: {config['training']['total_steps']} steps, batch={config['training']['batch_size']}")
print(f"  Data: {config['data']['num_stories']} stories")

# Load tokenizer
print(f"\n[LOADING] Tokenizer...")
tokenizer = BPETokenizer.load(TOKENIZER_PATH)
print(f"  Vocab size: {len(tokenizer.vocab)}")

# Load dataset (same 20 stories)
print(f"\n[LOADING] TinyStories (20 stories, seed=42)...")
from datasets import load_dataset
full_dataset = load_dataset("roneneldan/TinyStories", split="train")
random.seed(42)
indices = random.sample(range(len(full_dataset)), 20)
stories = [full_dataset[i]["text"] for i in indices]
print(f"  Stories loaded: {len(stories)}")

# Tokenize stories
print(f"\n[TOKENIZING] Stories...")
tokenized_stories = []
for i, story in enumerate(stories):
    tokens = tokenizer.encode(story)[:255]
    tokens = tokens + [tokenizer.eos_token_id]
    tokenized_stories.append(tokens)
    if i < 3:
        print(f"  Story {i+1}: {len(tokens)} tokens")
        print(f"    First 50 chars: {story[:50]}...")

print(f"\n  Total stories: {len(tokenized_stories)}")
print(f"  Total tokens: {sum(len(s) for s in tokenized_stories):,}")

# Create model
print(f"\n[CREATING] Model...")
model = BitLinearGPT(
    vocab_size=config["model"]["vocab_size"],
    d_model=config["model"]["d_model"],
    n_heads=config["model"]["n_heads"],
    n_layers=config["model"]["n_layers"],
    d_ff=config["model"]["d_ff"],
    max_seq_len=config["model"]["max_seq_len"],
)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check for checkpoint
checkpoint_path = Path("/root/agent-sandbox/ARTN/outputs/checkpoints")
checkpoints = list(checkpoint_path.glob("*.pt")) if checkpoint_path.exists() else []
print(f"\n[CHECKPOINTS] Found {len(checkpoints)} checkpoints:")
for ckpt in sorted(checkpoints)[-5:]:
    print(f"  {ckpt.name}")

# Since we didn't save checkpoint in monitored script, use untrained model for now
print(f"\n[NOTE] No checkpoint saved by monitored script.")
print(f"       Analyzing with UNTRAINED model (random weights).")

model.eval()

# Generate samples from each story prompt
print(f"\n" + "=" * 70)
print(f"GENERATED SAMPLES (first 50 tokens from each story)")
print(f"=" * 70)

for i, tokens in enumerate(tokenized_stories[:5]):
    # Use first 20 tokens as prompt
    prompt = tokens[:20]
    prompt_text = tokenizer.decode(prompt)
    
    # Generate continuation (greedy)
    with torch.no_grad():
        input_ids = torch.tensor([prompt], dtype=torch.long)
        # Generate 50 more tokens
        generated = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=1.0,
            top_k=None,
        )
    
    generated_tokens = generated[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    original_text = tokenizer.decode(tokens[:70])
    
    print(f"\n--- Story {i+1} ---")
    print(f"ORIGINAL ({len(tokens[:70])} tokens):")
    print(f"  {original_text[:200]}")
    print(f"\nGENERATED ({len(generated_tokens)} tokens):")
    print(f"  {generated_text[:200]}")

# Compute per-example loss
print(f"\n" + "=" * 70)
print(f"PER-EXAMPLE LOSS")
print(f"=" * 70)

model.train()  # Need dropout for loss computation
criterion = torch.nn.CrossEntropyLoss(reduction="none")

print(f"\n  Story | Tokens | Avg Loss | Max Loss | Min Loss")
print(f"  ------|--------|----------|----------|--------")

for i, tokens in enumerate(tokenized_stories):
    # Compute loss on this story
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(input_ids)
        # outputs: (1, seq_len, vocab_size)
        # targets: shift by 1
        logits = outputs[0, :-1, :]  # (seq_len-1, vocab)
        targets = input_ids[0, 1:]   # (seq_len-1,)
        
        # Per-token loss
        loss_per_token = criterion(logits, targets)
        
        avg_loss = loss_per_token.mean().item()
        max_loss = loss_per_token.max().item()
        min_loss = loss_per_token.min().item()
    
    print(f"  {i+1:5d} | {len(tokens):6d} | {avg_loss:8.4f} | {max_loss:8.4f} | {min_loss:7.4f}")

# Summary
print(f"\n" + "=" * 70)
print(f"SUMMARY")
print(f"=" * 70)
print(f"\n  Training results:")
print(f"    Initial loss: {results['initial_loss']:.4f}")
print(f"    Final loss: {results['final_loss']:.4f}")
print(f"    Best loss: {results['best_loss']:.4f} (step {results['best_step']})")
print(f"\n  NOTE: Without checkpoint, cannot verify memorization.")
print(f"       Training script should save checkpoint at step {results['best_step']}.")
print(f"\n" + "=" * 70)
