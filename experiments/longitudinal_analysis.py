#!/usr/bin/env python3
"""
COMPLETE LONGITUDINAL ANALYSIS: Ternary 50k Training Run

Analyzes ALL 50 checkpoints (every 1000 steps) to determine:
- Learning trajectory
- Plateau point
- Generation quality over time
- Whether model is still learning
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset
import re
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*80)
print("LONGITUDINAL ANALYSIS: Ternary 50k Training Run")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_DIR = Path("outputs/artn-50k-best-config/checkpoints")
CHECKPOINT_STEPS = [i * 1000 for i in range(1, 51)]  # 1000 to 50000
EVAL_SAMPLES = 200  # Stories to evaluate per checkpoint
MAX_SEQ_LEN = 256
tokenizer_path = "outputs/artn-optimized/tokenizer.json"

# Generation prompts (FIXED)
PROMPTS = [
    "Once upon a time",
    "There was a little girl",
    "The cat sat on the",
    "A boy named Tom",
]

# ============================================================
# LOAD TOKENIZER AND DATA
# ============================================================

print(f"\nLoading tokenizer...")
tokenizer = BPETokenizer.load(tokenizer_path)
print(f"Vocab: {tokenizer.vocab_size}, PAD={tokenizer.pad_token_id}, EOS={tokenizer.eos_token_id}")

print(f"\nLoading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")
train_data = dataset.select(range(50000))
heldout_data = dataset.select(range(50000, 50000 + EVAL_SAMPLES))
print(f"Training data: 50,000 stories")
print(f"Held-out data: {EVAL_SAMPLES} stories")

# ============================================================
# MODEL CONFIG
# ============================================================

model_config = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
}

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_model(model, tokenizer, dataset, max_samples, name=""):
    """Evaluate model on dataset."""
    model.eval()
    
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_correct_prob = 0.0
    total_real_tokens = 0
    
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            text = dataset[i]['text']
            tokens = tokenizer.encode(text)[:MAX_SEQ_LEN - 1]
            tokens = tokens + [tokenizer.eos_token_id]
            
            if len(tokens) < 2:
                continue
            
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            output = model(input_ids)
            logits = output["logits"]
            
            for pos in range(len(tokens) - 1):
                target = tokens[pos + 1]
                
                if target == tokenizer.pad_token_id:
                    continue
                
                pos_logits = logits[0, pos, :]
                loss = F.cross_entropy(pos_logits.unsqueeze(0), torch.tensor([target], device=device))
                total_loss += loss.item()
                
                probs = F.softmax(pos_logits, dim=-1)
                top1 = probs.argmax().item()
                top5 = probs.topk(5).indices.tolist()
                
                if top1 == target:
                    total_correct_top1 += 1
                if target in top5:
                    total_correct_top5 += 1
                
                total_correct_prob += probs[target].item()
                total_real_tokens += 1
        
        if (i + 1) % 100 == 0 and name:
            print(f"    {name}: {i+1}/{max_samples}")
    
    if total_real_tokens == 0:
        return None
    
    return {
        "loss": total_loss / total_real_tokens,
        "perplexity": torch.exp(torch.tensor(total_loss / total_real_tokens)).item(),
        "top1_accuracy": 100 * total_correct_top1 / total_real_tokens,
        "top5_accuracy": 100 * total_correct_top5 / total_real_tokens,
        "correct_prob": 100 * total_correct_prob / total_real_tokens,
        "real_tokens": total_real_tokens,
    }

def generate(model, tokenizer, prompt, max_new_tokens=30, temperature=1.0, device='cpu'):
    """Generate text from prompt."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(input_ids)
            logits = output["logits"]
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            if next_token == tokenizer.eos_token_id or len(generated) >= 100:
                break
    
    return tokenizer.decode(generated)

def analyze_repetition(text):
    """Analyze repetition in generated text."""
    words = text.split()
    if len(words) < 2:
        return {"word_diversity": 1.0, "bigram_repetition": 0.0}
    
    unique_words = len(set(words))
    word_diversity = unique_words / len(words) if words else 1.0
    
    bigrams = list(zip(words[:-1], words[1:]))
    if not bigrams:
        return {"word_diversity": word_diversity, "bigram_repetition": 0.0}
    
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
    bigram_repetition = repeated_bigrams / len(bigrams)
    
    return {
        "word_diversity": word_diversity,
        "bigram_repetition": bigram_repetition,
    }

# ============================================================
# LOAD TRAINING LOG
# ============================================================

print(f"\n{'='*80}")
print("LOADING TRAINING LOG")
print("="*80)

training_log_path = Path("outputs/artn-50k-best-config/monitor_log.txt")
training_log = {}

if training_log_path.exists():
    with open(training_log_path, 'r') as f:
        for line in f:
            # Parse lines like: "2026-08-16 11:04:57 | Step 50000 | Loss 7.1267 | ..."
            match = re.search(r'Step (\d+).*Loss ([\d.]+)', line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                training_log[step] = loss
    
    print(f"Loaded training log: {len(training_log)} entries")
    print(f"Steps: {sorted(training_log.keys())[:10]}...{sorted(training_log.keys())[-10:]}")
else:
    print(f"Training log not found at {training_log_path}")

# ============================================================
# ANALYZE ALL CHECKPOINTS
# ============================================================

print(f"\n{'='*80}")
print("ANALYZING ALL CHECKPOINTS")
print("="*80)

results = {
    "steps": [],
    "training_loss": [],
    "heldout_metrics": [],
    "generations": [],
}

for checkpoint_step in CHECKPOINT_STEPS:
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_step_{checkpoint_step}.pt"
    
    if not checkpoint_path.exists():
        print(f"Step {checkpoint_step}: Checkpoint not found")
        continue
    
    print(f"\n{'='*80}")
    print(f"CHECKPOINT STEP {checkpoint_step}")
    print("="*80)
    
    # Load model
    model = BitLinearGPT(**model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Get training loss from log
    train_loss = training_log.get(checkpoint_step, None)
    if train_loss is None:
        # Find closest step in log
        closest_steps = [s for s in training_log.keys() if abs(s - checkpoint_step) < 500]
        if closest_steps:
            train_loss = training_log[min(closest_steps, key=lambda s: abs(s - checkpoint_step))]
    
    results["steps"].append(checkpoint_step)
    results["training_loss"].append(train_loss)
    
    print(f"  Training loss (from log): {train_loss:.4f}" if train_loss else "  Training loss: N/A")
    
    # Evaluate on held-out
    print(f"  Evaluating on {EVAL_SAMPLES} held-out stories...")
    heldout_metrics = evaluate_model(model, tokenizer, heldout_data, EVAL_SAMPLES, "Held-out")
    results["heldout_metrics"].append(heldout_metrics)
    
    if heldout_metrics:
        print(f"  Held-out loss: {heldout_metrics['loss']:.4f}")
        print(f"  Held-out Top-1: {heldout_metrics['top1_accuracy']:.2f}%")
        print(f"  Held-out Top-5: {heldout_metrics['top5_accuracy']:.2f}%")
    
    # Generate samples
    print(f"  Generating samples...")
    gen_samples = []
    for prompt in PROMPTS:
        gen_text = generate(model, tokenizer, prompt, max_new_tokens=25, temperature=1.0, device=device)
        repetition = analyze_repetition(gen_text)
        gen_samples.append({
            "prompt": prompt,
            "generated": gen_text,
            "word_diversity": repetition["word_diversity"],
            "bigram_repetition": repetition["bigram_repetition"],
        })
    results["generations"].append(gen_samples)
    
    for sample in gen_samples[:2]:  # Print first 2
        print(f"    '{sample['prompt']}' → \"{sample['generated'][:60]}...\"")
    
    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================
# LONGITUDINAL ANALYSIS
# ============================================================

print(f"\n{'='*80}")
print("LONGITUDINAL ANALYSIS")
print("="*80)

# Calculate loss reduction rates
print(f"\n{'Step':<10} {'Train Loss':>12} {'Held Loss':>12} {'Δ Train':>10} {'Δ Held':>10} {'Top-1':>8} {'Top-5':>8}")
print(f"{'-'*75}")

for i, step in enumerate(results["steps"]):
    train_loss = results["training_loss"][i]
    heldout = results["heldout_metrics"][i]
    held_loss = heldout["loss"] if heldout else None
    top1 = heldout["top1_accuracy"] if heldout else None
    top5 = heldout["top5_accuracy"] if heldout else None
    
    if i == 0:
        delta_train = 0
        delta_held = 0
    else:
        prev_train = results["training_loss"][i-1]
        prev_held = results["heldout_metrics"][i-1]["loss"] if results["heldout_metrics"][i-1] else None
        delta_train = train_loss - prev_train if train_loss and prev_train else 0
        delta_held = held_loss - prev_held if held_loss and prev_held else 0
    
    print(f"{step:<10,} {train_loss:>12.4f} {held_loss:>12.4f} {delta_train:>10.4f} {delta_held:>10.4f} {top1:>8.2f}% {top5:>8.2f}%")

# Plateau analysis
print(f"\n{'='*80}")
print("PLATEAU ANALYSIS")
print("="*80)

if len(results["steps"]) >= 10:
    # Early phase: first 20% (steps 1-10k)
    early_indices = [i for i, s in enumerate(results["steps"]) if s <= 10000]
    early_train_losses = [results["training_loss"][i] for i in early_indices if results["training_loss"][i]]
    early_held_losses = [results["heldout_metrics"][i]["loss"] for i in early_indices if results["heldout_metrics"][i]]
    
    # Late phase: last 20% (steps 40-50k)
    late_indices = [i for i, s in enumerate(results["steps"]) if s >= 40000]
    late_train_losses = [results["training_loss"][i] for i in late_indices if results["training_loss"][i]]
    late_held_losses = [results["heldout_metrics"][i]["loss"] for i in late_indices if results["heldout_metrics"][i]]
    
    if early_train_losses and late_train_losses:
        early_train_rate = (early_train_losses[0] - early_train_losses[-1]) / (len(early_train_losses) - 1) if len(early_train_losses) > 1 else 0
        late_train_rate = (late_train_losses[0] - late_train_losses[-1]) / (len(late_train_losses) - 1) if len(late_train_losses) > 1 else 0
        
        print(f"\nTraining loss reduction rate:")
        print(f"  Early (1k-10k): {early_train_rate:.4f} per checkpoint")
        print(f"  Late (40k-50k): {late_train_rate:.4f} per checkpoint")
        print(f"  Ratio: {abs(early_train_rate / late_train_rate):.1f}x slower")
    
    if early_held_losses and late_held_losses:
        early_held_rate = (early_held_losses[0] - early_held_losses[-1]) / (len(early_held_losses) - 1) if len(early_held_losses) > 1 else 0
        late_held_rate = (late_held_losses[0] - late_held_losses[-1]) / (len(late_held_losses) - 1) if len(late_held_losses) > 1 else 0
        
        print(f"\nHeld-out loss reduction rate:")
        print(f"  Early (1k-10k): {early_held_rate:.4f} per checkpoint")
        print(f"  Late (40k-50k): {late_held_rate:.4f} per checkpoint")
        print(f"  Ratio: {abs(early_held_rate / late_held_rate):.1f}x slower")

# Generation quality analysis
print(f"\n{'='*80}")
print("GENERATION QUALITY OVER TIME")
print("="*80)

for i in [0, 9, 19, 29, 39, 49]:  # Steps 1k, 10k, 20k, 30k, 40k, 50k
    if i < len(results["steps"]):
        step = results["steps"][i]
        gen = results["generations"][i]
        
        print(f"\nStep {step}:")
        for sample in gen[:2]:
            print(f"  '{sample['prompt']}' → \"{sample['generated'][:80]}\"")
            print(f"    Diversity: {sample['word_diversity']:.2f}, Repetition: {sample['bigram_repetition']:.2f}")

# ============================================================
# SAVE RESULTS
# ============================================================

output_path = Path("experiments/results/longitudinal_analysis.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*80}")
print(f"RESULTS SAVED: {output_path}")
print("="*80)

# ============================================================
# FINAL ANSWERS
# ============================================================

print(f"\n{'='*80}")
print("FINAL ANSWERS")
print("="*80)

# A. Is the model still learning at 50k?
if results["training_loss"] and len(results["training_loss"]) >= 2:
    last_two_train = results["training_loss"][-2:]
    if last_two_train[0] and last_two_train[1]:
        train_diff = last_two_train[0] - last_two_train[1]
        print(f"\nA. Is model still learning at 50k?")
        print(f"   Training loss change (49k→50k): {train_diff:.4f}")
        if abs(train_diff) < 0.01:
            print(f"   Answer: NO - Training loss plateaued")
        else:
            print(f"   Answer: MARGINAL - Very small improvement")

if results["heldout_metrics"] and len(results["heldout_metrics"]) >= 2:
    last_two_held = results["heldout_metrics"][-2:]
    if last_two_held[0] and last_two_held[1]:
        held_diff = last_two_held[0]["loss"] - last_two_held[1]["loss"]
        print(f"   Held-out loss change (49k→50k): {held_diff:.4f}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print("="*80)
