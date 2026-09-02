#!/usr/bin/env python3
"""
DIAGNOSTIC: What Has the Model ACTUALLY Learned?

This script analyzes the 50k checkpoint to determine:
1. What tokens does the model predict correctly?
2. Are correct predictions just common words?
3. Has the model learned any multi-token patterns?
4. What patterns appear in gibberish generation?

NO TRAINING REQUIRED - Just analysis of existing checkpoint.
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from datasets import load_dataset
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*100)
print("DIAGNOSTIC: What Has the Model ACTUALLY Learned?")
print("="*100)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model and tokenizer
CHECKPOINT = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")
TOKENIZER_PATH = "outputs/artn-optimized/tokenizer.json"

tokenizer = BPETokenizer.load(TOKENIZER_PATH)
print(f"\nTokenizer loaded: vocab={tokenizer.vocab_size}")

checkpoint = torch.load(CHECKPOINT, map_location='cpu')

model_config = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
}

model = BitLinearGPT(**model_config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded (50k checkpoint)")

# Load evaluation data
dataset = load_dataset("roneneldan/TinyStories", split="train")
eval_data = dataset.select(range(50000, 50500))  # 500 held-out stories

print(f"\nEvaluation data: 500 held-out stories")

# ==============================================================================
# ANALYSIS 1: What tokens does the model predict correctly?
# ==============================================================================

print(f"\n{'='*100}")
print("ANALYSIS 1: Correct Predictions Analysis")
print("="*100)

correct_top1_tokens = Counter()
total_predictions = 0
correct_predictions = 0

MAX_SEQ_LEN = 256
MAX_STORIES = 100

with torch.no_grad():
    for i in range(min(MAX_STORIES, len(eval_data))):
        text = eval_data[i]['text']
        tokens = tokenizer.encode(text)[:MAX_SEQ_LEN-1]
        tokens = tokens + [tokenizer.eos_token_id]
        
        if len(tokens) < 5:
            continue
        
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        output = model(input_ids)
        logits = output["logits"]
        
        for pos in range(len(tokens) - 1):
            target = tokens[pos + 1]
            pos_logits = logits[0, pos, :]
            
            top1 = pos_logits.argmax().item()
            
            total_predictions += 1
            
            if top1 == target:
                correct_predictions += 1
                correct_top1_tokens[target] += 1

print(f"\nTotal predictions: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Top-1 accuracy: {100 * correct_predictions / total_predictions:.2f}%")

print(f"\nMost commonly predicted CORRECTLY tokens:")
print(f"{'Rank':<6} {'Token ID':<10} {'Count':<10} {'Token':<20} {'Decoded':<30}")
print("-"*80)

for rank, (token_id, count) in enumerate(correct_top1_tokens.most_common(20), 1):
    decoded = tokenizer.decode([token_id])
    token_str = repr(decoded[:20])
    print(f"{rank:<6} {token_id:<10} {count:<10} {token_str:<20} {decoded[:30]}")

# ==============================================================================
# ANALYSIS 2: Token frequency in training data vs correct predictions
# ==============================================================================

print(f"\n{'='*100}")
print("ANALYSIS 2: Are Correct Predictions Just Common Words?")
print("="*100)

# Count token frequencies in training data
print("\nCounting token frequencies in training data...")
train_data = dataset.select(range(10000))  # Sample 10k stories
train_token_freq = Counter()

for story in train_data:
    tokens = tokenizer.encode(story['text'])
    train_token_freq.update(tokens)

print(f"Total tokens in sample: {sum(train_token_freq.values())}")
print(f"Unique tokens: {len(train_token_freq)}")

# Compare
print(f"\nCorrelation: Correct predictions vs Training frequency")
print(f"{'Rank':<6} {'Token ID':<10} {'Correct Count':<15} {'Train Freq':<15} {'Decoded':<20}")
print("-"*70)

for rank, (token_id, count) in enumerate(correct_top1_tokens.most_common(20), 1):
    train_freq = train_token_freq.get(token_id, 0)
    decoded = tokenizer.decode([token_id])[:20]
    print(f"{rank:<6} {token_id:<10} {count:<15} {train_freq:<15} {decoded:<20}")

# Calculate correlation
common_correct = set([t for t, c in correct_top1_tokens.most_common(100)])
common_train = set([t for t, c in train_token_freq.most_common(100)])

overlap = len(common_correct & common_train)
print(f"\nOverlap in top-100 tokens: {overlap}/100")
print(f"  Correct predictions are {'mostly' if overlap > 80 else 'not'} common words")

# ==============================================================================
# ANALYSIS 3: Perplexity by token type
# ==============================================================================

print(f"\n{'='*100}")
print("ANALYSIS 3: Perplexity by Token Frequency")
print("="*100)

# Categorize tokens by frequency
sorted_tokens = [t for t, c in train_token_freq.most_common()]
top_100 = set(sorted_tokens[:100])
top_1000 = set(sorted_tokens[:1000])
rest = set(sorted_tokens[1000:])

# Calculate perplexity for each category
loss_by_category = {
    'top_100': [],
    'top_1000': [],
    'rest': [],
}

with torch.no_grad():
    for i in range(min(MAX_STORIES, len(eval_data))):
        text = eval_data[i]['text']
        tokens = tokenizer.encode(text)[:MAX_SEQ_LEN-1]
        tokens = tokens + [tokenizer.eos_token_id]
        
        if len(tokens) < 5:
            continue
        
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        output = model(input_ids)
        logits = output["logits"]
        
        for pos in range(len(tokens) - 1):
            target = tokens[pos + 1]
            pos_logits = logits[0, pos, :]
            loss = F.cross_entropy(pos_logits.unsqueeze(0), torch.tensor([target], device=device))
            
            if target in top_100:
                loss_by_category['top_100'].append(loss.item())
            elif target in top_1000:
                loss_by_category['top_1000'].append(loss.item())
            else:
                loss_by_category['rest'].append(loss.item())

print(f"\nPerplexity by token frequency:")
print(f"{'Category':<15} {'Count':<10} {'Avg Loss':<12} {'Perplexity':<12}")
print("-"*50)

for cat, losses in loss_by_category.items():
    if len(losses) > 0:
        avg_loss = sum(losses) / len(losses)
        ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        print(f"{cat:<15} {len(losses):<10} {avg_loss:<12.4f} {ppl:<12.2f}")

# ==============================================================================
# ANALYSIS 4: Generation pattern analysis
# ==============================================================================

print(f"\n{'='*100}")
print("ANALYSIS 4: Generation Pattern Analysis")
print("="*100)

# Generate multiple samples
print("\nGenerating 10 samples...")

generations = []
for i in range(10):
    prompt = "Once upon a time"
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(30):
            output = model(input_ids)
            logits = output["logits"]
            next_logits = logits[0, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            if len(generated) >= 100:
                break
    
    decoded = tokenizer.decode(generated)
    generations.append(decoded)
    print(f"\n{i+1}. {decoded[:100]}...")

# Analyze patterns
print(f"\n{'='*100}")
print("Pattern Analysis:")
print("="*100)

all_gen_tokens = []
for gen in generations:
    tokens = tokenizer.encode(gen)
    all_gen_tokens.extend(tokens)

gen_counter = Counter(all_gen_tokens)

print(f"\nMost common tokens in generations:")
print(f"{'Rank':<6} {'Token ID':<10} {'Count':<10} {'Decoded':<30}")
print("-"*60)

for rank, (token_id, count) in enumerate(gen_counter.most_common(20), 1):
    decoded = tokenizer.decode([token_id])
    print(f"{rank:<6} {token_id:<10} {count:<10} {decoded[:30]}")

# Check for repetition
repetition_detected = False
for gen in generations:
    words = gen.split()
    if len(words) > 3:
        # Check for repeated bigrams
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        for bigram, count in bigram_counts.items():
            if count > 2:
                repetition_detected = True
                print(f"\n⚠️ Repetition detected: '{bigram[0]} {bigram[1]}' repeated {count} times")
                break

if not repetition_detected:
    print("\n✓ No obvious repetition patterns detected")

# ==============================================================================
# SUMMARY
# ==============================================================================

print(f"\n{'='*100}")
print("DIAGNOSTIC SUMMARY")
print("="*100)

# Calculate key metrics
overlap_ratio = overlap / 100 if overlap else 0

print(f"""
KEY FINDINGS:

1. CORRECT PREDICTIONS:
   - Top-1 accuracy: {100 * correct_predictions / total_predictions:.2f}%
   - Most correct predictions are: {'common words' if overlap_ratio > 0.8 else 'diverse tokens'}

2. TOKEN FREQUENCY CORRELATION:
   - Overlap with top-100 training tokens: {overlap}/100 ({overlap_ratio*100:.0f}%)
   - {'Model predicts common words well, struggles with rare words' if overlap_ratio > 0.8 else 'Model has learned diverse patterns'}

3. PERPLEXITY BY CATEGORY:""")

top_100_avg = sum(loss_by_category['top_100'])/len(loss_by_category['top_100']) if loss_by_category['top_100'] else 0
rest_avg = sum(loss_by_category['rest'])/len(loss_by_category['rest']) if loss_by_category['rest'] else 0

print(f"""   - Top-100 tokens: {'LOWER perplexity (better)' if top_100_avg < rest_avg else 'SIMILAR perplexity'}

4. GENERATION QUALITY:
   - Gibberish level: HIGH
   - Repetition: {'DETECTED' if repetition_detected else 'NOT detected'}
   - Most common tokens in generation: {tokenizer.decode([gen_counter.most_common(1)[0][0]]) if gen_counter else 'N/A'}

HYPOTHESIS:
  The model has learned {'mostly common word statistics' if overlap_ratio > 0.8 else 'diverse token patterns'}
  but {'not multi-token coherence' if overlap_ratio > 0.8 else 'but generation fails for other reasons'}
  This suggests {'insufficient model capacity for multi-token patterns' if overlap_ratio > 0.8 else 'a different issue needs investigation'}
""")

# Save results
output_path = Path("experiments/results/what_model_learned_diagnostic.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

results = {
    "top1_accuracy": correct_predictions / total_predictions if total_predictions > 0 else 0,
    "correct_prediction_overlap_with_common": overlap,
    "loss_by_category": {k: sum(v)/len(v) if v else 0 for k, v in loss_by_category.items()},
    "most_common_correct_predictions": [(t, c, tokenizer.decode([t])) for t, c in correct_top1_tokens.most_common(20)],
    "generations_sample": generations,
}

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved: {output_path}")

