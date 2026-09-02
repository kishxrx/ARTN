#!/usr/bin/env python3
"""
FORENSIC DIAGNOSTIC: Does the Model Use Contextual Information?

This diagnostic determines whether the model has learned contextual patterns
beyond unigram statistics, WITHOUT any training.

CRITICAL: This script is READ-ONLY. It does NOT modify the checkpoint.
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from datasets import load_dataset
import math
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*100)
print("FORENSIC DIAGNOSTIC: Contextual Information Analysis")
print("="*100)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")
TOKENIZER_PATH = "outputs/artn-optimized/tokenizer.json"
OUTPUT_DIR = Path("experiments/results/context_forensic_diagnostic")

MAX_SEQ_LEN = 256
EVAL_STORIES = 200  # Number of held-out stories to evaluate
TRAIN_SAMPLE = 10000  # Sample for unigram baseline

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# ============================================================
# LOAD MODEL AND TOKENIZER (READ-ONLY)
# ============================================================

print(f"\n{'='*100}")
print("LOADING MODEL AND TOKENIZER (READ-ONLY)")
print("="*100)

tokenizer = BPETokenizer.load(TOKENIZER_PATH)
print(f"✓ Tokenizer loaded: vocab={tokenizer.vocab_size}")

checkpoint = torch.load(CHECKPOINT, map_location='cpu')
print(f"✓ Checkpoint loaded: step={checkpoint.get('step', 'N/A')}")

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
print(f"✓ Model loaded and set to eval mode")

# ============================================================
# LOAD DATA
# ============================================================

print(f"\n{'='*100}")
print("LOADING DATA")
print("="*100)

dataset = load_dataset("roneneldan/TinyStories", split="train")
train_data = dataset.select(range(TRAIN_SAMPLE))
heldout_data = dataset.select(range(50000, 50000 + EVAL_STORIES))

print(f"✓ Training sample: {TRAIN_SAMPLE} stories (for unigram baseline)")
print(f"✓ Held-out data: {EVAL_STORIES} stories (for evaluation)")

# ============================================================
# TEST 1: TOP-K ACCURACY
# ============================================================

print(f"\n{'='*100}")
print("TEST 1: TOP-K ACCURACY")
print("="*100)

topk_correct = {1: 0, 5: 0, 10: 0, 50: 0, 100: 0}
total_predictions = 0
all_losses = []

with torch.no_grad():
    for i in range(EVAL_STORIES):
        text = heldout_data[i]['text']
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
            
            # Cross-entropy loss
            loss = F.cross_entropy(pos_logits.unsqueeze(0), torch.tensor([target], device=device))
            all_losses.append(loss.item())
            
            # Top-k accuracy
            for k in topk_correct.keys():
                topk = pos_logits.topk(k).indices.tolist()
                if target in topk:
                    topk_correct[k] += 1
            
            total_predictions += 1

avg_loss = sum(all_losses) / len(all_losses)
perplexity = math.exp(avg_loss)

print(f"\nTotal predictions: {total_predictions}")
print(f"Average cross-entropy: {avg_loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")

print(f"\nTop-K Accuracy:")
print(f"{'K':<10} {'Correct':<15} {'Accuracy':<15}")
print("-"*40)
for k in sorted(topk_correct.keys()):
    acc = 100 * topk_correct[k] / total_predictions
    print(f"{k:<10} {topk_correct[k]:<15} {acc:.2f}%")

results = {
    "test1_topk_accuracy": {
        "total_predictions": total_predictions,
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "topk": {str(k): 100 * v / total_predictions for k, v in topk_correct.items()}
    }
}

# ============================================================
# TEST 2: UNIGRAM BASELINE
# ============================================================

print(f"\n{'='*100}")
print("TEST 2: UNIGRAM BASELINE")
print("="*100)

# Count unigram frequencies from training sample
print(f"Computing unigram frequencies from {TRAIN_SAMPLE} training stories...")
unigram_counts = Counter()
total_train_tokens = 0

for story in train_data:
    tokens = tokenizer.encode(story['text'])
    unigram_counts.update(tokens)
    total_train_tokens += len(tokens)

print(f"Total training tokens: {total_train_tokens}")
print(f"Unique tokens: {len(unigram_counts)}")

# Compute unigram probabilities
unigram_probs = {}
for token, count in unigram_counts.items():
    unigram_probs[token] = count / total_train_tokens

# Unigram baseline loss on held-out data
print(f"\nComputing unigram baseline on held-out data...")

unigram_losses = []
unigram_topk_correct = {1: 0, 5: 0, 10: 0, 50: 0, 100: 0}
unigram_total = 0

with torch.no_grad():
    for i in range(EVAL_STORIES):
        text = heldout_data[i]['text']
        tokens = tokenizer.encode(text)[:MAX_SEQ_LEN-1]
        tokens = tokens + [tokenizer.eos_token_id]
        
        if len(tokens) < 5:
            continue
        
        for pos in range(len(tokens) - 1):
            target = tokens[pos + 1]
            
            # Unigram probability
            p = unigram_probs.get(target, 1e-10)
            loss = -math.log(p)
            unigram_losses.append(loss)
            
            # Top-k based on unigram frequency
            sorted_tokens = [t for t, c in unigram_counts.most_common()]
            for k in unigram_topk_correct.keys():
                if target in sorted_tokens[:k]:
                    unigram_topk_correct[k] += 1
            
            unigram_total += 1

unigram_avg_loss = sum(unigram_losses) / len(unigram_losses)
unigram_perplexity = math.exp(unigram_avg_loss)

print(f"\nUnigram baseline:")
print(f"  Average cross-entropy: {unigram_avg_loss:.4f}")
print(f"  Perplexity: {unigram_perplexity:.2f}")

print(f"\nUnigram Top-K Accuracy:")
print(f"{'K':<10} {'Correct':<15} {'Accuracy':<15}")
print("-"*40)
for k in sorted(unigram_topk_correct.keys()):
    acc = 100 * unigram_topk_correct[k] / unigram_total
    print(f"{k:<10} {unigram_topk_correct[k]:<15} {acc:.2f}%")

# Model vs Unigram comparison
print(f"\n{'='*60}")
print("MODEL vs UNIGRAM COMPARISON")
print("="*60)

loss_improvement = unigram_avg_loss - avg_loss
ppl_improvement = unigram_perplexity - perplexity

print(f"\nCross-entropy:")
print(f"  Unigram: {unigram_avg_loss:.4f}")
print(f"  Model:   {avg_loss:.4f}")
print(f"  Improvement: {loss_improvement:.4f} ({100 * loss_improvement / unigram_avg_loss:.1f}%)")

print(f"\nPerplexity:")
print(f"  Unigram: {unigram_perplexity:.2f}")
print(f"  Model:   {perplexity:.2f}")
print(f"  Improvement: {ppl_improvement:.2f} ({100 * ppl_improvement / unigram_perplexity:.1f}%)")

print(f"\nTop-K Accuracy Improvement:")
print(f"{'K':<10} {'Unigram':<15} {'Model':<15} {'Improvement':<15}")
print("-"*55)
for k in sorted(topk_correct.keys()):
    uni_acc = 100 * unigram_topk_correct[k] / unigram_total
    model_acc = 100 * topk_correct[k] / total_predictions
    improvement = model_acc - uni_acc
    print(f"{k:<10} {uni_acc:.2f}%{'':<8} {model_acc:.2f}%{'':<8} {improvement:+.2f}%")

results["test2_unigram_baseline"] = {
    "unigram_avg_loss": unigram_avg_loss,
    "unigram_perplexity": unigram_perplexity,
    "model_vs_unigram_loss_improvement": loss_improvement,
    "model_vs_unigram_ppl_improvement": ppl_improvement,
    "relative_improvement_pct": 100 * loss_improvement / unigram_avg_loss
}

# ============================================================
# TEST 3: CONTEXT ABLATION
# ============================================================

print(f"\n{'='*100}")
print("TEST 3: CONTEXT ABLATION")
print("="*100)

def evaluate_with_context_limit(model, tokenizer, data, max_context_tokens, eval_stories=100):
    """Evaluate model with limited context."""
    losses = []
    top1_correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(min(eval_stories, len(data))):
            text = data[i]['text']
            tokens = tokenizer.encode(text)[:MAX_SEQ_LEN-1]
            tokens = tokens + [tokenizer.eos_token_id]
            
            if len(tokens) < 5:
                continue
            
            for pos in range(len(tokens) - 1):
                target = tokens[pos + 1]
                
                # Limit context
                start_pos = max(0, pos - max_context_tokens + 1)
                context = tokens[start_pos:pos + 1]
                
                if len(context) < 2:
                    continue
                
                input_ids = torch.tensor([context], dtype=torch.long, device=device)
                output = model(input_ids)
                logits = output["logits"]
                pos_logits = logits[0, -1, :]
                
                loss = F.cross_entropy(pos_logits.unsqueeze(0), torch.tensor([target], device=device))
                losses.append(loss.item())
                
                top1 = pos_logits.argmax().item()
                if top1 == target:
                    top1_correct += 1
                
                total += 1
    
    if total == 0:
        return None
    
    return {
        "avg_loss": sum(losses) / len(losses),
        "perplexity": math.exp(sum(losses) / len(losses)),
        "top1_accuracy": 100 * top1_correct / total,
        "total": total
    }

context_limits = [256, 8, 4, 1]

print(f"\nEvaluating with different context limits...")

context_results = {}
for limit in context_limits:
    print(f"  Context limit: {limit} tokens...")
    result = evaluate_with_context_limit(model, tokenizer, heldout_data, limit, EVAL_STORIES)
    if result:
        context_results[limit] = result

print(f"\n{'='*70}")
print("CONTEXT ABLATION RESULTS")
print("="*70)

print(f"\n{'Context':<15} {'Loss':<12} {'Perplexity':<12} {'Top-1 Acc':<12}")
print("-"*55)

for limit in context_limits:
    if limit in context_results:
        r = context_results[limit]
        print(f"{limit:<15} {r['avg_loss']:<12.4f} {r['perplexity']:<12.2f} {r['top1_accuracy']:<12.2f}%")

# Check if context matters
if 256 in context_results and 1 in context_results:
    full_loss = context_results[256]['avg_loss']
    single_loss = context_results[1]['avg_loss']
    context_benefit = single_loss - full_loss
    
    print(f"\n{'='*70}")
    print(f"Context Benefit Analysis:")
    print(f"  Full context (256): {full_loss:.4f}")
    print(f"  Single token:       {single_loss:.4f}")
    print(f"  Context benefit:    {context_benefit:.4f}")
    print(f"  {'✓ Model USES context' if context_benefit > 0.1 else '✗ Model DOES NOT effectively use context'}")

results["test3_context_ablation"] = {
    str(k): v for k, v in context_results.items()
}

# ============================================================
# TEST 4: CONTEXT SHUFFLING CONTROL
# ============================================================

print(f"\n{'='*100}")
print("TEST 4: CONTEXT SHUFFLING CONTROL")
print("="*100)

def evaluate_with_shuffled_context(model, tokenizer, data, eval_stories=100):
    """Evaluate with shuffled context."""
    losses = []
    top1_correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(min(eval_stories, len(data))):
            text = data[i]['text']
            tokens = tokenizer.encode(text)[:MAX_SEQ_LEN-1]
            tokens = tokens + [tokenizer.eos_token_id]
            
            if len(tokens) < 10:
                continue
            
            # Use only middle portion (need enough context to shuffle)
            for pos in range(5, min(len(tokens) - 1, 50)):
                target = tokens[pos + 1]
                
                # Get context and shuffle
                context = tokens[:pos + 1].copy()
                shuffled = context[:-1].copy()
                random.shuffle(shuffled)
                shuffled_context = shuffled + [context[-1]]
                
                input_ids = torch.tensor([shuffled_context], dtype=torch.long, device=device)
                output = model(input_ids)
                logits = output["logits"]
                pos_logits = logits[0, -1, :]
                
                loss = F.cross_entropy(pos_logits.unsqueeze(0), torch.tensor([target], device=device))
                losses.append(loss.item())
                
                top1 = pos_logits.argmax().item()
                if top1 == target:
                    top1_correct += 1
                
                total += 1
    
    if total == 0:
        return None
    
    return {
        "avg_loss": sum(losses) / len(losses),
        "perplexity": math.exp(sum(losses) / len(losses)),
        "top1_accuracy": 100 * top1_correct / total,
        "total": total
    }

print(f"\nEvaluating with shuffled context...")

shuffled_result = evaluate_with_shuffled_context(model, tokenizer, heldout_data, EVAL_STORIES)

# Compare normal vs shuffled
print(f"\n{'='*70}")
print("NORMAL vs SHUFFLED CONTEXT")
print("="*70)

normal_loss = avg_loss
shuffled_loss = shuffled_result['avg_loss'] if shuffled_result else None

print(f"\nNormal context:")
print(f"  Loss: {normal_loss:.4f}")
print(f"  Perplexity: {perplexity:.2f}")

if shuffled_loss:
    print(f"\nShuffled context:")
    print(f"  Loss: {shuffled_loss:.4f}")
    print(f"  Perplexity: {shuffled_result['perplexity']:.2f}")
    
    loss_increase = shuffled_loss - normal_loss
    
    print(f"\nLoss increase from shuffling: {loss_increase:.4f}")
    print(f"{'✓ Model DEPENDS on context order' if loss_increase > 0.1 else '✗ Model DOES NOT depend on context order'}")

results["test4_context_shuffling"] = {
    "shuffled_avg_loss": shuffled_loss,
    "normal_avg_loss": normal_loss,
    "loss_increase_from_shuffling": shuffled_loss - normal_loss if shuffled_loss else None
}

# ============================================================
# TEST 5: BIGRAM/TRIGRAM DIAGNOSTIC
# ============================================================

print(f"\n{'='*100}")
print("TEST 5: BIGRAM/TRIGRAM PATTERNS")
print("="*100)

# Count bigrams/trigrams from training
print("Computing bigram/trigram frequencies...")

bigram_counts = Counter()
trigram_counts = Counter()
bigram_totals = Counter()
trigram_totals = Counter()

for story in train_data:
    tokens = tokenizer.encode(story['text'])
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        bigram_counts[bigram] += 1
        bigram_totals[tokens[i]] += 1
    
    for i in range(len(tokens) - 2):
        trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
        trigram_counts[trigram] += 1
        trigram_totals[(tokens[i], tokens[i + 1])] += 1

# Evaluate bigram predictions
bigram_correct = 0
bigram_total = 0

with torch.no_grad():
    for i in range(min(100, EVAL_STORIES)):
        text = heldout_data[i]['text']
        tokens = tokenizer.encode(text)[:MAX_SEQ_LEN-1]
        
        for pos in range(len(tokens) - 1):
            prev_token = tokens[pos]
            target = tokens[pos + 1]
            
            # Most common bigram continuation
            if prev_token in bigram_totals:
                candidates = [bg[1] for bg in bigram_counts.keys() if bg[0] == prev_token]
                if candidates:
                    most_common = max(candidates, key=lambda t: bigram_counts.get((prev_token, t), 0))
                    if most_common == target:
                        bigram_correct += 1
                bigram_total += 1

bigram_acc = 100 * bigram_correct / bigram_total if bigram_total > 0 else 0

print(f"\nBigram baseline accuracy: {bigram_acc:.2f}%")
print(f"Model Top-1 accuracy: {100 * topk_correct[1] / total_predictions:.2f}%")
print(f"{'✓ Model BETTER than bigram' if (100 * topk_correct[1] / total_predictions) > bigram_acc + 1 else '⚠️ Model NOT better than bigram baseline'}")

results["test5_bigram_trigram"] = {
    "bigram_baseline_accuracy": bigram_acc,
    "model_top1_accuracy": 100 * topk_correct[1] / total_predictions
}

# ============================================================
# TEST 6: KNOWN-PREFIX GENERATION
# ============================================================

print(f"\n{'='*100}")
print("TEST 6: KNOWN-PREFIX GENERATION")
print("="*100)

def generate_continuation(model, tokenizer, prefix_tokens, max_new_tokens=20, temperature=1.0, greedy=False):
    """Generate continuation with specified decoding."""
    model.eval()
    generated = prefix_tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            if input_ids.shape[1] > MAX_SEQ_LEN:
                input_ids = input_ids[:, -MAX_SEQ_LEN:]
            
            output = model(input_ids)
            logits = output["logits"]
            next_logits = logits[0, -1, :]
            
            if greedy:
                next_token = next_logits.argmax().item()
            else:
                next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
    
    return generated

# Test with real prefixes from held-out data
print("\nTesting with REAL prefixes from held-out TinyStories...")

prefix_tests = []
for i in range(min(5, len(heldout_data))):
    text = heldout_data[i]['text']
    tokens = tokenizer.encode(text)
    
    if len(tokens) > 20:
        prefix = tokens[:10]
        actual_continuation = tokens[10:30]
        
        greedy_gen = generate_continuation(model, tokenizer, prefix, max_new_tokens=20, greedy=True)
        temp07_gen = generate_continuation(model, tokenizer, prefix, max_new_tokens=20, temperature=0.7)
        
        prefix_tests.append({
            "prefix": tokenizer.decode(prefix),
            "actual": tokenizer.decode(actual_continuation),
            "greedy": tokenizer.decode(greedy_gen[:30]),
            "temp0.7": tokenizer.decode(temp07_gen[:30])
        })

print(f"\n{'='*80}")
print("PREFIX GENERATION RESULTS")
print("="*80)

for i, test in enumerate(prefix_tests[:5], 1):
    print(f"\n--- Test {i} ---")
    print(f"Prefix: {test['prefix'][:60]}...")
    print(f"Actual continuation: {test['actual'][:60]}...")
    print(f"Greedy generation:   {test['greedy'][:60]}...")
    print(f"Temp 0.7 generation: {test['temp0.7'][:60]}...")

results["test6_prefix_generation"] = prefix_tests

# ============================================================
# TEST 7: DECODING COMPARISON
# ============================================================

print(f"\n{'='*100}")
print("TEST 7: DECODING METHOD COMPARISON")
print("="*100)

# Test with "Once upon a time"
once_upon = tokenizer.encode("Once upon a time")

print("\nGenerating with different decoding methods from 'Once upon a time':")

decoding_tests = {}

# Greedy
greedy_result = generate_continuation(model, tokenizer, once_upon, max_new_tokens=30, greedy=True)
print(f"\nGreedy:")
print(f"  {tokenizer.decode(greedy_result)[:100]}...")

# Temperature 0.5
temp05 = generate_continuation(model, tokenizer, once_upon, max_new_tokens=30, temperature=0.5)
print(f"\nTemperature 0.5:")
print(f"  {tokenizer.decode(temp05)[:100]}...")

# Temperature 0.7
temp07 = generate_continuation(model, tokenizer, once_upon, max_new_tokens=30, temperature=0.7)
print(f"\nTemperature 0.7:")
print(f"  {tokenizer.decode(temp07)[:100]}...")

# Temperature 1.0
temp10 = generate_continuation(model, tokenizer, once_upon, max_new_tokens=30, temperature=1.0)
print(f"\nTemperature 1.0 (default):")
print(f"  {tokenizer.decode(temp10)[:100]}...")

decoding_tests["greedy"] = tokenizer.decode(greedy_result)
decoding_tests["temp_0.5"] = tokenizer.decode(temp05)
decoding_tests["temp_0.7"] = tokenizer.decode(temp07)
decoding_tests["temp_1.0"] = tokenizer.decode(temp10)

results["test7_decoding_comparison"] = decoding_tests

# ============================================================
# TEST 8: TOKENIZER DIAGNOSTIC
# ============================================================

print(f"\n{'='*100}")
print("TEST 8: TOKENIZER DIAGNOSTIC")
print("="*100)

# Sample some predictions and show tokenization
print("\nSample predictions with token IDs and decoded strings:")

with torch.no_grad():
    text = heldout_data[0]['text']
    tokens = tokenizer.encode(text)[:30]
    
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    output = model(input_ids)
    logits = output["logits"]
    
    print(f"\nFirst 15 positions:")
    print(f"{'Pos':<5} {'Input Token':<12} {'Decoded':<15} {'Top-3 Predictions':<40}")
    print("-"*75)
    
    for pos in range(min(15, len(tokens) - 1)):
        input_token = tokens[pos]
        decoded_input = tokenizer.decode([input_token])
        
        pos_logits = logits[0, pos, :]
        top3 = pos_logits.topk(3).indices.tolist()
        top3_decoded = [tokenizer.decode([t]) for t in top3]
        
        target = tokens[pos + 1] if pos + 1 < len(tokens) else None
        target_decoded = tokenizer.decode([target]) if target else "N/A"
        
        print(f"{pos:<5} {input_token:<12} {repr(decoded_input[:12]):<15} {str(top3_decoded)[:40]}")
    
    print(f"\nNote: Target for pos {pos} would be token {target} = {repr(target_decoded)}")

results["test8_tokenizer_diagnostic"] = {
    "vocab_size": tokenizer.vocab_size,
    "sample_tokens": tokens[:15],
    "sample_decoded": tokenizer.decode(tokens[:15])
}

# ============================================================
# SUMMARY AND HYPOTHESIS CLASSIFICATION
# ============================================================

print(f"\n{'='*100}")
print("DIAGNOSTIC SUMMARY AND HYPOTHESIS CLASSIFICATION")
print("="*100)

# Calculate key metrics
model_top1 = 100 * topk_correct[1] / total_predictions
unigram_top1 = 100 * unigram_topk_correct[1] / unigram_total
model_vs_unigram = model_top1 - unigram_top1

context_benefit = 0
if 256 in context_results and 1 in context_results:
    context_benefit = context_results[1]['avg_loss'] - context_results[256]['avg_loss']

shuffling_effect = 0
if shuffled_result:
    shuffling_effect = shuffled_result['avg_loss'] - avg_loss

model_vs_bigram = model_top1 - bigram_acc

print(f"\n{'='*70}")
print("KEY METRICS")
print("="*70)

print(f"\n1. Model Performance:")
print(f"   Top-1 accuracy: {model_top1:.2f}%")
print(f"   Perplexity: {perplexity:.2f}")

print(f"\n2. Model vs Baselines:")
print(f"   vs Unigram: {model_top1:.2f}% vs {unigram_top1:.2f}% (gain: {model_vs_unigram:+.2f}%)")
print(f"   vs Bigram:  {model_top1:.2f}% vs {bigram_acc:.2f}% (gain: {model_vs_bigram:+.2f}%)")

print(f"\n3. Context Usage:")
print(f"   Context benefit (full vs single token): {context_benefit:.4f}")
print(f"   Shuffling effect: {shuffling_effect:.4f}")

print(f"\n{'='*70}")
print("HYPOTHESIS CLASSIFICATION")
print("="*70)

hypotheses = []

# H1: Model learned only unigram statistics
if model_vs_unigram < 0.5 and context_benefit < 0.1:
    h1_status = "STRONGLY SUPPORTED"
elif model_vs_unigram < 1.0:
    h1_status = "POSSIBLE"
else:
    h1_status = "NOT SUPPORTED"
hypotheses.append(("Model learned only unigram statistics", h1_status, f"Model vs unigram gain: {model_vs_unigram:.2f}%"))

# H2: Model uses contextual information
if context_benefit > 0.1 and shuffling_effect > 0.1:
    h2_status = "STRONGLY SUPPORTED"
elif context_benefit > 0.05 or shuffling_effect > 0.05:
    h2_status = "POSSIBLE"
else:
    h2_status = "NOT SUPPORTED"
hypotheses.append(("Model uses contextual information", h2_status, f"Context benefit: {context_benefit:.4f}, Shuffling effect: {shuffling_effect:.4f}"))

# H3: Model learned bigram/trigram patterns
if model_vs_bigram > 1.0:
    h3_status = "STRONGLY SUPPORTED"
elif model_vs_bigram > 0:
    h3_status = "POSSIBLE"
else:
    h3_status = "NOT SUPPORTED"
hypotheses.append(("Model learned bigram/trigram patterns", h3_status, f"Model vs bigram: {model_vs_bigram:+.2f}%"))

# H4: Generation problem is due to sampling
# Check if greedy is much better
greedy_quality = len([t for t in greedy_result if t < tokenizer.vocab_size])  # Valid tokens
h4_status = "UNKNOWN"  # Need qualitative assessment
hypotheses.append(("Generation problem is due to sampling/decoding", h4_status, "Requires qualitative comparison"))

# H5: Tokenizer/decoding issue
h5_status = "UNKNOWN"
hypotheses.append(("Tokenizer/decoding causes gibberish appearance", h5_status, "Check decoded samples"))

# H6: Data insufficient for multi-token learning
h6_status = "POSSIBLE"
hypotheses.append(("Insufficient data for multi-token patterns", h6_status, f"1.61 tokens/param (heuristic, not proven)"))

# H7: Ternary limits capacity
h7_status = "UNKNOWN"
hypotheses.append(("Ternary quantization limits learning capacity", h7_status, "Requires FP32 comparison"))

# H8: Architecture insufficient
h8_status = "UNKNOWN"
hypotheses.append(("Architecture capacity insufficient", h8_status, "6.8M params, 6 layers"))

print(f"\n{'Hypothesis':<50} {'Status':<20} Evidence")
print("-"*95)

for hyp, status, evidence in hypotheses:
    print(f"{hyp:<50} {status:<20} {evidence}")

# ============================================================
# FINAL CONCLUSIONS
# ============================================================

print(f"\n{'='*100}")
print("CONCLUSIONS")
print("="*100)

print(f"\nA. WHAT THIS DIAGNOSTIC PROVES:")
print(f"   1. Model Top-1 accuracy: {model_top1:.2f}%")
print(f"   2. Model vs Unigram baseline: {model_vs_unigram:+.2f}% gain")
print(f"   3. Model vs Bigram baseline: {model_vs_bigram:+.2f}% gain")
print(f"   4. Context benefit: {context_benefit:.4f} loss reduction")
print(f"   5. Shuffling effect: {shuffling_effect:.4f} loss increase")

if model_vs_unigram > 1.0 and context_benefit > 0.1:
    print(f"\n   ✓ Model DOES learn contextual information beyond unigram statistics")
else:
    print(f"\n   ✗ Model shows LIMITED contextual learning")

print(f"\nB. WHAT THIS DIAGNOSTIC DOES NOT PROVE:")
print(f"   1. Why generation is gibberish")
print(f"   2. Whether data is sufficient")
print(f"   3. Whether ternary limits capacity")
print(f"   4. Whether architecture is sufficient")
print(f"   5. Root cause of gibberish (beyond identifying what model learned)")

print(f"\nC. HIGHEST-VALUE NEXT EXPERIMENT:")
print(f"   Based on the diagnostic, the model {'HAS' if model_vs_unigram > 1.0 else 'HAS NOT'} learned contextual patterns.")
print(f"   If contextual learning is limited: investigate data/ternary/architecture bottlenecks")
print(f"   If contextual learning exists: investigate why generation still fails")

# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results["summary"] = {
    "model_top1_accuracy": model_top1,
    "model_perplexity": perplexity,
    "unigram_top1_accuracy": unigram_top1,
    "model_vs_unigram_gain": model_vs_unigram,
    "model_vs_bigram_gain": model_vs_bigram,
    "context_benefit": context_benefit,
    "shuffling_effect": shuffling_effect,
    "hypotheses": hypotheses
}

with open(OUTPUT_DIR / "context_forensic_diagnostic.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*100}")
print(f"RESULTS SAVED: {OUTPUT_DIR / 'context_forensic_diagnostic.json'}")
print("="*100)
