#!/usr/bin/env python3
"""
RIGOROUS CHECKPOINT REGRESSION ANALYSIS
========================================

Evaluates steps 1000, 2000, 5000, 8000, 10000 with:
- Train/val loss and gap
- Perplexity
- Fixed prompts with greedy decoding and deterministic seeds
- Repetition metrics
- Vocabulary diversity
- Token entropy/confidence
- Weight statistics
- Ternary weight distribution

DOES NOT ASSUME OVERFITTING OR CATASTROPHIC FORGETTING
ONLY REPORTS MEASURED METRICS
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
import json
import math
import numpy as np

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from experiments.bitlinear_adamw_controlled import BitLinearGPT

# =============================================================================
# CONFIGURATION
# =============================================================================

CHECKPOINT_STEPS = [1000, 2000, 5000, 8000, 10000]

# Fixed prompts for consistent evaluation
FIXED_PROMPTS = [
    "Once upon a time there was a little girl who",
    "There was a boy who loved to play with his",
    "One day, a happy dog went to the park and",
    "The little cat sat on the warm mat and",
    "A kind mother helped her son to",
    "Once upon a time there was a princess named",
    "There was a little bird who lived in a",
    "A baby bear loved to eat honey and",
    "The kind king ruled his kingdom with",
    "A little mouse found a big piece of",
]

# Deterministic seeds for each prompt
GENERATION_SEEDS = [42, 123, 456, 789, 1024]

MAX_GENERATION_TOKENS = 80

# =============================================================================
# LOAD UTILITIES
# =============================================================================

def load_tokenizer():
    """Load tokenizer."""
    tokenizer_path = BASE_DIR / 'outputs' / 'artn-50k-best-config' / 'tokenizer.json'
    return BPETokenizer.load(str(tokenizer_path))

def load_checkpoint(step):
    """Load checkpoint for given step."""
    ckpt_path = BASE_DIR / 'outputs' / 'bitlinear_adamw_20260910_143646' / 'checkpoints' / 'bitlinear_adamw' / f'step_{step}.pt'
    return torch.load(ckpt_path, map_location='cpu')

def load_model(checkpoint):
    """Load model from checkpoint."""
    model = BitLinearGPT(
        vocab_size=4000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_loss_metrics(checkpoint, tokenizer, device='cpu'):
    """Compute train/val loss metrics from checkpoint."""
    
    # Load cached data
    cache_dir = BASE_DIR / 'outputs' / 'tokenized_cache'
    train_data = torch.load(cache_dir / 'train_tokenized.pt')
    val_data = torch.load(cache_dir / 'val_tokenized.pt')
    
    # Sample for efficiency
    train_sample = train_data[:1000]
    val_sample = val_data[:500]
    
    # Load model
    model = load_model(checkpoint)
    model.eval()
    
    # Compute train loss
    train_losses = []
    with torch.no_grad():
        for tokens in train_sample:
            if len(tokens) < 2:
                continue
            input_ids = torch.tensor([tokens[:-1]])
            targets = torch.tensor([tokens[1:]])
            
            output = model(input_ids, targets=targets)
            if output['loss'] is not None and not torch.isnan(output['loss']):
                train_losses.append(output['loss'].item())
    
    # Compute val loss
    val_losses = []
    with torch.no_grad():
        for tokens in val_sample:
            if len(tokens) < 2:
                continue
            input_ids = torch.tensor([tokens[:-1]])
            targets = torch.tensor([tokens[1:]])
            
            output = model(input_ids, targets=targets)
            if output['loss'] is not None and not torch.isnan(output['loss']):
                val_losses.append(output['loss'].item())
    
    train_loss = np.mean(train_losses) if train_losses else float('inf')
    val_loss = np.mean(val_losses) if val_losses else float('inf')
    gap = abs(train_loss - val_loss)
    
    perplexity = math.exp(val_loss) if val_loss < 10 else float('inf')
    
    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_val_gap': gap,
        'perplexity': perplexity,
    }

def generate_greedy(model, tokenizer, prompt, max_tokens=80):
    """Generate with greedy decoding."""
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)['logits']
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())

def generate_with_seed(model, tokenizer, prompt, seed, temperature=0.9, max_tokens=80):
    """Generate with deterministic seed."""
    torch.manual_seed(seed)
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)['logits']
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())

def compute_repetition_metrics(text):
    """Compute repetition metrics for generated text."""
    
    words = text.lower().split()
    tokens = list(text)  # Character-level tokens
    
    if len(words) == 0 or len(tokens) == 0:
        return {
            'unique_token_ratio': 0.0,
            'repeated_bigram_rate': 0.0,
            'repeated_trigram_rate': 0.0,
            'repetition_score': 0.0,
        }
    
    # Unique token ratio
    unique_words = len(set(words))
    unique_token_ratio = unique_words / len(words) if len(words) > 0 else 0.0
    
    # Repeated n-gram rates
    def ngram_rate(items, n):
        if len(items) < n:
            return 0.0
        ngrams = [tuple(items[i:i+n]) for i in range(len(items) - n + 1)]
        ngram_counts = Counter(ngrams)
        repeated = sum(count - 1 for count in ngram_counts.values() if count > 1)
        return repeated / len(ngrams) if len(ngrams) > 0 else 0.0
    
    repeated_bigram_rate = ngram_rate(words, 2)
    repeated_trigram_rate = ngram_rate(words, 3)
    
    # Overall repetition score (lower is better)
    repetition_score = (1 - unique_token_ratio + repeated_bigram_rate + repeated_trigram_rate) / 3
    
    return {
        'unique_token_ratio': unique_token_ratio,
        'repeated_bigram_rate': repeated_bigram_rate,
        'repeated_trigram_rate': repeated_trigram_rate,
        'repetition_score': repetition_score,
    }

def compute_vocabulary_diversity(text, tokenizer):
    """Compute vocabulary diversity metrics."""
    
    tokens = tokenizer.encode(text)
    if len(tokens) == 0:
        return {
            'vocabulary_size': 0,
            'vocabulary_diversity': 0.0,
            'rare_token_ratio': 0.0,
        }
    
    token_counts = Counter(tokens)
    vocab_size = len(token_counts)
    vocab_diversity = vocab_size / len(tokens)
    
    # Rare tokens (used only once)
    rare_tokens = sum(1 for count in token_counts.values() if count == 1)
    rare_token_ratio = rare_tokens / len(tokens)
    
    return {
        'vocabulary_size': vocab_size,
        'vocabulary_diversity': vocab_diversity,
        'rare_token_ratio': rare_token_ratio,
    }

def compute_token_entropy_and_confidence(model, tokenizer, prompt):
    """Compute token-level entropy and confidence metrics."""
    
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    entropies = []
    max_probs = []
    
    with torch.no_grad():
        for _ in range(50):  # Sample 50 tokens
            logits = model(input_ids)['logits']
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Entropy
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).item()
            entropies.append(entropy)
            
            # Max probability (confidence)
            max_prob = probs.max(dim=-1)[0].item()
            max_probs.append(max_prob)
            
            # Greedy next token
            next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return {
        'avg_token_entropy': np.mean(entropies),
        'std_token_entropy': np.std(entropies),
        'avg_max_probability': np.mean(max_probs),
        'std_max_probability': np.std(max_probs),
    }

def compute_weight_statistics(model):
    """Compute weight and ternary distribution statistics."""
    
    total_params = 0
    ternary_stats = {
        'positive': 0,
        'zero': 0,
        'negative': 0,
        'sparsity': 0.0,
    }
    
    weight_norms = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            w = module.weight.data
            total_params += w.numel()
            weight_norms.append(w.norm().item())
            
            # Ternary distribution (threshold at 0.1)
            positive = (w > 0.1).sum().item()
            negative = (w < -0.1).sum().item()
            zero = w.numel() - positive - negative
            
            ternary_stats['positive'] += positive
            ternary_stats['negative'] += negative
            ternary_stats['zero'] += zero
    
    if total_params > 0:
        ternary_stats['sparsity'] = ternary_stats['zero'] / total_params
        ternary_stats['positive_ratio'] = ternary_stats['positive'] / total_params
        ternary_stats['negative_ratio'] = ternary_stats['negative'] / total_params
    
    return {
        'total_params': total_params,
        'ternary_distribution': ternary_stats,
        'avg_weight_norm': np.mean(weight_norms) if weight_norms else 0.0,
        'std_weight_norm': np.std(weight_norms) if weight_norms else 0.0,
    }

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_checkpoint(step, tokenizer):
    """Perform comprehensive analysis of a single checkpoint."""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING CHECKPOINT STEP {step}")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    checkpoint = load_checkpoint(step)
    
    # Load model
    model = load_model(checkpoint)
    
    # 1. Loss metrics
    print("Computing loss metrics...")
    loss_metrics = compute_loss_metrics(checkpoint, tokenizer)
    print(f"  Train loss: {loss_metrics['train_loss']:.4f}")
    print(f"  Val loss: {loss_metrics['val_loss']:.4f}")
    print(f"  Train-val gap: {loss_metrics['train_val_gap']:.4f}")
    print(f"  Perplexity: {loss_metrics['perplexity']:.2f}")
    
    # 2. Weight statistics
    print("\nComputing weight statistics...")
    weight_stats = compute_weight_statistics(model)
    print(f"  Total params: {weight_stats['total_params']:,}")
    print(f"  Sparsity: {weight_stats['ternary_distribution']['sparsity']:.2%}")
    print(f"  Positive ratio: {weight_stats['ternary_distribution']['positive_ratio']:.2%}")
    print(f"  Negative ratio: {weight_stats['ternary_distribution']['negative_ratio']:.2%}")
    print(f"  Avg weight norm: {weight_stats['avg_weight_norm']:.4f}")
    
    # 3. Generation with fixed prompts (greedy)
    print("\nGenerating with fixed prompts (greedy decoding)...")
    greedy_generations = []
    greedy_repetition_metrics = []
    greedy_vocab_metrics = []
    
    for i, prompt in enumerate(FIXED_PROMPTS):
        gen = generate_greedy(model, tokenizer, prompt, MAX_GENERATION_TOKENS)
        greedy_generations.append({
            'prompt': prompt,
            'generation': gen,
        })
        
        rep = compute_repetition_metrics(gen)
        greedy_repetition_metrics.append(rep)
        
        vocab = compute_vocabulary_diversity(gen, tokenizer)
        greedy_vocab_metrics.append(vocab)
        
        if i < 3:  # Show first 3
            print(f"\n  Prompt {i+1}: {prompt}")
            print(f"  Generated: {gen[:100]}...")
    
    # 4. Average repetition metrics
    avg_repetition = {
        'unique_token_ratio': np.mean([m['unique_token_ratio'] for m in greedy_repetition_metrics]),
        'repeated_bigram_rate': np.mean([m['repeated_bigram_rate'] for m in greedy_repetition_metrics]),
        'repeated_trigram_rate': np.mean([m['repeated_trigram_rate'] for m in greedy_repetition_metrics]),
        'repetition_score': np.mean([m['repetition_score'] for m in greedy_repetition_metrics]),
    }
    
    print(f"\nAverage Repetition Metrics:")
    print(f"  Unique token ratio: {avg_repetition['unique_token_ratio']:.4f}")
    print(f"  Repeated bigram rate: {avg_repetition['repeated_bigram_rate']:.4f}")
    print(f"  Repeated trigram rate: {avg_repetition['repeated_trigram_rate']:.4f}")
    print(f"  Repetition score: {avg_repetition['repetition_score']:.4f}")
    
    # 5. Average vocabulary diversity
    avg_vocab = {
        'vocabulary_size': np.mean([m['vocabulary_size'] for m in greedy_vocab_metrics]),
        'vocabulary_diversity': np.mean([m['vocabulary_diversity'] for m in greedy_vocab_metrics]),
        'rare_token_ratio': np.mean([m['rare_token_ratio'] for m in greedy_vocab_metrics]),
    }
    
    print(f"\nAverage Vocabulary Metrics:")
    print(f"  Vocabulary size: {avg_vocab['vocabulary_size']:.1f}")
    print(f"  Vocabulary diversity: {avg_vocab['vocabulary_diversity']:.4f}")
    print(f"  Rare token ratio: {avg_vocab['rare_token_ratio']:.4f}")
    
    # 6. Token entropy and confidence
    print("\nComputing token entropy and confidence...")
    entropy_metrics = compute_token_entropy_and_confidence(model, tokenizer, FIXED_PROMPTS[0])
    print(f"  Avg token entropy: {entropy_metrics['avg_token_entropy']:.4f}")
    print(f"  Std token entropy: {entropy_metrics['std_token_entropy']:.4f}")
    print(f"  Avg max probability: {entropy_metrics['avg_max_probability']:.4f}")
    print(f"  Std max probability: {entropy_metrics['std_max_probability']:.4f}")
    
    # Compile all results
    results = {
        'step': step,
        'loss_metrics': loss_metrics,
        'weight_stats': weight_stats,
        'repetition_metrics': avg_repetition,
        'vocabulary_metrics': avg_vocab,
        'entropy_metrics': entropy_metrics,
        'greedy_generations': greedy_generations,
    }
    
    return results

def create_comparison_table(all_results):
    """Create comparison table across all checkpoints."""
    
    print(f"\n{'='*80}")
    print(f"CHECKPOINT COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Step':<8} {'Train Loss':<12} {'Val Loss':<12} {'Gap':<10} {'PPL':<10} {'Sparsity':<10} {'Rep Score':<12}")
    print("-" * 80)
    
    for r in all_results:
        step = r['step']
        train = r['loss_metrics']['train_loss']
        val = r['loss_metrics']['val_loss']
        gap = r['loss_metrics']['train_val_gap']
        ppl = r['loss_metrics']['perplexity']
        sparsity = r['weight_stats']['ternary_distribution']['sparsity']
        rep = r['repetition_metrics']['repetition_score']
        
        print(f"{step:<8} {train:<12.4f} {val:<12.4f} {gap:<10.4f} {ppl:<10.2f} {sparsity:<10.2%} {rep:<12.4f}")
    
    print()
    
    # Additional metrics
    print(f"\n{'Step':<8} {'Unique%':<12} {'Bigram%':<12} {'Trigram%':<12} {'Vocab Div':<12} {'Entropy':<12} {'MaxProb':<12}")
    print("-" * 80)
    
    for r in all_results:
        step = r['step']
        unique = r['repetition_metrics']['unique_token_ratio']
        bigram = r['repetition_metrics']['repeated_bigram_rate']
        trigram = r['repetition_metrics']['repeated_trigram_rate']
        vocab = r['vocabulary_metrics']['vocabulary_diversity']
        entropy = r['entropy_metrics']['avg_token_entropy']
        maxprob = r['entropy_metrics']['avg_max_probability']
        
        print(f"{step:<8} {unique:<12.4f} {bigram:<12.4f} {trigram:<12.4f} {vocab:<12.4f} {entropy:<12.4f} {maxprob:<12.4f}")

def identify_first_degrading_metric(all_results):
    """Identify the FIRST metric that degrades (correlates with generation quality)."""
    
    print(f"\n{'='*80}")
    print(f"METRIC DEGRADATION ANALYSIS")
    print(f"{'='*80}\n")
    
    steps = [r['step'] for r in all_results]
    
    # Find peak step (usually 5000 based on generation quality)
    # We'll check which metrics change direction at the same point
    
    metrics_to_check = [
        ('val_loss', 'loss_metrics'),
        ('train_val_gap', 'loss_metrics'),
        ('perplexity', 'loss_metrics'),
        ('sparsity', 'weight_stats', 'ternary_distribution'),
        ('repetition_score', 'repetition_metrics'),
        ('unique_token_ratio', 'repetition_metrics'),
        ('vocabulary_diversity', 'vocabulary_metrics'),
        ('avg_token_entropy', 'entropy_metrics'),
        ('avg_max_probability', 'entropy_metrics'),
    ]
    
    print("Checking for metric degradation...\n")
    
    for metric_info in metrics_to_check:
        if len(metric_info) == 2:
            metric_name, category = metric_info
            values = [r[category][metric_name] for r in all_results]
        else:
            metric_name, category, subcategory = metric_info
            values = [r[category][subcategory][metric_name] for r in all_results]
        
        # Find direction changes
        increasing = []
        decreasing = []
        
        for i in range(1, len(values)):
            if values[i] > values[i-1]:
                increasing.append(steps[i])
            elif values[i] < values[i-1]:
                decreasing.append(steps[i])
        
        print(f"{metric_name}:")
        print(f"  Values: {[f'{v:.4f}' for v in values]}")
        print(f"  Increases at steps: {increasing}")
        print(f"  Decreases at steps: {decreasing}")
        print()

def main():
    print("="*80)
    print("RIGOROUS CHECKPOINT REGRESSION ANALYSIS")
    print("="*80)
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Analyze each checkpoint
    all_results = []
    
    for step in CHECKPOINT_STEPS:
        results = analyze_checkpoint(step, tokenizer)
        all_results.append(results)
    
    # Create comparison table
    create_comparison_table(all_results)
    
    # Identify first degrading metric
    identify_first_degrading_metric(all_results)
    
    # Save results
    output_path = BASE_DIR / 'outputs' / 'checkpoint_regression_analysis.json'
    with open(output_path, 'w') as f:
        # Convert to serializable format
        serializable = []
        for r in all_results:
            sr = {
                'step': r['step'],
                'loss_metrics': r['loss_metrics'],
                'repetition_metrics': r['repetition_metrics'],
                'vocabulary_metrics': r['vocabulary_metrics'],
                'entropy_metrics': r['entropy_metrics'],
            }
            serializable.append(sr)
        
        json.dump(serializable, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
