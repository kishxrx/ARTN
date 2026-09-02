#!/usr/bin/env python3
"""
Checkpoint Analysis Script

Analyzes all available checkpoints:
1. Generation quality
2. Token/word diversity
3. N-gram repetition
4. Ternary weight distribution layer-by-layer
5. Validation loss/perplexity
"""

import sys
import json
import torch
import torch.nn as nn
import math
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

# ============================================================
# Configuration
# ============================================================

MODEL_CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256
}

PROMPTS = [
    "Once upon a time",
    "The little girl",
    "A boy named",
    "There was a",
    "One day"
]

NUM_SAMPLES_PER_PROMPT = 5  # Reduced for faster analysis
MAX_TOKENS = 30  # Reduced token length

# Checkpoints to analyze
CHECKPOINTS = {
    "lr_001_constant": {
        "step_5000": "/root/agent-sandbox/ARTN/experiments/results/lr_experiments/lr_001_constant_20260818_194425/best_checkpoint.pt"
    },
    "lr_003_constant": {
        "step_5000": "/root/agent-sandbox/ARTN/experiments/results/lr_experiments/lr_003_constant_20260818_232652/best_checkpoint.pt"
    },
    "lr_003_cosine": {
        "step_5000": "/root/agent-sandbox/ARTN/experiments/results/lr_experiments/lr_003_cosine_20260819_100817/best_checkpoint.pt"
    },
    "lr_003_constant_fixed": {
        "step_5000": "/root/agent-sandbox/ARTN/experiments/results/lr_experiments/lr_003_constant_20260819_163222/best_checkpoint.pt"
    }
}

# ============================================================
# Helper Functions
# ============================================================

def create_model():
    """Create model architecture"""
    return BitLinearGPT(
        vocab_size=MODEL_CONFIG["vocab_size"],
        d_model=MODEL_CONFIG["d_model"],
        n_heads=MODEL_CONFIG["n_heads"],
        n_layers=MODEL_CONFIG["n_layers"],
        d_ff=MODEL_CONFIG["d_ff"],
        max_seq_len=MODEL_CONFIG["max_seq_len"]
    )

def generate_sample(model, tokenizer, prompt, max_tokens=50, temperature=1.0):
    """Generate sample from prompt"""
    model.eval()
    
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        
        for _ in range(max_tokens):
            if input_ids.size(1) >= MODEL_CONFIG["max_seq_len"]:
                break
            
            input_ids = torch.clamp(input_ids, 0, tokenizer.vocab_size - 1)
            
            output = model(input_ids)
            logits = output['logits']
            next_token_logits = logits[:, -1, :] / temperature
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        generated = tokenizer.decode(input_ids[0].tolist())
    
    model.train()
    return generated

def compute_word_diversity(text):
    """Compute word diversity (unique words / total words)"""
    words = text.split()
    if len(words) == 0:
        return 0.0
    unique_words = len(set(words))
    return unique_words / len(words)

def compute_ngram_repetition(text, n=1):
    """Compute n-gram repetition score"""
    words = text.split()
    if len(words) < n:
        return 0.0
    
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    if len(ngrams) == 0:
        return 0.0
    
    ngram_counts = Counter(ngrams)
    unique_ngrams = len(ngram_counts)
    total_ngrams = len(ngrams)
    
    # Repetition = 1 - uniqueness ratio
    return 1.0 - (unique_ngrams / total_ngrams)

def compute_structure_score(text):
    """Compute simple structure score (presence of punctuation, capitalization)"""
    # Check for sentence structure
    has_period = '.' in text
    has_comma = ',' in text
    has_capital = any(c.isupper() for c in text if c.isalpha())
    
    # Check for repeated structure
    words = text.split()
    if len(words) < 3:
        return 0.0
    
    # Score based on presence of structure markers
    score = 0.0
    if has_period:
        score += 0.33
    if has_comma:
        score += 0.33
    if has_capital:
        score += 0.34
    
    return score

def analyze_ternary_layer_by_layer(model):
    """Analyze ternary weight distribution for each layer"""
    layer_stats = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and 'BitLinear' in module.__class__.__name__:
            w = module.weight.data
            
            # Count ternary values
            neg_count = (w < -0.5).sum().item()
            zero_count = ((w >= -0.5) & (w <= 0.5)).sum().item()
            pos_count = (w > 0.5).sum().item()
            total = w.numel()
            
            # Compute scale if available
            scale = w.abs().mean().item() if total > 0 else 0.0
            
            # Compute sparsity
            sparsity = zero_count / total if total > 0 else 0.0
            
            # Store stats
            layer_stats[name] = {
                'neg_pct': 100 * neg_count / total if total > 0 else 0.0,
                'zero_pct': 100 * zero_count / total if total > 0 else 0.0,
                'pos_pct': 100 * pos_count / total if total > 0 else 0.0,
                'scale': scale,
                'sparsity': sparsity,
                'shape': tuple(w.shape)
            }
    
    return layer_stats

def compute_validation_loss(model, tokenizer, device='cpu'):
    """Compute validation loss on held-out synthetic data"""
    model.eval()
    
    # Generate simple validation data
    torch.manual_seed(999)  # Different seed for validation
    val_losses = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for _ in range(10):  # 10 validation batches
            # Create synthetic batch (same as training)
            input_ids = torch.randint(0, tokenizer.vocab_size, (4, 64))
            target_ids = torch.cat([input_ids[:, 1:], torch.zeros(4, 1, dtype=torch.long)], dim=1)
            
            # Forward pass
            output = model(input_ids)
            logits = output['logits']
            
            # Compute loss
            loss = criterion(
                logits.reshape(-1, MODEL_CONFIG['vocab_size']),
                target_ids.reshape(-1)
            )
            val_losses.append(loss.item())
    
    model.train()
    return sum(val_losses) / len(val_losses)

# ============================================================
# Main Analysis
# ============================================================

def analyze_checkpoint(checkpoint_path, checkpoint_name, tokenizer):
    """Analyze single checkpoint"""
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {checkpoint_name}")
    print(f"{'='*70}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model and load weights
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nCheckpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    # ============================================================
    # 1. Generate multiple samples
    # ============================================================
    
    print(f"\n--- Generation Quality ---")
    
    all_generations = []
    all_diversity = []
    all_1gram_rep = []
    all_2gram_rep = []
    all_structure = []
    
    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt}")
        
        prompt_samples = []
        for i in range(NUM_SAMPLES_PER_PROMPT):
            gen = generate_sample(model, tokenizer, prompt, MAX_TOKENS)
            prompt_samples.append(gen)
            
            # Compute metrics
            diversity = compute_word_diversity(gen)
            rep_1gram = compute_ngram_repetition(gen, n=1)
            rep_2gram = compute_ngram_repetition(gen, n=2)
            structure = compute_structure_score(gen)
            
            all_diversity.append(diversity)
            all_1gram_rep.append(rep_1gram)
            all_2gram_rep.append(rep_2gram)
            all_structure.append(structure)
        
        # Show first 3 samples
        for i, gen in enumerate(prompt_samples[:3]):
            print(f"  Sample {i+1}: {gen[:70]}...")
        
        all_generations.append({
            'prompt': prompt,
            'samples': prompt_samples
        })
    
    # Average metrics
    avg_diversity = sum(all_diversity) / len(all_diversity)
    avg_1gram_rep = sum(all_1gram_rep) / len(all_1gram_rep)
    avg_2gram_rep = sum(all_2gram_rep) / len(all_2gram_rep)
    avg_structure = sum(all_structure) / len(all_structure)
    
    print(f"\n--- Metrics Summary ---")
    print(f"Word diversity: {avg_diversity:.4f} (higher is better)")
    print(f"1-gram repetition: {avg_1gram_rep:.4f} (lower is better)")
    print(f"2-gram repetition: {avg_2gram_rep:.4f} (lower is better)")
    print(f"Structure score: {avg_structure:.4f} (higher is better)")
    
    # ============================================================
    # 2. Ternary weight distribution layer-by-layer
    # ============================================================
    
    print(f"\n--- Ternary Distribution Layer-by-Layer ---")
    
    layer_stats = analyze_ternary_layer_by_layer(model)
    
    # Group by layer type
    attention_layers = {}
    ff_layers = {}
    
    for name, stats in layer_stats.items():
        if 'attention' in name.lower() or 'attn' in name.lower():
            attention_layers[name] = stats
        elif 'feed_forward' in name.lower() or 'ff' in name.lower():
            ff_layers[name] = stats
        
        # Print each layer
        print(f"\n  {name}:")
        print(f"    Shape: {stats['shape']}")
        print(f"    Distribution: -1={stats['neg_pct']:.2f}%, 0={stats['zero_pct']:.2f}%, +1={stats['pos_pct']:.2f}%")
        print(f"    Scale: {stats['scale']:.4f}")
        print(f"    Sparsity: {stats['sparsity']:.4f}")
    
    # Average by layer type
    if attention_layers:
        avg_attn_zero = sum(s['zero_pct'] for s in attention_layers.values()) / len(attention_layers)
        print(f"\n  Average attention layer sparsity: {avg_attn_zero:.2f}%")
    
    if ff_layers:
        avg_ff_zero = sum(s['zero_pct'] for s in ff_layers.values()) / len(ff_layers)
        print(f"  Average feed-forward layer sparsity: {avg_ff_zero:.2f}%")
    
    # Overall statistics
    all_zero_pcts = [s['zero_pct'] for s in layer_stats.values()]
    all_scales = [s['scale'] for s in layer_stats.values()]
    
    print(f"\n  Overall statistics:")
    print(f"    Min zero%: {min(all_zero_pcts):.2f}%")
    print(f"    Max zero%: {max(all_zero_pcts):.2f}%")
    print(f"    Mean zero%: {sum(all_zero_pcts)/len(all_zero_pcts):.2f}%")
    print(f"    Scale range: [{min(all_scales):.4f}, {max(all_scales):.4f}]")
    
    # ============================================================
    # 3. Validation loss
    # ============================================================
    
    print(f"\n--- Validation Loss ---")
    val_loss = compute_validation_loss(model, tokenizer)
    val_ppl = math.exp(val_loss) if val_loss < 10 else 9999.0
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation perplexity: {val_ppl:.1f}")
    
    # Return results
    return {
        'checkpoint_name': checkpoint_name,
        'step': checkpoint.get('step', 'unknown'),
        'loss': checkpoint.get('loss', 'unknown'),
        'metrics': {
            'word_diversity': avg_diversity,
            '1gram_repetition': avg_1gram_rep,
            '2gram_repetition': avg_2gram_rep,
            'structure_score': avg_structure,
            'val_loss': val_loss,
            'val_perplexity': val_ppl
        },
        'ternary_stats': layer_stats,
        'generations': all_generations
    }

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("CHECKPOINT ANALYSIS")
    print("=" * 70)
    
    # Load tokenizer
    tokenizer_path = "/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"\nTokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # Analyze all checkpoints
    results = {}
    
    for exp_name, checkpoints in CHECKPOINTS.items():
        for step_name, checkpoint_path in checkpoints.items():
            checkpoint_label = f"{exp_name}_{step_name}"
            results[checkpoint_label] = analyze_checkpoint(checkpoint_path, checkpoint_label, tokenizer)
    
    # Save results
    output_path = Path(__file__).parent / "results" / "checkpoint_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            'checkpoint_name': data['checkpoint_name'],
            'step': data['step'],
            'loss': float(data['loss']) if isinstance(data['loss'], (int, float)) else data['loss'],
            'metrics': data['metrics'],
            'ternary_stats': {
                layer: {
                    'neg_pct': stats['neg_pct'],
                    'zero_pct': stats['zero_pct'],
                    'pos_pct': stats['pos_pct'],
                    'scale': stats['scale'],
                    'sparsity': stats['sparsity']
                }
                for layer, stats in data['ternary_stats'].items()
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    
    print(f"\n{'Checkpoint':<40} {'Div':<8} {'1-gram':<8} {'2-gram':<8} {'Struct':<8} {'Val Loss':<10}")
    print("-" * 70)
    
    for name, data in results.items():
        m = data['metrics']
        print(f"{name:<40} {m['word_diversity']:<8.4f} {m['1gram_repetition']:<8.4f} "
              f"{m['2gram_repetition']:<8.4f} {m['structure_score']:<8.4f} {m['val_loss']:<10.4f}")

if __name__ == "__main__":
    main()
