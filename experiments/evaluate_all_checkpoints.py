#!/usr/bin/env python3
"""
Comprehensive Checkpoint Evaluation

Compares all available checkpoints for next-token prediction and generation quality.
"""

import torch
import torch.nn as nn
import sys
import math
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

# Configuration
MODEL_CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256
}

DEVICE = 'cpu'

# Test prompts
PROMPTS = [
    "Once upon a time",
    "The little girl",
    "A boy named",
    "There was a",
    "One day"
]

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

def load_checkpoint(checkpoint_path):
    """Load checkpoint and return model"""
    model = create_model()
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, checkpoint

def evaluate_next_token_accuracy(model, tokenizer, num_samples=100):
    """Evaluate next-token prediction accuracy"""
    
    # Generate test sequences from tokenizer
    # Use a simple approach: randomly sample tokens and evaluate
    
    torch.manual_seed(42)
    
    total_loss = 0.0
    total_top1_correct = 0
    total_top5_correct = 0
    total_correct_prob = 0.0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Generate random sequence
            seq_len = 32
            input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len))
            
            # Split into input and target
            x = input_ids[:, :-1]
            y = input_ids[:, 1:]
            
            # Forward pass
            output = model(x)
            logits = output['logits']
            
            # Evaluate each position
            for i in range(y.size(1)):
                target = y[0, i].item()
                logits_i = logits[0, i, :]
                
                # Top-1 accuracy
                pred = logits_i.argmax().item()
                if pred == target:
                    total_top1_correct += 1
                
                # Top-5 accuracy
                top5 = logits_i.topk(5).indices.tolist()
                if target in top5:
                    total_top5_correct += 1
                
                # Probability assigned to correct token
                probs = torch.softmax(logits_i, dim=0)
                correct_prob = probs[target].item()
                total_correct_prob += correct_prob
                
                # Loss
                loss = criterion(logits_i.unsqueeze(0), torch.tensor([target]))
                total_loss += loss.item()
                
                total_samples += 1
    
    return {
        'loss': total_loss / total_samples,
        'top1_acc': 100 * total_top1_correct / total_samples,
        'top5_acc': 100 * total_top5_correct / total_samples,
        'correct_prob': 100 * total_correct_prob / total_samples,
        'samples': total_samples
    }

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
    
    return generated

def compute_diversity_metrics(text):
    """Compute diversity metrics for generated text"""
    words = text.split()
    if len(words) < 2:
        return {'word_diversity': 0.0, '1gram_rep': 0.0, '2gram_rep': 0.0}
    
    # Word diversity
    unique_words = len(set(words))
    word_diversity = unique_words / len(words)
    
    # 1-gram repetition
    word_counts = Counter(words)
    unique_1gram = len(word_counts)
    total_1gram = len(words)
    rep_1gram = 1.0 - (unique_1gram / total_1gram)
    
    # 2-gram repetition
    bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    unique_2gram = len(bigram_counts)
    total_2gram = len(bigrams) if bigrams else 1
    rep_2gram = 1.0 - (unique_2gram / total_2gram)
    
    return {
        'word_diversity': word_diversity,
        '1gram_rep': rep_1gram,
        '2gram_rep': rep_2gram
    }

def evaluate_checkpoint(checkpoint_path, name, tokenizer):
    """Evaluate a single checkpoint"""
    
    print(f"\n{'='*70}")
    print(f"EVALUATING: {name}")
    print(f"{'='*70}")
    
    try:
        model, checkpoint = load_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return None
    
    # Get checkpoint info
    step = checkpoint.get('step', 'unknown') if isinstance(checkpoint, dict) else 'unknown'
    loss = checkpoint.get('loss', 'unknown') if isinstance(checkpoint, dict) else 'unknown'
    
    print(f"Step: {step}")
    print(f"Training loss: {loss:.4f}" if isinstance(loss, (int, float)) else f"Training loss: {loss}")
    
    # Evaluate next-token prediction
    print(f"\n--- Next-Token Prediction ---")
    
    metrics = evaluate_next_token_accuracy(model, tokenizer, num_samples=50)
    
    print(f"Cross-entropy loss: {metrics['loss']:.4f}")
    print(f"Top-1 accuracy: {metrics['top1_acc']:.2f}%")
    print(f"Top-5 accuracy: {metrics['top5_acc']:.2f}%")
    print(f"Correct token probability: {metrics['correct_prob']:.2f}%")
    
    # Get perplexity
    perplexity = math.exp(metrics['loss']) if metrics['loss'] < 10 else float('inf')
    print(f"Perplexity: {perplexity:.1f}")
    
    # Generate samples
    print(f"\n--- Generation Samples ---")
    
    generations = []
    diversity_scores = []
    
    for prompt in PROMPTS[:3]:  # Use first 3 prompts
        gen = generate_sample(model, tokenizer, prompt, max_tokens=40)
        div = compute_diversity_metrics(gen)
        
        generations.append({'prompt': prompt, 'generated': gen})
        diversity_scores.append(div)
        
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {gen[:80]}...")
        print(f"Diversity: {div['word_diversity']:.3f}, 1-gram rep: {div['1gram_rep']:.3f}")
    
    # Average diversity
    avg_diversity = sum(d['word_diversity'] for d in diversity_scores) / len(diversity_scores)
    avg_1gram_rep = sum(d['1gram_rep'] for d in diversity_scores) / len(diversity_scores)
    avg_2gram_rep = sum(d['2gram_rep'] for d in diversity_scores) / len(diversity_scores)
    
    print(f"\nAverage diversity: {avg_diversity:.3f}")
    print(f"Average 1-gram repetition: {avg_1gram_rep:.3f}")
    print(f"Average 2-gram repetition: {avg_2gram_rep:.3f}")
    
    return {
        'name': name,
        'step': step,
        'training_loss': loss,
        'eval_loss': metrics['loss'],
        'top1_acc': metrics['top1_acc'],
        'top5_acc': metrics['top5_acc'],
        'correct_prob': metrics['correct_prob'],
        'perplexity': perplexity,
        'diversity': avg_diversity,
        '1gram_rep': avg_1gram_rep,
        '2gram_rep': avg_2gram_rep,
        'generations': generations
    }

def main():
    print("="*70)
    print("COMPREHENSIVE CHECKPOINT EVALUATION")
    print("="*70)
    
    # Load tokenizer
    tokenizer_path = "/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"\nTokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # Define all checkpoints to evaluate
    checkpoints = [
        # Early training (from experiments/results/checkpoints/)
        ("step_500", "/root/agent-sandbox/ARTN/experiments/results/checkpoints/step_500.pt"),
        ("step_1000", "/root/agent-sandbox/ARTN/experiments/results/checkpoints/step_1000.pt"),
        ("step_1500", "/root/agent-sandbox/ARTN/experiments/results/checkpoints/step_1500.pt"),
        ("step_2000", "/root/agent-sandbox/ARTN/experiments/results/checkpoints/step_2000.pt"),
        ("step_2500", "/root/agent-sandbox/ARTN/experiments/results/checkpoints/step_2500.pt"),
        
        # 3k-step model (final.pt from experiments)
        ("step_3000", "/root/agent-sandbox/ARTN/experiments/results/checkpoints/final.pt"),
        
        # 5k-step experiment models
        ("lr_001_constant_5k", "/root/agent-sandbox/ARTN/experiments/results/lr_experiments/lr_001_constant_20260818_194425/best_checkpoint.pt"),
        ("lr_003_constant_5k", "/root/agent-sandbox/ARTN/experiments/results/lr_experiments/lr_003_constant_20260819_163222/best_checkpoint.pt"),
        
        # 50k-step final model
        ("step_50k", "/root/agent-sandbox/ARTN/outputs/artn-50k-best-config/final_model/model.pt"),
    ]
    
    results = []
    
    for name, path in checkpoints:
        try:
            result = evaluate_checkpoint(path, name, tokenizer)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\nERROR evaluating {name}: {e}")
    
    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    
    print(f"\n{'Name':<25} {'Step':<8} {'Top-1%':<8} {'Top-5%':<8} {'Loss':<8} {'PPL':<8} {'Diversity':<10}")
    print("-" * 70)
    
    for r in results:
        step_str = str(r['step']) if r['step'] != 'unknown' else '?'
        print(f"{r['name']:<25} {step_str:<8} {r['top1_acc']:<8.2f} {r['top5_acc']:<8.2f} "
              f"{r['eval_loss']:<8.4f} {r['perplexity']:<8.1f} {r['diversity']:<10.3f}")
    
    print(f"\n{'='*70}")
    print("KEY QUESTION ANALYSIS")
    print(f"{'='*70}")
    
    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        
        print(f"\nCompared: {first['name']} vs {last['name']}")
        print(f"\nTop-1 accuracy change: {first['top1_acc']:.2f}% → {last['top1_acc']:.2f}%")
        print(f"Perplexity change: {first['perplexity']:.1f} → {last['perplexity']:.1f}")
        print(f"Diversity change: {first['diversity']:.3f} → {last['diversity']:.3f}")
        
        # Determine which case fits
        acc_improvement = last['top1_acc'] - first['top1_acc']
        
        print(f"\n--- DIAGNOSIS ---")
        
        if last['top1_acc'] > 30:  # Reasonable next-token prediction
            print(f"Case A: Good prediction but poor generation")
            print(f"  → Investigate generation/token decoding")
        elif acc_improvement > 10:  # Significant improvement
            print(f"Case C: Performance improves but remains low")
            print(f"  → Insufficient training or architecture issue")
        else:  # No significant improvement
            print(f"Case B: Poor next-token prediction throughout")
            print(f"  → Fundamental training/architecture issue")

if __name__ == "__main__":
    main()
