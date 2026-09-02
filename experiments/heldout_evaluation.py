#!/usr/bin/env python3
"""
Held-Out Evaluation for FP32 vs Ternary Models

Evaluates existing trained checkpoints on:
1. Original training subset (to measure memorization)
2. Held-out TinyStories (to measure generalization)

Verifies:
- No overlap between training and held-out
- Correct target shifting
- Identical preprocessing
- Same evaluation code for all models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.fp32_control_experiment import StandardGPT
from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*70)
print("HELD-OUT EVALUATION: FP32 vs Ternary")
print("="*70)

# ============================================================
# STEP 1: Define Training Subset (EXACTLY as used in training)
# ============================================================

TRAINING_SENTENCES = [
    "Once upon a time there was a little girl named Sarah.",
    "The little girl loved to play in the garden every day.",
    "One day she found a magic flower in the garden.",
    "The flower glowed with beautiful colors.",
    "Sarah picked the flower and made a wish.",
    "Suddenly a fairy appeared before her eyes.",
    "The fairy granted Sarah three special wishes.",
    "Sarah wished for happiness and peace.",
    "The fairy smiled and disappeared into the light.",
    "From that day on Sarah was always happy.",
    "There was once a brave little boy named Tom.",
    "Tom loved to explore the forest behind his house.",
    "One day he found a secret path in the woods.",
    "The path led to a hidden treasure chest.",
    "Tom opened the chest and found golden coins.",
    "He shared the treasure with his family.",
    "Everyone was proud of the brave boy.",
    "Tom learned that adventures are exciting.",
    "He promised to have more adventures.",
    "The end of the story made everyone smile.",
]

print(f"\nTraining subset: {len(TRAINING_SENTENCES)} sentences")

# ============================================================
# STEP 2: Define HELD-OUT TinyStories (NOT in training)
# ============================================================

# These are REAL TinyStories patterns that are NOT in the training set
HELD_OUT_SENTENCES = [
    "Once there was a little boy named Tim who loved animals.",
    "Tim had a pet dog named Max who was very playful.",
    "Every morning Tim and Max would go for a walk in the park.",
    "One sunny day they saw a beautiful butterfly in the garden.",
    "Tim tried to catch the butterfly but it flew away quickly.",
    "Max barked happily and chased the butterfly around the flowers.",
    "Tim laughed and clapped his hands with joy.",
    "The butterfly landed on a red rose and stayed very still.",
    "Tim watched quietly and smiled at the beautiful sight.",
    "He learned that watching animals was better than catching them.",
    "There was a little girl named Emma who loved to paint.",
    "Emma had many colorful paints and brushes in her room.",
    "She painted pictures of flowers and trees and birds.",
    "Her mother hung the paintings on the wall in the kitchen.",
    "Everyone said Emma was a very talented artist.",
    "Emma felt proud and wanted to paint more pictures.",
    "She painted a big picture of her family for her birthday.",
    "Her parents were very happy with the special gift.",
    "Emma knew that practice would make her even better.",
    "She promised to paint something new every single day.",
    "A little cat named Whiskers lived in a cozy house.",
    "Whiskers liked to sleep in the warm sunshine by the window.",
    "Sometimes she chased mice in the garden behind the house.",
    "Her owner gave her fresh milk and fish every evening.",
    "Whiskers purred happily when she got her favorite treats.",
    "She was a very happy cat with a loving home.",
    "All the neighbors knew Whiskers and loved to pet her.",
    "She was the friendliest cat in the whole neighborhood.",
    "Whiskers taught everyone that cats can be very loving pets.",
    "She lived a long and happy life with her family.",
]

print(f"Held-out subset: {len(HELD_OUT_SENTENCES)} sentences")

# ============================================================
# STEP 3: Verify NO OVERLAP
# ============================================================

print(f"\n{'='*70}")
print("VERIFYING NO OVERLAP BETWEEN TRAINING AND HELD-OUT")
print(f"{'='*70}")

# Check for exact matches
training_set = set(s.lower().strip() for s in TRAINING_SENTENCES)
heldout_set = set(s.lower().strip() for s in HELD_OUT_SENTENCES)

overlap = training_set & heldout_set

if overlap:
    print(f"\n✗ ERROR: Found {len(overlap)} overlapping sentences!")
    for s in overlap:
        print(f"  - {s}")
    sys.exit(1)
else:
    print(f"\n✓ VERIFIED: No overlap between training and held-out sets")

# Check for partial matches (substrings)
partial_matches = []
for ts in TRAINING_SENTENCES:
    for hs in HELD_OUT_SENTENCES:
        if ts.lower() in hs.lower() or hs.lower() in ts.lower():
            partial_matches.append((ts, hs))

if partial_matches:
    print(f"\n⚠ WARNING: Found {len(partial_matches)} partial matches (substrings)")
    for t, h in partial_matches[:3]:
        print(f"  Training: {t[:40]}...")
        print(f"  Held-out: {h[:40]}...")
else:
    print(f"✓ VERIFIED: No partial matches (substrings) found")

# ============================================================
# STEP 4: Load Tokenizer
# ============================================================

print(f"\n{'='*70}")
print("LOADING TOKENIZER")
print(f"{'='*70}")

tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

# ============================================================
# STEP 5: Preprocessing Function (IDENTICAL for all models)
# ============================================================

def preprocess_sentences(sentences, name):
    """Tokenize and pad sentences - IDENTICAL to training pipeline"""
    print(f"\nPreprocessing {name}...")
    
    # Tokenize
    sequences = []
    for sentence in sentences:
        tokens = tokenizer.encode(sentence)
        sequences.append(tokens)
    
    # Pad
    max_len = max(len(s) for s in sequences)
    print(f"  Max length: {max_len}")
    
    padded = []
    for seq in sequences:
        padded.append(seq + [0] * (max_len - len(seq)))
    
    input_ids = torch.tensor(padded)
    
    # Create targets (NEXT-TOKEN PREDICTION)
    # target[i] = input[i+1]
    targets = torch.cat([input_ids[:, 1:], torch.zeros(input_ids.size(0), 1, dtype=torch.long)], dim=1)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Target shape: {targets.shape}")
    
    return input_ids, targets, sequences

# ============================================================
# STEP 6: Verify Target Shifting
# ============================================================

print(f"\n{'='*70}")
print("VERIFYING TARGET SHIFTING")
print(f"{'='*70}")

# Create a small test
test_input = torch.tensor([[1, 2, 3, 4, 5]])
test_target = torch.cat([test_input[:, 1:], torch.zeros(1, 1, dtype=torch.long)], dim=1)

print(f"\nTest input:  {test_input[0].tolist()}")
print(f"Test target: {test_target[0].tolist()}")

expected_target = [2, 3, 4, 5, 0]
if test_target[0].tolist() == expected_target:
    print(f"✓ VERIFIED: Target[i] = Input[i+1] (correct next-token prediction)")
else:
    print(f"✗ ERROR: Target shifting incorrect!")
    sys.exit(1)

# ============================================================
# STEP 7: Preprocess Both Datasets
# ============================================================

train_input, train_targets, train_seqs = preprocess_sentences(TRAINING_SENTENCES, "Training Set")
heldout_input, heldout_targets, heldout_seqs = preprocess_sentences(HELD_OUT_SENTENCES, "Held-Out Set")

# ============================================================
# STEP 8: Evaluation Function (IDENTICAL for all models)
# ============================================================

def evaluate_model(model, model_name, input_ids, targets, tokenizer, dataset_name):
    """
    Evaluate a model on a dataset.
    Uses IDENTICAL evaluation code for all models.
    """
    model.eval()
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    with torch.no_grad():
        # Forward pass
        output = model(input_ids)
        logits = output['logits']
        
        # Loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        perplexity = math.exp(loss.item()) if loss.item() < 10 else float('inf')
        
        # Top-1 accuracy
        preds = logits.argmax(dim=-1)
        top1_correct = (preds == targets).sum().item()
        top1_acc = 100 * top1_correct / targets.numel()
        
        # Top-5 accuracy
        top5 = logits.topk(5, dim=-1).indices
        top5_correct = 0
        total = 0
        for i in range(targets.size(0)):
            for j in range(targets.size(1)):
                if targets[i, j].item() in top5[i, j].tolist():
                    top5_correct += 1
                total += 1
        top5_acc = 100 * top5_correct / total
        
        # Correct token probability
        probs = F.softmax(logits, dim=-1)
        correct_prob_sum = 0
        for i in range(targets.size(0)):
            for j in range(targets.size(1)):
                correct_prob_sum += probs[i, j, targets[i, j].item()].item()
        correct_prob = 100 * correct_prob_sum / total
        
        # Generation samples
        generation_prompts = ["Once upon a time", "The little girl", "A boy named"]
        generations = []
        
        for prompt in generation_prompts:
            input_tokens = torch.tensor([tokenizer.encode(prompt)])
            
            # Generate
            for _ in range(30):
                if input_tokens.size(1) >= 256:
                    break
                
                out = model(input_tokens)
                log = out['logits']
                next_token = log[:, -1, :].argmax(dim=-1, keepdim=True)
                input_tokens = torch.cat([input_tokens, next_token], dim=1)
            
            generated = tokenizer.decode(input_tokens[0].tolist())
            
            # Diversity metrics
            words = generated.split()
            if len(words) > 1:
                word_diversity = len(set(words)) / len(words)
                bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
                bigram_rep = 1.0 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0
            else:
                word_diversity = 0.0
                bigram_rep = 0.0
            
            generations.append({
                'prompt': prompt,
                'generated': generated,
                'word_diversity': word_diversity,
                'bigram_rep': bigram_rep,
            })
    
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'loss': loss.item(),
        'perplexity': perplexity,
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'correct_prob': correct_prob,
        'generations': generations,
    }
    
    return results

# ============================================================
# STEP 9: Load and Evaluate All Models
# ============================================================

print(f"\n{'='*70}")
print("LOADING AND EVALUATING MODELS")
print(f"{'='*70}")

results = []

# Define models to evaluate
models_to_eval = [
    ("FP32 5k", "experiments/results/fp32_control_20260819_232646/step_5000.pt", "fp32"),
    ("Ternary 5k", "experiments/results/lr_experiments/lr_003_constant_20260819_163222/best_checkpoint.pt", "ternary"),
    ("Ternary 50k", "outputs/artn-50k-best-config/final_model/model.pt", "ternary"),
]

for model_name, checkpoint_path, model_type in models_to_eval:
    print(f"\n--- {model_name} ---")
    print(f"Loading: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    if model_type == "fp32":
        model = StandardGPT(4000, 256, 8, 6, 1024, 256)
    else:
        model = BitLinearGPT(4000, 256, 8, 6, 1024, 256)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('step', '?')
        print(f"  Step: {step}")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Step: unknown")
    
    # Evaluate on TRAINING set
    print(f"\n  Evaluating on TRAINING set...")
    train_results = evaluate_model(model, model_name, train_input, train_targets, tokenizer, "Training")
    
    # Evaluate on HELD-OUT set
    print(f"  Evaluating on HELD-OUT set...")
    heldout_results = evaluate_model(model, model_name, heldout_input, heldout_targets, tokenizer, "Held-Out")
    
    results.append({
        'model': model_name,
        'training': train_results,
        'heldout': heldout_results,
    })

# ============================================================
# STEP 10: RESULTS COMPARISON
# ============================================================

print(f"\n{'='*70}")
print("RESULTS COMPARISON")
print(f"{'='*70}")

print(f"\n{'='*70}")
print("A. TRAINING SET PERFORMANCE")
print(f"{'='*70}")

print(f"\n{'Model':<20} {'Loss':<10} {'PPL':<10} {'Top-1%':<10} {'Top-5%':<10} {'Prob%':<10}")
print("-" * 70)

for r in results:
    tr = r['training']
    print(f"{r['model']:<20} {tr['loss']:<10.4f} {tr['perplexity']:<10.1f} "
          f"{tr['top1_acc']:<10.2f} {tr['top5_acc']:<10.2f} {tr['correct_prob']:<10.2f}")

print(f"\n{'='*70}")
print("B. HELD-OUT SET PERFORMANCE")
print(f"{'='*70}")

print(f"\n{'Model':<20} {'Loss':<10} {'PPL':<10} {'Top-1%':<10} {'Top-5%':<10} {'Prob%':<10}")
print("-" * 70)

for r in results:
    hr = r['heldout']
    print(f"{r['model']:<20} {hr['loss']:<10.4f} {hr['perplexity']:<10.1f} "
          f"{hr['top1_acc']:<10.2f} {hr['top5_acc']:<10.2f} {hr['correct_prob']:<10.2f}")

print(f"\n{'='*70}")
print("C. GENERALIZATION GAP (Training - Held-Out)")
print(f"{'='*70}")

print(f"\n{'Model':<20} {'Loss Gap':<15} {'Top-1 Gap':<15} {'Top-5 Gap':<15}")
print("-" * 65)

for r in results:
    tr = r['training']
    hr = r['heldout']
    
    loss_gap = tr['loss'] - hr['loss']
    top1_gap = tr['top1_acc'] - hr['top1_acc']
    top5_gap = tr['top5_acc'] - hr['top5_acc']
    
    print(f"{r['model']:<20} {loss_gap:<15.4f} {top1_gap:<15.2f} {top5_gap:<15.2f}")

print(f"\n{'='*70}")
print("D. FP32 vs TERNARY (5k steps)")
print(f"{'='*70}")

fp32_5k = results[0]
ternary_5k = results[1]

print(f"\nOn TRAINING set:")
print(f"  FP32:    Top-1={fp32_5k['training']['top1_acc']:.2f}%, Loss={fp32_5k['training']['loss']:.4f}")
print(f"  Ternary: Top-1={ternary_5k['training']['top1_acc']:.2f}%, Loss={ternary_5k['training']['loss']:.4f}")

print(f"\nOn HELD-OUT set:")
print(f"  FP32:    Top-1={fp32_5k['heldout']['top1_acc']:.2f}%, Loss={fp32_5k['heldout']['loss']:.4f}")
print(f"  Ternary: Top-1={ternary_5k['heldout']['top1_acc']:.2f}%, Loss={ternary_5k['heldout']['loss']:.4f}")

print(f"\n{'='*70}")
print("E. TERNARY 5k vs 50k")
print(f"{'='*70}")

ternary_50k = results[2]

print(f"\nOn TRAINING set:")
print(f"  Ternary 5k:  Top-1={ternary_5k['training']['top1_acc']:.2f}%, Loss={ternary_5k['training']['loss']:.4f}")
print(f"  Ternary 50k: Top-1={ternary_50k['training']['top1_acc']:.2f}%, Loss={ternary_50k['training']['loss']:.4f}")

print(f"\nOn HELD-OUT set:")
print(f"  Ternary 5k:  Top-1={ternary_5k['heldout']['top1_acc']:.2f}%, Loss={ternary_5k['heldout']['loss']:.4f}")
print(f"  Ternary 50k: Top-1={ternary_50k['heldout']['top1_acc']:.2f}%, Loss={ternary_50k['heldout']['loss']:.4f}")

# ============================================================
# STEP 11: GENERATION COMPARISON
# ============================================================

print(f"\n{'='*70}")
print("GENERATION COMPARISON")
print(f"{'='*70}")

for r in results:
    print(f"\n--- {r['model']} (Held-Out) ---")
    for gen in r['heldout']['generations'][:2]:
        print(f"\n  Prompt: {gen['prompt']}")
        print(f"  Generated: {gen['generated'][:70]}...")
        print(f"  Diversity: {gen['word_diversity']:.3f}, Repetition: {gen['bigram_rep']:.3f}")

# ============================================================
# STEP 12: CLASSIFICATION
# ============================================================

print(f"\n{'='*70}")
print("CLASSIFICATION")
print(f"{'='*70}")

fp32_heldout_top1 = fp32_5k['heldout']['top1_acc']
ternary_5k_heldout_top1 = ternary_5k['heldout']['top1_acc']
ternary_50k_heldout_top1 = ternary_50k['heldout']['top1_acc']

print(f"\nHeld-Out Top-1 Accuracy:")
print(f"  FP32 5k:      {fp32_heldout_top1:.2f}%")
print(f"  Ternary 5k:   {ternary_5k_heldout_top1:.2f}%")
print(f"  Ternary 50k:  {ternary_50k_heldout_top1:.2f}%")

# Determine case
if fp32_heldout_top1 > ternary_5k_heldout_top1 * 2:
    classification = "CASE A: FP32 clearly better on held-out data"
elif abs(fp32_heldout_top1 - ternary_5k_heldout_top1) < 5:
    classification = "CASE B: FP32 and ternary similar on held-out data"
elif max(fp32_heldout_top1, ternary_5k_heldout_top1, ternary_50k_heldout_top1) < 5:
    classification = "CASE C: Both perform poorly on held-out data"
else:
    classification = "CASE D: Results are inconclusive"

print(f"\n{classification}")

# ============================================================
# STEP 13: Save Results
# ============================================================

output_path = Path("experiments/results/heldout_evaluation_results.json")
with open(output_path, 'w') as f:
    # Convert to JSON-serializable
    results_save = []
    for r in results:
        results_save.append({
            'model': r['model'],
            'training': {
                'loss': r['training']['loss'],
                'perplexity': r['training']['perplexity'],
                'top1_acc': r['training']['top1_acc'],
                'top5_acc': r['training']['top5_acc'],
                'correct_prob': r['training']['correct_prob'],
            },
            'heldout': {
                'loss': r['heldout']['loss'],
                'perplexity': r['heldout']['perplexity'],
                'top1_acc': r['heldout']['top1_acc'],
                'top5_acc': r['heldout']['top5_acc'],
                'correct_prob': r['heldout']['correct_prob'],
            },
        })
    
    json.dump({
        'results': results_save,
        'classification': classification,
        'training_sentences': TRAINING_SENTENCES,
        'heldout_sentences': HELD_OUT_SENTENCES,
    }, f, indent=2)

print(f"\nResults saved to: {output_path}")

print(f"\n{'='*70}")
print("HELD-OUT EVALUATION COMPLETE")
print(f"{'='*70}")
