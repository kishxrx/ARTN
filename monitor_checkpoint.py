#!/usr/bin/env python3
"""
CHECKPOINT MONITOR - Run this while training is running.

Usage:
    python monitor_checkpoint.py --step 1000
    
Will check the checkpoint and report:
- Entropy (must be >30%)
- Top token probability (must be <50%)
- Sample quality
- Status: HEALTHY / WARNING / COLLAPSED
"""

import torch
import sys
import argparse
sys.path.insert(0, '.')

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

def top_k_filter(logits, k):
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits

def monitor_checkpoint(step):
    print("="*70)
    print(f"CHECKPOINT MONITOR - Step {step}")
    print("="*70)
    
    # Load checkpoint
    ckpt_path = f"outputs/artn-final/checkpoints/checkpoint_step_{step}.pt"
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        print(f"Training may not have reached step {step} yet.")
        return
    
    print(f"\n[Loading] Checkpoint found")
    print(f"  Loss: {ckpt['loss']:.4f}")
    
    # Load model
    tokenizer = BPETokenizer.load('outputs/artn-final/tokenizer.json')
    model = BitLinearGPT(
        vocab_size=4000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Check 1: Entropy
    print(f"\n[CHECK 1] ENTROPY")
    print("-"*70)
    
    prompts = ['Once upon a time', 'The dog', 'A girl named']
    entropies = []
    
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs['logits'][0, -1, :]
            probs = torch.softmax(logits, dim=0)
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            max_entropy = torch.log(torch.tensor(4000.0)).item()
            entropy_pct = (entropy / max_entropy) * 100
            entropies.append(entropy_pct)
            
            top_prob, top_idx = torch.max(probs, dim=0)
            top_token = tokenizer.decode([top_idx.item()])
            
            print(f"  '{prompt}': Entropy {entropy:.4f} ({entropy_pct:.1f}%), Top='{top_token}' ({top_prob.item()*100:.1f}%)")
    
    avg_entropy = sum(entropies) / len(entropies)
    print(f"\n  AVERAGE ENTROPY: {avg_entropy:.1f}%")
    
    # Check 2: Top token probabilities
    print(f"\n[CHECK 2] TOP TOKEN PROBABILITY")
    print("-"*70)
    
    prompt = 'Once upon a time'
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits'][0, -1, :]
        probs = torch.softmax(logits, dim=0)
        
        top5 = torch.topk(probs, k=5)
        print(f"  Top 5 tokens:")
        for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
            token = tokenizer.decode([idx.item()])
            print(f"    {i+1}. '{token}' ({prob.item()*100:.2f}%)")
    
    top_prob = top5.values[0].item() * 100
    
    # Check 3: Sample Generation
    print(f"\n[CHECK 3] SAMPLE GENERATION (temp=0.9, top-k=50)")
    print("-"*70)
    
    prompt = 'Once upon a time'
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    generated = input_ids.clone()
    
    torch.manual_seed(42)
    
    with torch.no_grad():
        for _ in range(40):
            outputs = model(generated)
            logits = outputs['logits'][0, -1, :] / 0.9
            logits = top_k_filter(logits.clone(), 50)
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    sample = tokenizer.decode(generated[0].tolist())
    print(f"  {sample}")
    
    # Check for repetition
    words = sample.split()
    unique_words = len(set(words))
    total_words = len(words)
    
    print(f"\n  Words: {total_words}, Unique: {unique_words} ({unique_words/total_words*100:.1f}%)")
    
    # STATUS DETERMINATION
    print(f"\n[STATUS]")
    print("="*70)
    
    status = "HEALTHY"
    warnings = []
    
    # Entropy check
    if avg_entropy < 20:
        status = "COLLAPSED"
        warnings.append(f"CRITICAL: Entropy {avg_entropy:.1f}% < 20% - COLLAPSED")
    elif avg_entropy < 30:
        status = "WARNING"
        warnings.append(f"WARNING: Entropy {avg_entropy:.1f}% < 30%")
    
    # Top probability check
    if top_prob > 70:
        status = "COLLAPSED"
        warnings.append(f"CRITICAL: Top token {top_prob:.1f}% > 70% - COLLAPSED")
    elif top_prob > 50:
        if status != "COLLAPSED":
            status = "WARNING"
        warnings.append(f"WARNING: Top token {top_prob:.1f}% > 50%")
    
    # Repetition check
    if unique_words / total_words < 0.3:
        if status != "COLLAPSED":
            status = "WARNING"
        warnings.append(f"WARNING: Low word variety {unique_words/total_words*100:.1f}%")
    
    print(f"  STATUS: {status}")
    
    if warnings:
        print(f"\n  WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
    
    # Recommendation
    print(f"\n[RECOMMENDATION]")
    print("-"*70)
    
    if status == "HEALTHY":
        print("  ✓ Continue training - all metrics healthy")
    elif status == "WARNING":
        print("  ! Monitor closely - some metrics concerning")
        print("  ! Continue but check next checkpoint carefully")
    else:  # COLLAPSED
        print("  ✗ STOP TRAINING - model has collapsed")
        print("  ✗ Do NOT continue - investigate immediately")
    
    print("="*70)
    
    return status

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step to check")
    args = parser.parse_args()
    
    monitor_checkpoint(args.step)
