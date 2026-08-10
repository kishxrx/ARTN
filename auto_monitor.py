#!/usr/bin/env python3
"""
AUTOMATIC CHECKPOINT MONITOR
Runs in background and monitors checkpoints as they are saved.

Usage:
    python auto_monitor.py
    
This will:
- Watch the checkpoints folder
- When new checkpoint appears, run full diagnostics
- Show results immediately
- Save log to file
"""

import torch
import sys
import time
import os
sys.path.insert(0, '.')

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

# Thresholds
ENTROPY_WARN = 30.0  # % of max entropy
ENTROPY_CRITICAL = 20.0
TOP_PROB_WARN = 50.0  # %
TOP_PROB_CRITICAL = 70.0

def top_k_filter(logits, k):
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits

def analyze_checkpoint(ckpt_path, step):
    """Analyze a checkpoint and return status."""
    
    print(f"\n{'='*70}")
    print(f"AUTOMATIC MONITOR - Step {step}")
    print(f"{'='*70}")
    
    # Load checkpoint
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return "ERROR"
    
    loss = ckpt.get('loss', 0)
    print(f"[Step {step}] Loss: {loss:.4f}")
    
    # Load model
    try:
        tokenizer = BPETokenizer.load('outputs/artn-final/tokenizer.json')
    except:
        print("ERROR: Tokenizer not found")
        return "ERROR"
    
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
    
    # ========== CHECK 1: ENTROPY ==========
    print(f"\n[CHECK 1] ENTROPY")
    print("-"*70)
    
    prompts = ['Once upon a time', 'The dog', 'A girl named']
    entropies = []
    top_tokens = []
    
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
            top_tokens.append((top_token, top_prob.item() * 100))
            
            print(f"  '{prompt}': {entropy_pct:.1f}%, Top='{top_token}' ({top_prob.item()*100:.1f}%)")
    
    avg_entropy = sum(entropies) / len(entropies)
    print(f"\n  AVERAGE ENTROPY: {avg_entropy:.1f}%")
    
    # ========== CHECK 2: TOP TOKEN PROBABILITY ==========
    print(f"\n[CHECK 2] TOP TOKEN PROBABILITY")
    print("-"*70)
    
    prompt = 'Once upon a time'
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits'][0, -1, :]
        probs = torch.softmax(logits, dim=0)
        
        top5 = torch.topk(probs, k=5)
        for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
            token = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{token}' ({prob.item()*100:.2f}%)")
    
    top_prob = top5.values[0].item() * 100
    
    # ========== CHECK 3: SAMPLE GENERATION ==========
    print(f"\n[CHECK 3] SAMPLE (temp=0.9, top-k=50)")
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
    
    words = sample.split()
    unique_words = len(set(words))
    total_words = len(words)
    variety_pct = unique_words / total_words * 100 if total_words > 0 else 0
    
    print(f"\n  Words: {total_words}, Unique: {unique_words} ({variety_pct:.1f}%)")
    
    # ========== STATUS DETERMINATION ==========
    print(f"\n[STATUS]")
    print("="*70)
    
    status = "HEALTHY"
    warnings = []
    
    # Entropy check
    if avg_entropy < ENTROPY_CRITICAL:
        status = "COLLAPSED"
        warnings.append(f"CRITICAL: Entropy {avg_entropy:.1f}% < {ENTROPY_CRITICAL}%")
    elif avg_entropy < ENTROPY_WARN:
        status = "WARNING"
        warnings.append(f"WARNING: Entropy {avg_entropy:.1f}% < {ENTROPY_WARN}%")
    
    # Top probability check
    if top_prob > TOP_PROB_CRITICAL:
        status = "COLLAPSED"
        warnings.append(f"CRITICAL: Top token {top_prob:.1f}% > {TOP_PROB_CRITICAL}%")
    elif top_prob > TOP_PROB_WARN:
        if status != "COLLAPSED":
            status = "WARNING"
        warnings.append(f"WARNING: Top token {top_prob:.1f}% > {TOP_PROB_WARN}%")
    
    # Variety check
    if variety_pct < 30:
        if status != "COLLAPSED":
            status = "WARNING"
        warnings.append(f"WARNING: Low variety {variety_pct:.1f}%")
    
    print(f"  STATUS: {status}")
    
    if warnings:
        print(f"\n  WARNINGS:")
        for w in warnings:
            print(f"    - {w}")
    
    # Recommendation
    print(f"\n[RECOMMENDATION]")
    print("-"*70)
    
    if status == "HEALTHY":
        print("  ✓ CONTINUE training")
    elif status == "WARNING":
        print("  ! MONITOR closely")
        print("  ! Check next checkpoint")
    else:  # COLLAPSED
        print("  ✗ STOP TRAINING IMMEDIATELY")
        print("  ✗ Model has COLLAPSED")
    
    print("="*70)
    
    # Save to log
    with open('outputs/artn-final/monitor_log.txt', 'a') as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp} | Step {step} | Loss {loss:.4f} | Entropy {avg_entropy:.1f}% | TopProb {top_prob:.1f}% | Variety {variety_pct:.1f}% | STATUS: {status}\n")
    
    return status

def main():
    print("="*70)
    print("AUTOMATIC CHECKPOINT MONITOR")
    print("="*70)
    print("\nThresholds:")
    print(f"  Entropy: HEALTHY >{ENTROPY_WARN}%, WARNING {ENTROPY_CRITICAL}-{ENTROPY_WARN}%, COLLAPSED <{ENTROPY_CRITICAL}%")
    print(f"  Top Prob: HEALTHY <{TOP_PROB_WARN}%, WARNING {TOP_PROB_WARN}-{TOP_PROB_CRITICAL}%, COLLAPSED >{TOP_PROB_CRITICAL}%")
    print(f"\nMonitoring: outputs/artn-final/checkpoints/")
    print(f"Log file: outputs/artn-final/monitor_log.txt")
    print("="*70)
    print("\nWaiting for checkpoints... (Press Ctrl+C to stop)")
    
    checkpoint_dir = 'outputs/artn-final/checkpoints'
    seen_checkpoints = set()
    
    # Create log file
    os.makedirs('outputs/artn-final', exist_ok=True)
    
    with open('outputs/artn-final/monitor_log.txt', 'w') as f:
        f.write(f"ARTN Training Monitor Log - Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
    
    while True:
        try:
            # Check for new checkpoints
            if os.path.exists(checkpoint_dir):
                for filename in os.listdir(checkpoint_dir):
                    if filename.startswith('checkpoint_step_') and filename.endswith('.pt'):
                        ckpt_path = os.path.join(checkpoint_dir, filename)
                        
                        if ckpt_path not in seen_checkpoints:
                            seen_checkpoints.add(ckpt_path)
                            
                            # Extract step number
                            step = int(filename.replace('checkpoint_step_', '').replace('.pt', ''))
                            
                            # Analyze
                            status = analyze_checkpoint(ckpt_path, step)
                            
                            # If collapsed, print big warning
                            if status == "COLLAPSED":
                                print("\n" + "!"*70)
                                print("!"*70)
                                print("!!! MODEL COLLAPSED - STOP TRAINING IMMEDIATELY !!!")
                                print("!"*70)
                                print("!"*70 + "\n")
            
            # Wait before checking again
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n\nMonitor stopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
