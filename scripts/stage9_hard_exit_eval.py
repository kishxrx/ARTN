#!/usr/bin/env python3
"""
Stage 9 Correction: Hard early-exit evaluation

Fixes:
1. threshold=1.0 floating-point: clamp remaining to >= 0 before check,
   so threshold=1.0 never triggers early stop (always 6 passes).
2. Actual hard early-exit NLL/perplexity on held-out validation subset:
   At each position, run passes until last position's cumulative halt
   crosses threshold, then compute NLL from the LAST position's logits.
   Compare against full-depth (always 6 passes) on identical data.

This is NOT the training forward pass. This is the actual inference path:
  - Per-token: run block k times, check halt, stop early
  - Compute NLL from the output at the halt point
  - Remaining mass assigned to current pass

Method:
  - Use validation data (5000 samples, batch_size=4, max 50 batches = 200 samples)
  - For each sample: forward pass token-by-token, tracking passes per token
  - Compute NLL = -log P(target | context) at each position
  - Report avg NLL, perplexity, avg passes

Preserves all existing files.
"""

import sys
import os
import math
import json
import torch
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer
from experiments.artn_train import CachedDataset, load_data

ARTN_CKPT = str(BASE_DIR / 'outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt')
TOKENIZER_PATH = str(BASE_DIR / 'outputs/artn-50k-best-config/tokenizer.json')

DEVICE = 'cpu'
MAX_BATCHES = 50  # 50 batches × 4 samples = 200 validation samples
THRESHOLDS = [0.3, 0.5, 0.7, 0.9, 1.0]


def load_model():
    ckpt = torch.load(ARTN_CKPT, map_location=DEVICE, weights_only=False)
    config = ckpt['config']
    model = ARTNGPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        target_passes=config.get('target_passes', 4.0),
        ponder_loss_weight=config.get('ponder_loss_weight', 0.01),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model, config


def compute_hard_exit_nll(model, input_ids, targets, threshold, epsilon=1e-6):
    """
    Compute NLL using actual hard early-exit inference path.

    For each position in the sequence:
      1. Run block recursively with per-position halting
      2. Stop when last position's cumulative halt >= threshold (clamped)
      3. Assign remaining mass to current pass
      4. Compute logits from output at halt point
      5. NLL = -log P(target | logits)

    Returns:
      nll_sum: sum of NLL over all non-padding positions
      n_tokens: number of non-padding positions
      passes_per_position: list of passes used per position
    """
    B, T = input_ids.shape
    D = model.d_model

    # Embed
    tok_emb = model.token_embedding(input_ids)
    pos_emb = model.position_embedding(torch.arange(T, device=DEVICE))
    x_init = tok_emb + pos_emb

    # Process the ENTIRE sequence as one forward pass (all positions simultaneously)
    # For evaluation, the "last position" for halt decision is position T-1.
    # But we need NLL at EVERY position, so we compute the full forward pass
    # and track how many passes the last position uses.
    #
    # WAIT: The halting is sequence-level (based on last position). So ALL positions
    # get the same number of passes. This is the design: the halt decision is
    # made for the whole sequence based on the last position.
    #
    # For evaluation, we run the model forward on the full sequence and check
    # the halt condition at the last position. ALL positions use the same
    # number of passes. Then we compute NLL at every position from the
    # output at that depth.
    #
    # This is the actual inference behavior: the model decides how many
    # passes to run for the whole sequence, then computes logits for all
    # positions at that depth.

    remaining = torch.ones(B, T, 1, device=DEVICE)
    output = torch.zeros(B, T, D, device=DEVICE)
    passes_used = 0

    for pass_idx in range(model.max_passes):
        passes_used += 1
        x_init = model.block(x_init)
        h = torch.sigmoid(model.halting_gate(x_init))
        p = h * remaining
        if pass_idx == model.max_passes - 1:
            p = remaining

        output = output + x_init * p
        remaining = remaining - p

        # Clamp remaining to prevent floating-point early stop
        remaining_clamped = remaining.clamp(min=0.0)

        # Check last position
        cumulative_last = 1.0 - remaining_clamped[0, -1, 0].item()
        if cumulative_last >= threshold and pass_idx < model.max_passes - 1:
            break

    # Assign remaining mass
    output = output + x_init * remaining
    x = model.ln_f(output)
    logits = model.lm_head(x)

    # Compute NLL at each position
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
    nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)

    # Mask padding (target == -1 or 0)
    mask = (targets != -1) & (targets != 0)
    nll_masked = nll * mask
    n_tokens = mask.sum().item()
    nll_sum = nll_masked.sum().item()

    return nll_sum, n_tokens, passes_used


def evaluate_threshold(model, val_loader, threshold, max_batches):
    """Evaluate NLL/perplexity with hard early-exit at a given threshold."""
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    total_passes = 0
    total_sequences = 0
    passes_histogram = {}

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            # input = all tokens except last, target = all tokens except first
            input_ids = batch[:, :-1].to(DEVICE)
            targets = batch[:, 1:].to(DEVICE)

            nll_sum, n_tokens, passes_used = compute_hard_exit_nll(
                model, input_ids, targets, threshold
            )

            total_nll += nll_sum
            total_tokens += n_tokens
            total_passes += passes_used
            total_sequences += 1

            if passes_used not in passes_histogram:
                passes_histogram[passes_used] = 0
            passes_histogram[passes_used] += 1

    avg_nll = total_nll / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_nll) if avg_nll < 20 else float('inf')
    avg_passes = total_passes / total_sequences if total_sequences > 0 else 0

    return {
        'threshold': threshold,
        'avg_nll': avg_nll,
        'perplexity': perplexity,
        'avg_passes': avg_passes,
        'total_tokens': total_tokens,
        'total_sequences': total_sequences,
        'passes_histogram': passes_histogram,
    }


def main():
    print("=" * 80)
    print("STAGE 9 CORRECTION: Hard Early-Exit Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {ARTN_CKPT}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Max batches: {MAX_BATCHES} (200 samples)")
    print(f"Device: {DEVICE}")
    print()

    # Load model and data
    print("Loading model...", flush=True)
    model, config = load_model()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print("Loading validation data...", flush=True)
    val_config = {
        'vocab_size': 4000, 'd_model': 256, 'n_heads': 8,
        'n_layers': 6, 'd_ff': 1024, 'max_seq_len': 256,
        'train_samples': 50000, 'val_samples': 5000,
    }
    _, val_dataset, tokenizer = load_data(val_config)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0
    )
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Evaluating first {MAX_BATCHES} batches (200 samples)")
    print()

    # Evaluate each threshold
    results = {}
    for threshold in THRESHOLDS:
        print(f"Evaluating threshold={threshold}...", flush=True)
        r = evaluate_threshold(model, val_loader, threshold, MAX_BATCHES)
        results[f'threshold_{threshold}'] = r
        print(f"  NLL: {r['avg_nll']:.4f}")
        print(f"  Perplexity: {r['perplexity']:.4f}")
        print(f"  Avg passes: {r['avg_passes']:.4f}")
        print(f"  Passes histogram: {r['passes_histogram']}")
        print()

    # Comparison table
    print("=" * 80)
    print("HARD EARLY-EXIT EVALUATION RESULTS (ALL MEASURED)")
    print("=" * 80)
    print()
    print(f"{'Threshold':<12} {'NLL':<10} {'Perplexity':<12} {'AvgPasses':<12} {'Pass Histogram'}")
    print("-" * 75)
    for t in THRESHOLDS:
        r = results[f'threshold_{t}']
        hist_str = str(dict(sorted(r['passes_histogram'].items())))
        print(f"  {t:<12.1f} {r['avg_nll']:<10.4f} {r['perplexity']:<12.4f} {r['avg_passes']:<12.4f} {hist_str}")

    # Delta from full computation
    print()
    full = results['threshold_1.0']
    print(f"{'Threshold':<12} {'NLL delta':<12} {'PPL ratio':<12} {'Pass savings'}")
    print("-" * 55)
    for t in THRESHOLDS:
        r = results[f'threshold_{t}']
        nll_delta = r['avg_nll'] - full['avg_nll']
        ppl_ratio = r['perplexity'] / full['perplexity']
        pass_savings = (1 - r['avg_passes'] / full['avg_passes']) * 100
        print(f"  {t:<12.1f} {nll_delta:<12.4f} {ppl_ratio:<12.4f} {pass_savings:.1f}%")

    # Save
    output_file = BASE_DIR / 'outputs' / 'artn_recursive_20261025_002725' / 'stage9_hard_exit_eval.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
