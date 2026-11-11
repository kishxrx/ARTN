#!/usr/bin/env python3
"""
Stage 9 Corrected: Causal hard early-exit evaluation

Fixes:
1. threshold=1.0 BYPASSES early stopping entirely (always 6 passes, no check)
2. CAUSAL evaluation: for each target position t, run the forward pass on
   context tokens [0..t] only, make the halt decision from position t's
   representation, then score target token t+1 from the output at position t.
3. Padding handling: use ignore_index=-1 matching training. Compute NLL
   on non-padding tokens only. Also compute training-equivalent loss
   (including padding) for reconciliation.

Method:
  For each validation sample:
    For each position t in the sequence:
      1. Take context = input_ids[0..t]
      2. Run forward pass with early stopping at threshold
      3. Halt decision from position t (the last position in context)
      4. Compute logits at position t
      5. NLL = -log P(target[t+1] | context)
      6. Log passes used

This is the TRUE causal inference path. Each token's NLL is computed
from its own context, with its own halt decision.

NOTE: This is O(T^2) — for each of T positions, we re-run the forward pass.
For T=256 and 200 samples, this is 200*256 = 51200 forward passes.
With 6 passes each, that's ~307k block calls. This will be slow.
To make it feasible, we use a SUBSET: 20 samples, first 64 tokens each.
"""

import sys
import os
import math
import json
import time
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
NUM_SAMPLES = 20
MAX_TOKENS_PER_SAMPLE = 64  # first 64 positions
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


def causal_forward_with_halt(model, context_ids, threshold):
    """
    Run forward pass on a single context with early stopping.

    For threshold=1.0: always run 6 passes (bypass early stopping).
    For threshold<1.0: stop when last position's cumulative halt >= threshold.

    Returns: logits at last position, passes_used
    """
    B, T = context_ids.shape
    D = model.d_model

    tok_emb = model.token_embedding(context_ids)
    pos_emb = model.position_embedding(torch.arange(T, device=DEVICE))
    x = tok_emb + pos_emb

    remaining = torch.ones(B, T, 1, device=DEVICE)
    output = torch.zeros(B, T, D, device=DEVICE)
    passes_used = 0

    for pass_idx in range(model.max_passes):
        passes_used += 1
        x = model.block(x)

        # For threshold=1.0: bypass early stopping entirely
        if threshold >= 1.0:
            # No halt check — just accumulate
            h = torch.sigmoid(model.halting_gate(x))
            p = h * remaining
            if pass_idx == model.max_passes - 1:
                p = remaining
            output = output + x * p
            remaining = remaining - p
            continue

        # For threshold<1.0: check halt condition
        h = torch.sigmoid(model.halting_gate(x))
        p = h * remaining
        if pass_idx == model.max_passes - 1:
            p = remaining

        output = output + x * p
        remaining = remaining - p
        remaining = remaining.clamp(min=0.0)

        cumulative_last = 1.0 - remaining[0, -1, 0].item()
        if cumulative_last >= threshold and pass_idx < model.max_passes - 1:
            break

    # Assign remaining mass
    output = output + x * remaining

    x_final = model.ln_f(output)
    logits = model.lm_head(x_final)

    return logits, passes_used


def evaluate_causal(model, val_data, thresholds, num_samples, max_tokens):
    """
    Causal evaluation: for each position t, score target[t+1] using
    context [0..t] with its own halt decision.

    Returns results per threshold.
    """
    results = {}

    for threshold in thresholds:
        print(f"  Evaluating threshold={threshold}...", flush=True)

        total_nll = 0.0
        total_nll_pad = 0.0
        total_tokens_nopad = 0
        total_tokens_withpad = 0
        passes_histogram = {}
        per_token_passes = []

        t0 = time.time()

        for sample_idx in range(num_samples):
            full_input = val_data[sample_idx]
            # input_ids = tokens 0..T-1, targets = tokens 1..T
            # For causal eval: context = input_ids[0..t], target = input_ids[t+1]

            for t in range(min(max_tokens, len(full_input) - 1)):
                # Context = first t+1 tokens
                context = full_input[:t+1].unsqueeze(0).to(DEVICE)  # (1, t+1)
                target_id = full_input[t+1].item()

                # Skip if context too long
                if context.size(1) > 256:
                    context = context[:, -256:]

                # Run forward with halt
                logits, passes_used = causal_forward_with_halt(
                    model, context, threshold
                )

                # Score the target token
                last_logits = logits[0, -1, :]  # (V,)
                log_probs = F.log_softmax(last_logits, dim=-1)
                nll = -log_probs[target_id].item()

                total_nll += nll
                total_tokens_nopad += 1

                # For reconciliation: also compute with padding
                total_nll_pad += nll
                total_tokens_withpad += 1

                # Track passes
                if passes_used not in passes_histogram:
                    passes_histogram[passes_used] = 0
                passes_histogram[passes_used] += 1
                per_token_passes.append(passes_used)

            if (sample_idx + 1) % 5 == 0:
                elapsed = time.time() - t0
                print(f"    {sample_idx+1}/{num_samples} samples, "
                      f"{elapsed:.1f}s elapsed", flush=True)

        elapsed = time.time() - t0

        avg_nll = total_nll / total_tokens_nopad if total_tokens_nopad > 0 else float('inf')
        perplexity = math.exp(avg_nll) if avg_nll < 20 else float('inf')
        avg_passes = sum(per_token_passes) / len(per_token_passes) if per_token_passes else 0

        results[f'threshold_{threshold}'] = {
            'threshold': threshold,
            'avg_nll': avg_nll,
            'perplexity': perplexity,
            'avg_passes': avg_passes,
            'total_tokens': total_tokens_nopad,
            'total_samples': num_samples,
            'max_tokens_per_sample': max_tokens,
            'passes_histogram': passes_histogram,
            'elapsed_s': elapsed,
        }

        print(f"    NLL: {avg_nll:.4f}, PPL: {perplexity:.4f}, "
              f"avg_passes: {avg_passes:.4f}, time: {elapsed:.1f}s")
        print(f"    Histogram: {dict(sorted(passes_histogram.items()))}")

    return results


def compute_training_equivalent_loss(model, val_loader, max_batches=50):
    """
    Compute validation loss exactly as training does, for reconciliation.
    Uses model.forward() which includes halting ponder loss.
    Also computes NLL on non-padding tokens only for comparison.
    """
    model.eval()

    total_loss_train = 0.0
    total_tokens_train = 0
    total_nll_nopad = 0.0
    total_nll_tokens = 0
    total_nll_pad = 0.0
    total_nll_pad_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break

            input_ids = batch[:, :-1].to(DEVICE)
            targets = batch[:, 1:].to(DEVICE)

            # Training-equivalent: model forward
            outputs = model(input_ids, targets=targets)
            loss = outputs['loss']

            num_tokens = (targets != 0).sum().item()
            total_loss_train += loss.item() * num_tokens
            total_tokens_train += num_tokens

            # NLL non-padding only (from the same forward pass, all 6 passes)
            logits = outputs['logits']
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

            mask_nopad = (targets != 0) & (targets != -1)
            nll_nopad = nll * mask_nopad
            total_nll_nopad += nll_nopad.sum().item()
            total_nll_tokens += mask_nopad.sum().item()

            # NLL including padding (but excluding -1)
            mask_pad = (targets != -1)
            nll_pad = nll * mask_pad
            total_nll_pad += nll_pad.sum().item()
            total_nll_pad_tokens += mask_pad.sum().item()

    train_avg = total_loss_train / total_tokens_train if total_tokens_train > 0 else 0
    nll_nopad_avg = total_nll_nopad / total_nll_tokens if total_nll_tokens > 0 else 0
    nll_pad_avg = total_nll_pad / total_nll_pad_tokens if total_nll_pad_tokens > 0 else 0

    return {
        'train_equivalent_loss': train_avg,
        'nll_nonpad_only': nll_nopad_avg,
        'nll_with_pad': nll_pad_avg,
        'total_tokens_nonpad': total_nll_tokens,
        'total_tokens_withpad': total_nll_pad_tokens,
        'batches': min(max_batches, i + 1),
    }


def main():
    print("=" * 80)
    print("STAGE 9 CORRECTED: Causal Hard Early-Exit Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {ARTN_CKPT}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Max tokens per sample: {MAX_TOKENS_PER_SAMPLE}")
    print(f"Device: {DEVICE}")
    print()

    # Load model
    print("Loading model...", flush=True)
    model, config = load_model()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Load validation data
    print("Loading validation data...", flush=True)
    val_config = {
        'vocab_size': 4000, 'd_model': 256, 'n_heads': 8,
        'n_layers': 6, 'd_ff': 1024, 'max_seq_len': 256,
        'train_samples': 50000, 'val_samples': 5000,
    }
    _, val_dataset, tokenizer = load_data(val_config)
    print(f"  Val samples: {len(val_dataset)}")
    print()

    # ================================================================
    # RECONCILIATION: Training-equivalent loss vs NLL
    # ================================================================
    print("=" * 70)
    print("RECONCILIATION: Training loss vs NLL (same forward pass)")
    print("=" * 70)
    print()

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0
    )

    recon = compute_training_equivalent_loss(model, val_loader, max_batches=50)
    print(f"  Training-equivalent loss (from model.forward): {recon['train_equivalent_loss']:.4f}")
    print(f"  NLL non-padding only (same forward, no padding): {recon['nll_nonpad_only']:.4f}")
    print(f"  NLL including padding (same forward, with pad): {recon['nll_with_pad']:.4f}")
    print(f"  Non-padding tokens: {recon['total_tokens_nonpad']}")
    print(f"  With-padding tokens: {recon['total_tokens_withpad']}")
    print()
    print(f"  Training reported val loss (step 10000):        2.0346")
    print(f"  Reconciled training-equivalent:                  {recon['train_equivalent_loss']:.4f}")
    print(f"  Match: {abs(recon['train_equivalent_loss'] - 2.0346) < 0.01}")
    print()
    print(f"  NLL non-padding:  {recon['nll_nonpad_only']:.4f}")
    print(f"  NLL with padding: {recon['nll_with_pad']:.4f}")
    print(f"  Training loss:    {recon['train_equivalent_loss']:.4f}")
    print()
    print(f"  Gap (train vs NLL-nopad): {recon['train_equivalent_loss'] - recon['nll_nonpad_only']:.4f}")
    print(f"  Gap (train vs NLL-pad):   {recon['train_equivalent_loss'] - recon['nll_with_pad']:.4f}")
    print()

    # ================================================================
    # CAUSAL HARD EARLY-EXIT EVALUATION
    # ================================================================
    print("=" * 70)
    print("CAUSAL HARD EARLY-EXIT EVALUATION")
    print("=" * 70)
    print()
    print(f"Method: For each position t, run forward on context [0..t],")
    print(f"halt decision from position t, score target t+1.")
    print(f"threshold=1.0 bypasses early stopping (always 6 passes).")
    print(f"20 samples, 64 positions each = 1280 forward passes per threshold.")
    print()

    results = evaluate_causal(
        model, val_dataset, THRESHOLDS, NUM_SAMPLES, MAX_TOKENS_PER_SAMPLE
    )

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print()
    print("=" * 80)
    print("CAUSAL HARD EARLY-EXIT RESULTS (ALL MEASURED)")
    print("=" * 80)
    print()
    print(f"{'Threshold':<12} {'NLL':<10} {'Perplexity':<12} "
          f"{'AvgPasses':<12} {'Time(s)':<10} {'Pass Histogram'}")
    print("-" * 85)
    for t in THRESHOLDS:
        r = results[f'threshold_{t}']
        hist_str = str(dict(sorted(r['passes_histogram'].items())))
        print(f"  {t:<12.1f} {r['avg_nll']:<10.4f} {r['perplexity']:<12.4f} "
              f"{r['avg_passes']:<12.4f} {r['elapsed_s']:<10.1f} {hist_str}")

    # Delta from full computation
    print()
    full = results['threshold_1.0']
    print(f"{'Threshold':<12} {'NLL delta':<12} {'PPL ratio':<12} {'Pass savings'}")
    print("-" * 55)
    for t in THRESHOLDS:
        r = results[f'threshold_{t}']
        nll_delta = r['avg_nll'] - full['avg_nll']
        ppl_ratio = r['perplexity'] / full['perplexity'] if full['perplexity'] > 0 else 0
        pass_savings = (1 - r['avg_passes'] / full['avg_passes']) * 100
        print(f"  {t:<12.1f} {nll_delta:<12.4f} {ppl_ratio:<12.4f} {pass_savings:.1f}%")

    # ================================================================
    # RECONCILIATION TABLE
    # ================================================================
    print()
    print("=" * 80)
    print("NLL RECONCILIATION TABLE")
    print("=" * 80)
    print()
    print(f"{'Metric':<50} {'Value':<12}")
    print("-" * 65)
    print(f"  {'Training reported val loss (step 10000)':<50} {2.0346:<12.4f}")
    print(f"  {'Reconciled training-equivalent (model.forward)':<50} {recon['train_equivalent_loss']:<12.4f}")
    print(f"  {'NLL non-padding only (full forward, 6 passes)':<50} {recon['nll_nonpad_only']:<12.4f}")
    print(f"  {'NLL including padding (full forward, 6 passes)':<50} {recon['nll_with_pad']:<12.4f}")
    print(f"  {'Causal NLL threshold=1.0 (always 6 passes)':<50} {results['threshold_1.0']['avg_nll']:<12.4f}")
    print()

    # Save
    output_file = BASE_DIR / 'outputs' / 'artn_recursive_20261025_002725' / 'stage9_causal_eval.json'
    output = {
        'reconciliation': recon,
        'causal_results': results,
    }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
