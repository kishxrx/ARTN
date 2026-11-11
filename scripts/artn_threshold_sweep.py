#!/usr/bin/env python3
"""
ARTN Inference Threshold Sweep

Sweeps halting thresholds {0.3, 0.5, 0.7, 0.9, 1.0} on 10 fixed prompts.
Measures: avg passes per token, tokens/sec, elapsed time, peak RSS.
Uses deterministic decoding (greedy / fixed seed) for fair quality comparison.

Does NOT modify any existing files. Writes results to a new JSON in the
ARTN output directory.
"""

import sys
import os
import time
import json
import gc
import torch
import psutil
from pathlib import Path

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

# === CONFIG ===
CKPT_PATH = str(BASE_DIR / 'outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt')
TOKENIZER_PATH = str(BASE_DIR / 'outputs/artn-50k-best-config/tokenizer.json')

THRESHOLDS = [0.3, 0.5, 0.7, 0.9, 1.0]

PROMPTS = [
    "Once upon a time",
    "A little girl named",
    "The boy went to the",
    "There was a big dog",
    "She looked at the sky and",
    "In the morning, the",
    "The cat sat on the",
    "He opened the door and saw",
    "Once there was a magical",
    "The children played in the",
]

MAX_NEW_TOKENS = 50
TEMPERATURE = 1.0  # Use 1.0 but with fixed seed for reproducibility
SEED = 42
DEVICE = 'cpu'


def get_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def load_model_and_tokenizer():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
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

    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    return model, tokenizer, config


def generate_with_passes_logged(model, tokenizer, prompt, max_new_tokens,
                                 threshold, seed=None):
    """Generate text while logging actual passes used per token."""
    if seed is not None:
        torch.manual_seed(seed)

    input_ids = torch.tensor([tokenizer.encode(prompt)], device=DEVICE)

    passes_per_token = []
    generated_tokens = []

    for _ in range(max_new_tokens):
        B, T = input_ids.shape

        tok_emb = model.token_embedding(input_ids)
        pos_emb = model.position_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb

        remaining = torch.ones(B, T, 1, device=DEVICE)
        output = torch.zeros(B, T, model.d_model, device=DEVICE)
        passes_this_token = 0

        for pass_idx in range(model.max_passes):
            passes_this_token += 1
            x = model.block(x)

            if threshold >= 1.0:
                # Full-depth bypass: always run all passes, no early stop
                h = torch.sigmoid(model.halting_gate(x))
                p = h * remaining
                if pass_idx == model.max_passes - 1:
                    p = remaining
                output = output + x * p
                remaining = remaining - p
                continue

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

        x = model.ln_f(output)
        logits = model.lm_head(x)
        logits = logits[:, -1, :]  # last position

        # Greedy decoding for determinism
        next_token = logits.argmax(dim=-1, keepdim=True)

        passes_per_token.append(passes_this_token)
        generated_tokens.append(next_token.item())

        input_ids = torch.cat([input_ids, next_token], dim=1)

    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text, passes_per_token


def run_sweep():
    print("=" * 80)
    print("ARTN INFERENCE THRESHOLD SWEEP")
    print("=" * 80)
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Decoding: Greedy (deterministic)")
    print(f"Seed: {SEED}")
    print()

    # Load model
    print("Loading model and tokenizer...", flush=True)
    model, tokenizer, config = load_model_and_tokenizer()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"  Max passes: {model.max_passes}", flush=True)
    print(f"  Target passes: {config.get('target_passes', 'N/A')}", flush=True)
    print()

    # Baseline RSS
    baseline_rss = get_rss_mb()
    print(f"  Baseline RSS (model + tokenizer loaded): {baseline_rss:.2f} MB")
    print()

    all_results = {}

    for threshold in THRESHOLDS:
        print(f"\n{'='*80}")
        print(f"THRESHOLD = {threshold}")
        print(f"{'='*80}")

        torch.manual_seed(SEED)  # reset seed before each threshold

        threshold_results = {
            'threshold': threshold,
            'prompts': [],
            'total_tokens': 0,
            'total_passes': 0,
            'total_time': 0.0,
            'peak_rss': baseline_rss,
        }

        gc.collect()
        start_rss = get_rss_mb()
        peak_rss = start_rss
        sweep_start = time.time()

        for i, prompt in enumerate(PROMPTS):
            torch.manual_seed(SEED)  # reset for each prompt (deterministic)

            t0 = time.time()
            generated_text, passes = generate_with_passes_logged(
                model, tokenizer, prompt, MAX_NEW_TOKENS, threshold, seed=SEED
            )
            elapsed = time.time() - t0

            current_rss = get_rss_mb()
            if current_rss > peak_rss:
                peak_rss = current_rss

            new_tokens = len(passes)
            total_passes = sum(passes)
            avg_passes = total_passes / new_tokens if new_tokens > 0 else 0
            tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0

            threshold_results['prompts'].append({
                'prompt': prompt,
                'generated': generated_text,
                'new_tokens': new_tokens,
                'total_passes': total_passes,
                'avg_passes': avg_passes,
                'elapsed_s': elapsed,
                'tokens_per_sec': tokens_per_sec,
                'passes_log': passes,
            })

            threshold_results['total_tokens'] += new_tokens
            threshold_results['total_passes'] += total_passes
            threshold_results['total_time'] += elapsed

            prompt_short = prompt[:30]
            print(f"  [{i+1}/{len(PROMPTS)}] {prompt_short:<30} "
                  f"tok={new_tokens} passes={avg_passes:.2f} "
                  f"t={elapsed:.2f}s speed={tokens_per_sec:.2f} tok/s", flush=True)

        sweep_elapsed = time.time() - sweep_start

        threshold_results['avg_passes_overall'] = (
            threshold_results['total_passes'] / threshold_results['total_tokens']
            if threshold_results['total_tokens'] > 0 else 0
        )
        threshold_results['avg_tokens_per_sec'] = (
            threshold_results['total_tokens'] / threshold_results['total_time']
            if threshold_results['total_time'] > 0 else 0
        )
        threshold_results['peak_rss_mb'] = peak_rss
        threshold_results['rss_delta_mb'] = peak_rss - start_rss
        threshold_results['total_elapsed_s'] = sweep_elapsed

        print(f"\n  SUMMARY (threshold={threshold}):")
        print(f"    Total tokens:     {threshold_results['total_tokens']}")
        print(f"    Total passes:     {threshold_results['total_passes']}")
        print(f"    Avg passes/token: {threshold_results['avg_passes_overall']:.4f}")
        print(f"    Avg tokens/sec:   {threshold_results['avg_tokens_per_sec']:.2f}")
        print(f"    Total time:       {threshold_results['total_time']:.2f}s")
        print(f"    Peak RSS:         {peak_rss:.2f} MB (delta: {peak_rss - start_rss:.2f} MB)")

        all_results[f"threshold_{threshold}"] = threshold_results

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print()
    print(f"{'Threshold':<12} {'AvgPasses':<12} {'Tok/s':<10} "
          f"{'Time(s)':<10} {'PeakRSS(MB)':<14} {'vs 1.0 speed'}")
    print("-" * 75)

    baseline_speed = all_results["threshold_1.0"]["avg_tokens_per_sec"]

    for thresh in THRESHOLDS:
        key = f"threshold_{thresh}"
        r = all_results[key]
        speedup = r["avg_tokens_per_sec"] / baseline_speed if baseline_speed > 0 else 0
        print(f"  {thresh:<12.1f} {r['avg_passes_overall']:<12.4f} "
              f"{r['avg_tokens_per_sec']:<10.2f} "
              f"{r['total_time']:<10.2f} "
              f"{r['peak_rss_mb']:<14.2f} "
              f"{speedup:.2f}x")

    # ================================================================
    # QUALITY COMPARISON (show first prompt at each threshold)
    # ================================================================
    print(f"\n{'='*80}")
    print("QUALITY COMPARISON (Prompt 1: 'Once upon a time')")
    print(f"{'='*80}")
    print()

    for thresh in THRESHOLDS:
        key = f"threshold_{thresh}"
        gen = all_results[key]["prompts"][0]["generated"]
        passes = all_results[key]["prompts"][0]["avg_passes"]
        print(f"  threshold={thresh}, avg_passes={passes:.2f}:")
        print(f"    {gen[:200]}")
        print()

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    output_file = BASE_DIR / 'outputs' / 'artn_recursive_20261025_002725' / 'threshold_sweep_results_corrected.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_sweep()
