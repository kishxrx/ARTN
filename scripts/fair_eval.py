#!/usr/bin/env python3
"""
Fair Evaluation: FP32, BitLinear GPT, ARTN on same validation data.

1. Excludes PAD (ID 0) and ignore_index (-1) from all CE calculations.
2. Separates CE from ARTN ponder loss.
3. Evaluates ARTN at thresholds 0.9 and 1.0 via causal per-position inference.
4. Full-depth batch evaluation for all three models (non-padding NLL only).

All results MEASURED. No existing files modified.
"""

import sys, os, math, json, time, gc
import torch
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from experiments.bitlinear_adamw_controlled import BitLinearGPT
from experiments.fp32_baseline_controlled import FP32GPT
from experiments.artn_model import ARTNGPT
from experiments.artn_train import CachedDataset, load_data

FP32_CKPT = str(BASE_DIR / 'outputs/fp32_baseline_20260908_142259/checkpoints/adamw/step_9500.pt')
BITLINEAR_CKPT = str(BASE_DIR / 'outputs/bitlinear_adamw_20260910_143646/checkpoints/bitlinear_adamw/step_9500.pt')
ARTN_CKPT = str(BASE_DIR / 'outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt')

DEVICE = 'cpu'
BATCH_SIZE = 4
MAX_BATCHES = 50  # 200 samples

# For causal eval
CAUSAL_SAMPLES = 20
CAUSAL_MAX_TOKENS = 64
THRESHOLDS = [0.9, 1.0]


def get_rss_mb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def load_model(model_type):
    if model_type == 'fp32':
        ckpt = torch.load(FP32_CKPT, map_location=DEVICE, weights_only=False)
        cfg = ckpt['config']
        model = FP32GPT(
            vocab_size=cfg['vocab_size'], d_model=cfg['d_model'],
            n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
            d_ff=cfg['d_ff'], max_seq_len=cfg['max_seq_len'],
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model_type_str = 'FP32 (6 blocks, FP32)'
    elif model_type == 'bitlinear':
        ckpt = torch.load(BITLINEAR_CKPT, map_location=DEVICE, weights_only=False)
        cfg = ckpt['config']
        model = BitLinearGPT(
            vocab_size=cfg['vocab_size'], d_model=cfg['d_model'],
            n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
            d_ff=cfg['d_ff'], max_seq_len=cfg['max_seq_len'],
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model_type_str = 'BitLinear GPT (6 blocks, ternary)'
    elif model_type == 'artn':
        ckpt = torch.load(ARTN_CKPT, map_location=DEVICE, weights_only=False)
        cfg = ckpt['config']
        model = ARTNGPT(
            vocab_size=cfg['vocab_size'], d_model=cfg['d_model'],
            n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
            d_ff=cfg['d_ff'], max_seq_len=cfg['max_seq_len'],
            target_passes=cfg.get('target_passes', 4.0),
            ponder_loss_weight=cfg.get('ponder_loss_weight', 0.01),
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model_type_str = 'ARTN (1 shared block, ternary + halting)'

    model.to(DEVICE)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    return model, cfg, params, model_type_str


def batch_eval_standard(model, val_loader, max_batches):
    """Evaluate FP32/BitLinear: full forward, non-padding NLL only."""
    total_nll = 0.0
    n_valid = 0
    n_pad = 0
    n_ignore = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch[:, :-1].to(DEVICE)
            targets = batch[:, 1:].to(DEVICE)

            outputs = model(input_ids, targets=targets)
            logits = outputs['logits']

            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

            # Non-padding, non-ignored
            mask_valid = (targets != 0) & (targets != -1)
            mask_pad = (targets == 0) & (targets != -1)
            mask_ignore = (targets == -1)

            total_nll += (nll * mask_valid).sum().item()
            n_valid += mask_valid.sum().item()
            n_pad += mask_pad.sum().item()
            n_ignore += mask_ignore.sum().item()

    avg_nll = total_nll / n_valid if n_valid > 0 else float('inf')
    ppl = math.exp(avg_nll) if avg_nll < 20 else float('inf')

    return {
        'nll': avg_nll,
        'perplexity': ppl,
        'n_valid_tokens': n_valid,
        'n_pad_tokens': n_pad,
        'n_ignore_tokens': n_ignore,
        'total_tokens': n_valid + n_pad + n_ignore,
    }


def batch_eval_artn(model, val_loader, max_batches):
    """Evaluate ARTN: full forward (6 passes), separate CE and ponder."""
    total_ce = 0.0
    total_ponder = 0.0
    n_valid = 0
    n_pad = 0
    n_ignore = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch[:, :-1].to(DEVICE)
            targets = batch[:, 1:].to(DEVICE)

            outputs = model(input_ids, targets=targets)
            logits = outputs['logits']
            info = outputs['halting_info']

            # Separate CE and ponder
            ce_loss = info['loss_ce']
            ponder_loss = info['loss_ponder']

            # NLL per token (non-padding)
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

            mask_valid = (targets != 0) & (targets != -1)
            mask_pad = (targets == 0) & (targets != -1)

            total_ce += (nll * mask_valid).sum().item()
            total_ponder += ponder_loss
            n_valid += mask_valid.sum().item()
            n_pad += mask_pad.sum().item()
            n_ignore += (targets == -1).sum().item()

    avg_nll = total_ce / n_valid if n_valid > 0 else float('inf')
    avg_ponder = total_ponder / min(max_batches, i + 1)
    ppl = math.exp(avg_nll) if avg_nll < 20 else float('inf')

    return {
        'nll': avg_nll,
        'perplexity': ppl,
        'ponder_loss': avg_ponder,
        'n_valid_tokens': n_valid,
        'n_pad_tokens': n_pad,
        'n_ignore_tokens': n_ignore,
        'total_tokens': n_valid + n_pad + n_ignore,
    }


def causal_forward_with_halt(model, context_ids, threshold):
    """Run forward with hard early-exit. threshold=1.0 bypasses halt (6 passes)."""
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

        if threshold >= 1.0:
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

    output = output + x * remaining
    x_final = model.ln_f(output)
    logits = model.lm_head(x_final)

    return logits, passes_used


def causal_eval_artn(model, val_data, num_samples, max_tokens, thresholds):
    """Causal per-position evaluation: each token scored from its own context."""
    results = {}

    for threshold in thresholds:
        print(f"  Causal threshold={threshold}...", flush=True)
        total_nll = 0.0
        n_valid = 0
        passes_hist = {}
        t0 = time.time()

        with torch.no_grad():
            for s_idx in range(min(num_samples, len(val_data))):
                full = val_data[s_idx]

                for t in range(min(max_tokens, len(full) - 1)):
                    context = full[:t+1].unsqueeze(0).to(DEVICE)
                    if context.size(1) > 256:
                        context = context[:, -256:]
                    target_id = full[t+1].item()

                    # Skip padding targets
                    if target_id == 0 or target_id == -1:
                        continue

                    logits, passes = causal_forward_with_halt(model, context, threshold)
                    log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                    nll = -log_probs[target_id].item()

                    total_nll += nll
                    n_valid += 1

                    if passes not in passes_hist:
                        passes_hist[passes] = 0
                    passes_hist[passes] += 1

                if (s_idx + 1) % 5 == 0:
                    print(f"    {s_idx+1}/{num_samples}, {time.time()-t0:.1f}s", flush=True)

        elapsed = time.time() - t0
        avg_nll = total_nll / n_valid if n_valid > 0 else float('inf')
        ppl = math.exp(avg_nll) if avg_nll < 20 else float('inf')
        avg_passes = sum(k*v for k,v in passes_hist.items()) / sum(passes_hist.values()) if passes_hist else 0

        results[f'threshold_{threshold}'] = {
            'threshold': threshold,
            'nll': avg_nll,
            'perplexity': ppl,
            'n_valid_tokens': n_valid,
            'avg_passes': avg_passes,
            'passes_histogram': passes_hist,
            'elapsed_s': elapsed,
        }
        print(f"    NLL={avg_nll:.4f}, PPL={ppl:.4f}, passes={avg_passes:.2f}, "
              f"tokens={n_valid}, {elapsed:.1f}s")

    return results


def main():
    print("=" * 80)
    print("FAIR EVALUATION: Non-Padding NLL (All Models)")
    print("=" * 80)
    print(f"Batch eval: {MAX_BATCHES} batches × {BATCH_SIZE} = {MAX_BATCHES*BATCH_SIZE} samples")
    print(f"Causal eval: {CAUSAL_SAMPLES} samples × {CAUSAL_MAX_TOKENS} positions")
    print(f"Excludes: PAD (ID 0) and ignore (-1) from all NLL")
    print(f"ARTN CE and ponder loss separated")
    print(f"Device: {DEVICE}")
    print()

    # Load data
    print("Loading validation data...", flush=True)
    cfg_data = {'vocab_size': 4000, 'd_model': 256, 'n_heads': 8, 'n_layers': 6,
                'd_ff': 1024, 'max_seq_len': 256, 'train_samples': 50000, 'val_samples': 5000}
    _, val_dataset, tokenizer = load_data(cfg_data)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Val samples: {len(val_dataset)}")
    print()

    batch_results = {}
    causal_results = {}

    # ================================================================
    # BATCH EVALUATION (full forward, non-padding NLL)
    # ================================================================
    print("=" * 70)
    print("BATCH EVALUATION (full forward, non-padding NLL only)")
    print("=" * 70)
    print()

    for mtype in ['fp32', 'bitlinear', 'artn']:
        print(f"Loading {mtype}...", flush=True)
        model, cfg, params, label = load_model(mtype)
        print(f"  {label}, params={params:,}")

        if mtype == 'artn':
            r = batch_eval_artn(model, val_loader, MAX_BATCHES)
        else:
            r = batch_eval_standard(model, val_loader, MAX_BATCHES)

        batch_results[mtype] = {**r, 'label': label, 'params': params}
        print(f"  NLL: {r['nll']:.4f}, PPL: {r['perplexity']:.4f}")
        print(f"  Valid tokens: {r['n_valid_tokens']}, "
              f"Pad: {r['n_pad_tokens']}, Ignore: {r['n_ignore_tokens']}")
        if 'ponder_loss' in r:
            print(f"  Ponder loss: {r['ponder_loss']:.4f}")
        print()

        del model
        gc.collect()

    # ================================================================
    # CAUSAL EVALUATION (ARTN, per-position, hard early-exit)
    # ================================================================
    print("=" * 70)
    print("CAUSAL EVALUATION (ARTN, per-position, hard early-exit)")
    print("=" * 70)
    print()

    model, cfg, params, label = load_model('artn')
    causal_results = causal_eval_artn(
        model, val_dataset, CAUSAL_SAMPLES, CAUSAL_MAX_TOKENS, THRESHOLDS)
    del model
    gc.collect()

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print()
    print("=" * 80)
    print("FAIR COMPARISON TABLE (ALL MEASURED, NON-PADDING NLL)")
    print("=" * 80)
    print()
    print("BATCH EVALUATION (full forward, 6 passes/blocks):")
    print(f"{'Model':<40} {'Params':<12} {'NLL':<10} {'PPL':<10} "
          f"{'ValidTok':<10} {'PadTok':<10}")
    print("-" * 95)
    for mtype in ['fp32', 'bitlinear', 'artn']:
        r = batch_results[mtype]
        print(f"  {r['label']:<38} {r['params']:<12,} "
              f"{r['nll']:<10.4f} {r['perplexity']:<10.4f} "
              f"{r['n_valid_tokens']:<10} {r['n_pad_tokens']:<10}")
    if 'ponder_loss' in batch_results['artn']:
        print(f"\n  ARTN ponder loss (separated): {batch_results['artn']['ponder_loss']:.4f}")
        print(f"  ARTN CE only (no ponder):     {batch_results['artn']['nll']:.4f}")

    print()
    print("CAUSAL EVALUATION (ARTN, per-position, hard early-exit):")
    print(f"{'Threshold':<12} {'NLL':<10} {'PPL':<10} {'AvgPasses':<12} "
          f"{'ValidTok':<10} {'Pass Histogram'}")
    print("-" * 85)
    for t in THRESHOLDS:
        r = causal_results[f'threshold_{t}']
        hist = str(dict(sorted(r['passes_histogram'].items())))
        print(f"  {t:<12.1f} {r['nll']:<10.4f} {r['perplexity']:<10.4f} "
              f"{r['avg_passes']:<12.4f} {r['n_valid_tokens']:<10} {hist}")

    # Delta from full
    print()
    full_causal = causal_results['threshold_1.0']
    print(f"{'Threshold':<12} {'NLL delta':<12} {'PPL ratio':<12} {'Pass savings'}")
    print("-" * 50)
    for t in THRESHOLDS:
        r = causal_results[f'threshold_{t}']
        delta = r['nll'] - full_causal['nll']
        ratio = r['perplexity'] / full_causal['perplexity'] if full_causal['perplexity'] > 0 else 0
        savings = (1 - r['avg_passes'] / full_causal['avg_passes']) * 100
        print(f"  {t:<12.1f} {delta:<12.4f} {ratio:<12.4f} {savings:.1f}%")

    # ================================================================
    # SAVE
    # ================================================================
    output = {
        'batch_evaluation': batch_results,
        'causal_evaluation': causal_results,
        'config': {
            'batch_samples': MAX_BATCHES * BATCH_SIZE,
            'causal_samples': CAUSAL_SAMPLES,
            'causal_max_tokens': CAUSAL_MAX_TOKENS,
            'excludes': 'PAD (ID 0) and ignore (-1)',
        },
    }
    out_file = BASE_DIR / 'outputs' / 'artn_recursive_20261025_002725' / 'fair_eval.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
