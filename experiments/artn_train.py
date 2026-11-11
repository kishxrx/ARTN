#!/usr/bin/env python3
"""
ARTN Training Script: Adaptive Recursive Ternary Network

Trains the full ARTN model:
  - 1 shared BitLinearBlock (recursive parameter tying, Stage 4 + 7)
  - Per-position adaptive halting gate (Stage 5)
  - Ternary BitLinear weights with STE (Stage 7)
  - AdamW optimizer (proven config from BitLinear GPT)

Reuses the proven TinyStories data pipeline, tokenizer, and cached data.
Saves to a new ARTN-specific directory. Never overwrites existing files.

Usage:
  cd /root/agent-sandbox/ARTN
  /root/agent-sandbox/.venv/bin/python experiments/artn_train.py
"""

import sys
import time
import math
import torch
import torch.nn as nn
import json
import gc
import os
from pathlib import Path
from datetime import datetime
from typing import Dict

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

# =============================================================================
# CONFIG — SAME PROVEN ADAMW CONFIG + ARTN-SPECIFIC ADDITIONS
# =============================================================================

CONFIG = {
    # Model architecture — SAME as proven BitLinear GPT
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,          # maps to max_passes (recursive depth)
    "d_ff": 1024,
    "max_seq_len": 256,
    "vocab_size": 4000,

    # ARTN-specific halting parameters
    "target_passes": 4.0,
    "ponder_loss_weight": 0.01,

    # Training parameters — SAME as proven BitLinear GPT
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "max_lr": 0.001,
    "min_lr": 0.00003,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "beta1": 0.5,
    "beta2": 0.98,

    # Training schedule — SAME as proven BitLinear GPT
    "total_steps": 10000,
    "warmup_steps": 1000,

    # Data — SAME as proven BitLinear GPT
    "train_samples": 50000,
    "val_samples": 5000,

    # Checkpointing
    "checkpoint_every": 500,
    "eval_every": 100,
    "keep_last_n_checkpoints": 0,    # 0 = keep ALL checkpoints (no deletion)

    # Seed — SAME as proven BitLinear GPT
    "seed": 42,

    # Generation
    "generation_prompt": "Once upon a time",
    "generation_max_tokens": 100,
    "generation_temperature": 0.9,
    "generation_threshold": 1.0,     # full computation during training samples
}

# =============================================================================
# DATA LOADING (SAME AS PROVEN BITLINEAR GPT)
# =============================================================================

class CachedDataset:
    """Dataset that loads pre-tokenized data from cache."""
    def __init__(self, tokenized_data, max_length=256):
        self.data = tokenized_data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens)


def load_data(config):
    """Load cached tokenized data — SAME pipeline as proven BitLinear GPT."""
    tokenizer_path = BASE_DIR / 'outputs' / 'artn-50k-best-config' / 'tokenizer.json'
    if not tokenizer_path.exists():
        raise RuntimeError(f"Tokenizer not found at {tokenizer_path}")

    print(f"Loading tokenizer from {tokenizer_path}", flush=True)
    tokenizer = BPETokenizer.load(str(tokenizer_path))

    cache_dir = BASE_DIR / 'outputs' / 'tokenized_cache'
    train_cache = cache_dir / 'train_tokenized.pt'
    val_cache = cache_dir / 'val_tokenized.pt'

    if not (train_cache.exists() and val_cache.exists()):
        raise RuntimeError("Tokenized cache not found. Run save_tokenized_cache.py first!")

    print(f"\nLoading CACHED tokenized data...", flush=True)
    train_tokenized = torch.load(train_cache)
    val_tokenized = torch.load(val_cache)
    print(f"  Train: {len(train_tokenized)} samples", flush=True)
    print(f"  Val: {len(val_tokenized)} samples", flush=True)

    train_dataset = CachedDataset(train_tokenized, config["max_seq_len"])
    val_dataset = CachedDataset(val_tokenized, config["max_seq_len"])

    return train_dataset, val_dataset, tokenizer


# =============================================================================
# LEARNING RATE SCHEDULER (SAME AS PROVEN BITLINEAR GPT)
# =============================================================================

class CosineWarmupScheduler:
    def __init__(self, warmup_steps, total_steps, max_lr, min_lr=1e-5):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# EVALUATION (ADAPTED FOR ARTN — LOGS HALTING STATS)
# =============================================================================

def compute_perplexity(model, dataloader, device, max_batches=50):
    """Compute validation perplexity and halting statistics."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_expected_passes = 0.0
    total_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            outputs = model(input_ids, targets=targets)
            loss = outputs["loss"]
            info = outputs["halting_info"]

            num_tokens = (targets != 0).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            total_expected_passes += info["expected_passes"]
            total_batches += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    avg_expected_passes = total_expected_passes / total_batches if total_batches > 0 else 0

    model.train()
    return perplexity, avg_loss, avg_expected_passes


def generate_sample(model, tokenizer, prompt, max_tokens=100, temperature=0.9,
                    threshold=1.0, device='cpu'):
    """Generate text sample with specified halting threshold."""
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        output_ids = model.generate(input_ids, max_tokens, temperature, threshold)
        generated_text = tokenizer.decode(output_ids[0].tolist())
    model.train()
    return generated_text


# =============================================================================
# CHECKPOINT MANAGEMENT (NEVER OVERWRITE EXISTING)
# =============================================================================

def save_checkpoint(model, optimizer, step, config, metrics, halting_stats,
                   ternary_stats, output_dir, keep_last_n=10):
    """Save checkpoint. Never overwrite existing files. Clean old checkpoints."""
    ckpt_path = output_dir / "checkpoints" / "artn" / f"step_{step}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Never overwrite
    if ckpt_path.exists():
        print(f"  WARNING: Checkpoint {ckpt_path} already exists — skipping save", flush=True)
        return ckpt_path

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": metrics.get("train_loss"),
        "val_loss": metrics.get("val_loss"),
        "halting_stats": halting_stats,
        "ternary_stats": ternary_stats,
        "config": config,
    }, ckpt_path)

    print(f"  Checkpoint saved: {ckpt_path} ({ckpt_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)

    if keep_last_n > 0:
        ckpt_dir = output_dir / "checkpoints" / "artn"
        ckpts = sorted(ckpt_dir.glob("step_*.pt"),
                      key=lambda p: int(p.stem.split('_')[1]))
        if len(ckpts) > keep_last_n:
            for old_ckpt in ckpts[:-keep_last_n]:
                old_ckpt.unlink()
                print(f"  Removed old checkpoint: {old_ckpt.name}", flush=True)

    return ckpt_path


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(config: Dict, output_dir: Path, device: torch.device):
    """Run ARTN training experiment with comprehensive logging."""

    print(f"\n{'='*80}", flush=True)
    print(f"EXPERIMENT: ARTN (Adaptive Recursive Ternary Network)", flush=True)
    print(f"Model: 1 shared BitLinearBlock + per-position halting gate + ternary", flush=True)
    print(f"Optimizer: AdamW (beta1={config['beta1']}, beta2={config['beta2']})", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Disk space check
    disk_usage = os.statvfs(str(BASE_DIR))
    free_mb = (disk_usage.f_bavail * disk_usage.f_frsize) / 1024 / 1024
    print(f"Available disk: {free_mb:.0f} MB", flush=True)
    est_ckpt_mb = (2904065 * 4 * 3) / 1024 / 1024  # params + AdamW state
    est_n_ckpts = config["total_steps"] // config["checkpoint_every"]
    est_total = est_n_ckpts * est_ckpt_mb
    print(f"Estimated checkpoint storage: {est_ckpt_mb:.1f} MB each, "
          f"{est_n_ckpts} checkpoints = {est_total:.0f} MB (ALL kept)", flush=True)
    if free_mb < est_total + 200:
        print(f"  WARNING: Low disk space! Need {est_total:.0f} MB, have {free_mb:.0f} MB", flush=True)
    print()

    # Set seed
    torch.manual_seed(config["seed"])

    # Load data
    train_dataset, val_dataset, tokenizer = load_data(config)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=0,
    )

    # Create model
    print("Creating ARTN model...", flush=True)
    model = ARTNGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        target_passes=config["target_passes"],
        ponder_loss_weight=config["ponder_loss_weight"],
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    block_params = sum(p.numel() for p in model.block.parameters())
    gate_params = sum(p.numel() for p in model.halting_gate.parameters())
    print(f"Parameters: {total_params:,}", flush=True)
    print(f"  Shared block: {block_params:,}", flush=True)
    print(f"  Halting gate: {gate_params:,}", flush=True)
    print(f"  Max passes: {model.max_passes}", flush=True)
    print(f"  Target passes: {config['target_passes']}", flush=True)

    # Ternary stats
    ternary_stats = model.get_ternary_stats()
    print(f"BitLinear layers: {len(ternary_stats['weight_stats'])}", flush=True)

    # Create optimizer (AdamW — SAME proven config)
    print(f"\nCreating AdamW optimizer...", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["max_lr"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )

    # Scheduler (SAME as proven)
    scheduler = CosineWarmupScheduler(
        warmup_steps=config["warmup_steps"],
        total_steps=config["total_steps"],
        max_lr=config["max_lr"],
        min_lr=config["min_lr"],
    )

    # Training
    print("\nStarting training...", flush=True)
    header = (f"{'Step':<8} {'TrLoss':<10} {'ValLoss':<10} {'PPL':<8} "
              f"{'LR':<10} {'GradNorm':<10} {'EPasses':<8} {'Ponder':<8} "
              f"{'Ternary%':<9} {'Status'}")
    print(header, flush=True)
    print("-" * 110, flush=True)

    all_metrics = []
    start_time = time.time()
    data_iter = iter(train_loader)

    for step in range(1, config["total_steps"] + 1):
        # Learning rate
        lr = scheduler.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_ponder_loss = 0.0
        accum_expected_passes = 0.0

        for _ in range(config["gradient_accumulation_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            outputs = model(input_ids, targets=targets)
            loss = outputs["loss"]
            info = outputs["halting_info"]

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: NaN/Inf loss at step {step}", flush=True)
                continue

            scaled_loss = loss / config["gradient_accumulation_steps"]
            scaled_loss.backward()
            accum_loss += loss.item()
            accum_ponder_loss += info["loss_ponder"]
            accum_expected_passes += info["expected_passes"]

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["max_grad_norm"]
        )

        # Optimizer step
        optimizer.step()

        train_loss = accum_loss / config["gradient_accumulation_steps"]
        avg_ponder = accum_ponder_loss / config["gradient_accumulation_steps"]
        avg_epasses = accum_expected_passes / config["gradient_accumulation_steps"]

        # Ternary statistics
        ternary_stats = model.get_ternary_stats()
        avg_sparsity = (
            sum(s["sparsity"] for s in ternary_stats["weight_stats"]) /
            len(ternary_stats["weight_stats"])
            if ternary_stats["weight_stats"] else 0
        )

        # Metrics
        metrics = {
            "step": step,
            "train_loss": train_loss,
            "lr": lr,
            "grad_norm": grad_norm.item(),
            "expected_passes": avg_epasses,
            "ponder_loss": avg_ponder,
            "ternary_sparsity": avg_sparsity,
        }

        # Validation
        if step % config["eval_every"] == 0 or step == 1:
            val_ppl, val_loss, val_epasses = compute_perplexity(
                model, val_loader, device
            )
            metrics["val_loss"] = val_loss
            metrics["val_perplexity"] = val_ppl
            metrics["val_expected_passes"] = val_epasses

            # Generation sample (full computation, threshold=1.0)
            if step % config["checkpoint_every"] == 0 or step == 1:
                sample = generate_sample(
                    model, tokenizer,
                    config["generation_prompt"],
                    config["generation_max_tokens"],
                    config["generation_temperature"],
                    threshold=config["generation_threshold"],
                    device=device,
                )
                metrics["generation_sample"] = sample
                print(f"\n  Generated: {sample[:200]}...\n", flush=True)

        all_metrics.append(metrics)

        # Progress
        status = "OK"
        if train_loss > 9:
            status = "HIGH_LOSS"

        if step % config["eval_every"] == 0 or step == 1:
            print(
                f"{step:<8} {train_loss:<10.4f} "
                f"{metrics.get('val_loss', '-'):<10} "
                f"{metrics.get('val_perplexity', '-'):<8.2f} "
                f"{lr:<10.6f} {grad_norm.item():<10.4f} "
                f"{avg_epasses:<8.2f} {avg_ponder:<8.4f} "
                f"{avg_sparsity:<9.2%} {status}",
                flush=True,
            )
        elif step % 100 == 0:
            print(
                f"{step:<8} {train_loss:<10.4f} {'-':<10} {'-':<8} "
                f"{lr:<10.6f} {grad_norm.item():<10.4f} "
                f"{avg_epasses:<8.2f} {avg_ponder:<8.4f} "
                f"{avg_sparsity:<9.2%} {status}",
                flush=True,
            )

        # Checkpoint
        if step % config["checkpoint_every"] == 0:
            halting_stats = {
                "expected_passes": avg_epasses,
                "ponder_loss": avg_ponder,
                "val_expected_passes": metrics.get("val_expected_passes"),
            }
            save_checkpoint(
                model, optimizer, step, config, metrics,
                halting_stats, ternary_stats, output_dir,
                keep_last_n=config["keep_last_n_checkpoints"],
            )

    # Final summary
    elapsed = time.time() - start_time

    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY: ARTN Training", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)", flush=True)
    print(f"Final train loss: {all_metrics[-1]['train_loss']:.4f}", flush=True)

    val_losses = [m.get("val_loss") for m in all_metrics if "val_loss" in m]
    if val_losses:
        print(f"Final val loss: {val_losses[-1]:.4f}", flush=True)
        print(f"Best val loss: {min(val_losses):.4f}", flush=True)

    epasses = [m.get("expected_passes") for m in all_metrics if "expected_passes" in m]
    if epasses:
        print(f"Final expected passes: {epasses[-1]:.4f}", flush=True)
        print(f"Target passes: {config['target_passes']}", flush=True)

    print(f"Final ternary sparsity: {all_metrics[-1].get('ternary_sparsity', 0):.2%}", flush=True)

    # Save results
    results = {
        "experiment": "artn_recursive_halting",
        "model": "ARTNGPT",
        "optimizer": "adamw",
        "config": config,
        "total_time_seconds": elapsed,
        "total_params": total_params,
        "block_params": block_params,
        "gate_params": gate_params,
        "metrics": all_metrics,
    }

    results_file = output_dir / "artn_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}", flush=True)

    # Comparison to baselines
    print(f"\n{'='*80}", flush=True)
    print(f"COMPARISON TO BASELINES", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"\nFP32 + AdamW baseline (6 blocks):")
    print(f"  Val loss: 1.69, Perplexity: 5.44")
    print(f"  Parameters: 6,852,608")
    print(f"\nBitLinear + AdamW (6 blocks):")
    print(f"  Val loss: 1.92, Perplexity: 6.81")
    print(f"  Parameters: 6,852,682")
    print(f"\nARTN (1 shared block + halting):")
    print(f"  Val loss: {val_losses[-1] if val_losses else 'N/A'}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Expected passes: {epasses[-1] if epasses else 'N/A'} / {config['target_passes']}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("ARTN TRAINING: Adaptive Recursive Ternary Network")
    print("1 shared BitLinearBlock + per-position halting gate + ternary STE")
    print("Optimizer: AdamW (proven config)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}", flush=True)

    # Create new output directory (never overwrite existing)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / "outputs" / f"artn_recursive_{timestamp}"

    if output_dir.exists():
        raise RuntimeError(f"Output directory already exists: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}", flush=True)

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)

    # Disk space check
    disk_usage = os.statvfs(str(BASE_DIR))
    free_mb = (disk_usage.f_bavail * disk_usage.f_frsize) / 1024 / 1024
    est_ckpt_mb = (2904065 * 4 * 3) / 1024 / 1024
    est_n_ckpts = CONFIG["total_steps"] // CONFIG["checkpoint_every"]
    est_total = est_n_ckpts * est_ckpt_mb
    print(f"\nDisk space:", flush=True)
    print(f"  Available: {free_mb:.0f} MB", flush=True)
    print(f"  Per checkpoint: ~{est_ckpt_mb:.1f} MB", flush=True)
    print(f"  All {est_n_ckpts} checkpoints: ~{est_total:.0f} MB (no deletion)", flush=True)
    if free_mb < est_total + 200:
        print(f"  WARNING: Low disk space!", flush=True)

    # Verify model can be created and forward pass works
    print(f"\nVerifying model instantiation...", flush=True)
    test_model = ARTNGPT(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        d_ff=CONFIG["d_ff"],
        max_seq_len=CONFIG["max_seq_len"],
        target_passes=CONFIG["target_passes"],
        ponder_loss_weight=CONFIG["ponder_loss_weight"],
    )
    test_params = sum(p.numel() for p in test_model.parameters())
    print(f"  Parameters: {test_params:,}", flush=True)
    assert test_params == 2904065, f"Parameter count mismatch: {test_params} != 2904065"
    print(f"  Parameter count verified: 2,904,065", flush=True)

    # Quick forward pass test
    test_input = torch.randint(0, 4000, (2, 32))
    test_targets = torch.randint(0, 4000, (2, 32))
    test_output = test_model(test_input, targets=test_targets)
    assert test_output['loss'].dim() == 0
    assert torch.isfinite(test_output['loss'])
    print(f"  Forward pass: loss={test_output['loss'].item():.4f} (finite, scalar)", flush=True)

    # Quick backward pass test
    test_output['loss'].backward()
    gate_grad = test_model.halting_gate.weight.grad
    assert gate_grad is not None and gate_grad.abs().max() > 0
    print(f"  Backward pass: gate gradient non-zero", flush=True)
    print(f"  Model verification PASSED", flush=True)

    del test_model, test_output
    gc.collect()

    # Run experiment
    results = run_experiment(CONFIG, output_dir, device)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"\nFinal Results:", flush=True)
    print(f"  Train loss: {results['metrics'][-1]['train_loss']:.4f}", flush=True)
    print(f"  Val loss: {results['metrics'][-1].get('val_loss', 'N/A')}", flush=True)
    print(f"  Perplexity: {results['metrics'][-1].get('val_perplexity', 'N/A')}", flush=True)
    print(f"  Expected passes: {results['metrics'][-1].get('expected_passes', 'N/A')}", flush=True)
    print(f"  Time: {results['total_time_seconds']/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
