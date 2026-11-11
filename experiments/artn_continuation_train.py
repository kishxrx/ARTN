#!/usr/bin/env python3
"""
ARTN Controlled Training Continuation Script
Loads from step_10000 checkpoint, trains 5000 continuation steps.

Three configurations:
  control      — original dataset (49,988 stories)
  treatment_a  — original + 500 duplicate entries (50,488 stories)
  treatment_b  — original + 500 augmented stories (50,488 stories)

Checkpoint policy:
  - Model-only every 500 steps (steps 10500-14500, skip 12500)
  - Full (model+optimizer) at step 12500 (midpoint recovery)
  - Full (model+optimizer) at step 15000 (final)

Usage:
  cd /root/agent-sandbox/ARTN
  python experiments/artn_continuation_train.py --experiment control
  python experiments/artn_continuation_train.py --experiment treatment_a
  python experiments/artn_continuation_train.py --experiment treatment_b

  Or to resume from midpoint:
  python experiments/artn_continuation_train.py --experiment control --resume
"""

import sys
import time
import math
import torch
import torch.nn as nn
import json
import gc
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

# =============================================================================
# CONFIGURATION
# =============================================================================

ORIGINAL_CHECKPOINT = BASE_DIR / 'outputs' / 'artn_recursive_20261025_002725' / 'checkpoints' / 'artn' / 'step_10000.pt'
TOKENIZER_PATH = BASE_DIR / 'outputs' / 'artn-50k-best-config' / 'tokenizer.json'

# Dataset cache locations (each independent)
DATASET_PATHS = {
    'control':     BASE_DIR / 'outputs' / 'control_continuation' / 'tokenized_cache',
    'treatment_a': BASE_DIR / 'outputs' / 'treatment_oversample' / 'tokenized_cache',
    'treatment_b': BASE_DIR / 'outputs' / 'treatment_augment' / 'tokenized_cache',
}

# Output directory naming convention (follows existing pattern)
# Timestamp added at runtime to prevent collision

# Training config (matches original, with continuation-specific scheduler)
CONFIG = {
    # Model architecture — SAME as original
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    "vocab_size": 4000,

    # ARTN halting
    "target_passes": 4.0,
    "ponder_loss_weight": 0.01,

    # Training parameters — SAME as original
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "max_grad_norm": 1.0,
    "beta1": 0.5,
    "beta2": 0.98,
    "weight_decay": 0.01,

    # Continuation schedule — NEW
    "total_steps": 5000,       # continuation steps
    "warmup_steps": 500,
    "max_lr": 0.0005,          # peak LR for continuation
    "min_lr": 0.00001,         # min LR for continuation

    # Seed — SAME as original
    "seed": 42,

    # Checkpointing — NEW policy
    "checkpoint_every": 500,
    "eval_every": 100,
    "full_checkpoint_steps": [2500, 5000],  # steps where we save model+optimizer

    # Generation samples
    "generation_prompt": "Once upon a time",
    "generation_max_tokens": 100,
    "generation_temperature": 0.9,
    "generation_threshold": 1.0,
}

# =============================================================================
# DATA LOADING
# =============================================================================

class CachedDataset:
    """Dataset that loads pre-tokenized data from cache. SAME as original."""
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


def load_data(experiment_name, config):
    """Load cached tokenized data for the specified experiment."""
    cache_dir = DATASET_PATHS[experiment_name]
    train_cache = cache_dir / 'train_tokenized.pt'
    val_cache = cache_dir / 'val_tokenized.pt'

    if not train_cache.exists():
        raise RuntimeError(f"Train cache not found: {train_cache}")
    if not val_cache.exists():
        raise RuntimeError(f"Val cache not found: {val_cache}")

    print(f"Loading tokenizer from {TOKENIZER_PATH}", flush=True)
    tokenizer = BPETokenizer.load(str(TOKENIZER_PATH))

    print(f"Loading cached data from {cache_dir}", flush=True)
    train_tokenized = torch.load(train_cache, weights_only=False)
    val_tokenized = torch.load(val_cache, weights_only=False)
    print(f"  Train: {len(train_tokenized)} samples", flush=True)
    print(f"  Val: {len(val_tokenized)} samples", flush=True)

    train_dataset = CachedDataset(train_tokenized, config["max_seq_len"])
    val_dataset = CachedDataset(val_tokenized, config["max_seq_len"])

    return train_dataset, val_dataset, tokenizer


# =============================================================================
# LEARNING RATE SCHEDULER (SAME as original, supports offset for resume)
# =============================================================================

class CosineWarmupScheduler:
    def __init__(self, warmup_steps, total_steps, max_lr, min_lr=1e-5):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr

    def get_lr(self, step):
        """Get LR for a given continuation step (0-indexed)."""
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# CHECKPOINT MANAGEMENT (NEVER overwrite, NEVER delete)
# =============================================================================

def save_checkpoint(model, optimizer, cont_step, abs_step, config, metrics,
                    halting_stats, ternary_stats, rng_state, output_dir,
                    save_optimizer=True):
    """Save checkpoint. Never overwrite existing files. Never delete."""
    ckpt_dir = output_dir / "checkpoints" / "artn"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"step_{abs_step}.pt"

    if ckpt_path.exists():
        print(f"  WARNING: Checkpoint {ckpt_path} already exists — skipping", flush=True)
        return ckpt_path

    save_dict = {
        "step": abs_step,
        "continuation_step": cont_step,
        "model_state_dict": model.state_dict(),
        "train_loss": metrics.get("train_loss"),
        "val_loss": metrics.get("val_loss"),
        "halting_stats": halting_stats,
        "ternary_stats": ternary_stats,
        "config": config,
        "rng_state": rng_state,
    }

    if save_optimizer:
        save_dict["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(save_dict, ckpt_path)

    ckpt_type = "FULL (model+optimizer)" if save_optimizer else "model-only"
    size_mb = ckpt_path.stat().st_size / 1024 / 1024
    print(f"  Checkpoint saved: {ckpt_path.name} [{ckpt_type}] ({size_mb:.1f} MB)", flush=True)

    return ckpt_path


# =============================================================================
# TRAINING
# =============================================================================

def run_training(experiment_name, config, output_dir, device, resume=False):
    """Run continuation training for one experiment."""

    print(f"\n{'=' * 80}", flush=True)
    print(f"CONTROLLED TRAINING: {experiment_name.upper()}", flush=True)
    print(f"Checkpoint: {ORIGINAL_CHECKPOINT}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"{'=' * 80}\n", flush=True)

    # Disk space check
    disk_usage = os.statvfs(str(BASE_DIR))
    free_mb = (disk_usage.f_bavail * disk_usage.f_frsize) / 1024 / 1024
    print(f"Available disk: {free_mb:.0f} MB", flush=True)

    # Set seed
    torch.manual_seed(config["seed"])

    # Load data
    train_dataset, val_dataset, tokenizer = load_data(experiment_name, config)

    # Data loaders — SAME as original (shuffle=True, batch_size=4)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=0,
    )

    # Create model — SAME architecture
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
    print(f"Parameters: {total_params:,}", flush=True)

    # Load original checkpoint (read-only)
    print(f"\nLoading checkpoint: {ORIGINAL_CHECKPOINT}", flush=True)
    ckpt = torch.load(str(ORIGINAL_CHECKPOINT), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Model loaded from step {ckpt['step']}", flush=True)

    # Create optimizer — SAME as original
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["max_lr"],  # will be overridden by scheduler
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )

    # Load optimizer state from checkpoint
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"  Optimizer state loaded (AdamW momentum preserved)", flush=True)
    else:
        print(f"  WARNING: No optimizer state in checkpoint — starting fresh", flush=True)

    # Scheduler — NEW continuation schedule
    scheduler = CosineWarmupScheduler(
        warmup_steps=config["warmup_steps"],
        total_steps=config["total_steps"],
        max_lr=config["max_lr"],
        min_lr=config["min_lr"],
    )

    # Show initial LR (from checkpoint, before scheduler takes over)
    print(f"\nCheckpoint LR (saved): {optimizer.param_groups[0]['lr']:.6f}", flush=True)
    init_lr = scheduler.get_lr(0)
    print(f"Continuation step 0 LR (scheduler): {init_lr:.6f}", flush=True)
    peak_lr = scheduler.get_lr(config["warmup_steps"])
    print(f"Continuation step {config['warmup_steps']} LR (peak): {peak_lr:.6f}", flush=True)
    final_lr = scheduler.get_lr(config["total_steps"] - 1)
    print(f"Continuation step {config['total_steps']-1} LR (final): {final_lr:.6f}", flush=True)

    # Resume logic
    start_step = 0
    if resume:
        midpoint_ckpt = output_dir / "checkpoints" / "artn" / "step_12500.pt"
        if midpoint_ckpt.exists():
            print(f"\nResuming from midpoint: {midpoint_ckpt}", flush=True)
            resume_ckpt = torch.load(str(midpoint_ckpt), map_location=device, weights_only=False)
            model.load_state_dict(resume_ckpt["model_state_dict"])
            if "optimizer_state_dict" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
                print(f"  Optimizer state restored from step {resume_ckpt['step']}", flush=True)
            start_step = resume_ckpt.get("continuation_step", 2500)
            print(f"  Resuming from continuation step {start_step}", flush=True)
            # Restore RNG state if available
            if "rng_state" in resume_ckpt:
                torch.set_rng_state(resume_ckpt["rng_state"])
                print(f"  RNG state restored", flush=True)
        else:
            print(f"  No midpoint checkpoint found at {midpoint_ckpt} — starting from step 0", flush=True)

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\nStarting training from continuation step {start_step + 1}...", flush=True)
    header = (f"{'ContStep':<9} {'AbsStep':<9} {'TrLoss':<10} {'ValLoss':<10} "
              f"{'LR':<10} {'GradNorm':<10} {'EPasses':<8} {'Status'}")
    print(header, flush=True)
    print("-" * 90, flush=True)

    all_metrics = []
    start_time = time.time()
    data_iter = iter(train_loader)

    for cont_step in range(start_step, config["total_steps"]):
        step = cont_step + 1  # 1-indexed for LR and logging
        abs_step = 10000 + step

        # Learning rate from scheduler
        lr = scheduler.get_lr(cont_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation — SAME as original
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
                print(f"  WARNING: NaN/Inf loss at step {abs_step}", flush=True)
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
            "continuation_step": step,
            "absolute_step": abs_step,
            "train_loss": train_loss,
            "lr": lr,
            "grad_norm": grad_norm.item(),
            "expected_passes": avg_epasses,
            "ponder_loss": avg_ponder,
            "ternary_sparsity": avg_sparsity,
        }

        # Validation
        if step % config["eval_every"] == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_tokens = 0
                val_epasses_sum = 0.0
                val_batches = 0
                for i, batch in enumerate(val_loader):
                    if i >= 50:
                        break
                    input_ids = batch[:, :-1].to(device)
                    targets = batch[:, 1:].to(device)
                    outputs = model(input_ids, targets=targets)
                    loss = outputs["loss"]
                    info = outputs["halting_info"]
                    num_tokens = (targets != 0).sum().item()
                    val_loss_sum += loss.item() * num_tokens
                    val_tokens += num_tokens
                    val_epasses_sum += info["expected_passes"]
                    val_batches += 1

                val_loss = val_loss_sum / val_tokens if val_tokens > 0 else float('inf')
                val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
                val_epasses = val_epasses_sum / val_batches if val_batches > 0 else 0

            metrics["val_loss"] = val_loss
            metrics["val_perplexity"] = val_ppl
            metrics["val_expected_passes"] = val_epasses
            model.train()

        all_metrics.append(metrics)

        # Progress
        status = "OK"
        if train_loss > 9:
            status = "HIGH_LOSS"

        if step % config["eval_every"] == 0 or step == 1:
            print(
                f"{step:<9} {abs_step:<9} {train_loss:<10.4f} "
                f"{metrics.get('val_loss', '-'):<10} "
                f"{lr:<10.6f} {grad_norm.item():<10.4f} "
                f"{avg_epasses:<8.2f} {status}",
                flush=True,
            )
        elif step % 100 == 0:
            print(
                f"{step:<9} {abs_step:<9} {train_loss:<10.4f} {'-':<10} "
                f"{lr:<10.6f} {grad_norm.item():<10.4f} "
                f"{avg_epasses:<8.2f} {status}",
                flush=True,
            )

        # Checkpoint
        if step % config["checkpoint_every"] == 0:
            halting_stats = {
                "expected_passes": avg_epasses,
                "ponder_loss": avg_ponder,
                "val_expected_passes": metrics.get("val_expected_passes"),
            }

            # Determine if this is a full or model-only checkpoint
            save_full = step in config["full_checkpoint_steps"]

            # Save RNG state for reproducible recovery
            rng_state = torch.get_rng_state()

            save_checkpoint(
                model, optimizer,
                cont_step=step,
                abs_step=abs_step,
                config=config,
                metrics=metrics,
                halting_stats=halting_stats,
                ternary_stats=ternary_stats,
                rng_state=rng_state,
                output_dir=output_dir,
                save_optimizer=save_full,
            )

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}", flush=True)
    print(f"SUMMARY: {experiment_name.upper()} Continuation Training", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)", flush=True)
    print(f"Final train loss: {all_metrics[-1]['train_loss']:.4f}", flush=True)

    val_losses = [m.get("val_loss") for m in all_metrics if "val_loss" in m]
    if val_losses:
        print(f"Final val loss: {val_losses[-1]:.4f}", flush=True)
        print(f"Best val loss: {min(val_losses):.4f}", flush=True)

    print(f"Final ternary sparsity: {all_metrics[-1].get('ternary_sparsity', 0):.2%}", flush=True)

    # Save results
    results = {
        "experiment": experiment_name,
        "model": "ARTNGPT",
        "optimizer": "adamw",
        "config": config,
        "total_time_seconds": elapsed,
        "total_params": total_params,
        "metrics": all_metrics,
    }

    results_file = output_dir / f"{experiment_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}", flush=True)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ARTN Controlled Training Continuation")
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['control', 'treatment_a', 'treatment_b'],
                        help='Which experiment to run')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from midpoint checkpoint (step 12500)')
    args = parser.parse_args()

    print("=" * 80)
    print(f"ARTN CONTROLLED TRAINING CONTINUATION")
    print(f"Experiment: {args.experiment}")
    print(f"Resume: {args.resume}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Create output directory (follows existing naming convention, never overwrite)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_prefix = {
        'control': 'artn_control',
        'treatment_a': 'artn_treatment_a',
        'treatment_b': 'artn_treatment_b',
    }[args.experiment]
    output_dir = BASE_DIR / "outputs" / f"{exp_prefix}_{timestamp}"

    if output_dir.exists():
        raise RuntimeError(f"Output directory already exists: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}", flush=True)

    # Verify original checkpoint exists
    if not ORIGINAL_CHECKPOINT.exists():
        raise RuntimeError(f"Original checkpoint not found: {ORIGINAL_CHECKPOINT}")

    # Verify dataset cache exists
    cache_dir = DATASET_PATHS[args.experiment]
    if not (cache_dir / 'train_tokenized.pt').exists():
        raise RuntimeError(f"Dataset cache not found: {cache_dir}")

    print(f"Original checkpoint: {ORIGINAL_CHECKPOINT} (read-only)", flush=True)
    print(f"Dataset cache: {cache_dir}", flush=True)

    # Run training
    results = run_training(args.experiment, CONFIG, output_dir, device, resume=args.resume)

    print(f"\n{'=' * 80}", flush=True)
    print(f"TRAINING COMPLETE: {args.experiment}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"{'=' * 80}", flush=True)


if __name__ == "__main__":
    main()
