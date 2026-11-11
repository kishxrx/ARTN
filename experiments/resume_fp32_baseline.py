#!/usr/bin/env python3
"""
RESUME FP32 BASELINE FROM LAST GOOD CHECKPOINT
Run this to continue the experiment after disk space issue.
"""

import sys
import torch
from pathlib import Path

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

# Import the original experiment
from experiments.fp32_baseline_controlled import (
    run_experiment,
    load_data,
    FP32GPT,
    CONFIG,
)
import json
import gc

def main():
    print("="*80)
    print("RESUMING FP32 BASELINE EXPERIMENT")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}", flush=True)
    
    # Find existing output directory (filter for directories only)
    output_dirs = sorted([d for d in Path(BASE_DIR / 'outputs').glob('fp32_baseline_*') if d.is_dir()])
    if not output_dirs:
        raise RuntimeError("No existing fp32_baseline directory found!")
    
    output_dir = output_dirs[-1]  # Most recent
    print(f"Output directory: {output_dir}", flush=True)
    
    # Check Adafactor checkpoints
    adafactor_dir = output_dir / 'checkpoints' / 'adafactor'
    adafactor_checkpoints = sorted(adafactor_dir.glob('step_*.pt')) if adafactor_dir.exists() else []
    
    # Find last GOOD checkpoint (file size > 40MB)
    last_good_step = 0
    for ckpt_path in reversed(adafactor_checkpoints):
        file_size = ckpt_path.stat().st_size
        if file_size > 40_000_000:  # > 40MB is good
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                if 'step' in ckpt:
                    last_good_step = ckpt['step']
                    print(f"\n✓ Last good Adafactor checkpoint: step {last_good_step}", flush=True)
                    break
            except:
                continue
    
    # Clean up corrupted checkpoints (< 40MB)
    removed_count = 0
    for ckpt_path in adafactor_checkpoints:
        file_size = ckpt_path.stat().st_size
        if file_size < 40_000_000 and file_size > 0:
            print(f"✗ Removing corrupted checkpoint: {ckpt_path.name} ({file_size/1e6:.1f}MB)", flush=True)
            ckpt_path.unlink()
            removed_count += 1
    
    if removed_count > 0:
        print(f"\nRemoved {removed_count} corrupted checkpoints", flush=True)
    
    # Load data
    print("\nLoading cached data...", flush=True)
    train_dataset, val_dataset, tokenizer = load_data(CONFIG)
    
    # Load initial model state
    init_state_path = output_dir / 'initial_model_state.pt'
    if init_state_path.exists():
        print("Loading initial model state...", flush=True)
        model_init_state = torch.load(init_state_path, map_location='cpu')
    else:
        print("Creating new initial model state...", flush=True)
        torch.manual_seed(CONFIG["seed"])
        init_model = FP32GPT(
            vocab_size=CONFIG["vocab_size"],
            d_model=CONFIG["d_model"],
            n_heads=CONFIG["n_heads"],
            n_layers=CONFIG["n_layers"],
            d_ff=CONFIG["d_ff"],
            max_seq_len=CONFIG["max_seq_len"],
        )
        model_init_state = init_model.state_dict()
        torch.save(model_init_state, init_state_path)
    
    # Check AdamW status
    adamw_dir = output_dir / 'checkpoints' / 'adamw'
    adamw_exists = adamw_dir.exists() and len(list(adamw_dir.glob('step_*.pt'))) > 0
    
    print(f"\n{'='*80}")
    print("EXPERIMENT STATUS")
    print(f"{'='*80}")
    print(f"Adafactor: Steps 1-{last_good_step} complete, need {last_good_step+1}-10000")
    print(f"AdamW: {'NOT STARTED' if not adamw_exists else 'ALREADY STARTED'}")
    print(f"{'='*80}\n")
    
    # Run Adafactor (resume from last good step)
    if last_good_step < CONFIG["total_steps"]:
        print(f"\n{'='*80}")
        print(f"RESUMING ADAFACTOR FROM STEP {last_good_step}")
        print(f"{'='*80}\n", flush=True)
        
        run_experiment(
            exp_name='adafactor',
            optimizer_type='adafactor',
            model_init_state=model_init_state,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            config=CONFIG,
            output_dir=output_dir,
            device=device,
            resume_step=last_good_step,
        )
    else:
        print("\n✓ Adafactor already complete!", flush=True)
    
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run AdamW (from scratch or resume)
    print(f"\n{'='*80}")
    print(f"RUNNING ADAMW")
    print(f"{'='*80}\n", flush=True)
    
    run_experiment(
        exp_name='adamw',
        optimizer_type='adamw',
        model_init_state=model_init_state,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        config=CONFIG,
        output_dir=output_dir,
        device=device,
        resume_step=0,  # Start from scratch
    )
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80, flush=True)


if __name__ == "__main__":
    main()
