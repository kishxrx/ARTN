#!/usr/bin/env python3
"""
RESUME BITLINEAR + ADAMW EXPERIMENT
Run this to continue after disk space issue or interruption.
"""

import sys
import torch
from pathlib import Path

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from experiments.bitlinear_adamw_controlled import (
    run_experiment,
    CONFIG,
)
import json

def main():
    print("="*80)
    print("RESUMING BITLINEAR + ADAMW EXPERIMENT")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}", flush=True)
    
    # Find existing output directory (filter for directories only)
    output_dirs = sorted([d for d in Path(BASE_DIR / 'outputs').glob('bitlinear_adamw_*') if d.is_dir()])
    if not output_dirs:
        raise RuntimeError("No existing bitlinear_adamw directory found!")
    
    output_dir = output_dirs[-1]  # Most recent
    print(f"Output directory: {output_dir}", flush=True)
    
    # Check checkpoints
    ckpt_dir = output_dir / 'checkpoints' / 'bitlinear_adamw'
    checkpoints = sorted(ckpt_dir.glob('step_*.pt')) if ckpt_dir.exists() else []
    
    # Find last GOOD checkpoint (file size > 40MB)
    last_good_step = 0
    for ckpt_path in reversed(checkpoints):
        file_size = ckpt_path.stat().st_size
        if file_size > 40_000_000:  # > 40MB is good
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                if 'step' in ckpt:
                    last_good_step = ckpt['step']
                    print(f"\n✓ Last good checkpoint: step {last_good_step}", flush=True)
                    break
            except:
                continue
    
    # Clean up corrupted checkpoints (< 40MB)
    removed_count = 0
    for ckpt_path in checkpoints:
        file_size = ckpt_path.stat().st_size
        if file_size < 40_000_000 and file_size > 0:
            print(f"✗ Removing corrupted checkpoint: {ckpt_path.name} ({file_size/1e6:.1f}MB)", flush=True)
            ckpt_path.unlink()
            removed_count += 1
    
    if removed_count > 0:
        print(f"\nRemoved {removed_count} corrupted checkpoints", flush=True)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT STATUS")
    print(f"{'='*80}")
    print(f"BitLinear + AdamW: Steps 1-{last_good_step} complete, need {last_good_step+1}-10000")
    print(f"{'='*80}\n")
    
    # Run experiment
    run_experiment(
        exp_name="bitlinear_adamw",
        config=CONFIG,
        output_dir=output_dir,
        device=device,
        resume_step=last_good_step,
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80, flush=True)


if __name__ == "__main__":
    main()
