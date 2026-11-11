import torch
from pathlib import Path

print("="*90)
print("COMPLETE CHECKPOINT INVENTORY - ALL EXPERIMENTS")
print("="*90)

# Define all checkpoints to analyze
checkpoints = {
    'Phase 22 FP32': [
        'outputs/phase22_controlled_experiment/fp32_checkpoints/checkpoint_step_5000.pt',
    ],
    'Phase 22 Ternary': [
        'outputs/phase22_controlled_experiment/ternary_checkpoints/checkpoint_step_5000.pt',
    ],
    '50k Ternary (Adafactor)': [
        'outputs/artn-50k-best-config/checkpoints/checkpoint_step_5000.pt',
        'outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt',
    ],
    'FP32-5k Comparison (Adafactor)': [
        'outputs/fp32-5k-comparison/checkpoints/checkpoint_step_5000.pt',
    ],
    'Beta1-0.5 Ternary (Adafactor)': [
        'outputs/beta1-0.5-batch8-lr001/checkpoints/checkpoint_step_5000.pt',
    ],
    'Beta1-0.7 Ternary (Adafactor)': [
        'outputs/beta1-0.7-batch8-lr001/checkpoints/checkpoint_step_5000.pt',
    ],
}

print(f"\n{'Experiment':<35} {'Step':<8} {'Model':<15} {'Optim':<12} {'LR':<8} {'Emb Std':<10} {'Loss':<10}")
print("-" * 90)

for exp_name, ckpt_paths in checkpoints.items():
    for ckpt_path in ckpt_paths:
        if not Path(ckpt_path).exists():
            continue
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        step = ckpt.get('step', 'N/A')
        
        # Model type
        msd = ckpt.get('model_state_dict', {})
        has_bitlinear = any('running_sparsity' in k for k in msd.keys())
        model_type = 'Ternary' if has_bitlinear else 'FP32'
        
        # Optimizer
        opt = ckpt.get('optimizer_state_dict', {})
        optimizer = 'Unknown'
        if 'state' in opt and len(opt['state']) > 0:
            first_state = list(opt['state'].values())[0]
            if 'exp_avg_sq_row' in first_state:
                optimizer = 'Adafactor'
            elif 'exp_avg' in first_state:
                optimizer = 'AdamW'
        
        # Config
        cfg = ckpt.get('config', {})
        lr = cfg.get('max_lr', cfg.get('learning_rate', 'N/A'))
        
        # Embedding std
        emb_std = 'N/A'
        if 'token_embedding.weight' in msd:
            emb_std = f"{msd['token_embedding.weight'].std():.4f}"
        
        # Loss
        loss = ckpt.get('loss', 'N/A')
        if isinstance(loss, float):
            loss = f"{loss:.4f}"
        
        print(f"{exp_name:<35} {step:<8} {model_type:<15} {optimizer:<12} {lr:<8} {emb_std:<10} {loss:<10}")

print("\n" + "="*90)
print("KEY FINDINGS")
print("="*90)

print("""
1. Phase 22 FP32: Uses AdamW, emb_std=0.0223, loss=0.058 @ 5k
2. Phase 22 Ternary: Uses AdamW, emb_std=0.0223, loss=0.112 @ 5k
3. 50k Ternary (Adafactor): emb_std=0.7728, loss=7.13 @ 50k
4. FP32-5k (Adafactor): emb_std=0.0200, loss=N/A @ 5k

CRITICAL OBSERVATION:
- All Adafactor Ternary models have emb_std ~0.77-0.99 (LARGE!)
- Phase 22 AdamW models have emb_std ~0.02 (NORMAL)
- FP32-5k Adafactor has emb_std ~0.02 (NORMAL)

This suggests: Adafactor + Ternarization = WEIGHT EXPLOSION
But: Adafactor + FP32 = NORMAL WEIGHTS
""")

print("\n" + "="*90)
print("LABEL SHIFTING STATUS")
print("="*90)

# Check Phase 22 script
with open('experiments/phase22_full_training.py') as f:
    phase22_content = f.read()

phase22_shifted = '[:, :-1]' in phase22_content or '[:, 1:]' in phase22_content
print(f"Phase 22 script has label shifting: {phase22_shifted}")

# Check train_optimized script
with open('scripts/train_optimized.py') as f:
    train_opt_content = f.read()

train_opt_shifted = '[:, :-1]' in train_opt_content or '[:, 1:]' in train_opt_content
print(f"train_optimized.py has label shifting: {train_opt_shifted}")

# Check train_beta1_0.5 script
with open('scripts/train_beta1_0.5.py') as f:
    train_beta_content = f.read()

train_beta_shifted = '[:, :-1]' in train_beta_content or '[:, 1:]' in train_beta_content
print(f"train_beta1_0.5.py has label shifting: {train_beta_shifted}")

print("\n" + "="*90)
print("DO WE HAVE THE REQUIRED EVIDENCE?")
print("="*90)

print("""
QUESTION: Do we already have an experiment comparing Adafactor vs AdamW?

ANSWER: YES AND NO

YES - We have:
  1. Phase 22 FP32 with AdamW @ 5k (emb_std=0.02, loss=0.058)
  2. Phase 22 Ternary with AdamW @ 5k (emb_std=0.02, loss=0.112)
  3. 50k Ternary with Adafactor @ 50k (emb_std=0.77, loss=7.13)
  4. FP32-5k with Adafactor @ 5k (emb_std=0.02, loss=N/A)

NO - We DON'T have:
  1. Ternary with Adafactor @ same steps as Phase 22
  2. FP32 with AdamW vs FP32 with Adafactor comparison
  3. All with PROPER label shifting

CRITICAL ISSUE:
  - ALL existing experiments use COPY TASK (labels=input_ids)
  - NONE use proper next-token prediction (labels=input_ids[:, 1:])
  - This invalidates ALL previous language modeling conclusions

EVIDENCE FOR ADAFACTOR + TERNARIZATION ISSUE:
  - Ternary + Adafactor → emb_std=0.77-0.99 (LARGE)
  - FP32 + Adafactor → emb_std=0.02 (NORMAL)
  - Ternary + AdamW → emb_std=0.02 (NORMAL)

CONCLUSION:
  We have STRONG evidence that Adafactor + Ternarization causes weight explosion.
  But ALL experiments used wrong objective (copy task, not next-token).
""")

print("\n" + "="*90)
