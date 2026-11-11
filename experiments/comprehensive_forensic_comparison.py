import torch
import sys
sys.path.insert(0, '.')

print("="*70)
print("COMPREHENSIVE COMPARISON: 50k vs Phase 22")
print("="*70)

# Load 50k checkpoint
ckpt_50k = torch.load('outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt', map_location='cpu')
config_50k = ckpt_50k['config']

# Load Phase 22 checkpoint
ckpt_p22 = torch.load('outputs/phase22_controlled_experiment/fp32_checkpoints/checkpoint_step_5000.pt', map_location='cpu')

print("\n" + "="*70)
print("1. TRAINING SCRIPT CONFIGURATION")
print("="*70)

print("\n50k CONFIG:")
print(f"  optimizer: Adafactor (from train_optimized.py)")
print(f"  beta1: {config_50k['beta1']}")
print(f"  beta2: 0.999 (in code)")
print(f"  learning_rate: {config_50k['max_lr']}")
print(f"  weight_decay: {config_50k['weight_decay']}")
print(f"  total_steps: {config_50k['total_steps']}")
print(f"  warmup_steps: {config_50k['warmup_steps']}")
print(f"  batch_size: {config_50k['batch_size']}")
print(f"  gradient_accumulation: {config_50k['gradient_accumulation_steps']}")
print(f"  max_samples: {config_50k['max_samples']}")
print(f"  label construction: labels = input_ids.clone() (COPY TASK)")

print("\nPhase 22 CONFIG:")
print(f"  optimizer: AdamW")
print(f"  beta1: 0.9 (default)")
print(f"  beta2: 0.999 (default)")
print(f"  learning_rate: 0.001")
print(f"  weight_decay: 0.01")
print(f"  total_steps: 5000")
print(f"  warmup_steps: 100")
print(f"  batch_size: 4")
print(f"  gradient_accumulation: 1")
print(f"  max_samples: 50000")
print(f"  label construction: labels = input_ids.clone() (COPY TASK)")

print("\n" + "="*70)
print("2. KEY DIFFERENCES")
print("="*70)

print("\n| Aspect | 50k | Phase 22 |")
print("|--------|-----|----------|")
print("| Optimizer | Adafactor | AdamW |")
print("| beta1 | 0.5 | 0.9 |")
print("| Label shifting | NO (copy) | NO (copy) |")
print("| Training loss @ 5k | ~7.9 | ~0.05 |")
print("| Entropy @ 5k | 95.7% | N/A |")
print("| Status | FAILED | LEARNED |")

print("\n" + "="*70)
print("3. WEIGHT STATISTICS COMPARISON")
print("="*70)

# 50k weights
msd_50k = ckpt_50k['model_state_dict']
w_50k_emb = msd_50k['token_embedding.weight']
w_50k_lm = msd_50k['lm_head.weight']

print("\n50k Model:")
print(f"  token_embedding: mean={w_50k_emb.mean():.6f}, std={w_50k_emb.std():.6f}")
print(f"  lm_head: mean={w_50k_lm.mean():.6f}, std={w_50k_lm.std():.6f}")

# Check weight magnitude distribution
print(f"  token_embedding abs mean: {w_50k_emb.abs().mean():.6f}")
print(f"  lm_head abs mean: {w_50k_lm.abs().mean():.6f}")

# Phase 22 FP32 weights
msd_p22 = ckpt_p22['model_state_dict']
w_p22_emb = msd_p22['token_embedding.weight']
w_p22_lm = msd_p22['lm_head.weight']

print("\nPhase 22 FP32:")
print(f"  token_embedding: mean={w_p22_emb.mean():.6f}, std={w_p22_emb.std():.6f}")
print(f"  lm_head: mean={w_p22_lm.mean():.6f}, std={w_p22_lm.std():.6f}")
print(f"  token_embedding abs mean: {w_p22_emb.abs().mean():.6f}")
print(f"  lm_head abs mean: {w_p22_lm.abs().mean():.6f}")

# COMPARISON
print("\nWEIGHT MAGNITUDE RATIO:")
print(f"  token_embedding: 50k std / Phase22 std = {w_50k_emb.std() / w_p22_emb.std():.2f}x")
print(f"  lm_head: 50k std / Phase22 std = {w_50k_lm.std() / w_p22_lm.std():.2f}x")

print("\n" + "="*70)
print("4. OPTIMIZER STATE ANALYSIS (50k only)")
print("="*70)

# 50k optimizer
opt_50k = ckpt_50k['optimizer_state_dict']
print(f"\n50k Optimizer (Adafactor):")
print(f"  Param groups: {len(opt_50k.get('param_groups', []))}")

if 'state' in opt_50k and len(opt_50k['state']) > 0:
    # Get first parameter state
    first_key = list(opt_50k['state'].keys())[0]
    first_state = opt_50k['state'][first_key]
    print(f"  State keys: {list(first_state.keys())}")
    
    # Analyze moment statistics
    if 'exp_avg' in first_state:
        exp_avg = first_state['exp_avg']
        print(f"  exp_avg (first moment): shape={exp_avg.shape}, mean={exp_avg.mean():.6f}, std={exp_avg.std():.6f}")
    
    if 'exp_avg_sq_row' in first_state:
        exp_avg_sq_row = first_state['exp_avg_sq_row']
        print(f"  exp_avg_sq_row: shape={exp_avg_sq_row.shape}, mean={exp_avg_sq_row.mean():.6f}")
    
    if 'exp_avg_sq_col' in first_state:
        exp_avg_sq_col = first_state['exp_avg_sq_col']
        print(f"  exp_avg_sq_col: shape={exp_avg_sq_col.shape}, mean={exp_avg_sq_col.mean():.6f}")

print("\n" + "="*70)
print("5. TERNARIZATION STATISTICS")
print("="*70)

# Check BitLinear running statistics
bitlinear_keys = [k for k in msd_50k.keys() if 'running_sparsity' in k or 'num_quantizations' in k]
if bitlinear_keys:
    print(f"\n50k BitLinear Statistics (first 10):")
    for k in bitlinear_keys[:10]:
        v = msd_50k[k]
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item() if v.numel() == 1 else v}")
        else:
            print(f"  {k}: {v}")
    
    # Calculate average sparsity
    sparsity_keys = [k for k in bitlinear_keys if 'running_sparsity' in k]
    if sparsity_keys:
        sparsities = [msd_50k[k].item() if isinstance(msd_50k[k], torch.Tensor) else msd_50k[k] for k in sparsity_keys]
        print(f"\n  Average running_sparsity: {sum(sparsities)/len(sparsities):.6f}")
        print(f"  Min running_sparsity: {min(sparsities):.6f}")
        print(f"  Max running_sparsity: {max(sparsities):.6f}")
else:
    print(f"\n50k: No BitLinear running statistics found")

# Phase 22 doesn't use BitLinear for FP32
print(f"\nPhase 22 FP32: Uses standard Linear (no ternarization statistics)")

print("\n" + "="*70)
print("6. CRITICAL FINDING: WEIGHT SCALE MISMATCH")
print("="*70)

print(f"""
50k token_embedding std: {w_50k_emb.std():.4f}
Phase 22 token_embedding std: {w_p22_emb.std():.4f}
RATIO: {w_50k_emb.std() / w_p22_emb.std():.2f}x

50k lm_head std: {w_50k_lm.std():.4f}
Phase 22 lm_head std: {w_p22_lm.std():.4f}
RATIO: {w_50k_lm.std() / w_p22_lm.std():.2f}x

CRITICAL: 50k token embeddings are 35x LARGER than Phase 22!
""")

print("\n" + "="*70)
print("7. SUMMARY")
print("="*70)

print("""
KEY DIFFERENCES BETWEEN 50k AND PHASE 22:

1. OPTIMIZER:
   - 50k: Adafactor with beta1=0.5
   - Phase 22: AdamW with beta1=0.9

2. LABEL CONSTRUCTION:
   - BOTH use copy task (labels = input_ids.clone())
   - NEITHER shifts labels for next-token prediction

3. WEIGHT SCALE:
   - 50k token_embedding std = 0.77 (35x larger!)
   - Phase 22 token_embedding std = 0.02 (normal)

4. TRAINING OUTCOME:
   - 50k: FAILED to learn (loss ~7, entropy 97%)
   - Phase 22: LEARNED (loss ~0.05)

5. ROOT CAUSE HYPOTHESIS:
   - 50k weights EXPLODED during training
   - Large weights cause near-uniform softmax distribution
   - Adafactor may not handle ternarization well
   - OR initialization was different
""")

print("\n" + "="*70)
