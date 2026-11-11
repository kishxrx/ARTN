import torch

print("="*80)
print("FP32-5k vs Ternary-5k - BOTH ADAFACTOR")
print("="*80)

# Load FP32-5k checkpoint
fp32_ckpt = torch.load('outputs/fp32-5k-comparison/checkpoints/checkpoint_step_5000.pt', map_location='cpu')
fp32_msd = fp32_ckpt.get('model_state_dict', {})

# Load Ternary-5k checkpoint  
ternary_ckpt = torch.load('outputs/artn-50k-best-config/checkpoints/checkpoint_step_5000.pt', map_location='cpu')
ternary_msd = ternary_ckpt.get('model_state_dict', {})

print("\n" + "="*80)
print("EMBEDDING WEIGHT STATISTICS")
print("="*80)

# FP32 embeddings
if 'token_embedding.weight' in fp32_msd:
    fp32_emb = fp32_msd['token_embedding.weight']
    print(f"\nFP32-5k token_embedding:")
    print(f"  mean: {fp32_emb.mean():.6f}")
    print(f"  std:  {fp32_emb.std():.6f}")
    print(f"  abs mean: {fp32_emb.abs().mean():.6f}")

# Ternary embeddings
if 'token_embedding.weight' in ternary_msd:
    ternary_emb = ternary_msd['token_embedding.weight']
    print(f"\nTernary-5k token_embedding:")
    print(f"  mean: {ternary_emb.mean():.6f}")
    print(f"  std:  {ternary_emb.std():.6f}")
    print(f"  abs mean: {ternary_emb.abs().mean():.6f}")

print("\n" + "="*80)
print("LM HEAD WEIGHT STATISTICS")
print("="*80)

# FP32 lm_head
if 'lm_head.weight' in fp32_msd:
    fp32_lm = fp32_msd['lm_head.weight']
    print(f"\nFP32-5k lm_head:")
    print(f"  mean: {fp32_lm.mean():.6f}")
    print(f"  std:  {fp32_lm.std():.6f}")

# Ternary lm_head
if 'lm_head.weight' in ternary_msd:
    ternary_lm = ternary_msd['lm_head.weight']
    print(f"\nTernary-5k lm_head:")
    print(f"  mean: {ternary_lm.mean():.6f}")
    print(f"  std:  {ternary_lm.std():.6f}")

print("\n" + "="*80)
print("OPTIMIZER STATE")
print("="*80)

# Check optimizer states
fp32_opt = fp32_ckpt.get('optimizer_state_dict', {})
ternary_opt = ternary_ckpt.get('optimizer_state_dict', {})

if 'state' in fp32_opt and len(fp32_opt['state']) > 0:
    first_state = list(fp32_opt['state'].values())[0]
    print(f"\nFP32-5k optimizer state keys: {list(first_state.keys())}")
    if 'exp_avg' in first_state:
        print(f"  exp_avg mean: {first_state['exp_avg'].abs().mean():.6f}")

if 'state' in ternary_opt and len(ternary_opt['state']) > 0:
    first_state = list(ternary_opt['state'].values())[0]
    print(f"\nTernary-5k optimizer state keys: {list(first_state.keys())}")
    if 'exp_avg' in first_state:
        print(f"  exp_avg mean: {first_state['exp_avg'].abs().mean():.6f}")

print("\n" + "="*80)
print("KEY FINDING")
print("="*80)

fp32_std = fp32_emb.std().item() if 'token_embedding.weight' in fp32_msd else 0
ternary_std = ternary_emb.std().item() if 'token_embedding.weight' in ternary_msd else 0

print(f"""
FP32-5k (Adafactor):
  - Emb std: {fp32_std:.4f}
  - Loss: 5.99
  - Accuracy: 16.4%

Ternary-5k (Adafactor):
  - Emb std: {ternary_std:.4f}
  - Loss: 8.21
  - Accuracy: 1.5%

RATIO: Ternary emb std / FP32 emb std = {ternary_std/fp32_std:.2f}x

If BOTH models have similar emb std but different performance,
then the issue is NOT just weight explosion from Adafactor + Ternary.

If Ternary has MUCH larger emb std than FP32, then the issue
IS Adafactor + Ternary interaction.
""")

print("="*80)
