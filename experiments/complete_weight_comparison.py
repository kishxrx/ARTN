import torch
import sys

print("="*80)
print("COMPLETE WEIGHT AND LOSS COMPARISON")
print("="*80)

# FP32 AdamW @ 5k
fp32_adamw = torch.load('experiments/results/fp32_control_20260819_232646/step_5000.pt', map_location='cpu')
fp32_adamw_emb = fp32_adamw['model_state_dict']['token_embedding.weight']

# FP32 Adafactor @ 5k
fp32_adaf = torch.load('outputs/fp32-5k-comparison/checkpoints/checkpoint_step_5000.pt', map_location='cpu')
fp32_adaf_emb = fp32_adaf['model_state_dict']['token_embedding.weight']

# Ternary Adafactor @ 5k
ternary_adaf = torch.load('outputs/artn-50k-best-config/checkpoints/checkpoint_step_5000.pt', map_location='cpu')
ternary_adaf_emb = ternary_adaf['model_state_dict']['token_embedding.weight']

# Phase 22 Ternary AdamW @ 5k
p22_ternary = torch.load('outputs/phase22_controlled_experiment/ternary_checkpoints/checkpoint_step_5000.pt', map_location='cpu')
p22_ternary_emb = p22_ternary['model_state_dict']['token_embedding.weight']

print(f"\n{'Model':<30} {'Optim':<12} {'Emb Std':<12} {'Loss':<12} {'Accuracy':<12}")
print("-"*80)
print(f"{'FP32 AdamW @ 5k':<30} {'AdamW':<12} {fp32_adamw_emb.std():<12.4f} {'0.07':<12} {'95%':<12}")
print(f"{'FP32 Adafactor @ 5k':<30} {'Adafactor':<12} {fp32_adaf_emb.std():<12.4f} {'5.99':<12} {'16.4%':<12}")
print(f"{'Ternary Adafactor @ 5k':<30} {'Adafactor':<12} {ternary_adaf_emb.std():<12.4f} {'7.94':<12} {'~15%':<12}")
print(f"{'Phase 22 Ternary AdamW @ 5k':<30} {'AdamW':<12} {p22_ternary_emb.std():<12.4f} {'0.11':<12} {'~100%':<12}")

print("\n" + "="*80)
print("CRITICAL OBSERVATION")
print("="*80)

print(f"""
WAIT! FP32 AdamW has emb std = 0.86 (LARGE!) but learned WELL (loss 0.07, acc 95%)
     FP32 Adafactor has emb std = 0.02 (small) but learned POORLY (loss 5.99, acc 16%)
     
This means EMBEDDING SCALE IS NOT THE ISSUE!

The key difference is:
  - FP32 AdamW: Loss 0.07 (LEARNED WELL)
  - FP32 Adafactor: Loss 5.99 (LEARNED POORLY)
  
ADAMW vs ADAFACTOR is the real difference, NOT weight scale!
""")

print("\n" + "="*80)
print("CORRECTED CONCLUSION")
print("="*80)

print("""
BOTH AdamW models (FP32 and Ternary) learned the copy task:
  - FP32 AdamW: Loss 0.07, Acc 95%
  - Ternary AdamW (Phase 22): Loss 0.11, Acc 100%

BOTH Adafactor models (FP32 and Ternary) learned POORLY:
  - FP32 Adafactor: Loss 5.99, Acc 16%
  - Ternary Adafactor: Loss 7.94, Acc ~15%

The Ternary Adafactor model has EMB std 0.96 which is 48x larger than FP32 Adafactor,
but this is NOT the primary cause of failure - Adafactor itself performs poorly!
""")

print("="*80)
