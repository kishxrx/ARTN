import torch
c=torch.load('experiments/results/fp32_control_20260819_232646/step_5000.pt',map_location='cpu')
e=c['model_state_dict']['token_embedding.weight']
print(f'FP32 AdamW emb std: {e.std():.4f}')
