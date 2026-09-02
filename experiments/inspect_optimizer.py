#!/usr/bin/env python3
import torch

checkpoint = torch.load('outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt', map_location='cpu')

opt_state = checkpoint['optimizer_state_dict']

print('=== OPTIMIZER PARAM GROUPS ===')
for i, pg in enumerate(opt_state.get('param_groups', [])):
    print(f'\nParam group {i}:')
    for k, v in pg.items():
        if k == 'params':
            print(f'  {k}: [{len(v)} param IDs]')
        else:
            print(f'  {k}: {v}')

print('\n=== STATE SAMPLE ===')
state = opt_state.get('state', {})
print(f'Total state entries: {len(state)}')
if state:
    first_key = list(state.keys())[0]
    print(f'\nFirst state entry (key={first_key}):')
    for k, v in state[first_key].items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: tensor {v.shape}')
        else:
            print(f'  {k}: {type(v).__name__}')
