import torch
from torch.optim import Adafactor
import inspect

print('PyTorch Adafactor signature:')
sig = inspect.signature(Adafactor.__init__)
for name, param in sig.parameters.items():
    if name != 'params':
        default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
        print(f'  {name}: {default}')
