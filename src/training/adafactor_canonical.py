"""
Canonical Adafactor Implementation - Matching PyTorch Reference
================================================================

Based on Shazeer & Stern 2018 Algorithm 4 and torch.optim.Adafactor.

KEY FEATURES:
- NO momentum (beta1 = None) - canonical Adafactor
- RELATIVE step size: αt = max(eps2, RMS(weight)) × lr
- RMS clipping: if RMS(update) > d, scale by d/RMS(update)
- Factored second moment for 2D parameters
- Full second moment for 1D parameters

DO NOT MODIFY existing implementations - this is a NEW file.
"""

from typing import Optional, Iterable
import torch
import torch.nn as nn
from torch.optim import Optimizer


class CanonicalAdafactor(Optimizer):
    """
    Canonical Adafactor optimizer matching PyTorch's torch.optim.Adafactor.
    
    Reference:
    - Shazeer & Stern 2018: "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
    - PyTorch implementation: torch.optim.Adafactor
    
    Key differences from Adam:
    - NO momentum (β1 = 0) by default
    - RELATIVE step size based on parameter RMS
    - Factored second moment for memory efficiency
    - RMS-based clipping
    
    Args:
        params: Model parameters
        lr: Relative step size ρ (default: 1e-3)
        beta2_decay: Decay rate for β2 scheduling (default: -0.8)
                     β2t = 1 - t^beta2_decay (paper uses -0.8)
        eps: Tuple of (eps1, eps2) for numerical stability
        d: Clipping threshold for RMS of update (default: 1.0)
        weight_decay: L2 regularization (default: 0.0)
    """
    
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        beta2_decay: float = -0.8,
        eps: tuple = (1e-30, 1e-3),
        d: float = 1.0,
        weight_decay: float = 0.0,
    ):
        defaults = {
            'lr': lr,
            'beta2_decay': beta2_decay,
            'eps': eps,
            'd': d,
            'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """
        Perform single optimization step.
        
        Returns:
            Loss if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    if grad.dim() == 2:
                        # Factored second moment for 2D
                        state['exp_avg_sq_row'] = torch.zeros(
                            grad.shape[0], dtype=grad.dtype, device=grad.device
                        )
                        state['exp_avg_sq_col'] = torch.zeros(
                            grad.shape[1], dtype=grad.dtype, device=grad.device
                        )
                    else:
                        # Full second moment for 1D
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                
                state['step'] += 1
                step = state['step']
                
                # Get hyperparameters
                lr = group['lr']
                beta2_decay = group['beta2_decay']
                eps1, eps2 = group['eps']
                d = group['d']
                weight_decay = group['weight_decay']
                
                # Compute β2t with optional decay scheduling
                # Paper: β2t = 1 - t^beta2_decay
                if beta2_decay < 0:
                    beta2 = 1.0 - step ** beta2_decay
                    beta2 = max(0.0, min(1.0, beta2))
                else:
                    beta2 = beta2_decay  # Fixed value if positive
                
                # Weight decay
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                
                # Compute second moment
                grad_sq = grad * grad
                
                if grad.dim() == 2:
                    # Factored representation for 2D
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    
                    # Update row and column sums
                    exp_avg_sq_row.mul_(beta2).add_(
                        grad_sq.mean(dim=1), alpha=1 - beta2
                    )
                    exp_avg_sq_col.mul_(beta2).add_(
                        grad_sq.mean(dim=0), alpha=1 - beta2
                    )
                    
                    # Factored variance: V̂t = Rt × Ct / mean(Rt)
                    # This matches PyTorch's implementation
                    row_mean = exp_avg_sq_row.mean()
                    factored_var = (
                        exp_avg_sq_row.unsqueeze(1) * 
                        exp_avg_sq_col.unsqueeze(0) / 
                        row_mean.add(eps1)
                    )
                    
                    # Update = grad / sqrt(variance + eps2)
                    update = grad / (factored_var.sqrt() + eps2)
                else:
                    # Full second moment for 1D
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    update = grad / (exp_avg_sq.sqrt() + eps2)
                
                # RMS clipping
                update_rms = (update ** 2).mean().sqrt()
                if update_rms > d:
                    update.mul_(d / update_rms)
                
                # Relative step size: αt = max(eps2, RMS(weight)) × lr
                weight_rms = (p ** 2).mean().sqrt()
                alpha_t = max(eps2, weight_rms.item()) * lr
                
                # Apply update
                p.add_(update, alpha=-alpha_t)
        
        return loss
