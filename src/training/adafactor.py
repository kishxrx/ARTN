"""
Adafactor optimizer for ARTN.

Memory-efficient optimizer that uses ~60% less memory than AdamW.
Factors the second moment matrix into row/column sums.

Math:
    Standard Adam/AdamW momentum: stores full R×C matrix per parameter
    Adafactor: stores R + C values (row and column sums)
    
For 256×1024 matrix:
    Adam: 262,144 values
    Adafactor: 1,280 values (~200× reduction)
"""

from typing import Optional, Iterable

import torch
import torch.nn as nn
from torch.optim import Optimizer


class Adafactor(Optimizer):
    """
    Adafactor optimizer implementation.
    
    Reference: Adafactor: Adaptive Learning Rates with Sublinear Memory Cost
    (Shazeer & Stern, 2018)
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 1e-3)
        beta1: First moment decay (default: 0.9, use None to disable)
        beta2: Second moment decay (default: 0.999)
        eps1: Regularization for second moment (default: 1e-30)
        eps2: Epsilon in update denominator (default: 1e-3)
        weight_decay: L2 regularization (default: 0.0)
        clip_threshold: Gradient clipping threshold (default: 1.0)
    
    Memory savings vs Adam:
        Adam: 2 × parameters (m + v)
        Adafactor: 2 × (row_sum + col_sum) per 2D parameter
                  or 1 × parameter for 1D
    
    Note:
        PyTorch provides Adafactor in torch.optim, but this
        implementation is for educational purposes and customization.
    """
    
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        beta1: Optional[float] = 0.9,
        beta2: float = 0.999,
        eps1: float = 1e-30,
        eps2: float = 1e-3,
        weight_decay: float = 0.0,
        clip_threshold: float = 1.0,
    ):
        defaults = {
            'lr': lr,
            'beta1': beta1,
            'beta2': beta2,
            'eps1': eps1,
            'eps2': eps2,
            'weight_decay': weight_decay,
            'clip_threshold': clip_threshold,
        }
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns loss
        
        Returns:
            Loss value if closure provided, else None
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
                    
                    # For 2D parameters: use row/col factored moments
                    if grad.dim() == 2:
                        state['exp_avg_sq_row'] = torch.zeros(
                            grad.shape[0], dtype=grad.dtype, device=grad.device
                        )
                        state['exp_avg_sq_col'] = torch.zeros(
                            grad.shape[1], dtype=grad.dtype, device=grad.device
                        )
                    else:
                        # For 1D parameters: use standard moments
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                    
                    # First moment (optional)
                    if group['beta1'] is not None:
                        state['exp_avg'] = torch.zeros_like(grad)
                
                state['step'] += 1
                step = state['step']
                
                beta1 = group['beta1']
                beta2 = group['beta2']
                eps1 = group['eps1']
                eps2 = group['eps2']
                lr = group['lr']
                weight_decay = group['weight_decay']
                clip_threshold = group['clip_threshold']
                
                # Apply weight decay
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                
                # Compute second moment estimate
                if grad.dim() == 2:
                    # Factored representation for 2D
                    grad_sq = grad * grad
                    
                    # Row average
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_row.mul_(beta2).add_(
                        grad_sq.mean(dim=1), alpha=1 - beta2
                    )
                    
                    # Column average
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    exp_avg_sq_col.mul_(beta2).add_(
                        grad_sq.mean(dim=0), alpha=1 - beta2
                    )
                    
                    # Factorized estimate: row × col / global_mean
                    global_mean = grad_sq.mean()
                    numerator = exp_avg_sq_row.unsqueeze(1) * exp_avg_sq_col.unsqueeze(0)
                    factored_var = numerator / global_mean.add(eps1)
                    update = grad / (factored_var.sqrt() + eps2)
                    
                else:
                    # Standard update for 1D
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    update = grad / (exp_avg_sq.sqrt() + eps2)
                
                # Clip update
                update_norm = update.norm()
                if update_norm > clip_threshold:
                    update.mul_(clip_threshold / update_norm)
                
                # First moment (momentum)
                if beta1 is not None:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    bias_correction = 1 - beta1 ** step
                    update = exp_avg / bias_correction
                
                # Apply update
                p.add_(update, alpha=-lr)
        
        return loss


def create_adafactor(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs,
) -> Adafactor:
    """
    Factory function to create Adafactor optimizer.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: L2 regularization
        **kwargs: Additional Adafactor parameters
    
    Returns:
        Adafactor optimizer instance
    """
    return Adafactor(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        **kwargs,
    )
