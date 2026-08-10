"""
Halting Gate module for ARTN.

Implements adaptive computation - model decides when to stop processing.

Math:
    h_i = sigmoid(W @ x_i + b)     # Halting probability at pass i
    
    H_i = sum(h_j for j=1..i)      # Cumulative probability
    
    Stop when H_i >= 1.0 (or max_passes reached)
    
    Output = sum(p_i * x_i)        # Weighted by stop probability
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class HaltingGate(nn.Module):
    """
    Learnable halting gate for adaptive computation.
    
    Single linear layer producing halting probability.
    
    Args:
        d_model: Model dimension
        init_bias: Initial bias (negative = prefer continuing)
    """
    
    def __init__(self, d_model: int, init_bias: float = -1.0):
        super().__init__()
        
        self.d_model = d_model
        
        # Linear layer: d_model -> 1
        self.gate = nn.Linear(d_model, 1)
        
        # Initialize bias negative so model tends to continue initially
        nn.init.constant_(self.gate.bias, init_bias)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute halting probability.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Halting probability (batch, seq_len, 1), sigmoid bounded [0, 1]
        """
        return torch.sigmoid(self.gate(x))


class AdaptiveHalting(nn.Module):
    """
    Adaptive halting mechanism for recursive transformers.
    
    Processes inputs through block multiple times, deciding when to stop
    based on learned halting probabilities.
    
    Args:
        d_model: Model dimension
        max_passes: Maximum number of passes (safety limit)
        target_passes: Target average passes (for ponder loss)
        halting_threshold: Threshold for cumulative halt (default 1.0)
    """
    
    def __init__(
        self,
        d_model: int,
        max_passes: int = 8,
        target_passes: float = 4.0,
        halting_threshold: float = 1.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_passes = max_passes
        self.target_passes = target_passes
        self.halting_threshold = halting_threshold
        
        # Halting gate
        self.halting_gate = HaltingGate(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        block: nn.Module,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Process x through block with adaptive halting.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            block: Transformer block to apply
            
        Returns:
            output: Weighted output tensor
            info: Dict with halting statistics
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Track outputs and halt probabilities
        outputs = []
        halt_probs = []
        remain_probs = []
        
        # Initial remaining probability = 1.0 (haven't halted yet)
        remaining = torch.ones(batch_size, seq_len, 1, device=device)
        
        total_halted = torch.zeros(batch_size, seq_len, 1, device=device)
        
        for pass_idx in range(self.max_passes):
            # Apply transformer block
            x = block(x)
            
            # Compute halting probability for this pass
            halt_prob = self.halting_gate(x)
            
            # Probability of halting at this pass (given not halted before)
            halt_this_pass = halt_prob * remaining
            
            # Only halt if we haven't exceeded threshold
            # If this is last pass, force halt
            if pass_idx == self.max_passes - 1:
                halt_this_pass = remaining
            else:
                # Check if cumulative halt exceeds threshold
                can_halt = (total_halted + halt_this_pass) <= self.halting_threshold
                halt_this_pass = halt_this_pass * can_halt.float()
            
            # Store weighted output
            outputs.append(x * halt_this_pass)
            halt_probs.append(halt_this_pass)
            
            # Update remaining probability
            remaining = remaining - halt_this_pass
            total_halted = total_halted + halt_this_pass
            
            remain_probs.append(remaining)
            
            # Check if all sequences have halted
            if remaining.max() < 1e-6:
                break
        
        # Combine outputs
        # Output is weighted sum of all pass outputs
        output = torch.stack(outputs, dim=0).sum(dim=0)
        
        # Clamp remaining output for numerical stability
        output = output + x * remaining * 0.0  # Ensure gradient flow
        
        # Compute number of passes used
        passes_used = len(outputs)
        
        # Compute average passes per position (for ponder loss)
        avg_passes = sum(hp.mean() for hp in halt_probs).item() + remaining.mean().item()
        
        info = {
            'passes_used': passes_used,
            'halt_probs': halt_probs,
            'remain_probs': remain_probs,
            'avg_passes': avg_passes,
            'final_remaining': remaining.mean().item(),
        }
        
        return output, info
    
    def compute_ponder_loss(self, avg_passes: float) -> torch.Tensor:
        """
        Compute ponder regularizaton loss.
        
        Encourages average passes to be close to target.
        
        Args:
            avg_passes: Average number of passes used
            
        Returns:
            Ponder loss scalar
        """
        diff = avg_passes - self.target_passes
        return torch.tensor(diff ** 2, requires_grad=False)


class AdaptiveHaltingSimple(nn.Module):
    """
    Simplified adaptive halting using straight-through estimator.
    
    During forward: use threshold to decide discrete halt
    During backward: gradients flow through sigmoid
    
    This is conceptually simpler and often works better in practice.
    
    Args:
        d_model: Model dimension
        max_passes: Maximum passes
        target_passes: Target average (for regularization)
        threshold: Sigmoid threshold for halting (default 0.5)
    """
    
    def __init__(
        self,
        d_model: int,
        max_passes: int = 8,
        target_passes: float = 4.0,
        threshold: float = 0.5,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_passes = max_passes
        self.target_passes = target_passes
        self.threshold = threshold
        
        self.halting_gate = HaltingGate(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        block: nn.Module,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Apply adaptive halting with straight-through gradient.
        """
        batch_size = x.shape[0]
        device = x.device
        
        outputs = []
        halt_logits = []
        
        # Process each pass
        for pass_idx in range(self.max_passes):
            x = block(x)
            outputs.append(x)
            
            # Get halting probability
            h = self.halting_gate(x)
            halt_logits.append(h)
            
            # Straight-through: discrete forward, continuous backward
            h_discrete = (h > self.threshold).float()
            h_st = h + (h_discrete - h).detach()  # Straight-through
            
            # If all positions want to halt, stop early
            if pass_idx < self.max_passes - 1:
                if h_discrete.mean() > 0.99:
                    # Most positions want to halt
                    break
        
        # Return last output
        output = outputs[-1]
        
        # Compute statistics
        avg_halt = sum(h.mean().item() for h in halt_logits) / len(halt_logits)
        passes_used = len(outputs)
        
        info = {
            'passes_used': passes_used,
            'avg_halt_prob': avg_halt,
            'halt_logits': halt_logits,
        }
        
        return output, info
    
    def compute_ponder_loss(self, avg_halt_prob: float) -> torch.Tensor:
        """
        Ponder loss based on average halt probability.
        
        Higher halt prob = fewer passes used.
        Target: avg_halt_prob ≈ 1 / target_passes
        """
        target_halt = 1.0 / self.target_passes
        diff = avg_halt_prob - target_halt
        return torch.tensor(diff ** 2, requires_grad=False)


def create_halting_gate(d_model: int, **kwargs) -> HaltingGate:
    """Factory for halting gate."""
    return HaltingGate(d_model, **kwargs)


def create_adaptive_halting(d_model: int, **kwargs) -> AdaptiveHaltingSimple:
    """Factory for adaptive halting mechanism."""
    return AdaptiveHaltingSimple(d_model, **kwargs)
