"""
Feed-Forward Network module for ARTN.

Implements the position-wise FFN used in transformer blocks.

Math:
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    
Where:
    - W1: (d_model, d_ff), typically d_ff = 4 * d_model
    - W2: (d_ff, d_model)
    - GELU: Gaussian Error Linear Unit, smoother than ReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Two linear transformations with GELU activation.
    Applied to each position independently.
    
    Args:
        d_model: Model dimension (input/output size)
        d_ff: Hidden dimension, typically 4 * d_model
        dropout: Dropout rate after hidden layer
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # First linear: d_model → d_ff
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        
        # Second linear: d_ff → d_model
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
        
        # Dropout after GELU
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # x: (batch, seq, d_model)
        
        # Project up to d_ff
        x = self.W1(x)
        
        # Apply GELU activation
        # GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
        # PyTorch's GELU uses the exact formulation
        x = F.gelu(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Project back down to d_model
        x = self.W2(x)
        
        return x


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU variant of Feed-Forward Network (optional advanced version).
    
    Used in LLaMA and other modern transformers.
    
    Math:
        FFN_SwiGLU(x) = Swish(W1(x)) ⊙ (W2(x))
    
    Where:
        - Swish/SiLU = x * sigmoid(x)
        - ⊙ = element-wise multiplication
        - Two gates instead of one expansion
    
    Note:
        This is more memory efficient but we'll use standard FFN for baseline.
        Included here for reference/future stages.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # SwiGLU uses two projections for gating
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)
        self.W_out = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate: silu(W1(x))
        gate = F.silu(self.W1(x))
        # Value: W2(x)
        value = self.W2(x)
        # Combine: gate * value
        x = gate * value
        x = self.dropout(x)
        x = self.W_out(x)
        return x
