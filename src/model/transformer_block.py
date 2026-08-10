"""
Transformer Block module for ARTN.

Combines all components into a complete transformer block:
- Layer Normalization
- Causal Self-Attention
- Residual Connection
- Layer Normalization
- Feed-Forward Network
- Residual Connection

Architecture (Pre-Norm variant):
    output = x + Attention(LayerNorm(x))
    output = output + FFN(LayerNorm(output))

Pre-Norm (normalize before) is more stable than Post-Norm (normalize after).
"""

import torch
import torch.nn as nn

from src.model.attention import CausalSelfAttention
from src.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Single Transformer Decoder Block.
    
    Pre-norm architecture with residual connections.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Layer norms (pre-norm architecture)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Attention
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Attention sublayer with residual
        # Pre-norm: normalize BEFORE attention
        residual = x
        x = self.ln1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = residual + x
        
        # FFN sublayer with residual
        # Pre-norm: normalize BEFORE FFN
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x


class TransformerBlockPostNorm(nn.Module):
    """
    Post-Norm variant of Transformer Block (for comparison).
    
    Architecture:
        output = LayerNorm(x + Attention(x))
        output = LayerNorm(output + FFN(output))
    
    Note:
        Pre-norm is more stable for training. This is included
        for reference/comparison, not used in baseline.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-norm: normalize AFTER attention+residual
        x = self.ln1(x + self.dropout(self.attention(x)))
        x = self.ln2(x + self.dropout(self.ffn(x)))
        return x
