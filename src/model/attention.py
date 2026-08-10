"""
Multi-Head Self-Attention module for ARTN.

Implements causal (masked) multi-head attention.

Math:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V
    
    Where:
    - Q = Query: "What am I looking for?"
    - K = Key: "What do I contain?"
    - V = Value: "What will I output?"
    - d_k = Dimension per head (d_model / n_heads)
    
Multi-Head:
    Instead of one big attention, run h smaller attentions in parallel,
    then concatenate and project.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with causal masking.
    
    Args:
        d_model: Model dimension (input/output size)
        n_heads: Number of attention heads
        dropout: Dropout rate after attention
    
    Note:
        d_model must be divisible by n_heads.
        Each head operates on d_model / n_heads dimensions.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        # Each projects from d_model to d_model
        # We'll split the output into heads later
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection (after concatenating heads)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Scaling factor: 1/sqrt(d_k)
        self.scale = 1.0 / math.sqrt(self.d_k)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (seq_len, seq_len)
                  Positions with 1 are masked (not attended to)
                  Positions with 0 are kept
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Step 1: Linear projections
        # Q, K, V: (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Step 2: Reshape for multi-head attention
        # Split d_model into n_heads × d_k
        # Shape: (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Step 3: Compute attention scores
        # Q @ K^T: (batch, n_heads, seq_len, d_k) @ (batch, n_heads, d_k, seq_len)
        #        = (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Scale by sqrt(d_k)
        scores = scores * self.scale
        
        # Step 4: Apply causal mask
        if mask is not None:
            # Mask shape: (seq_len, seq_len)
            # We expand it to match scores: (1, 1, seq_len, seq_len)
            # Fill masked positions with -inf so softmax outputs 0 there
            scores = scores.masked_fill(mask == 1, float('-inf'))
        
        # Step 5: Softmax over last dimension (keys)
        # attention_weights: (batch, n_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Handle case where all positions are masked (softmax returns NaN)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Weighted sum of values
        # attention_weights @ V: (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, d_k)
        #                      = (batch, n_heads, seq_len, d_k)
        output = torch.matmul(attention_weights, V)
        
        # Step 7: Concatenate heads
        # Reshape: (batch, n_heads, seq_len, d_k) → (batch, seq_len, n_heads, d_k)
        #         → (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Step 8: Final projection
        output = self.W_o(output)
        
        return output


def generate_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate a causal attention mask.
    
    The mask prevents positions from attending to future positions.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
    
    Returns:
        Mask tensor of shape (seq_len, seq_len)
        1 = masked (cannot attend), 0 = not masked (can attend)
    
    Example for seq_len=4:
        [[0, 1, 1, 1],    ← Position 0 can only see position 0
         [0, 0, 1, 1],    ← Position 1 can see positions 0-1
         [0, 0, 0, 1],    ← Position 2 can see positions 0-2
         [0, 0, 0, 0]]    ← Position 3 can see all positions
    """
    # Create upper triangular matrix of ones
    # diagonal=1 means start from 1 above main diagonal
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.int()


class CausalSelfAttention(nn.Module):
    """
    Convenience wrapper that automatically applies causal masking.
    
    Same as MultiHeadAttention but always uses causal mask.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal self-attention.
        
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        mask = generate_causal_mask(seq_len, device=x.device)
        return self.attention(x, mask)
