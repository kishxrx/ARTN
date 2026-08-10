"""
Embeddings module for ARTN.

Implements:
- TokenEmbedding: Learnable lookup table for token vectors
- PositionalEncoding: Sinusoidal position signals

Math:
    Token Embed: E[token_id] → vector of shape (d_model,)
    Positional: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
                PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Learnable token embeddings.
    
    Each token ID maps to a learnable vector.
    
    Args:
        vocab_size: Size of vocabulary (number of unique tokens)
        d_model: Dimension of embedding vectors
    
    Shape:
        Input: (batch_size, seq_len) - token IDs
        Output: (batch_size, seq_len, d_model) - embedded vectors
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for each token.
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Embedded vectors of shape (batch_size, seq_len, d_model)
        """
        # nn.Embedding returns shape (batch, seq, d_model)
        # We scale by sqrt(d_model) as in original transformer paper
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    
    Adds position information to embeddings.
    Uses sin/cos functions so model can learn relative positions.
    
    Not learned - computed once and registered as buffer.
    
    Args:
        d_model: Dimension of model (must match embedding dim)
        max_seq_len: Maximum sequence length to support
        dropout: Dropout rate applied after adding positional encoding
    
    Shape:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        # Shape: (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        
        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Divisor term: 10000^(2i/d_model) = exp(2i * -log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dims
        
        # Add batch dimension: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a learnable parameter, but saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output with positional encoding added, same shape as input
        """
        # x: (batch, seq_len, d_model)
        # pe: (1, max_seq_len, d_model)
        # We only use the first seq_len positions
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class Embeddings(nn.Module):
    """
    Combined token embedding + positional encoding.
    
    This is the complete input embedding pipeline.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        max_seq_len: int = 512, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Transform token IDs to embeddings with position info.
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        x = self.token_embedding(token_ids)
        x = self.positional_encoding(x)
        return x
