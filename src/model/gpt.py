"""
GPT Language Model for ARTN.

Complete GPT decoder architecture:
- Token Embedding + Positional Encoding
- N Transformer Blocks
- Final Layer Norm
- Output Head (project to vocabulary)

This is the Float32 baseline model.
"""

import torch
import torch.nn as nn

from src.model.embeddings import Embeddings
from src.model.transformer_block import TransformerBlock


class GPT(nn.Module):
    """
    GPT-style Decoder-Only Transformer.
    
    For Stage 2 baseline, this is a standard Float32 architecture.
    No quantization, no recursion, no adaptive halting.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    
    Config from experiments/001_baseline/config.yaml:
        vocab_size: 4000
        d_model: 256
        n_heads: 8
        n_layers: 6
        d_ff: 1024
        max_seq_len: 256
        dropout: 0.1
    
    Estimated parameters: ~6.77M
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Embeddings: token + positional
        self.embeddings = Embeddings(vocab_size, d_model, max_seq_len, dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm (before output head)
        self.ln_final = nn.LayerNorm(d_model)
        
        # Output head: project to vocabulary
        # Weight tying: reuse embedding weights (optional, saves params)
        # For baseline, we use a separate linear
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights.
        
        Standard initialization for transformers:
        - Linear: normal(0, 0.02)
        - Embedding: normal(0, 0.02)
        - LayerNorm: weight=1, bias=0
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        token_ids: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass through GPT.
        
        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)
            targets: Optional target token IDs for loss computation
                     Shape: (batch_size, seq_len)
        
        Returns:
            Dictionary with:
                - 'logits': Output logits of shape (batch, seq_len, vocab_size)
                - 'loss': Cross-entropy loss if targets provided, else None
        """
        batch_size, seq_len = token_ids.size()
        
        # Check sequence length
        assert seq_len <= self.max_seq_len, \
            f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
        
        # Embed tokens + add positional encoding
        # x: (batch, seq_len, d_model)
        x = self.embeddings(token_ids)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Project to vocabulary
        # logits: (batch, seq_len, vocab_size)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Cross-entropy loss
            # Reshape for CrossEntropyLoss: (batch * seq_len, vocab_size)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,  # Ignore special padding tokens
            )
        
        return {'logits': logits, 'loss': loss}
    
    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        This is a basic generation for testing.
        Full generation with all sampling options in Stage 8.
        
        Args:
            token_ids: Prompt tokens of shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
        
        Returns:
            Generated token sequence including prompt
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = token_ids if token_ids.size(1) <= self.max_seq_len \
                       else token_ids[:, -self.max_seq_len:]
            
            # Forward pass
            outputs = self(idx_cond)
            logits = outputs['logits']
            
            # Get logits for last position
            logits = logits[:, -1, :]  # (batch, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Convert to probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            token_ids = torch.cat([token_ids, next_token], dim=1)
        
        return token_ids
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_memory_mb(self) -> float:
        """
        Estimate memory needed for weights in Float32.
        
        Returns:
            Memory in MB
        """
        params = self.count_parameters()
        bytes_per_param = 4  # Float32
        return params * bytes_per_param / (1024 * 1024)


def create_baseline_gpt(config: dict | None = None) -> GPT:
    """
    Factory function to create baseline GPT from config dict.
    
    Args:
        config: Config dict with keys:
            - vocab_size
            - d_model
            - n_heads
            - n_layers
            - d_ff
            - max_seq_len
            - dropout
        If None, uses default values.
    
    Returns:
        GPT model instance
    """
    default_config = {
        'vocab_size': 4000,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'max_seq_len': 256,
        'dropout': 0.1,
    }
    if config is None:
        config = {}
    cfg = {**default_config, **config}
    
    return GPT(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        d_ff=cfg['d_ff'],
        max_seq_len=cfg['max_seq_len'],
        dropout=cfg['dropout'],
    )
