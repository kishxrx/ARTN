"""
Recursive Transformer module for ARTN.

Implements ALBERT-style parameter tying - single transformer block
looped multiple times instead of multiple distinct blocks.

Memory savings: N× block params → 1× block params
"""

from typing import Optional

import torch
import torch.nn as nn

from src.model.embeddings import Embeddings
from src.model.transformer_block import TransformerBlock


class RecursiveTransformer(nn.Module):
    """
    Recursive (parameter-tying) Transformer.
    
    Single block is executed multiple times sequentially.
    Same weights, different input at each pass.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        num_passes: Number of times to loop through block (depth)
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    
    Comparing to baseline:
        Baseline (6 blocks): 6 × block_params
        Recursive (6 passes): 1 × block_params
        Memory savings: ~83% for blocks
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_passes: int = 6,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_passes = num_passes
        
        # Embeddings (not shared)
        self.embeddings = Embeddings(vocab_size, d_model, max_seq_len, dropout)
        
        # Single transformer block (shared across all passes)
        self.block = TransformerBlock(d_model, n_heads, d_ff, dropout)
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(d_model)
        
        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights."""
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
        targets: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass through recursive transformer.
        
        Args:
            token_ids: Input token IDs (batch, seq_len)
            targets: Optional target IDs for loss computation
        
        Returns:
            Dict with 'logits' and 'loss' (if targets provided)
        """
        batch_size, seq_len = token_ids.size()
        
        # Embed tokens
        x = self.embeddings(token_ids)
        
        # Pass through block multiple times (recursive)
        for pass_idx in range(self.num_passes):
            x = self.block(x)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Output projection
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        
        return {'logits': logits, 'loss': loss}
    
    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = token_ids if token_ids.size(1) <= self.embeddings.positional_encoding.pe.size(1) \
                       else token_ids[:, -self.embeddings.positional_encoding.pe.size(1):]
            
            outputs = self(idx_cond)
            logits = outputs['logits'][:, -1, :]
            
            if temperature > 0:
                logits = logits / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            token_ids = torch.cat([token_ids, next_token], dim=1)
        
        return token_ids
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def estimate_memory_mb(self) -> float:
        """Estimate memory in Float32 MB."""
        return self.count_parameters() * 4 / (1024 * 1024)
    
    def get_block_parameter_count(self) -> int:
        """Count parameters in just the shared block."""
        return sum(p.numel() for p in self.block.parameters())


class RecursiveGPT(nn.Module):
    """
    Alternative interface matching GPT class name.
    
    Wrapper around RecursiveTransformer for API consistency.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,  # Maps to num_passes for API consistency
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = RecursiveTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_passes=n_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        # Expose common attributes
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
    
    def forward(self, token_ids, targets=None):
        return self.transformer(token_ids, targets)
    
    def generate(self, token_ids, max_new_tokens, **kwargs):
        return self.transformer.generate(token_ids, max_new_tokens, **kwargs)
    
    def count_parameters(self):
        return self.transformer.count_parameters()
    
    def estimate_memory_mb(self):
        return self.transformer.estimate_memory_mb()


def create_recursive_gpt(config: Optional[dict] = None) -> RecursiveGPT:
    """
    Factory function to create recursive GPT.
    
    Args:
        config: Config dict (same keys as baseline GPT)
    
    Returns:
        RecursiveGPT instance
    """
    default_config = {
        'vocab_size': 4000,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,  # num_passes
        'd_ff': 1024,
        'max_seq_len': 256,
        'dropout': 0.1,
    }
    if config is None:
        config = {}
    cfg = {**default_config, **config}
    
    return RecursiveGPT(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        d_ff=cfg['d_ff'],
        max_seq_len=cfg['max_seq_len'],
        dropout=cfg['dropout'],
    )
