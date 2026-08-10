"""
Adaptive Recursive Transformer for ARTN.

Combines:
- Recursive parameter tying (shared block)
- Adaptive halting (variable depth)
- Memory-efficient for 4GB constraint
"""

from typing import Optional

import torch
import torch.nn as nn

from src.model.embeddings import Embeddings
from src.model.transformer_block import TransformerBlock
from src.model.halting import AdaptiveHaltingSimple, create_adaptive_halting


class AdaptiveGPT(nn.Module):
    """
    GPT with adaptive halting and recursive parameter tying.
    
    Architecture:
    - Embeddings (token + position)
    - Single transformer block (shared across passes)
    - Adaptive halting gate
    - Output projection
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_passes: Maximum recursive passes
        target_passes: Target average passes
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_passes: int = 8,
        target_passes: float = 4.0,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_passes = max_passes
        self.target_passes = target_passes
        
        # Embeddings
        self.embeddings = Embeddings(vocab_size, d_model, max_seq_len, dropout)
        
        # Single block (shared across passes)
        self.block = TransformerBlock(d_model, n_heads, d_ff, dropout)
        
        # Adaptive halting
        self.halting = create_adaptive_halting(
            d_model,
            max_passes=max_passes,
            target_passes=target_passes,
            threshold=0.3,  # Lower threshold = more likely to halt early
        )
        
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
        return_info: bool = False,
    ) -> dict:
        """
        Forward pass with adaptive halting.
        
        Args:
            token_ids: Input tokens (batch, seq_len)
            targets: Optional targets for loss
            return_info: Return halting info
            
        Returns:
            Dict with 'logits', 'loss', and optionally 'info'
        """
        batch_size, seq_len = token_ids.size()
        
        # Embed
        x = self.embeddings(token_ids)
        
        # Adaptive halting loop
        x, halt_info = self.halting(x, self.block)
        
        # Final norm + projection
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        # Loss computation
        loss = None
        if targets is not None:
            # Task loss
            task_loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            
            # Ponder loss (encourage target_passes)
            ponder_loss = self.halting.compute_ponder_loss(halt_info['avg_halt_prob'])
            
            # Combined loss (small weight for ponder)
            loss = task_loss + 0.01 * ponder_loss
        
        result = {'logits': logits, 'loss': loss}
        
        if return_info:
            result['info'] = halt_info
        
        return result
    
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
            # Truncate if needed
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
        """Estimate Float32 memory in MB."""
        return self.count_parameters() * 4 / (1024 * 1024)


def create_adaptive_gpt(config: Optional[dict] = None) -> AdaptiveGPT:
    """
    Factory function to create adaptive GPT.
    
    Args:
        config: Config dict with model parameters
        
    Returns:
        AdaptiveGPT instance
    """
    default_config = {
        'vocab_size': 4000,
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 1024,
        'max_passes': 8,
        'target_passes': 4.0,
        'max_seq_len': 256,
        'dropout': 0.1,
    }
    if config is None:
        config = {}
    cfg = {**default_config, **config}
    
    return AdaptiveGPT(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        d_ff=cfg['d_ff'],
        max_passes=cfg['max_passes'],
        target_passes=cfg['target_passes'],
        max_seq_len=cfg['max_seq_len'],
        dropout=cfg['dropout'],
    )
