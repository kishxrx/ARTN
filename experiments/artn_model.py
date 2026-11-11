#!/usr/bin/env python3
"""
ARTN Model: Adaptive Recursive Ternary Network

Implements the full ARTN architecture per blueprint:
- Stage 4: Recursive parameter tying (1 shared BitLinear block, 6 passes)
- Stage 5: Adaptive halting gate (per-position causal, early stop at inference)
- Stage 7: Ternary BitLinear weights with STE

Architecture:
  token_embedding: nn.Embedding(vocab_size, d_model)
  position_embedding: nn.Embedding(max_seq_len, d_model)
  block: BitLinearBlock (ONE, shared — recursive)
  halting_gate: nn.Linear(d_model, 1)
  ln_f: nn.LayerNorm(d_model)
  lm_head: BitLinear(d_model, vocab_size, bias=False)

Total params: ~2.9M (58% reduction vs 6.85M with 6 separate blocks)

Training (per-position, soft, all passes — causal, differentiable):
  Each position gets its own halt probability from its own representation.
  No future-token leakage. All max_passes execute for exact gradients.

Inference (last-position early stop — genuine compute savings):
  Break loop when last position's cumulative halt crosses threshold.
  Assign remaining mass to current pass (weights sum to 1.0).
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import BitLinearBlock and components from the proven checkpoint code
BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from src.model.bitlinear import BitLinear
from experiments.bitlinear_adamw_controlled import BitLinearBlock


class ARTNGPT(nn.Module):
    """
    Adaptive Recursive Ternary Network (ARTN).

    One shared BitLinearBlock executed recursively (Stage 4 + 7),
    with a per-position adaptive halting gate (Stage 5).

    Training: per-position causal halting, all passes, differentiable.
    Inference: last-position early stop, genuine compute savings.
    """

    def __init__(
        self,
        vocab_size: int = 4000,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,       # maps to max_passes for API consistency
        d_ff: int = 1024,
        max_seq_len: int = 256,
        target_passes: float = 4.0,
        ponder_loss_weight: float = 0.01,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_passes = n_layers
        self.target_passes = target_passes
        self.ponder_loss_weight = ponder_loss_weight

        # Embeddings (not shared)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # ONE shared BitLinearBlock (recursive parameter tying)
        self.block = BitLinearBlock(d_model, n_heads, d_ff)

        # Adaptive halting gate: per-position halt probability
        self.halting_gate = nn.Linear(d_model, 1)

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = BitLinear(d_model, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, BitLinear):
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        """
        Training forward pass: per-position causal halting, all passes.

        Each position gets its own halt probability from its own causal
        representation. No future-token leakage. All max_passes execute
        for exact gradients. No compute savings during training.

        Returns:
            dict with 'logits', 'loss', 'halting_info'
        """
        B, T = input_ids.shape

        # Embed
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb

        # Per-position halting
        remaining = torch.ones(B, T, 1, device=x.device)
        output = torch.zeros(B, T, self.d_model, device=x.device)
        remaining_history = []
        halt_weights = []
        conditional_halt_probs = []

        for pass_idx in range(self.max_passes):
            x = self.block(x)   # shared block, full causal attention

            # Per-position halt probability — each position's own causal representation
            h = torch.sigmoid(self.halting_gate(x))  # (B, T, 1)
            conditional_halt_probs.append(h)

            # Halting weight: p_i = h_i * R_i
            p = h * remaining
            if pass_idx == self.max_passes - 1:
                p = remaining  # force halt on last pass

            output = output + x * p
            halt_weights.append(p)

            remaining = remaining - p
            remaining_history.append(remaining)

        # Final projection
        x = self.ln_f(output)
        logits = self.lm_head(x)

        # Loss
        loss = None
        halting_info = {}
        if targets is not None:
            loss_ce = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-1
            )

            # Ponder loss: global average expected passes (differentiable)
            expected_passes = 1.0 + sum(r.mean() for r in remaining_history)
            loss_ponder = (expected_passes - self.target_passes) ** 2

            loss = loss_ce + self.ponder_loss_weight * loss_ponder

            halting_info['loss_ce'] = loss_ce.item()
            halting_info['loss_ponder'] = loss_ponder.item()
            halting_info['expected_passes'] = expected_passes.item()

        halting_info['conditional_halt_probs'] = conditional_halt_probs
        halting_info['halt_weights'] = halt_weights
        halting_info['remaining_history'] = remaining_history
        halting_info['passes_executed'] = self.max_passes

        return {'logits': logits, 'loss': loss, 'halting_info': halting_info}

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=0.9,
        threshold=1.0,
    ):
        """
        Inference: autoregressive generation with last-position early stop.

        At each generation step:
          1. Run block recursively with per-position halting
          2. Break when last position's cumulative halt >= threshold
          3. Assign remaining mass to current pass (weights sum to 1)
          4. Use last position's logits to predict next token

        threshold=1.0: full computation (all max_passes, no early stop)
        threshold<1.0: genuine early stop, compute savings

        Returns: output token ids tensor
        """
        for _ in range(max_new_tokens):
            B, T = input_ids.shape

            # Embed
            tok_emb = self.token_embedding(input_ids)
            pos_emb = self.position_embedding(
                torch.arange(T, device=input_ids.device)
            )
            x = tok_emb + pos_emb

            # Per-position halting with early stop
            remaining = torch.ones(B, T, 1, device=x.device)
            output = torch.zeros(B, T, self.d_model, device=x.device)

            for pass_idx in range(self.max_passes):
                x = self.block(x)
                h = torch.sigmoid(self.halting_gate(x))  # (B, T, 1)
                p = h * remaining
                if pass_idx == self.max_passes - 1:
                    p = remaining

                output = output + x * p
                remaining = remaining - p

                # Check last position for early stop
                cumulative_halt_last = 1.0 - remaining[0, -1, 0].item()
                if cumulative_halt_last >= threshold and pass_idx < self.max_passes - 1:
                    break

            # Assign remaining mass to current pass (weights sum to 1)
            output = output + x * remaining
            remaining = torch.zeros_like(remaining)

            # Predict next token from last position
            x = self.ln_f(output)
            logits = self.lm_head(x)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self):
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_block_parameters(self):
        """Parameters in the shared block only."""
        return sum(p.numel() for p in self.block.parameters())

    def get_ternary_stats(self):
        """Get statistics about ternary weight distribution."""
        stats = {
            "total_params": 0,
            "ternary_params": 0,
            "weight_stats": [],
        }

        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                if hasattr(module, 'weight') and module.weight is not None:
                    w = module.weight.data
                    stats["total_params"] += w.numel()
                    stats["ternary_params"] += w.numel()

                    # Get quantized weights
                    weight_q, scale = module.quantize_weights()
                    qw = weight_q / scale.clamp(min=1e-8)  # dequantized

                    positive = (qw > 0.1).sum().item()
                    negative = (qw < -0.1).sum().item()
                    zero = qw.numel() - positive - negative

                    stats["weight_stats"].append({
                        "name": name,
                        "shape": list(w.shape),
                        "positive": positive,
                        "negative": negative,
                        "zero": zero,
                        "sparsity": zero / qw.numel() if qw.numel() > 0 else 0,
                    })

        return stats
