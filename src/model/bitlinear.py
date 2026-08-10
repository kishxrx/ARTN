"""
BitLinear: 1.58-bit Ternary Quantization for Neural Networks.

Implements BitNet b158-style quantization where weights are constrained
to {-1, 0, +1} using Straight-Through Estimator (STE) for gradient flow.

This is the core innovation enabling extreme memory compression while
maintaining model quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math


class TernarySTE(torch.autograd.Function):
    """
    Straight-Through Estimator for Ternary Quantization.
    
    Forward: Quantize weights to {-1, 0, +1}
    Backward: Pass gradient through unchanged
    
    This enables gradient-based optimization despite the discrete
    quantization operation.
    """
    
    @staticmethod
    def forward(
        ctx, 
        weight: Tensor, 
        scale: Tensor,
        threshold: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """
        Quantize weight to ternary values.
        
        Args:
            weight: Input weight tensor
            scale: Scale factor (abs-mean of weight)
            threshold: Quantization threshold (default 0.5)
        
        Returns:
            Tuple of (quantized_weight, scale)
        """
        # Scale and quantize
        weight_scaled = weight / scale.clamp(min=1e-8)
        
        # Ternary quantization
        #   |x| < 0.5 → 0
        #   x >= 0.5 → 1
        #   x <= -0.5 → -1
        weight_quantized = torch.zeros_like(weight)
        weight_quantized[weight_scaled > threshold] = 1.0
        weight_quantized[weight_scaled < -threshold] = -1.0
        
        # Save for backward (though we use STE, save for debugging)
        ctx.save_for_backward(weight, scale)
        
        return weight_quantized, scale
    
    @staticmethod
    def backward(
        ctx, 
        grad_quantized: Tensor, 
        grad_scale: Tensor
    ) -> Tuple[Tensor, Tensor, None]:
        """
        Backward pass - STE: pass gradient through unchanged.
        
        The gradient for scale is None (we don't learn scale directly).
        The gradient for weight is the same as gradient for quantized weight.
        """
        # STE: gradient passes through unchanged
        return grad_quantized, None, None


def compute_abs_mean_scale(weight: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute scale using abs-mean method.
    
    This is the core scaling strategy from BitNet b158:
    scale = mean(|weight|)
    
    Args:
        weight: Weight tensor
        eps: Minimum value to prevent division by zero
    
    Returns:
        Scale tensor
    """
    return weight.abs().mean().clamp(min=eps)


def compute_per_channel_scale(weight: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute per-channel scale (optional, for better quality).
    
    Args:
        weight: Weight tensor (out_features, in_features)
        eps: Minimum value
    
    Returns:
        Scale tensor (out_features, 1)
    """
    return weight.abs().mean(dim=1, keepdim=True).clamp(min=eps)


class BitLinear(nn.Module):
    """
    Linear layer with 1.58-bit ternary weight quantization.
    
    During training:
        Forward: Weights quantized to {-1, 0, +1}
        Backward: Gradients flow through via STE
    
    During inference:
        Weights can be stored as 2-bit packed integers
        
    Memory savings:
        Float32: 32 bits per weight
        Ternary: 2 bits per weight (theoretically)
        Compression: 16× (inference) or 4× (training with dequant)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 0.5,
        per_channel: bool = False,
        eps: float = 1e-8,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to include bias
            threshold: Quantization threshold (0.5 for standard)
            per_channel: Use per-channel scaling (better quality)
            eps: Minimum scale value
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.per_channel = per_channel
        self.eps = eps
        
        # Initialize weights with Kaiming normal
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_in')
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        
        # Running statistics for monitoring
        self.register_buffer("running_sparsity", torch.tensor(0.0))
        self.register_buffer("num_quantizations", torch.tensor(0))
    
    def quantize_weights(self) -> Tuple[Tensor, Tensor]:
        """
        Quantize weights to ternary {-1, 0, +1}.
        
        Returns:
            Tuple of (quantized_weight, scale)
        """
        # Compute scale
        if self.per_channel:
            scale = compute_per_channel_scale(self.weight, self.eps)
        else:
            scale = compute_abs_mean_scale(self.weight, self.eps)
        
        # Quantize using STE
        weight_quantized, scale = TernarySTE.apply(
            self.weight, scale, self.threshold
        )
        
        # Dequantize: multiply back by scale for computation
        weight_dequantized = weight_quantized * scale
        
        # Update running sparsity (for monitoring)
        if self.training:
            with torch.no_grad():
                sparsity = (weight_quantized == 0).float().mean()
                momentum = 0.1
                self.running_sparsity.mul_(1 - momentum).add_(sparsity * momentum)
                self.num_quantizations.add_(1)
        
        return weight_dequantized, scale
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with quantized weights."""
        # Quantize weights
        weight_q, scale = self.quantize_weights()
        
        # Standard linear operation
        return F.linear(x, weight_q, self.bias)
    
    def get_sparsity(self) -> float:
        """Get current weight sparsity (fraction of weights that are 0)."""
        with torch.no_grad():
            scale = compute_abs_mean_scale(self.weight, self.eps)
            weight_scaled = self.weight / scale
            weight_quantized = torch.zeros_like(self.weight)
            weight_quantized[weight_scaled > self.threshold] = 1.0
            weight_quantized[weight_scaled < -self.threshold] = -1.0
            return (weight_quantized == 0).float().mean().item()
    
    def get_ternary_distribution(self) -> dict:
        """Get distribution of ternary values."""
        with torch.no_grad():
            scale = compute_abs_mean_scale(self.weight, self.eps)
            weight_scaled = self.weight / scale
            weight_quantized = torch.zeros_like(self.weight)
            weight_quantized[weight_scaled > self.threshold] = 1.0
            weight_quantized[weight_scaled < -self.threshold] = -1.0
            
            total = self.weight.numel()
            return {
                "negative": (weight_quantized == -1).sum().item() / total,
                "zero": (weight_quantized == 0).sum().item() / total,
                "positive": (weight_quantized == 1).sum().item() / total,
            }
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"threshold={self.threshold}, "
            f"per_channel={self.per_channel}"
        )


class BitLinearEmbedding(nn.Module):
    """
    Embedding layer with ternary weight quantization.
    
    Note: Embeddings are typically NOT quantized as aggressively
    as linear layers. This is experimental.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        
        self.weight = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim) * 0.02
        )
    
    def forward(self, indices: Tensor) -> Tensor:
        """Forward with quantized embeddings."""
        # Quantize
        scale = compute_abs_mean_scale(self.weight)
        weight_scaled = self.weight / scale.clamp(min=1e-8)
        
        weight_quantized = torch.zeros_like(self.weight)
        weight_quantized[weight_scaled > self.threshold] = 1.0
        weight_quantized[weight_scaled < -self.threshold] = -1.0
        
        weight_dequantized = weight_quantized * scale
        
        return F.embedding(indices, weight_dequantized)


class BitLinearAttention(nn.Module):
    """
    Multi-head attention with BitLinear projections.
    
    Replaces standard linear projections with ternary quantized versions.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        threshold: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)
        
        # BitLinear projections
        self.q_proj = BitLinear(d_model, d_model, threshold=threshold)
        self.k_proj = BitLinear(d_model, d_model, threshold=threshold)
        self.v_proj = BitLinear(d_model, d_model, threshold=threshold)
        self.o_proj = BitLinear(d_model, d_model, threshold=threshold)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: Tensor, 
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass with ternary projections."""
        batch_size, seq_len, _ = x.shape
        
        # Project with BitLinear
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply causal mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.o_proj(out)
        
        return out


class BitLinearFeedForward(nn.Module):
    """
    Feed-forward network with BitLinear layers.
    
    Two BitLinear layers with GELU activation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        threshold: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.w1 = BitLinear(d_model, d_ff, threshold=threshold)
        self.w2 = BitLinear(d_ff, d_model, threshold=threshold)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.w1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class BitLinearTransformerBlock(nn.Module):
    """
    Complete transformer block with BitLinear throughout.
    
    Combines BitLinearAttention + BitLinearFeedForward
    with residual connections and LayerNorm.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        threshold: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BitLinearAttention(d_model, n_heads, threshold, dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = BitLinearFeedForward(d_model, d_ff, threshold, dropout)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass with pre-norm residual."""
        # Attention with residual
        x = x + self.attn(self.ln1(x), mask)
        
        # Feed-forward with residual
        x = x + self.ff(self.ln2(x))
        
        return x


class BitLinearGPT(nn.Module):
    """
    GPT model with BitLinear (1.58-bit) throughout.
    
    This is the FULL ternary model ready for training.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        threshold: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings (keep float for stability)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # BitLinear transformer blocks
        self.blocks = nn.ModuleList([
            BitLinearTransformerBlock(
                d_model, n_heads, d_ff, threshold, dropout
            )
            for _ in range(n_layers)
        ])
        
        # Output
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = BitLinear(d_model, vocab_size, threshold=threshold)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        input_ids: Tensor, 
        targets: Optional[Tensor] = None
    ) -> dict:
        """Forward pass."""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Causal mask
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=input_ids.device)
        ).unsqueeze(0).unsqueeze(0)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        
        return {"logits": logits, "loss": loss}
    
    def get_sparsity_stats(self) -> dict:
        """Get sparsity statistics for all BitLinear layers."""
        stats = {
            "layers": [],
            "total_params": 0,
            "total_zeros": 0,
        }
        
        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                sparsity = module.get_sparsity()
                distrib = module.get_ternary_distribution()
                stats["layers"].append({
                    "name": name,
                    "sparsity": sparsity,
                    "distribution": distrib,
                })
                stats["total_params"] += module.weight.numel()
                zeros = int(module.weight.numel() * sparsity)
                stats["total_zeros"] += zeros
        
        stats["overall_sparsity"] = (
            stats["total_zeros"] / stats["total_params"] 
            if stats["total_params"] > 0 else 0
        )
        
        return stats
    
    def count_parameters(self) -> dict:
        """Count parameters (for comparison with Float32)."""
        total = sum(p.numel() for p in self.parameters())
        
        # Count BitLinear parameters
        bitlinear_params = 0
        for module in self.modules():
            if isinstance(module, BitLinear):
                bitlinear_params += module.weight.numel()
        
        return {
            "total": total,
            "bitlinear": bitlinear_params,
            "other": total - bitlinear_params,
            "theoretical_bits": 1.58 * bitlinear_params + 32 * (total - bitlinear_params),
            "theoretical_mb": (
                1.58 * bitlinear_params + 32 * (total - bitlinear_params)
            ) / 8 / 1024 / 1024,
        }


def create_bitlinear_gpt(
    vocab_size: int = 4000,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 1024,
    max_seq_len: int = 256,
    threshold: float = 0.5,
) -> BitLinearGPT:
    """Create a BitLinear GPT model with default config."""
    return BitLinearGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        threshold=threshold,
    )
