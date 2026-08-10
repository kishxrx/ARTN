"""
INT8 Quantization module for ARTN.

Implements symmetric quantization with straight-through estimator (STE)
for gradient flow through quantization operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


class StraightThroughQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator for quantization.
    
    Forward: Quantize to discrete values
    Backward: Pass gradient through unchanged
    """
    
    @staticmethod
    def forward(
        ctx, x: Tensor, scale: Tensor, min_val: float, max_val: float
    ) -> Tensor:
        """
        Quantize tensor using scale factor.
        
        Args:
            x: Input tensor
            scale: Scale factor (max_abs / 127)
            min_val: Minimum quantized value (-127)
            max_val: Maximum quantized value (127)
        
        Returns:
            Quantized tensor (in float but representing integers)
        """
        # Quantize: round, clamp to valid range, dequantize
        x_quantized = torch.round(x / scale)
        x_quantized = torch.clamp(x_quantized, min_val, max_val)
        x_dequantized = x_quantized * scale
        return x_dequantized
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """Pass gradient through unchanged (STE)."""
        return grad_output, None, None, None


def compute_scale(x: Tensor, bits: int = 8) -> Tensor:
    """
    Compute symmetric quantization scale.
    
    Args:
        x: Input tensor
        bits: Number of bits (default 8 for INT8)
    
    Returns:
        Scale factor
    """
    max_val = 2 ** (bits - 1) - 1  # 127 for INT8
    max_abs = x.abs().max().clamp(min=1e-8)  # Prevent division by zero
    return max_abs / max_val


def quantize_symmetric(
    x: Tensor, scale: Optional[Tensor] = None, bits: int = 8
) -> Tuple[Tensor, Tensor]:
    """
    Symmetric quantization to INT8.
    
    Args:
        x: Input tensor
        scale: Optional pre-computed scale (computed if None)
        bits: Number of bits (default 8)
    
    Returns:
        Tuple of (quantized tensor, scale)
    """
    max_val = 2 ** (bits - 1) - 1
    min_val = -(2 ** (bits - 1) - 1)
    
    if scale is None:
        scale = compute_scale(x, bits)
    
    quantized = StraightThroughQuantize.apply(x, scale, min_val, max_val)
    return quantized, scale


class HysteresisScale(nn.Module):
    """
    Running scale tracker for stable quantization.
    
    Uses exponential moving average of max absolute values
    to prevent scale jitter during training.
    """
    
    def __init__(self, momentum: float = 0.9, eps: float = 1e-8):
        """
        Args:
            momentum: EMA momentum (higher = slower change)
            eps: Minimum value to prevent division by zero
        """
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_max", torch.tensor(1.0))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Compute scale with hysteresis.
        
        Args:
            x: Input tensor
        
        Returns:
            Stable scale factor
        """
        current_max = x.abs().max().clamp(min=self.eps)
        
        # Update running max with EMA (no gradient needed for running stats)
        with torch.no_grad():
            if self.training:
                self.running_max.copy_(
                    self.momentum * self.running_max + (1 - self.momentum) * current_max
                )
        
        return self.running_max.detach() / 127.0


class QuantizedLinear(nn.Module):
    """
    Linear layer with INT8 weight quantization.
    
    Weight is quantized to INT8 during forward pass,
    but gradients flow through via STE.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_hysteresis: bool = True,
        bits: int = 8,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to include bias
            use_hysteresis: Use running scale for stability
            bits: Quantization bits (default 8)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Standard weight and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        
        # Hysteresis scale for weights
        if use_hysteresis:
            self.weight_scale = HysteresisScale()
        else:
            self.weight_scale = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with quantized weights."""
        # Compute scale
        if self.weight_scale is not None:
            scale = self.weight_scale(self.weight)
        else:
            scale = compute_scale(self.weight, self.bits)
        
        # Quantize weights using STE
        weight_quantized, _ = quantize_symmetric(self.weight, scale, self.bits)
        
        # Standard linear operation
        return F.linear(x, weight_quantized, self.bias)
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"bits={self.bits}"
        )


class QuantizedEmbedding(nn.Module):
    """
    Embedding layer with INT8 quantization.
    
    Note: Embeddings are typically quantized after training.
    During training, we use fake quantization for analysis.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bits: int = 8,
    ):
        """
        Args:
            num_embeddings: Vocabulary size
            embedding_dim: Embedding dimension
            bits: Quantization bits (default 8)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.bits = bits
        
        # Embedding weights
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.02)
    
    def forward(self, indices: Tensor) -> Tensor:
        """Forward pass with quantized embeddings."""
        # Quantize embeddings
        scale = compute_scale(self.weight, self.bits)
        weight_quantized, _ = quantize_symmetric(self.weight, scale, self.bits)
        
        # Look up embeddings (indices don't need quantization)
        return F.embedding(indices, weight_quantized)


class QuantizedLinearFunction(torch.autograd.Function):
    """
    Quantized linear operation with fused quantization.
    
    Fuses: quantize weights -> linear -> dequantize output
    """
    
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        weight_scale: Tensor,
    ) -> Tensor:
        """
        Forward pass with quantized weights.
        
        Args:
            x: Input (batch, in_features)
            weight: Weight matrix (out_features, in_features)
            bias: Optional bias
            weight_scale: Scale factor for weights
        """
        # Quantize weights
        weight_q = torch.round(weight / weight_scale)
        weight_q = torch.clamp(weight_q, -127, 127)
        weight_dq = weight_q * weight_scale
        
        # Compute output
        output = F.linear(x, weight_dq, bias)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """Backward pass - STE for weights."""
        # Gradient flows through weight_scale unchanged
        return grad_output, grad_output, None, None


def measure_quantization_error(
    original: Tensor, quantized: Tensor
) -> dict:
    """
    Measure quantization quality metrics.
    
    Args:
        original: Original tensor
        quantized: Quantized tensor
    
    Returns:
        Dictionary of error metrics
    """
    error = original - quantized
    
    return {
        "max_error": error.abs().max().item(),
        "mean_error": error.abs().mean().item(),
        "mse": (error ** 2).mean().item(),
        "relative_error": (error.abs().mean() / original.abs().mean()).item(),
    }
