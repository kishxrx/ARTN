"""ARTN Inference module.

Efficient autoregressive generation:
- Top-K / Top-P sampling
- Temperature scaling
- Quantized weight loading
"""

from .generate import (
    load_model_from_checkpoint,
    generate_text,
    generate_batch,
    verify_ternary_weights,
)

__all__ = [
    "load_model_from_checkpoint",
    "generate_text",
    "generate_batch",
    "verify_ternary_weights",
]
