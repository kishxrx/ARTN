"""ARTN Model module.

Core neural network components:
- Float32 baseline transformer blocks
- Recursive parameter-tying architecture
- BitLinear 1.58-bit quantization layers
- Adaptive halting gates
"""

from src.model.embeddings import Embeddings, TokenEmbedding, PositionalEncoding
from src.model.attention import MultiHeadAttention, CausalSelfAttention, generate_causal_mask
from src.model.feedforward import FeedForward
from src.model.transformer_block import TransformerBlock
from src.model.gpt import GPT, create_baseline_gpt
from src.model.recursive import RecursiveGPT, RecursiveTransformer, create_recursive_gpt
from src.model.halting import HaltingGate, AdaptiveHaltingSimple, create_adaptive_halting
from src.model.adaptive import AdaptiveGPT, create_adaptive_gpt
from src.model.quantization import (
    QuantizedLinear,
    QuantizedEmbedding,
    quantize_symmetric,
    compute_scale,
    HysteresisScale,
)
from src.model.bitlinear import (
    BitLinear,
    BitLinearEmbedding,
    BitLinearAttention,
    BitLinearFeedForward,
    BitLinearTransformerBlock,
    BitLinearGPT,
    create_bitlinear_gpt,
)

__all__ = [
    "Embeddings",
    "TokenEmbedding", 
    "PositionalEncoding",
    "MultiHeadAttention",
    "CausalSelfAttention",
    "generate_causal_mask",
    "FeedForward",
    "TransformerBlock",
    "GPT",
    "create_baseline_gpt",
    "RecursiveGPT",
    "RecursiveTransformer",
    "create_recursive_gpt",
    "HaltingGate",
    "AdaptiveHaltingSimple",
    "create_adaptive_halting",
    "AdaptiveGPT",
    "create_adaptive_gpt",
    "QuantizedLinear",
    "QuantizedEmbedding",
    "quantize_symmetric",
    "compute_scale",
    "HysteresisScale",
    "BitLinear",
    "BitLinearEmbedding",
    "BitLinearAttention",
    "BitLinearFeedForward",
    "BitLinearTransformerBlock",
    "BitLinearGPT",
    "create_bitlinear_gpt",
]
