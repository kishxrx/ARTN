"""
ARTN - Adaptive Recursive Ternary Networks
===========================================

Building Transformer Language Models Under Extreme Hardware Constraints.

This package provides:
- Memory-efficient transformer architectures for CPU-only training
- BitLinear 1.58-bit quantization layers
- Recursive parameter tying
- Adaptive halting gates
"""

__version__ = "0.1.0"
__author__ = "ARTN Project"

from importlib.metadata import version

try:
    __version__ = version("artn")
except Exception:
    pass

__all__ = ["__version__", "__author__"]
