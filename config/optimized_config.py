"""

ARTN Optimized Training Configuration.

Optimized for ternary (1.58-bit) quantization on low-end hardware.
Key changes:
- Higher learning rate (0.003)
- Learning rate warmup + decay
- Larger effective batch size
- Gradient accumulation
"""

# ...[truncated]