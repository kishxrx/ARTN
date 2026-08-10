"""ARTN Training module.

Memory-efficient training infrastructure:
- Gradient accumulation
- Adafactor optimizer
- Memory-mapped data streaming
- Metrics logging
"""

from src.training.dataset import MemmapDataset, TokenDataset, create_mock_dataset, save_tokens, load_tokens
from src.training.adafactor import Adafactor, create_adafactor
from src.training.loop import Trainer, overfit_small_dataset

__all__ = [
    "MemmapDataset",
    "TokenDataset",
    "create_mock_dataset",
    "save_tokens",
    "load_tokens",
    "Adafactor",
    "create_adafactor",
    "Trainer",
    "overfit_small_dataset",
]
