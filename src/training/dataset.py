"""
Memory-mapped dataset for ARTN.

Streams tokenized data from disk without loading everything into RAM.
Critical for 4GB constraint - dataset can be larger than available memory.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    """
    Memory-mapped token dataset.
    
    Data lives on disk, loaded on-demand into RAM.
    Transparent to training loop - looks like a normal tensor.
    
    Args:
        filepath: Path to .npy file containing token IDs
        max_seq_len: Maximum sequence length per sample
        stride: Stride between sequences (default: max_seq_len for non-overlapping)
    
    Shape:
        Each sample: (max_seq_len,) token IDs
        
    Note:
        Total samples = (total_tokens - max_seq_len) // stride + 1
    """
    
    def __init__(
        self,
        filepath: str | Path,
        max_seq_len: int,
        stride: Optional[int] = None,
    ):
        self.filepath = Path(filepath)
        self.max_seq_len = max_seq_len
        self.stride = stride if stride is not None else max_seq_len
        
        # Open memory-mapped file (read-only)
        self.data = np.load(self.filepath, mmap_mode='r')
        
        # Calculate number of samples
        total_tokens = len(self.data)
        self.num_samples = (total_tokens - max_seq_len) // self.stride + 1
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sequence of tokens.
        
        Args:
            idx: Sample index
            
        Returns:
            Token IDs of shape (max_seq_len,)
        """
        start = idx * self.stride
        end = start + self.max_seq_len
        
        tokens = self.data[start:end]
        return torch.tensor(tokens, dtype=torch.long)


class TokenDataset(Dataset):
    """
    Simple in-memory token dataset for small datasets.
    
    When data fits in RAM, this is faster than memmap.
    Use for testing and small training runs.
    
    Args:
        tokens: Token IDs (numpy array or tensor)
        max_seq_len: Maximum sequence length
        stride: Stride between sequences
    """
    
    def __init__(
        self,
        tokens: np.ndarray | torch.Tensor,
        max_seq_len: int,
        stride: Optional[int] = None,
    ):
        self.max_seq_len = max_seq_len
        self.stride = stride if stride is not None else max_seq_len
        
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy()
        
        self.data = tokens.astype(np.int64)
        
        total_tokens = len(self.data)
        self.num_samples = (total_tokens - max_seq_len) // self.stride + 1
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.stride
        end = start + self.max_seq_len
        return torch.tensor(self.data[start:end], dtype=torch.long)


def create_mock_dataset(
    num_tokens: int,
    vocab_size: int,
    max_seq_len: int,
    seed: int = 42,
) -> TokenDataset:
    """
    Create a mock dataset for testing.
    
    Args:
        num_tokens: Total number of tokens
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        seed: Random seed
    
    Returns:
        TokenDataset with random tokens
    """
    np.random.seed(seed)
    tokens = np.random.randint(0, vocab_size, size=num_tokens)
    return TokenDataset(tokens, max_seq_len)


def save_tokens(tokens: np.ndarray, filepath: str | Path) -> None:
    """
    Save tokens to numpy file.
    
    Args:
        tokens: Token IDs as numpy array
        filepath: Output path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, tokens.astype(np.int32))


def load_tokens(filepath: str | Path) -> np.ndarray:
    """
    Load tokens from numpy file.
    
    Args:
        filepath: Path to .npy file
    
    Returns:
        Token IDs as numpy array
    """
    return np.load(filepath)
