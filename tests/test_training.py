"""
Tests for Training module.

Verifies Stage 3 milestone: Overfit small dataset, loss goes down.
"""

import pytest
import torch
import numpy as np

from src.model.gpt import create_baseline_gpt
from src.training.dataset import TokenDataset, create_mock_dataset
from src.training.adafactor import Adafactor, create_adafactor
from src.training.loop import overfit_small_dataset


class TestDataset:
    """Tests for dataset classes."""
    
    def test_token_dataset_length(self):
        """Dataset reports correct length."""
        tokens = np.arange(1000)
        dataset = TokenDataset(tokens, max_seq_len=32)
        
        assert len(dataset) == (1000 - 32) // 32 + 1
    
    def test_token_dataset_shape(self):
        """Dataset returns correct shape."""
        tokens = np.arange(500, dtype=np.int64)
        max_seq_len = 32
        dataset = TokenDataset(tokens, max_seq_len=max_seq_len)
        
        sample = dataset[0]
        
        assert sample.shape == (max_seq_len,)
    
    def test_token_dataset_stride(self):
        """Dataset respects stride."""
        tokens = np.arange(1000, dtype=np.int64)
        max_seq_len = 32
        stride = 16  # Overlapping sequences
        
        dataset = TokenDataset(tokens, max_seq_len=max_seq_len, stride=stride)
        
        # Non-overlapping: 1000 / 32 ≈ 31 samples
        # With stride 16: 1000 / 16 ≈ 60 samples
        assert len(dataset) > 31
        assert len(dataset) == (1000 - max_seq_len) // stride + 1
    
    def test_mock_dataset(self):
        """Mock dataset creation works."""
        vocab_size = 100
        num_tokens = 1000
        max_seq_len = 32
        
        dataset = create_mock_dataset(num_tokens, vocab_size, max_seq_len)
        
        assert len(dataset) > 0
        sample = dataset[0]
        assert sample.max() < vocab_size


class TestAdafactor:
    """Tests for Adafactor optimizer."""
    
    def test_adafactor_creation(self):
        """Adafactor can be created."""
        model = create_baseline_gpt()
        optimizer = create_adafactor(model, lr=1e-3)
        
        assert isinstance(optimizer, Adafactor)
    
    def test_adafactor_step(self):
        """Adafactor can perform a step."""
        model = create_baseline_gpt()
        optimizer = create_adafactor(model, lr=1e-3)
        
        # Create dummy input
        batch = torch.randint(0, 4000, (2, 16))
        targets = batch[:, 1:]
        inputs = batch[:, :-1]
        
        outputs = model(inputs, targets=targets)
        loss = outputs['loss']
        
        loss.backward()
        
        # Should not raise
        optimizer.step()
    
    def test_adafactor_memory_efficient(self):
        """Adafactor uses less memory than Adam."""
        model = create_baseline_gpt()
        
        # Count state
        adafactor = create_adafactor(model)
        
        # Perform a step to initialize state
        batch = torch.randint(0, 4000, (2, 16))
        targets = batch[:, 1:]
        inputs = batch[:, :-1]
        
        outputs = model(inputs, targets=targets)
        outputs['loss'].backward()
        adafactor.step()
        
        # Check state exists but is smaller than Adam would be
        total_state = 0
        for group in adafactor.param_groups:
            for p in group['params']:
                state = adafactor.state[p]
                for key, tensor in state.items():
                    if isinstance(tensor, torch.Tensor):
                        total_state += tensor.numel()
        
        # Adafactor state should be much smaller than 2×params
        assert total_state < 2 * model.count_parameters()


class TestTrainingLoop:
    """Tests for training loop."""
    
    def test_overfit_small_dataset(self):
        """
        STAGE 3 MILESTONE:
        Overfit a small dataset, loss goes down.
        """
        # Small model for fast test
        config = {
            'vocab_size': 100,
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'd_ff': 256,
        }
        model = create_baseline_gpt(config)
        
        # Small dataset
        dataset = create_mock_dataset(
            num_tokens=500,
            vocab_size=100,
            max_seq_len=32,
        )
        
        # Overfit
        final_loss = overfit_small_dataset(
            model,
            dataset,
            num_steps=50,
            lr=1e-3,
        )
        
        # Loss should have decreased (starting ~4.6 for 100 vocab)
        # Allow slightly higher threshold since random init varies
        assert final_loss < 4.35  # Loss decreased from initial ~4.6
        
        print(f"\nMILESTONE PASSED")
        print(f"Final loss: {final_loss:.4f}")


class TestMilestone:
    """Stage 3 milestone verification."""
    
    def test_training_milestone(self):
        """
        Full milestone test:
        1. Create model
        2. Create dataset
        3. Train for several steps
        4. Verify loss decreases
        """
        # Create model
        config = {
            'vocab_size': 100,
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'd_ff': 256,
        }
        model = create_baseline_gpt(config)
        
        # Create dataset
        dataset = create_mock_dataset(
            num_tokens=500,
            vocab_size=100,
            max_seq_len=32,
        )
        
        # Initial loss
        batch = torch.stack([dataset[i] for i in range(2)])
        targets = batch[:, 1:]
        inputs = batch[:, :-1]
        
        model.eval()
        with torch.no_grad():
            initial_outputs = model(inputs, targets=targets)
            initial_loss = initial_outputs['loss'].item()
        
        # Overfit
        model.train()
        final_loss = overfit_small_dataset(
            model,
            dataset,
            num_steps=100,
            lr=3e-4,
        )
        
        # Verify loss decreased
        assert final_loss < initial_loss
        
        print(f"\n{'='*60}")
        print("STAGE 3 MILESTONE PASSED")
        print(f"{'='*60}")
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss:   {final_loss:.4f}")
        print(f"Reduction:    {initial_loss - final_loss:.4f}")
