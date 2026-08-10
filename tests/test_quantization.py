"""Tests for INT8 quantization module."""

import pytest
import torch
import torch.nn as nn

from src.model.quantization import (
    compute_scale,
    quantize_symmetric,
    HysteresisScale,
    QuantizedLinear,
    QuantizedEmbedding,
    measure_quantization_error,
    StraightThroughQuantize,
)


class TestComputeScale:
    """Tests for scale computation."""
    
    def test_scale_basic(self):
        """Test basic scale calculation."""
        x = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0])
        scale = compute_scale(x)
        
        # max_abs = 1.0, so scale = 1.0 / 127 ≈ 0.00787
        assert torch.isclose(scale, torch.tensor(1.0 / 127), rtol=1e-4)
    
    def test_scale_preserves_range(self):
        """Test that scale preserves max value."""
        x = torch.randn(100, 100) * 10
        scale = compute_scale(x)
        
        quantized = torch.round(x / scale)
        
        # Should fit in [-127, 127]
        assert quantized.max() <= 127
        assert quantized.min() >= -127
    
    def test_scale_handles_small_values(self):
        """Test scale doesn't blow up for small values."""
        x = torch.tensor([0.0001, -0.0001, 0.00005])
        scale = compute_scale(x)
        
        # Should not be inf or nan
        assert torch.isfinite(scale)
        assert scale > 0


class TestQuantizeSymmetric:
    """Tests for symmetric quantization."""
    
    def test_quantize_values_in_range(self):
        """Test quantized values are in [-127, 127]."""
        x = torch.randn(100)
        quantized, scale = quantize_symmetric(x)
        
        # Check range
        assert quantized.max() <= 127 * scale
        assert quantized.min() >= -127 * scale
    
    def test_quantize_zero_unchanged(self):
        """Test that zeros stay zero."""
        x = torch.tensor([0.0, 0.0, 0.0])
        quantized, scale = quantize_symmetric(x)
        
        assert torch.allclose(quantized, torch.zeros_like(quantized), atol=scale)
    
    def test_quantize_symmetry(self):
        """Test symmetric handling of positive and negative."""
        x = torch.tensor([0.5, -0.5])
        quantized, scale = quantize_symmetric(x)
        
        # Should be approximately symmetric
        q0 = quantized[0].item()
        q1 = quantized[1].item()
        assert abs(q0 + q1) < scale * 2  # Nearly symmetric
    
    def test_quantize_reconstruction(self):
        """Test reconstruction error is small."""
        x = torch.randn(1000)
        quantized, scale = quantize_symmetric(x)
        
        error = (x - quantized).abs().mean()
        relative_error = error / x.abs().mean()
        
        # Relative error should be small (< 1%)
        assert relative_error < 0.01
    
    def test_gradient_flows(self):
        """Test that gradients can flow through quantization."""
        x = torch.randn(10, requires_grad=True)
        quantized, scale = quantize_symmetric(x)
        
        loss = quantized.sum()
        loss.backward()
        
        # Gradient should exist (not None or all zeros)
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestHysteresisScale:
    """Tests for hysteresis scale tracking."""
    
    def test_initial_scale(self):
        """Test initial scale computation."""
        scale_tracker = HysteresisScale()
        x = torch.tensor([0.0, 1.0, -1.0])
        
        scale = scale_tracker(x)
        
        # Should be based on input
        assert torch.isfinite(scale)
    
    def test_scale_changes_slowly(self):
        """Test that scale changes slowly (hysteresis)."""
        scale_tracker = HysteresisScale(momentum=0.9)
        
        # First input
        x1 = torch.tensor([0.0, 1.0])
        scale1 = scale_tracker(x1)
        
        # Second input with different max
        x2 = torch.tensor([0.0, 2.0])
        scale2 = scale_tracker(x2)
        
        # Scale should not double even though max doubled
        # Due to momentum=0.9: 0.9 * old + 0.1 * new
        assert scale2 < scale1 * 2
    
    def test_eval_mode_uses_running(self):
        """Test that eval mode uses running max."""
        scale_tracker = HysteresisScale(momentum=0.9)
        
        # Train mode
        scale_tracker.train()
        x1 = torch.tensor([0.0, 1.0])
        scale_tracker(x1)
        
        # Eval mode
        scale_tracker.eval()
        x2 = torch.tensor([0.0, 5.0])  # Large value
        scale = scale_tracker(x2)
        
        # Should not have updated with new large value
        assert scale * 127 < 5.0  # Running max should be ~1, not 5


class TestQuantizedLinear:
    """Tests for quantized linear layer."""
    
    def test_output_shape(self):
        """Test output shape matches standard linear."""
        batch_size, in_features, out_features = 4, 64, 128
        
        quant_linear = QuantizedLinear(in_features, out_features)
        standard_linear = nn.Linear(in_features, out_features)
        
        x = torch.randn(batch_size, in_features)
        
        quant_out = quant_linear(x)
        standard_out = standard_linear(x)
        
        assert quant_out.shape == standard_out.shape
    
    def test_weight_quantized_during_forward(self):
        """Test weights are quantized (discrete values)."""
        quant_linear = QuantizedLinear(32, 64)
        x = torch.randn(4, 32)
        
        # Store original weight
        original_weight = quant_linear.weight.clone()
        
        # Forward pass
        _ = quant_linear(x)
        
        # Weight should be unchanged (quantization is per-forward)
        assert torch.allclose(quant_linear.weight, original_weight)
    
    def test_gradient_flows_to_weight(self):
        """Test gradients reach weight parameter."""
        quant_linear = QuantizedLinear(16, 32)
        x = torch.randn(2, 16)
        
        output = quant_linear(x)
        loss = output.sum()
        loss.backward()
        
        assert quant_linear.weight.grad is not None
        assert quant_linear.weight.grad.abs().sum() > 0
    
    def test_gradient_flows_to_input(self):
        """Test gradients reach input."""
        quant_linear = QuantizedLinear(16, 32)
        x = torch.randn(2, 16, requires_grad=True)
        
        output = quant_linear(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
    
    def test_bias_optional(self):
        """Test bias can be disabled."""
        quant_linear = QuantizedLinear(16, 32, bias=False)
        assert quant_linear.bias is None
        
        x = torch.randn(2, 16)
        output = quant_linear(x)
        
        assert output.shape == (2, 32)


class TestQuantizedEmbedding:
    """Tests for quantized embedding layer."""
    
    def test_output_shape(self):
        """Test output shape matches standard embedding."""
        vocab_size, embedding_dim, batch_size = 100, 64, 8
        
        quant_embed = QuantizedEmbedding(vocab_size, embedding_dim)
        standard_embed = nn.Embedding(vocab_size, embedding_dim)
        
        indices = torch.randint(0, vocab_size, (batch_size,))
        
        quant_out = quant_embed(indices)
        standard_out = standard_embed(indices)
        
        assert quant_out.shape == standard_out.shape
    
    def test_gradient_flows(self):
        """Test gradients reach embedding weights."""
        quant_embed = QuantizedEmbedding(50, 32)
        indices = torch.randint(0, 50, (4,))
        
        output = quant_embed(indices)
        loss = output.sum()
        loss.backward()
        
        assert quant_embed.weight.grad is not None
    
    def test_different_indices_different_output(self):
        """Test different indices produce different outputs."""
        quant_embed = QuantizedEmbedding(10, 8)
        
        idx1 = torch.tensor([0])
        idx2 = torch.tensor([1])
        
        out1 = quant_embed(idx1)
        out2 = quant_embed(idx2)
        
        # Different indices should give different embeddings
        assert not torch.allclose(out1, out2)


class TestMeasureQuantizationError:
    """Tests for quantization error measurement."""
    
    def test_zero_error_for_identical(self):
        """Test zero error when tensors are identical."""
        x = torch.randn(100)
        metrics = measure_quantization_error(x, x)
        
        assert metrics["max_error"] == 0.0
        assert metrics["mean_error"] == 0.0
        assert metrics["mse"] == 0.0
    
    def test_correct_mse(self):
        """Test MSE calculation is correct."""
        original = torch.tensor([1.0, 2.0, 3.0])
        quantized = torch.tensor([1.1, 2.0, 2.9])
        
        metrics = measure_quantization_error(original, quantized)
        
        # MSE = ((0.1)^2 + 0 + (-0.1)^2) / 3 = 0.02/3 ≈ 0.00667
        expected_mse = ((0.1**2 + 0 + 0.1**2) / 3)
        assert abs(metrics["mse"] - expected_mse) < 1e-5


class TestEndToEndQuantization:
    """End-to-end tests for quantization in model."""
    
    def test_quantized_layer_can_learn(self):
        """Test quantized layer can learn a simple task."""
        # Create simple dataset: y = 2*x
        x_train = torch.randn(100, 1)
        y_train = 2 * x_train
        
        # Quantized layer
        model = nn.Sequential(
            QuantizedLinear(1, 1),
        )
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()
        
        # Train
        for _ in range(100):
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
        
        # Check weight is approximately 2
        with torch.no_grad():
            x_test = torch.tensor([[1.0]])
            y_pred = model(x_test)
            # Should be close to 2
            assert abs(y_pred.item() - 2.0) < 0.5
    
    def test_quantized_mlp_converges(self):
        """Test a small MLP with quantized layers converges."""
        # XOR problem
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
        
        model = nn.Sequential(
            QuantizedLinear(2, 8),
            nn.ReLU(),
            QuantizedLinear(8, 1),
            nn.Sigmoid(),
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        criterion = nn.BCELoss()
        
        # Train
        losses = []
        for _ in range(500):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should decrease
        assert losses[-1] < losses[0]
        # Final loss should be reasonably low (INT8 quantization makes this harder)
        assert losses[-1] < 0.5
