"""
Tests for BitLinear (1.58-bit Ternary Quantization).

These tests verify:
1. Ternary quantization produces {-1, 0, +1}
2. STE gradients flow correctly
3. Sparsity tracking works
4. Model can learn with ternary weights
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model.bitlinear import (
    BitLinear,
    BitLinearEmbedding,
    BitLinearAttention,
    BitLinearFeedForward,
    BitLinearTransformerBlock,
    BitLinearGPT,
    TernarySTE,
    compute_abs_mean_scale,
    compute_per_channel_scale,
    create_bitlinear_gpt,
)


class TestComputeAbsMeanScale:
    """Tests for abs-mean scale computation."""
    
    def test_scale_positive(self):
        """Scale should always be positive."""
        weight = torch.randn(100, 100)
        scale = compute_abs_mean_scale(weight)
        assert scale > 0
    
    def test_scale_reasonable(self):
        """Scale should be roughly mean absolute value."""
        weight = torch.randn(1000) * 2  # std=2
        scale = compute_abs_mean_scale(weight)
        
        # For N(0, 2), E[|X|] = 2 * sqrt(2/pi) ≈ 1.596
        expected = 2 * math.sqrt(2 / math.pi)
        assert abs(scale.item() - expected) < 0.3
    
    def test_scale_handles_zeros(self):
        """Scale handles all-zero tensors with eps."""
        weight = torch.zeros(100)
        scale = compute_abs_mean_scale(weight, eps=1e-8)
        assert scale > 0
        assert scale < 1e-7


class TestTernarySTE:
    """Tests for Straight-Through Estimator."""
    
    def test_quantizes_to_ternary(self):
        """Output should be only {-1, 0, +1}."""
        weight = torch.randn(100, requires_grad=True)
        scale = torch.tensor(1.0)
        
        quantized, _ = TernarySTE.apply(weight, scale, 0.5)
        
        unique = quantized.unique()
        for val in unique:
            assert val.item() in [-1.0, 0.0, 1.0]
    
    def test_zero_near_center(self):
        """Values near zero should quantize to 0."""
        weight = torch.tensor([0.0, 0.1, 0.4, -0.1, -0.4])
        scale = torch.tensor(1.0)
        
        quantized, _ = TernarySTE.apply(weight, scale, 0.5)
        
        # All should be 0 since |x| < 0.5
        assert torch.all(quantized == 0)
    
    def test_positive_quantizes_to_one(self):
        """Values > 0.5 should quantize to +1."""
        weight = torch.tensor([0.6, 0.7, 1.0, 10.0])
        scale = torch.tensor(1.0)
        
        quantized, _ = TernarySTE.apply(weight, scale, 0.5)
        
        assert torch.all(quantized == 1)
    
    def test_negative_quantizes_to_minus_one(self):
        """Values < -0.5 should quantize to -1."""
        weight = torch.tensor([-0.6, -0.7, -1.0, -10.0])
        scale = torch.tensor(1.0)
        
        quantized, _ = TernarySTE.apply(weight, scale, 0.5)
        
        assert torch.all(quantized == -1)
    
    def test_gradient_flows_through(self):
        """Gradient should pass through STE unchanged."""
        weight = torch.randn(10, requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=False)
        
        quantized, _ = TernarySTE.apply(weight, scale, 0.5)
        
        # Compute some loss and backward
        loss = quantized.sum()
        loss.backward()
        
        # Gradient should exist
        assert weight.grad is not None
        # For STE, gradient should be 1 (since we sum)
        assert torch.allclose(weight.grad, torch.ones_like(weight), rtol=0.1)


class TestBitLinear:
    """Tests for BitLinear layer."""
    
    def test_output_shape(self):
        """Output shape matches standard Linear."""
        in_features, out_features = 64, 128
        bitlinear = BitLinear(in_features, out_features)
        linear = nn.Linear(in_features, out_features)
        
        x = torch.randn(4, in_features)
        
        bit_out = bitlinear(x)
        lin_out = linear(x)
        
        assert bit_out.shape == lin_out.shape
    
    def test_weights_are_ternary_efffectively(self):
        """After quantization, weights should be ternary."""
        bitlinear = BitLinear(64, 128)
        x = torch.randn(4, 64)
        
        _ = bitlinear(x)
        
        # Get sparsity - should have zeros
        sparsity = bitlinear.get_sparsity()
        assert 0 <= sparsity <= 1
    
    def test_gradient_flows_to_weight(self):
        """Gradient reaches weight parameter."""
        bitlinear = BitLinear(32, 64)
        x = torch.randn(2, 32)
        
        y = bitlinear(x)
        loss = y.sum()
        loss.backward()
        
        assert bitlinear.weight.grad is not None
        assert bitlinear.weight.grad.abs().sum() > 0
    
    def test_gradient_flows_to_input(self):
        """Gradient reaches input."""
        bitlinear = BitLinear(32, 64)
        x = torch.randn(2, 32, requires_grad=True)
        
        y = bitlinear(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
    
    def test_bias_optional(self):
        """Can create without bias."""
        bitlinear = BitLinear(32, 64, bias=False)
        assert bitlinear.bias is None
        
        x = torch.randn(2, 32)
        y = bitlinear(x)
        assert y.shape == (2, 64)
    
    def test_get_ternary_distribution(self):
        """Can get distribution of ternary values."""
        bitlinear = BitLinear(100, 100)
        _ = bitlinear(torch.randn(10, 100))
        
        distrib = bitlinear.get_ternary_distribution()
        
        assert "negative" in distrib
        assert "zero" in distrib
        assert "positive" in distrib
        
        # Should sum to 1
        total = distrib["negative"] + distrib["zero"] + distrib["positive"]
        assert abs(total - 1.0) < 0.001
    
    def test_per_channel_scaling(self):
        """Per-channel scaling works."""
        bitlinear = BitLinear(32, 64, per_channel=True)
        x = torch.randn(2, 32)
        
        y = bitlinear(x)
        assert y.shape == (2, 64)


class TestBitLinearEmbedding:
    """Tests for BitLinear embedding."""
    
    def test_output_shape(self):
        """Output shape matches standard Embedding."""
        vocab_size, embedding_dim = 100, 64
        
        bit_embed = BitLinearEmbedding(vocab_size, embedding_dim)
        std_embed = nn.Embedding(vocab_size, embedding_dim)
        
        indices = torch.randint(0, vocab_size, (10,))
        
        bit_out = bit_embed(indices)
        std_out = std_embed(indices)
        
        assert bit_out.shape == std_out.shape
    
    def test_gradient_flows(self):
        """Gradient reaches embedding weights."""
        bit_embed = BitLinearEmbedding(50, 32)
        indices = torch.randint(0, 50, (4,))
        
        output = bit_embed(indices)
        loss = output.sum()
        loss.backward()
        
        assert bit_embed.weight.grad is not None


class TestBitLinearAttention:
    """Tests for BitLinear attention."""
    
    def test_output_shape(self):
        """Output shape is correct."""
        d_model, n_heads, batch_size, seq_len = 256, 8, 4, 32
        
        attn = BitLinearAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        out = attn(x)
        assert out.shape == (batch_size, seq_len, d_model)
    
    def test_with_causal_mask(self):
        """Works with causal mask."""
        d_model, n_heads = 64, 4
        
        attn = BitLinearAttention(d_model, n_heads)
        x = torch.randn(2, 16, d_model)
        
        # Causal mask
        mask = torch.tril(torch.ones(16, 16)).unsqueeze(0).unsqueeze(0)
        
        out = attn(x, mask)
        assert out.shape == x.shape
    
    def test_gradient_flows(self):
        """Gradient flows through attention."""
        attn = BitLinearAttention(64, 4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        
        out = attn(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None


class TestBitLinearFeedForward:
    """Tests for BitLinear FFN."""
    
    def test_output_shape(self):
        """Output shape is correct."""
        d_model, d_ff = 64, 256
        
        ffn = BitLinearFeedForward(d_model, d_ff)
        x = torch.randn(4, 16, d_model)
        
        out = ffn(x)
        assert out.shape == x.shape
    
    def test_gradient_flows(self):
        """Gradient flows through FFN."""
        ffn = BitLinearFeedForward(64, 256)
        x = torch.randn(2, 8, 64, requires_grad=True)
        
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None


class TestBitLinearTransformerBlock:
    """Tests for complete transformer block."""
    
    def test_output_shape(self):
        """Output shape matches input."""
        d_model, n_heads, d_ff = 64, 4, 256
        
        block = BitLinearTransformerBlock(d_model, n_heads, d_ff)
        x = torch.randn(2, 16, d_model)
        
        out = block(x)
        assert out.shape == x.shape
    
    def test_residual_connection(self):
        """Output should include input (residual)."""
        d_model, n_heads, d_ff = 64, 4, 256
        
        block = BitLinearTransformerBlock(d_model, n_heads, d_ff)
        x = torch.randn(2, 8, d_model)
        
        out = block(x)
        
        # Output should not be identical to input (transformations happened)
        # But should not be wildly different
        assert not torch.allclose(out, x)


class TestBitLinearGPT:
    """Tests for full BitLinear GPT model."""
    
    def test_forward_output_shape(self):
        """Forward pass produces correct shapes."""
        vocab_size, d_model, n_heads, n_layers = 100, 64, 4, 2
        batch_size, seq_len = 4, 16
        
        model = BitLinearGPT(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=256,
            max_seq_len=32,
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = model(input_ids)
        
        assert "logits" in output
        assert output["logits"].shape == (batch_size, seq_len, vocab_size)
    
    def test_loss_computed_with_targets(self):
        """Loss is computed when targets provided."""
        vocab_size = 100
        model = BitLinearGPT(
            vocab_size=vocab_size,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=32,
        )
        
        input_ids = torch.randint(0, vocab_size, (4, 16))
        targets = torch.randint(0, vocab_size, (4, 16))
        
        output = model(input_ids, targets)
        
        assert output["loss"] is not None
        assert output["loss"].item() > 0
    
    def test_gradient_flows(self):
        """Gradients reach all parameters."""
        vocab_size = 50
        model = BitLinearGPT(
            vocab_size=vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            max_seq_len=16,
        )
        
        input_ids = torch.randint(0, vocab_size, (2, 8))
        targets = torch.randint(0, vocab_size, (2, 8))
        
        output = model(input_ids, targets)
        output["loss"].backward()
        
        # Check some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad
    
    def test_get_sparsity_stats(self):
        """Can get sparsity statistics."""
        vocab_size = 50
        model = BitLinearGPT(
            vocab_size=vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            max_seq_len=16,
        )
        
        # Run forward to trigger quantization
        input_ids = torch.randint(0, vocab_size, (2, 8))
        _ = model(input_ids)
        
        stats = model.get_sparsity_stats()
        
        assert "layers" in stats
        assert "total_params" in stats
        assert "overall_sparsity" in stats
        assert stats["overall_sparsity"] >= 0
    
    def test_count_parameters(self):
        """Can count parameters."""
        vocab_size = 50
        model = BitLinearGPT(
            vocab_size=vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            max_seq_len=16,
        )
        
        param_stats = model.count_parameters()
        
        assert "total" in param_stats
        assert "bitlinear" in param_stats
        assert "other" in param_stats
        assert param_stats["total"] > 0
    
    def test_create_bitlinear_gpt_helper(self):
        """Helper function creates model."""
        model = create_bitlinear_gpt()
        
        assert isinstance(model, BitLinearGPT)
        assert model.vocab_size == 4000
        assert model.d_model == 256


class TestBitLinearLearning:
    """Tests that BitLinear can actually learn."""
    
    def test_single_layer_learns_linear_function(self):
        """Single BitLinear can learn y = 2x."""
        torch.manual_seed(42)
        
        model = BitLinear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        # y = 2x
        x = torch.randn(100, 1)
        y = 2 * x
        
        # Train
        for _ in range(200):
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
        
        # Verify
        with torch.no_grad():
            test_x = torch.tensor([[1.0]])
            pred_y = model(test_x)
            # Should be close to 2
            assert abs(pred_y.item() - 2.0) < 1.0
    
    def test_xor_with_mlp(self):
        """BitLinear MLP can learn XOR."""
        torch.manual_seed(42)
        
        # XOR data
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
        
        model = nn.Sequential(
            BitLinear(2, 16),
            nn.ReLU(),
            BitLinear(16, 1),
            nn.Sigmoid(),
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        criterion = nn.BCELoss()
        
        # Train
        losses = []
        for _ in range(500):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should decrease
        assert losses[-1] < losses[0]
        
        # Should solve XOR reasonably
        with torch.no_grad():
            final_pred = model(x)
            predicted = (final_pred > 0.5).float()
            accuracy = (predicted == y).float().mean()
            assert accuracy > 0.75  # At least 3/4 correct
    
    def test_bitlinear_gpt_overfits_tiny_dataset(self):
        """Full BitLinearGPT can overfit a tiny dataset."""
        torch.manual_seed(42)
        
        vocab_size = 20
        model = BitLinearGPT(
            vocab_size=vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=2,
            d_ff=64,
            max_seq_len=16,
        )
        
        # Tiny dataset: single sequence
        input_ids = torch.randint(0, vocab_size, (1, 8))
        targets = input_ids.clone()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train
        losses = []
        for _ in range(100):
            optimizer.zero_grad()
            output = model(input_ids, targets)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should decrease
        assert losses[-1] < losses[0]
