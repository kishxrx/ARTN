"""
Tests for GPT Model components.

Verifies Stage 2 milestone: Pass dummy tensor through network with correct output shape.
"""

import pytest
import torch

from src.model.embeddings import TokenEmbedding, PositionalEncoding, Embeddings
from src.model.attention import MultiHeadAttention, CausalSelfAttention, generate_causal_mask
from src.model.feedforward import FeedForward
from src.model.transformer_block import TransformerBlock
from src.model.gpt import GPT, create_baseline_gpt


class TestEmbeddings:
    """Tests for embedding layer."""
    
    def test_token_embedding_shape(self):
        """Token embedding produces correct shape."""
        vocab_size = 4000
        d_model = 256
        batch_size = 2
        seq_len = 64
        
        embedding = TokenEmbedding(vocab_size, d_model)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = embedding(token_ids)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_positional_encoding_shape(self):
        """Positional encoding preserves shape."""
        d_model = 256
        max_seq_len = 512
        batch_size = 2
        seq_len = 64
        
        pe = PositionalEncoding(d_model, max_seq_len)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pe(x)
        
        assert output.shape == x.shape
    
    def test_positional_encoding_values(self):
        """Positional encoding adds to input, doesn't replace."""
        d_model = 256
        pe = PositionalEncoding(d_model)
        x = torch.zeros(1, 10, d_model)
        
        output = pe(x)
        
        # Should be non-zero (sin/cos added)
        assert not torch.allclose(output, x)
    
    def test_combined_embeddings(self):
        """Full embedding pipeline works."""
        vocab_size = 4000
        d_model = 256
        batch_size = 2
        seq_len = 64
        
        embeddings = Embeddings(vocab_size, d_model)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = embeddings(token_ids)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestAttention:
    """Tests for attention mechanisms."""
    
    def test_causal_mask_shape(self):
        """Causal mask has correct shape."""
        seq_len = 10
        mask = generate_causal_mask(seq_len)
        
        assert mask.shape == (seq_len, seq_len)
    
    def test_causal_mask_upper_triangular(self):
        """Causal mask blocks future positions."""
        mask = generate_causal_mask(4)
        
        expected = torch.tensor([
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        
        assert torch.equal(mask, expected)
    
    def test_multi_head_attention_shape(self):
        """Multi-head attention preserves input shape."""
        d_model = 256
        n_heads = 8
        batch_size = 2
        seq_len = 64
        
        attention = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_causal_attention_no_future_leak(self):
        """Causal attention cannot attend to future."""
        d_model = 32
        n_heads = 4
        batch_size = 1
        seq_len = 4
        
        attention = CausalSelfAttention(d_model, n_heads)
        attention.eval()  # Set to eval mode to disable dropout
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x)
        
        # Output at position i should not depend on position i+1
        # We test this by checking that changing future doesn't affect past
        x_modified = x.clone()
        x_modified[:, 3, :] = 999.0  # Corrupt future position
        
        output_modified = attention(x_modified)
        
        # First 3 positions should be unchanged (within floating point tolerance)
        assert torch.allclose(output[:, :3, :], output_modified[:, :3, :], atol=1e-4)


class TestFeedForward:
    """Tests for FFN."""
    
    def test_feedforward_shape(self):
        """FFN preserves sequence shape."""
        d_model = 256
        d_ff = 1024
        batch_size = 2
        seq_len = 64
        
        ffn = FeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ffn(x)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestTransformerBlock:
    """Tests for complete transformer block."""
    
    def test_block_shape(self):
        """Block preserves input shape."""
        d_model = 256
        n_heads = 8
        d_ff = 1024
        batch_size = 2
        seq_len = 64
        
        block = TransformerBlock(d_model, n_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestGPT:
    """Tests for complete GPT model."""
    
    def test_gpt_forward_shape(self):
        """GPT forward pass produces correct output shape."""
        vocab_size = 4000
        d_model = 256
        n_heads = 8
        n_layers = 6
        d_ff = 1024
        batch_size = 2
        seq_len = 64
        
        model = GPT(vocab_size, d_model, n_heads, n_layers, d_ff)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        outputs = model(token_ids)
        logits = outputs['logits']
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_gpt_loss_computation(self):
        """GPT computes loss when targets provided."""
        vocab_size = 100
        d_model = 64
        n_heads = 4
        n_layers = 2
        d_ff = 256
        batch_size = 2
        seq_len = 16
        
        model = GPT(vocab_size, d_model, n_heads, n_layers, d_ff)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        outputs = model(token_ids, targets=targets)
        
        assert outputs['loss'] is not None
        assert outputs['loss'].item() > 0
    
    def test_gpt_parameter_count(self):
        """Parameter count is reasonable."""
        model = create_baseline_gpt()
        params = model.count_parameters()
        
        # Should be around 6.77M
        assert 6_000_000 < params < 8_000_000
    
    def test_gpt_memory_estimate(self):
        """Memory estimate is calculated."""
        model = create_baseline_gpt()
        memory_mb = model.estimate_memory_mb()
        
        # Should be around 27MB for Float32
        assert 20 < memory_mb < 35
    
    def test_gpt_generate_shape(self):
        """Generation produces correct sequence length."""
        vocab_size = 100
        d_model = 64
        n_heads = 4
        n_layers = 2
        d_ff = 256
        batch_size = 1
        seq_len = 10
        
        model = GPT(vocab_size, d_model, n_heads, n_layers, d_ff)
        model.eval()
        
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        max_new = 5
        
        generated = model.generate(token_ids, max_new_tokens=max_new)
        
        assert generated.shape == (batch_size, seq_len + max_new)


class TestMilestone:
    """Stage 2 milestone verification."""
    
    def test_milestone_dummy_tensor(self):
        """
        STAGE 2 MILESTONE:
        Pass dummy tensor (batch=2, seq_len=64) through network.
        Verify correct output shape.
        """
        batch = 2
        seq_len = 64
        vocab_size = 4000
        
        model = create_baseline_gpt()
        token_ids = torch.randint(0, vocab_size, (batch, seq_len))
        
        outputs = model(token_ids)
        
        # Check output shape
        assert outputs['logits'].shape == (batch, seq_len, vocab_size)
        
        # Check no errors - logits exist
        assert outputs['logits'] is not None
        
        # Print success info
        print(f"\n✓ MILESTONE PASSED")
        print(f"  Input: batch={batch}, seq_len={seq_len}")
        print(f"  Output: {outputs['logits'].shape}")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Memory (Float32): {model.estimate_memory_mb():.1f} MB")
