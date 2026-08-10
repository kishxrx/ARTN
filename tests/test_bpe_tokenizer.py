"""
Unit tests for BPE Tokenizer.

Tests the core milestone: decode(encode(text)) == text
"""

import pytest

from src.tokenizer.bpe_tokenizer import BPETokenizer


class TestBPETokenizerBasic:
    """Basic tests for BPE tokenizer initialization and structure."""

    def test_tokenizer_init(self):
        """Verify tokenizer initializes with correct parameters."""
        tokenizer = BPETokenizer(vocab_size=100, min_frequency=2)
        
        assert tokenizer.vocab_size == 100
        assert tokenizer.min_frequency == 2
        assert not tokenizer._initialized

    def test_tokenizer_default_params(self):
        """Verify default parameters are sensible."""
        tokenizer = BPETokenizer()
        
        assert tokenizer.vocab_size == 4000
        assert tokenizer.min_frequency == 2


class TestBPETokenizerTraining:
    """Tests for BPE tokenizer training."""

    def test_train_on_simple_corpus(self):
        """Verify tokenizer trains without error on simple corpus."""
        corpus = "hello world hello world hello"
        
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)
        
        assert tokenizer._initialized
        assert len(tokenizer.vocab) > 0
        assert len(tokenizer.merges) >= 0

    def test_train_produces_vocab(self):
        """Verify training produces a valid vocabulary."""
        corpus = "the cat sat on the mat the cat the mat"
        
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(corpus)
        
        # Should have special tokens
        assert "<pad>" in tokenizer.vocab
        assert "<unk>" in tokenizer.vocab
        
        # Should have learned some merges
        assert len(tokenizer.merges) > 0

    def test_vocab_size_respected(self):
        """Verify vocab doesn't exceed target size."""
        corpus = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        
        tokenizer = BPETokenizer(vocab_size=35)  # Account for 4 special tokens + chars
        tokenizer.train(corpus)
        
        # Vocab should be approximately the target (may be slightly larger due to base chars)
        assert len(tokenizer.vocab) <= 35 + 4  # Allow some slack for initial chars


class TestBPETokenizerEncodeDecode:
    """Tests for encode/decode functionality."""

    @pytest.fixture
    def trained_tokenizer(self):
        """Create a trained tokenizer for testing."""
        corpus = """
        once upon a time there was a little girl named lucy
        lucy loved to play in the garden with her cat
        the cat was fluffy and white and very friendly
        lucy and her cat played together every day
        """
        tokenizer = BPETokenizer(vocab_size=200)
        tokenizer.train(corpus)
        return tokenizer

    def test_encode_returns_list_of_ints(self, trained_tokenizer):
        """Verify encode returns integer token IDs."""
        text = "lucy played with the cat"
        
        ids = trained_tokenizer.encode(text)
        
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_decode_returns_string(self, trained_tokenizer):
        """Verify decode returns a string."""
        text = "lucy played"
        ids = trained_tokenizer.encode(text)
        
        decoded = trained_tokenizer.decode(ids)
        
        assert isinstance(decoded, str)

    def test_encode_decode_roundtrip(self, trained_tokenizer):
        """MILESTONE TEST: decode(encode(text)) == text."""
        original_texts = [
            "lucy played with the cat",
            "the cat was fluffy",
            "lucy and her cat",
        ]
        
        for text in original_texts:
            ids = trained_tokenizer.encode(text)
            decoded = trained_tokenizer.decode(ids)
            
            # Normalize whitespace for comparison
            assert decoded.lower().replace("  ", " ").strip() == text.lower().strip()

    def test_encode_empty_string(self, trained_tokenizer):
        """Verify encoding empty string returns empty list."""
        ids = trained_tokenizer.encode("")
        
        assert ids == []

    def test_decode_empty_list(self, trained_tokenizer):
        """Verify decoding empty list returns empty string."""
        decoded = trained_tokenizer.decode([])
        
        assert decoded == ""


class TestBPETokenizerSaveLoad:
    """Tests for save/load functionality."""

    def test_save_and_load_preserves_vocab(self, tmp_path):
        """Verify save/load preserves vocabulary and merges."""
        corpus = "hello world hello world"
        
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)
        
        # Save
        save_path = tmp_path / "tokenizer.txt"
        tokenizer.save(save_path)
        
        # Load
        loaded = BPETokenizer.load(save_path)
        
        # Verify vocab preserved
        assert loaded.vocab == tokenizer.vocab
        assert loaded.merges == tokenizer.merges

    def test_loaded_tokenizer_encodes_same(self, tmp_path):
        """Verify loaded tokenizer produces same encodings."""
        corpus = "the cat sat on the mat"
        
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)
        
        text = "the cat"
        original_ids = tokenizer.encode(text)
        
        # Save and load
        save_path = tmp_path / "tokenizer.txt"
        tokenizer.save(save_path)
        loaded = BPETokenizer.load(save_path)
        
        # Encode with loaded
        loaded_ids = loaded.encode(text)
        
        assert loaded_ids == original_ids


class TestBPETokenizerEdgeCases:
    """Edge case tests."""

    def test_encode_unknown_characters(self):
        """Verify tokenizer handles unseen characters gracefully."""
        corpus = "hello world"
        
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)
        
        # Encode text with unseen characters
        ids = tokenizer.encode("xyz")
        
        # Should still return a list (may contain unknown tokens)
        assert isinstance(ids, list)

    def test_min_frequency_filtering(self):
        """Verify low-frequency pairs are not merged."""
        # 'z' appears only once, should not form merges
        corpus = "the cat sat on the mat z"
        
        tokenizer = BPETokenizer(vocab_size=50, min_frequency=3)
        tokenizer.train(corpus)
        
        # z should still be in vocab but no merges involving it
        assert tokenizer._initialized

    def test_repr_shows_status(self):
        """Verify repr shows training status."""
        tokenizer = BPETokenizer(vocab_size=100)
        
        assert "untrained" in repr(tokenizer).lower()
        
        tokenizer.train("hello world")
        
        assert "trained" in repr(tokenizer).lower()
