"""
Unit tests for Regex Tokenizer (SMILES).

Tests the core milestone: decode(encode(smiles)) == smiles
"""

import pytest

from src.tokenizer.regex_tokenizer import RegexTokenizer


class TestRegexTokenizerBasic:
    """Basic tests for regex tokenizer initialization."""

    def test_tokenizer_init(self):
        """Verify tokenizer initializes correctly."""
        tokenizer = RegexTokenizer()
        
        assert not tokenizer._initialized
        assert len(tokenizer.special_tokens) >= 4

    def test_tokenizer_with_custom_special_tokens(self):
        """Verify custom special tokens are added."""
        tokenizer = RegexTokenizer(special_tokens=["<mol>", "</mol>"])
        
        assert "<mol>" in tokenizer.special_tokens
        assert "</mol>" in tokenizer.special_tokens


class TestRegexTokenizerPatterns:
    """Tests for SMILES tokenization patterns."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer with a simple vocab."""
        tokenizer = RegexTokenizer()
        # Build vocab with common SMILES tokens
        smiles_corpus = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "[NH3+]C",  # bracketed group
            "C1CCCCC1",  # cyclohexane
            "ClCCCl",  # with chlorine
            "BrCCBr",  # with bromine
        ]
        tokenizer.build_vocab(smiles_corpus)
        return tokenizer

    def test_tokenize_simple_molecule(self):
        """Verify basic SMILES tokenization."""
        tokenizer = RegexTokenizer()
        
        # Manually check tokenization
        tokens = tokenizer._tokenize_smiles("CCO")
        
        # C, C, O (three tokens)
        assert tokens == ["C", "C", "O"]

    def test_tokenize_multi_char_atoms(self):
        """Verify Cl and Br are single tokens."""
        tokenizer = RegexTokenizer()
        
        tokens = tokenizer._tokenize_smiles("ClCCBr")
        
        # Cl, C, C, Br (four tokens, not six)
        assert tokens == ["Cl", "C", "C", "Br"]

    def test_tokenize_bracketed_groups(self):
        """Verify bracketed groups stay together."""
        tokenizer = RegexTokenizer()
        
        tokens = tokenizer._tokenize_smiles("[NH3+]CC")
        
        # [NH3+], C, C (three tokens)
        assert "[NH3+]" in tokens

    def test_tokenize_ring_closures(self):
        """Verify ring closure digits are separate tokens."""
        tokenizer = RegexTokenizer()
        
        tokens = tokenizer._tokenize_smiles("C1CCCCC1")
        
        # C, 1, C, C, C, C, C, 1
        assert "1" in tokens

    def test_tokenize_bond_types(self):
        """Verify bond type symbols are tokens."""
        tokenizer = RegexTokenizer()
        
        tokens = tokenizer._tokenize_smiles("C=C")
        
        assert "=" in tokens

    def test_tokenize_branches(self):
        """Verify branch markers are tokens."""
        tokenizer = RegexTokenizer()
        
        tokens = tokenizer._tokenize_smiles("CC(=O)O")
        
        assert "(" in tokens
        assert ")" in tokens

    def test_tokenize_two_digit_ring_closure(self):
        """Verify %10, %11 style ring closures."""
        tokenizer = RegexTokenizer()
        
        # A ring with closure > 9
        tokens = tokenizer._tokenize_smiles("C%12CC%12")
        
        assert "%12" in tokens


class TestRegexTokenizerEncodeDecode:
    """Tests for encode/decode roundtrip."""

    @pytest.fixture
    def trained_tokenizer(self):
        """Create a trained tokenizer with diverse SMILES."""
        smiles_corpus = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "CCN(CC)CC",  # triethylamine
            "CC(=O)Nc1ccc(O)cc1",  # paracetamol-like
            "ClCCCl",  # dichloroethane
            "C1CCCCC1",  # cyclohexane
            "[NH3+]C",  # methylammonium
            "C@H(C)O",  # chiral center
        ]
        
        tokenizer = RegexTokenizer()
        tokenizer.build_vocab(smiles_corpus)
        return tokenizer

    def test_encode_returns_ints(self, trained_tokenizer):
        """Verify encode returns integer IDs."""
        ids = trained_tokenizer.encode("CCO")
        
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_decode_returns_string(self, trained_tokenizer):
        """Verify decode returns a string."""
        ids = trained_tokenizer.encode("CCO")
        decoded = trained_tokenizer.decode(ids)
        
        assert isinstance(decoded, str)

    def test_encode_decode_roundtrip(self, trained_tokenizer):
        """MILESTONE TEST: decode(encode(smiles)) == smiles."""
        test_smiles = [
            "CCO",
            "CC(=O)O",
            "c1ccccc1",
            "CCN(CC)CC",
            "ClCCCl",
            "C1CCCCC1",
        ]
        
        for smiles in test_smiles:
            ids = trained_tokenizer.encode(smiles)
            decoded = trained_tokenizer.decode(ids)
            
            assert decoded == smiles, f"Roundtrip failed for {smiles}: got {decoded}"

    def test_encode_decode_complex_brackets(self, trained_tokenizer):
        """Verify roundtrip preserves bracketed groups."""
        smiles = "[NH3+]C"
        
        ids = trained_tokenizer.encode(smiles)
        decoded = trained_tokenizer.decode(ids)
        
        assert decoded == smiles


class TestRegexTokenizerVocab:
    """Tests for vocabulary building."""

    def test_vocab_size(self):
        """Verify vocab size is correct after building."""
        smiles_corpus = ["CCO", "CC", "OO"]
        
        tokenizer = RegexTokenizer()
        tokenizer.build_vocab(smiles_corpus)
        
        # Should have special tokens + unique atoms (C, O)
        # + any other tokens from the SMILES
        assert tokenizer.vocab_size >= 6  # At least 4 special + C + O

    def test_special_tokens_get_first_ids(self):
        """Verify special tokens get IDs 0, 1, 2, 3..."""
        smiles_corpus = ["C"]
        
        tokenizer = RegexTokenizer()
        tokenizer.build_vocab(smiles_corpus)
        
        # Special tokens should be at the start
        assert tokenizer.vocab["<pad>"] == 0
        assert tokenizer.vocab["<unk>"] == 1


class TestRegexTokenizerSaveLoad:
    """Tests for save/load functionality."""

    def test_save_and_load_preserves_vocab(self, tmp_path):
        """Verify save/load preserves vocabulary."""
        smiles_corpus = ["CCO", "CC(=O)O", "ClCCCl"]
        
        tokenizer = RegexTokenizer()
        tokenizer.build_vocab(smiles_corpus)
        
        # Save
        save_path = tmp_path / "regex_tokenizer.txt"
        tokenizer.save(save_path)
        
        # Load
        loaded = RegexTokenizer.load(save_path)
        
        # Verify vocab preserved
        assert loaded.vocab == tokenizer.vocab
        assert loaded.vocab_size == tokenizer.vocab_size

    def test_loaded_tokenizer_encodes_same(self, tmp_path):
        """Verify loaded tokenizer produces same encodings."""
        smiles_corpus = ["CCO", "CC(=O)O"]
        
        tokenizer = RegexTokenizer()
        tokenizer.build_vocab(smiles_corpus)
        
        smiles = "CCO"
        original_ids = tokenizer.encode(smiles)
        
        # Save and load
        save_path = tmp_path / "regex_tokenizer.txt"
        tokenizer.save(save_path)
        loaded = RegexTokenizer.load(save_path)
        
        # Encode with loaded
        loaded_ids = loaded.encode(smiles)
        
        assert loaded_ids == original_ids


class TestRegexTokenizerEdgeCases:
    """Edge case tests."""

    def test_encode_unknown_tokens(self):
        """Verify tokenizer handles unseen tokens."""
        smiles_corpus = ["CCO"]
        
        tokenizer = RegexTokenizer()
        tokenizer.build_vocab(smiles_corpus)
        
        # Encode SMILES with token not in vocab
        ids = tokenizer.encode("CCBr")  # Br not trained
        
        # Should still return a list with unknown token for Br
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_empty_smiles(self):
        """Verify encoding empty string returns empty list."""
        smiles_corpus = ["CCO"]
        
        tokenizer = RegexTokenizer()
        tokenizer.build_vocab(smiles_corpus)
        
        ids = tokenizer.encode("")
        
        assert ids == []

    def test_decode_empty_list(self):
        """Verify decoding empty list returns empty string."""
        smiles_corpus = ["CCO"]
        
        tokenizer = RegexTokenizer()
        tokenizer.build_vocab(smiles_corpus)
        
        decoded = tokenizer.decode([])
        
        assert decoded == ""

    def test_repr_shows_status(self):
        """Verify repr shows initialization status."""
        tokenizer = RegexTokenizer()
        
        assert "uninitialized" in repr(tokenizer).lower()
        
        tokenizer.build_vocab(["C"])
        
        assert "initialized" in repr(tokenizer).lower()
