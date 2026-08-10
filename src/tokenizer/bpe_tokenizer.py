"""
BPE (Byte-Pair Encoding) Tokenizer

A pure Python implementation of Byte-Pair Encoding for text tokenization.
Trained on corpus to build vocabulary of subword units.

Reference: Sennrich et al., 2016 - Neural Machine Translation of Rare Words with Subword Units
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class BPETokenizer:
    """Byte-Pair Encoding tokenizer for text.
    
    Attributes:
        vocab_size: Target vocabulary size
        merges: List of merge operations (pairs -> merged token)
        vocab: Mapping from token string to integer ID
        inverse_vocab: Mapping from integer ID to token string
    """

    def __init__(
        self,
        vocab_size: int = 4000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ):
        """Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (including special tokens)
            min_frequency: Minimum frequency for a pair to be merged
            special_tokens: Additional special tokens beyond defaults
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Merge operations learned during training
        self.merges: List[Tuple[str, str]] = []
        
        # Vocabulary mappings
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        
        # Initialize special tokens
        self.special_tokens = SPECIAL_TOKENS.copy()
        if special_tokens:
            self.special_tokens.extend(special_tokens)
        
        self._initialized = False
    
    @property
    def pad_token_id(self) -> int:
        """Get pad token ID."""
        return self.vocab.get(PAD_TOKEN, 0)
    
    @property
    def unk_token_id(self) -> int:
        """Get unk token ID."""
        return self.vocab.get(UNK_TOKEN, 1)
    
    @property
    def bos_token_id(self) -> int:
        """Get bos token ID."""
        return self.vocab.get(BOS_TOKEN, 2)
    
    @property
    def eos_token_id(self) -> int:
        """Get eos token ID."""
        return self.vocab.get(EOS_TOKEN, 3)

    def _get_pairs(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """Count all adjacent pairs in the corpus.
        
        Args:
            word_freqs: Dictionary mapping word tuples to frequencies
            
        Returns:
            Counter of (token1, token2) pairs with their frequencies
        """
        pairs = Counter()
        for word, freq in word_freqs.items():
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        return pairs

    def _merge_pair(
        self,
        word_freqs: Dict[Tuple[str, ...], int],
        pair: Tuple[str, str],
    ) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of a pair in the corpus.
        
        Args:
            word_freqs: Current word frequency dictionary
            pair: The pair to merge (token1, token2)
            
        Returns:
            Updated word frequency dictionary with merged pairs
        """
        new_word_freqs = {}
        merged_token = pair[0] + pair[1]
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
            
        return new_word_freqs

    def _tokenize_to_chars(self, text: str) -> Dict[Tuple[str, ...], int]:
        """Convert text to character-level word frequencies.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping word tuples (split by whitespace) to frequencies
        """
        # Simple whitespace tokenization, preserving word boundaries
        # Each word becomes a tuple of characters with a word-end marker
        words = re.findall(r"\S+", text)
        word_freqs = Counter()
        
        for word in words:
            # Convert word to tuple of characters
            chars = tuple(word) + ("</w>",)
            word_freqs[chars] += 1
            
        return dict(word_freqs)

    def train(self, corpus: str) -> "BPETokenizer":
        """Train BPE tokenizer on corpus.
        
        Args:
            corpus: Training text corpus
            
        Returns:
            Self for method chaining
        """
        # Initialize with character-level tokenization
        word_freqs = self._tokenize_to_chars(corpus)
        
        # Track merge operations
        self.merges = []
        
        # Build initial vocabulary from characters
        chars = set()
        for word in word_freqs:
            chars.update(word)
        
        # Add special tokens to vocab (they get IDs 0, 1, 2, 3...)
        vocab = list(self.special_tokens)
        
        # Add character tokens
        vocab.extend(sorted(chars))
        
        # Perform BPE merges until vocab_size is reached
        num_merges = self.vocab_size - len(vocab)
        
        for _ in range(num_merges):
            # Get all pairs and their frequencies
            pairs = self._get_pairs(word_freqs)
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Stop if frequency is too low
            if pairs[best_pair] < self.min_frequency:
                break
            
            # Record merge operation
            self.merges.append(best_pair)
            
            # Merge the pair in all words
            word_freqs = self._merge_pair(word_freqs, best_pair)
            
            # Add merged token to vocab
            merged = best_pair[0] + best_pair[1]
            if merged not in vocab:
                vocab.append(merged)
        
        # Build vocab mappings
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        self._initialized = True
        return self

    def _apply_bpe(self, word: str) -> List[str]:
        """Apply learned BPE merges to a single word.
        
        Args:
            word: Input word string
            
        Returns:
            List of BPE tokens
        """
        # Start with characters
        tokens = list(word) + ["</w>"]
        
        # Apply merges in order
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens = tokens[:i] + [pair[0] + pair[1]] + tokens[i + 2:]
                else:
                    i += 1
        
        return tokens

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        if not self._initialized:
            raise RuntimeError("Tokenizer must be trained before encoding")
        
        tokens = []
        
        # Tokenize by words
        words = re.findall(r"\S+", text)
        
        for word in words:
            bpe_tokens = self._apply_bpe(word)
            for tok in bpe_tokens:
                if tok in self.vocab:
                    tokens.append(self.vocab[tok])
                else:
                    # Handle unknown tokens via character fallback
                    for char in tok:
                        if char in self.vocab:
                            tokens.append(self.vocab[char])
                        else:
                            tokens.append(self.vocab[UNK_TOKEN])
        
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        if not self._initialized:
            raise RuntimeError("Tokenizer must be trained before decoding")
        
        tokens = []
        for tid in token_ids:
            if tid in self.inverse_vocab:
                tokens.append(self.inverse_vocab[tid])
            else:
                tokens.append(UNK_TOKEN)
        
        # Join tokens and clean up word-boundary markers
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        
        # Clean up special tokens from output
        for special in self.special_tokens:
            text = text.replace(special, "")
        
        return text.strip()

    @property
    def vocab_size_actual(self) -> int:
        """Get actual vocabulary size."""
        return len(self.vocab)

    def save(self, path: str | Path) -> None:
        """Save tokenizer to file.
        
        Args:
            path: File path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            # Write parameters
            f.write(f"{self.vocab_size}\n")
            f.write(f"{self.min_frequency}\n")
            
            # Write special tokens
            f.write(" ".join(self.special_tokens) + "\n")
            
            # Write merges count
            f.write(f"{len(self.merges)}\n")
            
            # Write merges
            for pair in self.merges:
                # Escape spaces in tokens using tab as separator
                f.write(f"{pair[0]}\t{pair[1]}\n")
            
            # Write full vocabulary (ensures exact reconstruction)
            for idx in range(len(self.vocab)):
                token = self.inverse_vocab[idx]
                # Escape newlines in tokens
                escaped_token = token.replace("\n", "\\n")
                f.write(f"{escaped_token}\n")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load tokenizer from file.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded BPETokenizer instance
        """
        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        vocab_size = int(lines[0].strip())
        min_frequency = int(lines[1].strip())
        special_tokens = lines[2].strip().split()
        num_merges = int(lines[3].strip())
        
        tokenizer = cls(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )
        
        # Load merges (lines 4 to 4+num_merges)
        merges = []
        for i in range(4, 4 + num_merges):
            line = lines[i].rstrip("\n")
            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
        
        tokenizer.merges = merges
        
        # Load vocabulary from remaining lines
        vocab = []
        vocab_start = 4 + num_merges
        for line in lines[vocab_start:]:
            token = line.rstrip("\n")
            # Unescape newlines
            token = token.replace("\\n", "\n")
            vocab.append(token)
        
        tokenizer.vocab = {token: idx for idx, token in enumerate(vocab)}
        tokenizer.inverse_vocab = {idx: token for token, idx in tokenizer.vocab.items()}
        tokenizer._initialized = True
        
        return tokenizer

    def __repr__(self) -> str:
        status = "trained" if self._initialized else "untrained"
        return f"BPETokenizer(vocab_size={self.vocab_size}, status={status})"
