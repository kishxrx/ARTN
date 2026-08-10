"""
Regex Tokenizer for SMILES (Simplified Molecular Input Line Entry System)

A pattern-based tokenizer that preserves chemical structure semantics by
isolating multi-character atoms (Cl, Br), bracketed groups ([NH3+]), and
bond notation.

Used for ARTN-Chem (Era 2) training on molecular structures.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class RegexTokenizer:
    """Regex-based tokenizer for SMILES molecular strings.
    
    Preserves chemical semantics by isolating:
    - Multi-character atoms: Cl, Br, etc.
    - Bracketed groups: [NH3+], [O-], [C@H], etc.
    - Bond types: -, =, #, etc.
    - Branch markers: (, )
    - Ring closures: 1, 2, ..., %10, %11, etc.
    
    Attributes:
        vocab: Mapping from token string to integer ID
        inverse_vocab: Mapping from integer ID to token string
    """

    # Regex pattern for SMILES tokenization
    # Order matters - match longer patterns first
    SMILES_PATTERN = re.compile(
        r"(\[[^\]]+\])"  # Bracketed groups like [NH3+], [C@H]
        r"|(Br|Cl|Si|Se|se|As|as)"  # Two-char organic atoms
        r"|(%\d{2})"  # Two-digit ring closures (%10, %11, etc.) - MUST come before single digit
        r"|([%=#$:./@])"  # Bond types and special chars
        r"|([()])"  # Branch markers
        r"|([A-Za-z])"  # Single char atoms (C, N, O, etc.) - includes lowercase
        r"|(\d)"  # Single digit ring closures
        r"|(\\)"  # Escape characters
    )

    def __init__(
        self,
        special_tokens: Optional[List[str]] = None,
    ):
        """Initialize regex tokenizer.
        
        Args:
            special_tokens: Additional special tokens beyond defaults
        """
        self.special_tokens = SPECIAL_TOKENS.copy()
        if special_tokens:
            self.special_tokens.extend(special_tokens)
        
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self._initialized = False

    def _tokenize_smiles(self, smiles: str) -> List[str]:
        """Tokenize a single SMILES string.
        
        Args:
            smiles: SMILES molecular string
            
        Returns:
            List of tokens
        """
        tokens = []
        pos = 0
        
        while pos < len(smiles):
            match = self.SMILES_PATTERN.match(smiles, pos)
            
            if match:
                # Get the matched group (non-None)
                for group in match.groups():
                    if group:
                        tokens.append(group)
                        pos = match.end()
                        break
            else:
                # Unrecognized character, skip or treat as unknown
                pos += 1
        
        return tokens

    def build_vocab(self, smiles_corpus: List[str]) -> "RegexTokenizer":
        """Build vocabulary from SMILES corpus.
        
        Args:
            smiles_corpus: List of SMILES strings
            
        Returns:
            Self for method chaining
        """
        # Collect all unique tokens
        all_tokens: set = set()
        
        for smiles in smiles_corpus:
            tokens = self._tokenize_smiles(smiles)
            all_tokens.update(tokens)
        
        # Build vocabulary: special tokens first, then sorted regular tokens
        vocab = list(self.special_tokens)
        vocab.extend(sorted(all_tokens))
        
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        self._initialized = True
        return self

    def encode(self, text: str) -> List[int]:
        """Encode SMILES string to token IDs.
        
        Args:
            text: SMILES molecular string
            
        Returns:
            List of token IDs
        """
        if not self._initialized:
            raise RuntimeError("Tokenizer must have vocabulary built before encoding")
        
        tokens = self._tokenize_smiles(text)
        
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab[UNK_TOKEN])
        
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to SMILES string.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded SMILES string
        """
        if not self._initialized:
            raise RuntimeError("Tokenizer must have vocabulary built before decoding")
        
        tokens = []
        for tid in token_ids:
            if tid in self.inverse_vocab:
                token = self.inverse_vocab[tid]
                # Skip special tokens in output
                if token not in self.special_tokens:
                    tokens.append(token)
            # Unknown IDs are silently skipped
        
        return "".join(tokens)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def save(self, path: str | Path) -> None:
        """Save tokenizer vocabulary to file.
        
        Args:
            path: File path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            # Write special tokens
            f.write(" ".join(self.special_tokens) + "\n")
            
            # Write all tokens in order
            for idx in range(len(self.vocab)):
                f.write(self.inverse_vocab[idx] + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "RegexTokenizer":
        """Load tokenizer from file.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded RegexTokenizer instance
        """
        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        special_tokens = lines[0].strip().split()
        
        tokenizer = cls(special_tokens=special_tokens)
        
        # Build vocab from remaining lines
        vocab = []
        for line in lines[1:]:
            token = line.rstrip("\n")
            vocab.append(token)
        
        tokenizer.vocab = {token: idx for idx, token in enumerate(vocab)}
        tokenizer.inverse_vocab = {idx: token for token, idx in tokenizer.vocab.items()}
        tokenizer._initialized = True
        
        return tokenizer

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "uninitialized"
        vocab_size = len(self.vocab) if self._initialized else 0
        return f"RegexTokenizer(vocab_size={vocab_size}, status={status})"
