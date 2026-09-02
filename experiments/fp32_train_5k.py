#!/usr/bin/env python3
"""
FP32 Training Script - Identical to Ternary except FP32 Linear layers

This trains a Standard FP32 GPT model with IDENTICAL configuration to the
ternary model to enable controlled comparison.

CRITICAL: This is training only. No checkpoint modification of existing models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer.bpe_tokenizer import BPETokenizer

# ============================================================
# Configuration - IDENTICAL to ternary 50k training
# ============================================================

CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    "batch_size": 8,
    "learning_rate": 0.003,
    "max_steps": 5000,
    "seed": 42,
    "checkpoint_steps": [1000, 2000, 3000, 4000, 5000],
    "eval_interval": 500,
    "device": "cpu",
}

# ============================================================
# FP32 Model Implementation - Same architecture, FP32 weights
# ============================================================

class StandardLinear(nn.Linear):
    """Standard FP32 Linear layer - no quantization"""
    pass


class StandardMultiHeadAttention(nn.Module):
    """Standard multi-head attention with FP32 linear layers"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Standard Linear layers (FP32)
        self.q_proj = StandardLinear(d_model, d_model)
        self.k_proj = StandardLinear(d_model, d_model)
        self.v_proj = StandardLinear(d_model, d_model)
        self.o_proj = StandardLinear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.o_proj(context)


class StandardFeedForward(nn.Module):
    """Standard feed-forward network with FP32 linear layers"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = StandardLinear(d_model, d_ff)
        self.w2 = StandardLinear(d_ff, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))


class StandardTransformerBlock(nn.Module):
    """Standard transformer block with FP32 layers"""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StandardMultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = StandardFeedForward(d_model, d_ff)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class StandardGPT(nn.Module):
    """Standard GPT model with FP32 layers - identical architecture to BitLinearGPT"""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Embeddings (FP32)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = StandardLinear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                reduction='mean'
            )
        
        return logits, loss


# ============================================================
# AdaFactor Optimizer - Use the SAME custom AdaFactor as ternary
# ============================================================

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from src.training.adafactor import Adafactor  # Note: lowercase 'f'


# ============================================================
# Training Functions
# ============================================================

def load_training_data(tokenizer, num_stories=50000):
    """Load and tokenize training data - IDENTICAL to ternary training"""
    print(f"\n[Loading training data...]")
    
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    stories = []
    for i, item in enumerate(dataset):
        if i >= num_stories:
            break
        stories.append(item["text"])
    
    print(f"  Loaded {len(stories)} stories")
    
    # Tokenize
    sequences = []
    for story in stories:
        tokens = tokenizer.encode(story)[:255]
        tokens = tokens + [tokenizer.eos_token_id]
        if len(tokens) >= 20:
            sequences.append(tokens)
    
    print(f"  Usable sequences: {len(sequences)}")
    
    return sequences


def create_batch(sequences, batch_size, max_seq_len, pad_token_id):
    """Create a batch from sequences"""
    batch_seqs = []
    for _ in range(batch_size):
        seq = sequences[torch.randint(0, len(sequences), (1,)).item()]
        batch_seqs.append(seq)
    
    # Pad to same length
    max_len = min(max(len(s) for s in batch_seqs), max_seq_len)
    
    input_ids = []
    targets = []
    
    for seq in batch_seqs:
        seq = seq[:max_len]
        # Pad
        padding_length = max_len - len(seq)
        seq_padded = seq + [pad_token_id] * padding_length
        
        input_ids.append(seq_padded[:-1])
        targets.append(seq_padded[1:])
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    # Mask padding in targets
    targets[targets == pad_token_id] = -100
    
    return input_ids, targets


def train_model(config, tokenizer, sequences, output_dir):
    """Train FP32 model with IDENTICAL configuration to ternary"""
    
    print(f"\n{'='*100}")
    print(f"TRAINING FP32 MODEL")
    print(f"{'='*100}")
    
    # Set seed
    torch.manual_seed(config["seed"])
    
    # Create model
    model = StandardGPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model]")
    print(f"  Total parameters: {total_params:,}")
    
    # Create optimizer - SAME AdaFactor as ternary
    optimizer = Adafactor(
        model.parameters(),
        lr=config["learning_rate"],
        beta1=0.5,
        beta2=0.999,
    )
    
    print(f"\n[Optimizer]")
    print(f"  Type: AdaFactor (custom from src/training/adafactor.py)")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Beta1: 0.5, Beta2: 0.999")
    
    # Training loop
    model.train()
    
    print(f"\n[Training]")
    print(f"  Max steps: {config['max_steps']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Checkpoint steps: {config['checkpoint_steps']}")
    
    losses = []
    start_time = time.time()
    
    for step in range(1, config["max_steps"] + 1):
        # Create batch
        input_ids, targets = create_batch(
            sequences,
            config["batch_size"],
            config["max_seq_len"],
            tokenizer.pad_token_id
        )
        
        # Forward pass
        logits, loss = model(input_ids, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Logging
        if step % 100 == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta = (config["max_steps"] - step) / steps_per_sec / 60
            
            print(f"  Step {step:5d}/{config['max_steps']}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, ETA={eta:.1f}min")
        
        # Evaluation
        if step % config["eval_interval"] == 0:
            model.eval()
            eval_losses = []
            
            for _ in range(50):
                input_ids, targets = create_batch(
                    sequences,
                    config["batch_size"],
                    config["max_seq_len"],
                    tokenizer.pad_token_id
                )
                with torch.no_grad():
                    _, loss = model(input_ids, targets)
                    eval_losses.append(loss.item())
            
            avg_eval_loss = sum(eval_losses) / len(eval_losses)
            print(f"  [EVAL] Step {step}: eval_loss={avg_eval_loss:.4f}")
            model.train()
        
        # Save checkpoint
        if step in config["checkpoint_steps"]:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, checkpoint_path)
            
            print(f"  [CHECKPOINT] Saved to {checkpoint_path}")
    
    # Save final model
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "step": config["max_steps"],
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_dir / "model.pt")
    
    # Save config
    with open(final_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[DONE] Model saved to {final_dir}")
    
    return model, losses


# ============================================================
# Main
# ============================================================

def main():
    print("="*100)
    print("FP32 TRAINING SCRIPT")
    print("="*100)
    print("""
This trains a Standard FP32 GPT model with IDENTICAL configuration to the
ternary model to enable controlled comparison.

CRITICAL: This is training only. No modification of existing models.
""")
    
    # Load tokenizer
    tokenizer_path = Path("outputs/artn-optimized/tokenizer.json")
    tokenizer = BPETokenizer.load(str(tokenizer_path))
    
    print(f"\n[Tokenizer]")
    print(f"  Vocab size: {len(tokenizer.vocab)}")
    print(f"  Pad token ID: {tokenizer.pad_token_id}")
    print(f"  EOS token ID: {tokenizer.eos_token_id}")
    
    # Load training data
    sequences = load_training_data(tokenizer, num_stories=50000)
    
    # Set output directory
    output_dir = Path("outputs/fp32-5k-comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    model, losses = train_model(CONFIG, tokenizer, sequences, output_dir)
    
    print("\n" + "="*100)
    print("TRAINING COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
