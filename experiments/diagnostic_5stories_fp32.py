#!/usr/bin/env python3
"""
DIAGNOSTIC: FP32 Baseline vs Ternary Comparison

Purpose: Test if replacing ternary layers with FP32 improves learning.

ONLY CHANGE from previous ternary test:
  - ALL BitLinear layers → nn.Linear (FP32)

EVERYTHING else identical:
  - Same 5 TinyStories
  - Same architecture dimensions
  - Same optimizer, LR, etc.

Date: September 4, 2026
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import random
from typing import Optional
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer.bpe_tokenizer import BPETokenizer


# =============================================================================
# FP32 MODEL - IDENTICAL ARCHITECTURE TO BITLINEARGPT BUT WITH nn.Linear
# =============================================================================

class FP32Attention(nn.Module):
    """Multi-head attention with FP32 nn.Linear (identical structure to BitLinearAttention)."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # FP32 projections (replaces BitLinear)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out


class FP32FeedForward(nn.Module):
    """Feed-forward network with FP32 nn.Linear (identical structure to BitLinearFeedForward)."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        
        # FP32 layers (replaces BitLinear)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU activation (same as BitLinear version)
        x = F.gelu(self.w1(x))
        x = self.dropout(x)
        x = self.w2(x)
        return x


class FP32TransformerBlock(nn.Module):
    """Transformer block with FP32 nn.Linear (identical structure to BitLinearTransformerBlock)."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        
        # Pre-norm (same as BitLinear version)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = FP32Attention(d_model, n_heads, dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FP32FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention with residual
        x = x + self.attn(self.ln1(x), mask)
        
        # Feed-forward with residual
        x = x + self.ff(self.ln2(x))
        
        return x


class FP32GPT(nn.Module):
    """
    GPT model with FP32 nn.Linear throughout.
    
    IDENTICAL architecture to BitLinearGPT but with nn.Linear instead of BitLinear.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings (same as BitLinearGPT - already FP32)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # FP32 transformer blocks (replaces BitLinearTransformerBlock)
        self.blocks = nn.ModuleList([
            FP32TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output (same as BitLinearGPT)
        self.ln_final = nn.LayerNorm(d_model)
        
        # FP32 lm_head (replaces BitLinear)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> dict:
        """Forward pass - identical to BitLinearGPT."""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Causal mask
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=input_ids.device)
        ).unsqueeze(0).unsqueeze(0)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        
        return {"logits": logits, "loss": loss}
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """Generate tokens - identical to BitLinearGPT."""
        self.eval()
        token_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = token_ids if token_ids.size(1) <= self.max_seq_len \
                           else token_ids[:, -self.max_seq_len:]
                
                outputs = self(idx_cond)
                logits = outputs["logits"][:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_ids = torch.cat([token_ids, next_token], dim=1)
        
        return token_ids


# =============================================================================
# CONFIGURATION - IDENTICAL TO TERNARY TEST
# =============================================================================

SEED = 42

# Model - IDENTICAL dimensions
MODEL_CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 32,
    "dropout": 0.0,
}

# Optimizer - IDENTICAL
OPTIMIZER_CONFIG = {
    "optimizer": "Adam",
    "lr": 0.01,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}

# Training - IDENTICAL
TRAINING_CONFIG = {
    "num_stories": 5,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "total_steps": 500,
    "max_seq_len": 32,
}

# Tokenizer - IDENTICAL
TOKENIZER_PATH = "/root/agent-sandbox/ARTN/outputs/artn-50k-best-config/tokenizer.json"

# Output
OUTPUT_DIR = Path("/root/agent-sandbox/ARTN/outputs/diagnostic_5stories_fp32")


# =============================================================================
# DATASET - IDENTICAL TO TERNARY TEST
# =============================================================================

class TinyStoriesMemorizationDataset(Dataset):
    """Dataset - identical to ternary test."""
    
    def __init__(self, stories, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []
        
        for story in stories:
            text = story["text"]
            tokens = tokenizer.encode(text)[:max_length - 1]
            tokens = tokens + [tokenizer.eos_token_id]
            
            padding_length = max_length - len(tokens)
            tokens = tokens + [tokenizer.pad_token_id] * padding_length
            
            self.sequences.append(tokens)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.sequences[idx], dtype=torch.long)
        
        inputs = input_ids[:-1].clone()
        targets = input_ids[1:].clone()
        
        targets[targets == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": inputs, "targets": targets}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("DIAGNOSTIC: FP32 Baseline vs Ternary Comparison")
    print("="*80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print("="*80)
    
    # ==========================================================================
    # CONFIGURATION COMPARISON
    # ==========================================================================
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON: FP32 vs Ternary")
    print("="*80)
    
    print("""
| Component | Ternary Test | FP32 Test (This) |
|-----------|--------------|------------------|
| Linear layers | BitLinear (ternary) | nn.Linear (FP32) |
| Architecture | Same | Same |
| d_model | 256 | 256 |
| n_heads | 8 | 8 |
| n_layers | 6 | 6 |
| d_ff | 1024 | 1024 |
| max_seq_len | 32 | 32 |
| vocab_size | 4000 | 4000 |
| Embeddings | FP32 | FP32 |
| LayerNorm | Same | Same |
| Residuals | Same | Same |
| Causal mask | Same | Same |
| Activation | GELU | GELU |
| Optimizer | Adam | Adam |
| LR | 0.01 | 0.01 |
| beta1 | 0.9 | 0.9 |
| beta2 | 0.999 | 0.999 |
| weight_decay | 0.01 | 0.01 |
| grad_clip | 1.0 | 1.0 |
| Data | 5 TinyStories | 5 TinyStories |
| Steps | 500 | 500 |
| Seed | 42 | 42 |

ONLY CHANGE: BitLinear → nn.Linear (FP32)
""")
    
    torch.manual_seed(SEED)
    random.seed(SEED)
    
    # ==========================================================================
    # Load tokenizer
    # ==========================================================================
    print("\n[Tokenizer] Loading...")
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    print(f"  Vocab size: {len(tokenizer.vocab)}")
    print(f"  PAD: {tokenizer.pad_token_id}, EOS: {tokenizer.eos_token_id}")
    
    # ==========================================================================
    # Load 5 TinyStories - IDENTICAL TO TERNARY TEST
    # ==========================================================================
    print("\n[Dataset] Loading 5 TinyStories (same as ternary test)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    random.seed(SEED)
    indices = sorted(random.sample(range(len(dataset)), TRAINING_CONFIG["num_stories"]))
    stories = [dataset[i] for i in indices]
    
    print(f"  Selected {len(stories)} stories")
    
    train_dataset = TinyStoriesMemorizationDataset(
        stories, tokenizer, TRAINING_CONFIG["max_seq_len"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    
    print(f"  DataLoader: {len(train_dataset)} sequences, batch_size={TRAINING_CONFIG['batch_size']}")
    
    # ==========================================================================
    # Create FP32 model
    # ==========================================================================
    print("\n[Model] Creating FP32 GPT (same architecture, nn.Linear instead of BitLinear)...")
    torch.manual_seed(SEED)
    model = FP32GPT(
        vocab_size=MODEL_CONFIG["vocab_size"],
        d_model=MODEL_CONFIG["d_model"],
        n_heads=MODEL_CONFIG["n_heads"],
        n_layers=MODEL_CONFIG["n_layers"],
        d_ff=MODEL_CONFIG["d_ff"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
        dropout=MODEL_CONFIG["dropout"],
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Count linear layer params
    linear_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_params += module.weight.numel()
    print(f"  Linear layer params: {linear_params:,}")
    
    # ==========================================================================
    # Create optimizer - IDENTICAL TO TERNARY TEST
    # ==========================================================================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=OPTIMIZER_CONFIG["lr"],
        betas=(OPTIMIZER_CONFIG["beta1"], OPTIMIZER_CONFIG["beta2"]),
        weight_decay=OPTIMIZER_CONFIG["weight_decay"],
    )
    
    print(f"\n[Optimizer] Adam")
    print(f"  LR: {OPTIMIZER_CONFIG['lr']}")
    print(f"  Beta1: {OPTIMIZER_CONFIG['beta1']}, Beta2: {OPTIMIZER_CONFIG['beta2']}")
    print(f"  Weight decay: {OPTIMIZER_CONFIG['weight_decay']}")
    print(f"  Grad clip: {OPTIMIZER_CONFIG['max_grad_norm']}")
    
    # ==========================================================================
    # Training
    # ==========================================================================
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    results = {
        "config": {
            "model": MODEL_CONFIG,
            "optimizer": OPTIMIZER_CONFIG,
            "training": TRAINING_CONFIG,
        },
        "metrics": {},
    }
    
    model.train()
    data_iter = iter(train_loader)
    
    for step in range(1, TRAINING_CONFIG["total_steps"] + 1):
        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_active_loss = 0.0
        
        for _ in range(TRAINING_CONFIG["gradient_accumulation_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            input_ids = batch["input_ids"]
            targets = batch["targets"]
            
            outputs = model(input_ids, targets=targets)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            if torch.isnan(loss):
                print(f"\n[ERROR] NaN loss at step {step}")
                continue
            
            with torch.no_grad():
                active_mask = targets != -100
                active_logits = logits[active_mask]
                active_targets = targets[active_mask]
                if len(active_targets) > 0:
                    active_loss = F.cross_entropy(active_logits, active_targets)
                    accumulated_active_loss += active_loss.item()
            
            scaled_loss = loss / TRAINING_CONFIG["gradient_accumulation_steps"]
            scaled_loss.backward()
            accumulated_loss += loss.item()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), OPTIMIZER_CONFIG["max_grad_norm"]
        )
        
        optimizer.step()
        
        # Report at key steps
        if step in [1, 50, 100, 250, 500]:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                correct = (predictions == targets) & (targets != -100)
                accuracy = correct.sum().item() / (targets != -100).sum().item()
                
                perplexity = math.exp(accumulated_loss / TRAINING_CONFIG["gradient_accumulation_steps"])
                active_loss_avg = accumulated_active_loss / TRAINING_CONFIG["gradient_accumulation_steps"]
                
                # Logit statistics
                logit_rms = logits.float().pow(2).mean().sqrt().item()
                logit_max = logits.float().max().item()
            
            print(f"\nStep {step:4d}:")
            print(f"  Loss: {accumulated_loss / TRAINING_CONFIG['gradient_accumulation_steps']:.4f}")
            print(f"  Active-token loss: {active_loss_avg:.4f}")
            print(f"  Perplexity: {perplexity:.2f}")
            print(f"  Accuracy: {accuracy*100:.2f}%")
            print(f"  Grad norm: {grad_norm:.4f}")
            print(f"  Logit RMS: {logit_rms:.4f}, max: {logit_max:.4f}")
            
            results["metrics"][step] = {
                "loss": accumulated_loss / TRAINING_CONFIG["gradient_accumulation_steps"],
                "active_loss": active_loss_avg,
                "perplexity": perplexity,
                "accuracy": accuracy,
                "grad_norm": grad_norm.item(),
                "logit_rms": logit_rms,
                "logit_max": logit_max,
            }
    
    # ==========================================================================
    # Final evaluation
    # ==========================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    model.eval()
    total_correct = 0
    total_active = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for i in range(len(train_dataset)):
            batch = train_dataset[i]
            input_ids = batch["input_ids"].unsqueeze(0)
            targets = batch["targets"].unsqueeze(0)
            
            outputs = model(input_ids, targets=targets)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            predictions = logits.argmax(dim=-1)
            correct = (predictions == targets) & (targets != -100)
            
            total_correct += correct.sum().item()
            total_active += (targets != -100).sum().item()
            total_loss += loss.item()
    
    final_loss = total_loss / len(train_dataset)
    final_accuracy = total_correct / total_active if total_active > 0 else 0
    final_perplexity = math.exp(final_loss) if final_loss < 10 else float('inf')
    
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Final perplexity: {final_perplexity:.2f}")
    print(f"  Final accuracy: {final_accuracy*100:.2f}%")
    
    results["final_loss"] = final_loss
    results["final_accuracy"] = final_accuracy
    results["final_perplexity"] = final_perplexity
    
    # ==========================================================================
    # Generation - SAME PROMPT AS TERNARY TEST
    # ==========================================================================
    print("\n[Generation] From first training story prompt (same as ternary test)...")
    
    first_story_tokens = train_dataset.sequences[0]
    if tokenizer.pad_token_id in first_story_tokens:
        pad_start = first_story_tokens.index(tokenizer.pad_token_id)
    else:
        pad_start = len(first_story_tokens)
    
    real_tokens = first_story_tokens[:pad_start]
    prompt_len = min(15, len(real_tokens) - 1)
    prompt_tokens = real_tokens[:prompt_len]
    prompt_text = tokenizer.decode(prompt_tokens)
    print(f"  Prompt: {prompt_text}")
    
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        generated = model.generate(
            input_ids,
            max_new_tokens=15,
            temperature=1.0,
            top_k=50,
        )
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"  Generated: {generated_text}")
    
    results["generation"] = {
        "prompt": prompt_text,
        "generated": generated_text,
    }
    
    # ==========================================================================
    # Save results
    # ==========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "results.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Results] Saved to: {results_file}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    initial_loss = results["metrics"][1]["loss"]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss reduction: {loss_reduction:.1f}%")
    print(f"Final accuracy: {final_accuracy*100:.2f}%")
    
    print("\nDIRECT COMPARISON:")
    print(f"  Ternary (BitLinear): loss 8.8550 → 6.7483, accuracy 9.03%, reduction 23.8%")
    print(f"  FP32 (nn.Linear):    loss {initial_loss:.4f} → {final_loss:.4f}, accuracy {final_accuracy*100:.2f}%, reduction {loss_reduction:.1f}%")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
