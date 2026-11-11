#!/usr/bin/env python3
"""
Phase 22: Controlled Ternarization Experiment - FULL TRAINING SCRIPT

Uses SAME data pipeline as train_optimized.py for experimental control.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sys
import json
import math
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from src.model.bitlinear import BitLinear, BitLinearGPT, compute_abs_mean_scale
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*100)
print("PHASE 22: CONTROLLED TERNARIZATION - FULL TRAINING")
print("="*100)

# Configuration
CONFIG = {
    'vocab_size': 4000, 'd_model': 256, 'n_heads': 8, 'n_layers': 6,
    'd_ff': 1024, 'max_seq_len': 256, 'batch_size': 4,
    'gradient_accumulation_steps': 8, 'max_steps': 5000,
    'learning_rate': 3e-4, 'min_lr': 3e-5, 'warmup_steps': 500,
    'weight_decay': 0.01, 'max_grad_norm': 1.0,
    'checkpoint_steps': [1000, 2000, 5000],
    'seed': 42, 'max_samples': 50000, 'max_length': 256,
    'eval_ratio': 0.05,
    'tokenizer_path': 'outputs/artn-optimized/tokenizer.json',
    'dataset': 'roneneldan/TinyStories',
    'output_dir': 'outputs/phase22_controlled_experiment',
    'shared_init_path': 'outputs/phase22_controlled_experiment/shared_initialization.pt',
}

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# Dataset Class - SAME AS train_optimized.py
# =============================================================================

class TinyStoriesDataset(Dataset):
    """Dataset for TinyStories - IDENTICAL to train_optimized.py"""
    
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)[:self.max_length - 1]
        
        # Add EOS token
        tokens = tokens + [self.tokenizer.eos_token_id]
        
        # Pad
        padding_length = self.max_length - len(tokens)
        tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
        
        # Create tensors
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        
        # Mask padding (but NOT EOS)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": input_ids, "labels": labels}

# =============================================================================
# Model Classes
# =============================================================================

class StandardLinear(nn.Linear):
    pass

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.q_proj = StandardLinear(d_model, d_model)
        self.k_proj = StandardLinear(d_model, d_model)
        self.v_proj = StandardLinear(d_model, d_model)
        self.o_proj = StandardLinear(d_model, d_model)
    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None: scores = scores.masked_fill(mask == 0, float('-inf'))
        out = torch.matmul(F.softmax(scores, dim=-1), v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

class StandardFF(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1, self.w2 = StandardLinear(d_model, d_ff), StandardLinear(d_ff, d_model)
    def forward(self, x): return self.w2(F.gelu(self.w1(x)))

class StandardBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1, self.attn, self.ln2, self.ff = nn.LayerNorm(d_model), StandardAttention(d_model, n_heads), nn.LayerNorm(d_model), StandardFF(d_model, d_ff)
    def forward(self, x, mask=None): return x + self.ff(self.ln2(x + self.attn(self.ln1(x), mask)))

class StandardGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([StandardBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = StandardLinear(d_model, vocab_size)
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        x = self.token_embedding(input_ids) + self.position_embedding(torch.arange(T, device=input_ids.device))
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks: x = block(x, mask)
        logits = self.lm_head(self.ln_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

# =============================================================================
# Load Data - SAME AS train_optimized.py
# =============================================================================

print("\n[Loading data from HuggingFace]")
dataset = load_dataset(CONFIG['dataset'], split="train")
print(f"   Dataset size: {len(dataset):,}")

# Load tokenizer
tokenizer = BPETokenizer.load(CONFIG['tokenizer_path'])
print(f"   Tokenizer vocab_size: {tokenizer.vocab_size}")
print(f"   EOS token ID: {tokenizer.eos_token_id}")
print(f"   PAD token ID: {tokenizer.pad_token_id}")

# Split into train/eval
eval_size = int(len(dataset) * CONFIG['eval_ratio'])
train_data = dataset.select(range(len(dataset) - eval_size))
eval_data = dataset.select(range(len(dataset) - eval_size, len(dataset)))
print(f"   Train samples: {len(train_data):,}")
print(f"   Eval samples: {len(eval_data):,}")

# Create datasets
train_dataset = TinyStoriesDataset(train_data, tokenizer, CONFIG['max_length'])
eval_dataset = TinyStoriesDataset(eval_data, tokenizer, CONFIG['max_length'])

print(f"   Train dataset size: {len(train_dataset):,}")

# =============================================================================
# Load Shared Initialization
# =============================================================================

print("\n[Loading shared initialization]")
init_data = torch.load(CONFIG['shared_init_path'], map_location='cpu')
shared_weights, weight_mapping = init_data['shared_weights'], init_data['weight_mapping']
print(f"   Loaded {len(shared_weights)} shared tensors")

# =============================================================================
# Build Models
# =============================================================================

print("\n[Building models]")

# Build FP32 model
fp32_model = StandardGPT(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'], CONFIG['d_ff'], CONFIG['max_seq_len'])
fp32_state = fp32_model.state_dict()
for k in shared_weights:
    if k in fp32_state: fp32_state[k] = shared_weights[k].clone()
fp32_model.load_state_dict(fp32_state)
print(f"   FP32 model: {sum(p.numel() for p in fp32_model.parameters()):,} params")

# Build Ternary model
ternary_model = BitLinearGPT(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'], CONFIG['d_ff'], CONFIG['max_seq_len'])
ternary_state = ternary_model.state_dict()
for src, dst in weight_mapping.items():
    if src in shared_weights and dst in ternary_state: ternary_state[dst] = shared_weights[src].clone()
ternary_model.load_state_dict(ternary_state)
print(f"   Ternary model: {sum(p.numel() for p in ternary_model.parameters()):,} params")

# =============================================================================
# Verify Weights Identical
# =============================================================================

print("\n[Verifying underlying weights are identical]")
identical = sum(1 for src, dst in weight_mapping.items() if src in shared_weights and dst in ternary_state and torch.allclose(shared_weights[src], ternary_state[dst], atol=1e-8))
total = len([1 for src, dst in weight_mapping.items() if src in shared_weights and dst in ternary_state])
print(f"   Identical: {identical}/{total}")

if identical != total:
    print("   ERROR: Weights not identical!")
    sys.exit(1)

# =============================================================================
# Optimizers and Schedulers
# =============================================================================

fp32_opt = torch.optim.AdamW(fp32_model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
ternary_opt = torch.optim.AdamW(ternary_model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])

def get_lr(step):
    if step < CONFIG['warmup_steps']: return CONFIG['learning_rate'] * step / CONFIG['warmup_steps']
    progress = (step - CONFIG['warmup_steps']) / (CONFIG['max_steps'] - CONFIG['warmup_steps'])
    return CONFIG['min_lr'] + (CONFIG['learning_rate'] - CONFIG['min_lr']) * 0.5 * (1 + math.cos(math.pi * progress))

# =============================================================================
# DataLoaders
# =============================================================================

torch.manual_seed(42)  # Same seed for shuffling
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

# =============================================================================
# Output Directory
# =============================================================================

out_dir = Path(CONFIG['output_dir'])
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / 'fp32_checkpoints').mkdir(exist_ok=True)
(out_dir / 'ternary_checkpoints').mkdir(exist_ok=True)

# =============================================================================
# Metrics Storage
# =============================================================================

metrics = {'fp32': [], 'ternary': []}

# =============================================================================
# Training Loop
# =============================================================================

print("\n[Training]")
fp32_model.train()
ternary_model.train()
train_iter = iter(train_loader)

for step in range(1, CONFIG['max_steps'] + 1):
    # Get batch ONCE - both models use SAME batch
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)
    
    input_ids = batch["input_ids"]
    targets = batch["labels"]
    
    # FP32 forward/backward
    fp32_out = fp32_model(input_ids, targets)
    fp32_loss = fp32_out['loss'] / CONFIG['gradient_accumulation_steps']
    fp32_loss.backward()
    
    # Ternary forward/backward
    ternary_out = ternary_model(input_ids, targets)
    ternary_loss = ternary_out['loss'] / CONFIG['gradient_accumulation_steps']
    ternary_loss.backward()
    
    # Gradient accumulation
    if step % CONFIG['gradient_accumulation_steps'] == 0:
        # Update LR
        lr = get_lr(step)
        for pg in fp32_opt.param_groups: pg['lr'] = lr
        for pg in ternary_opt.param_groups: pg['lr'] = lr
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(fp32_model.parameters(), CONFIG['max_grad_norm'])
        torch.nn.utils.clip_grad_norm_(ternary_model.parameters(), CONFIG['max_grad_norm'])
        
        # Step optimizers
        fp32_opt.step()
        ternary_opt.step()
        fp32_opt.zero_grad()
        ternary_opt.zero_grad()
    
    # Logging
    if step % 100 == 0:
        print(f"   Step {step}: FP32 loss={fp32_out['loss'].item():.4f}, Ternary loss={ternary_out['loss'].item():.4f}")
    
    # Checkpoints
    if step in CONFIG['checkpoint_steps']:
        print(f"\n[Checkpoint at step {step}]")
        
        # Save checkpoints
        torch.save({'step': step, 'model_state_dict': fp32_model.state_dict(), 'loss': fp32_out['loss'].item()}, out_dir / 'fp32_checkpoints' / f'checkpoint_step_{step}.pt')
        torch.save({'step': step, 'model_state_dict': ternary_model.state_dict(), 'loss': ternary_out['loss'].item()}, out_dir / 'ternary_checkpoints' / f'checkpoint_step_{step}.pt')
        
        # Measure quantization error
        q_stats = []
        for name, param in ternary_model.named_parameters():
            if 'weight' in name and 'embedding' not in name and 'ln' not in name and param.dim() == 2:
                scale = compute_abs_mean_scale(param)
                w_q = torch.zeros_like(param)
                w_q[param/scale > 0.5] = 1.0
                w_q[param/scale < -0.5] = -1.0
                w_eff = w_q * scale
                cos_sim = F.cosine_similarity(param.flatten().unsqueeze(0), w_eff.flatten().unsqueeze(0)).item()
                q_stats.append(cos_sim)
        
        mean_q = np.mean(q_stats)
        
        metrics['fp32'].append({'step': step, 'loss': fp32_out['loss'].item()})
        metrics['ternary'].append({'step': step, 'loss': ternary_out['loss'].item(), 'quant_cos_sim': mean_q})
        
        print(f"   FP32 loss: {fp32_out['loss'].item():.4f}")
        print(f"   Ternary loss: {ternary_out['loss'].item():.4f}")
        print(f"   Quantization cos_sim: {mean_q:.4f}")

# Save final metrics
with open(out_dir / 'training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n" + "="*100)
print("TRAINING COMPLETE")
print("="*100)
print(f"Checkpoints saved to: {out_dir}")
