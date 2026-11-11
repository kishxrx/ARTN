#!/usr/bin/env python3
"""Phase 22: Fast Evaluation - Key metrics only"""
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

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*80)
print("PHASE 22: FAST EVALUATION")
print("="*80)

# Config
CONFIG = {
    'vocab_size': 4000, 'd_model': 256, 'n_heads': 8, 'n_layers': 6,
    'd_ff': 1024, 'max_seq_len': 256, 'max_length': 256,
    'batch_size': 64,
    'output_dir': 'outputs/phase22_controlled_experiment',
    'tokenizer_path': 'outputs/artn-optimized/tokenizer.json',
}

# Model classes
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
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100) if targets is not None else None
        return {"logits": logits, "loss": loss}

class TinyStoriesDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data, self.tokenizer, self.max_length = data, tokenizer, max_length
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        tokens = self.tokenizer.encode(text)[:self.max_length - 1]
        tokens = tokens + [self.tokenizer.eos_token_id]
        padding_length = self.max_length - len(tokens)
        tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}

# Load data
print("\n[Loading data]")
tokenizer = BPETokenizer.load(CONFIG['tokenizer_path'])
dataset = load_dataset("roneneldan/TinyStories", split="train")
eval_size = int(len(dataset) * 0.05)
eval_data = dataset.select(range(len(dataset) - eval_size, len(dataset)))
eval_dataset = TinyStoriesDataset(eval_data, tokenizer, CONFIG['max_length'])
eval_loader = DataLoader(eval_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
print(f"   Eval samples: {len(eval_dataset):,}")

# Build models
fp32_model = StandardGPT(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'], CONFIG['d_ff'], CONFIG['max_seq_len'])
ternary_model = BitLinearGPT(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'], CONFIG['d_ff'], CONFIG['max_seq_len'])

# Load checkpoints
out_dir = Path(CONFIG['output_dir'])
ckpts = {'fp32': {}, 'ternary': {}}
for step in [1000, 2000, 5000]:
    ckpts['fp32'][step] = torch.load(out_dir / 'fp32_checkpoints' / f'checkpoint_step_{step}.pt', map_location='cpu')
    ckpts['ternary'][step] = torch.load(out_dir / 'ternary_checkpoints' / f'checkpoint_step_{step}.pt', map_location='cpu')

# Evaluation function
def evaluate(model, loader, max_batches=50):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches: break
            outputs = model(batch["input_ids"], batch["labels"])
            logits = outputs['logits']
            loss = outputs['loss']
            labels = batch["labels"]
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            total_correct += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()
            total_loss += loss.item()
    n = min(i+1, max_batches)
    return {'loss': total_loss/n, 'perplexity': math.exp(total_loss/n), 'accuracy': total_correct/total_tokens}

# Run evaluation
print("\n[Evaluation Results]")
print(f"\n{'Step':<8} {'Model':<10} {'Eval Loss':<12} {'PPL':<10} {'Accuracy':<12}")
print("-" * 55)

results = {'fp32': {}, 'ternary': {}}

for step in [1000, 2000, 5000]:
    # FP32
    fp32_model.load_state_dict(ckpts['fp32'][step]['model_state_dict'])
    results['fp32'][step] = evaluate(fp32_model, eval_loader, max_batches=50)
    r = results['fp32'][step]
    print(f"{step:<8} {'FP32':<10} {r['loss']:<12.4f} {r['perplexity']:<10.2f} {r['accuracy']*100:<12.2f}%")
    
    # Ternary
    ternary_model.load_state_dict(ckpts['ternary'][step]['model_state_dict'])
    results['ternary'][step] = evaluate(ternary_model, eval_loader, max_batches=50)
    r = results['ternary'][step]
    print(f"{step:<8} {'Ternary':<10} {r['loss']:<12.4f} {r['perplexity']:<10.2f} {r['accuracy']*100:<12.2f}%")
    
    # Ratio
    fp32_r = results['fp32'][step]
    ternary_r = results['ternary'][step]
    print(f"{'Ratio':<8} {ternary_r['loss']/fp32_r['loss']:.2f}x{'':<6} {ternary_r['perplexity']/fp32_r['perplexity']:.2f}x{'':<5} {ternary_r['accuracy']/fp32_r['accuracy']:.2f}x")
    print()

# Context benefit (simplified)
print("\n[Context Benefit - Last Token Accuracy]")
print(f"{'Step':<8} {'Model':<10} {'Last Token Acc':<15}")
print("-" * 35)

for step in [1000, 2000, 5000]:
    for model_name, model in [('FP32', fp32_model), ('Ternary', ternary_model)]:
        if model_name == 'FP32':
            model.load_state_dict(ckpts['fp32'][step]['model_state_dict'])
        else:
            model.load_state_dict(ckpts['ternary'][step]['model_state_dict'])
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                if i >= 30: break
                outputs = model(batch["input_ids"])
                logits = outputs['logits']
                labels = batch["labels"]
                
                # Get last non-padded position
                for b in range(labels.shape[0]):
                    valid = (labels[b] != -100).nonzero(as_tuple=True)[0]
                    if len(valid) > 0:
                        last_pos = valid[-1].item()
                        pred = logits[b, last_pos].argmax().item()
                        target = labels[b, last_pos].item()
                        if pred == target:
                            correct += 1
                        total += 1
        
        acc = correct / total if total > 0 else 0
        print(f"{step:<8} {model_name:<10} {acc*100:<15.2f}%")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)

# Save results
with open(out_dir / 'phase22_fast_eval_results.json', 'w') as f:
    json.dump({k: {kk: {kkk: float(vvv) if isinstance(vvv, (int, float)) else vvv for kkk, vvv in vv.items()} for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
print(f"\nResults saved to: {out_dir / 'phase22_fast_eval_results.json'}")
