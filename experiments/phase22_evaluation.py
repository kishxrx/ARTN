#!/usr/bin/env python3
"""
Phase 22: Comprehensive Controlled Evaluation

Evaluate FP32 and Ternary checkpoints at 1k, 2k, 5k on:
1. Held-out evaluation CE
2. Perplexity
3. Top-1 accuracy
4. SAME-TARGET context benefit
5. SAME-TARGET shuffle effect
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
from collections import defaultdict

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*100)
print("PHASE 22: COMPREHENSIVE CONTROLLED EVALUATION")
print("="*100)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'vocab_size': 4000,
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,
    'd_ff': 1024,
    'max_seq_len': 256,
    'max_length': 256,
    'batch_size': 32,
    'eval_samples': 5000,  # Samples for evaluation
    'output_dir': 'outputs/phase22_controlled_experiment',
    'tokenizer_path': 'outputs/artn-optimized/tokenizer.json',
    'dataset': 'roneneldan/TinyStories',
}

# =============================================================================
# MODEL CLASSES
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
# DATASET
# =============================================================================

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

# =============================================================================
# LOAD DATA AND TOKENIZER
# =============================================================================

print("\n[Loading data and tokenizer]")
tokenizer = BPETokenizer.load(CONFIG['tokenizer_path'])
print(f"   Tokenizer: vocab_size={tokenizer.vocab_size}, EOS={tokenizer.eos_token_id}, PAD={tokenizer.pad_token_id}")

dataset = load_dataset(CONFIG['dataset'], split="train")
print(f"   Dataset size: {len(dataset):,}")

# Split: last 5% for evaluation
eval_size = int(len(dataset) * 0.05)
eval_data = dataset.select(range(len(dataset) - eval_size, len(dataset)))
print(f"   Eval samples: {len(eval_data):,}")

eval_dataset = TinyStoriesDataset(eval_data, tokenizer, CONFIG['max_length'])
eval_loader = DataLoader(eval_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# =============================================================================
# BUILD MODELS
# =============================================================================

print("\n[Building models]")
fp32_model = StandardGPT(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'], CONFIG['d_ff'], CONFIG['max_seq_len'])
ternary_model = BitLinearGPT(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'], CONFIG['d_ff'], CONFIG['max_seq_len'])

# =============================================================================
# LOAD CHECKPOINTS
# =============================================================================

print("\n[Loading checkpoints]")
out_dir = Path(CONFIG['output_dir'])

checkpoints = {'fp32': {}, 'ternary': {}}
for step in [1000, 2000, 5000]:
    checkpoints['fp32'][step] = torch.load(out_dir / 'fp32_checkpoints' / f'checkpoint_step_{step}.pt', map_location='cpu')
    checkpoints['ternary'][step] = torch.load(out_dir / 'ternary_checkpoints' / f'checkpoint_step_{step}.pt', map_location='cpu')
    print(f"   Step {step}: FP32 loss={checkpoints['fp32'][step]['loss']:.4f}, Ternary loss={checkpoints['ternary'][step]['loss']:.4f}")

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_standard(model, loader, max_batches=None):
    """Standard evaluation: CE, perplexity, top-1 accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            if max_batches and total_batches >= max_batches:
                break
            
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            
            outputs = model(input_ids, targets=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Accuracy
            predictions = logits.argmax(dim=-1)
            mask = labels != -100
            correct = (predictions == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
    }

def evaluate_context_benefit(model, loader, tokenizer, max_batches=50):
    """
    SAME-TARGET context benefit evaluation.
    
    Compare predictions on SAME target token/position with:
    - Short context (first N tokens)
    - Long context (full sequence)
    
    Returns: accuracy difference for same targets.
    """
    model.eval()
    
    short_correct = 0
    long_correct = 0
    total_targets = 0
    
    short_context_len = 32  # Use first 32 tokens as "short context"
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            
            # Find positions where we have valid targets
            # Target: positions after short_context_len
            for sample_idx in range(input_ids.shape[0]):
                sample_labels = labels[sample_idx]
                sample_input = input_ids[sample_idx]
                
                # Find valid target positions (after short context, not padding)
                valid_positions = []
                for pos in range(short_context_len, len(sample_labels)):
                    if sample_labels[pos] != -100:
                        valid_positions.append(pos)
                
                if len(valid_positions) == 0:
                    continue
                
                # Pick one target position (the middle one)
                target_pos = valid_positions[len(valid_positions) // 2]
                target_token = sample_labels[target_pos].item()
                
                # Short context prediction
                short_input = sample_input[:target_pos].unsqueeze(0)
                if short_input.shape[1] > short_context_len:
                    short_input = sample_input[:short_context_len].unsqueeze(0)
                    short_target_pos = short_input.shape[1] - 1
                else:
                    short_target_pos = target_pos
                
                # Long context prediction
                long_input = sample_input[:target_pos].unsqueeze(0)
                
                # Get predictions
                short_out = model(short_input)
                short_pred = short_out['logits'][0, -1].argmax().item()
                
                long_out = model(long_input)
                long_pred = long_out['logits'][0, -1].argmax().item()
                
                if short_pred == target_token:
                    short_correct += 1
                if long_pred == target_token:
                    long_correct += 1
                total_targets += 1
    
    short_acc = short_correct / total_targets if total_targets > 0 else 0
    long_acc = long_correct / total_targets if total_targets > 0 else 0
    
    return {
        'short_acc': short_acc,
        'long_acc': long_acc,
        'context_benefit': long_acc - short_acc,
        'total_targets': total_targets,
    }

def evaluate_shuffle_effect(model, loader, tokenizer, max_batches=50):
    """
    SAME-TARGET shuffle effect evaluation.
    
    Compare predictions on SAME target token/position with:
    - Original context
    - Shuffled context
    
    Returns: accuracy difference for same targets.
    """
    model.eval()
    
    original_correct = 0
    shuffled_correct = 0
    total_targets = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            
            for sample_idx in range(input_ids.shape[0]):
                sample_labels = labels[sample_idx]
                sample_input = input_ids[sample_idx]
                
                # Find valid target positions
                valid_positions = []
                for pos in range(10, len(sample_labels)):  # At least 10 tokens of context
                    if sample_labels[pos] != -100:
                        valid_positions.append(pos)
                
                if len(valid_positions) == 0:
                    continue
                
                # Pick one target position
                target_pos = valid_positions[len(valid_positions) // 2]
                target_token = sample_labels[target_pos].item()
                
                # Original context
                original_input = sample_input[:target_pos].unsqueeze(0)
                
                # Shuffled context (shuffle the input, keeping target position same)
                context = sample_input[:target_pos].clone()
                if len(context) > 1:
                    perm = torch.randperm(len(context))
                    shuffled_input = context[perm].unsqueeze(0)
                else:
                    shuffled_input = original_input
                
                # Get predictions
                original_out = model(original_input)
                original_pred = original_out['logits'][0, -1].argmax().item()
                
                shuffled_out = model(shuffled_input)
                shuffled_pred = shuffled_out['logits'][0, -1].argmax().item()
                
                if original_pred == target_token:
                    original_correct += 1
                if shuffled_pred == target_token:
                    shuffled_correct += 1
                total_targets += 1
    
    original_acc = original_correct / total_targets if total_targets > 0 else 0
    shuffled_acc = shuffled_correct / total_targets if total_targets > 0 else 0
    
    return {
        'original_acc': original_acc,
        'shuffled_acc': shuffled_acc,
        'shuffle_effect': original_acc - shuffled_acc,
        'total_targets': total_targets,
    }

# =============================================================================
# RUN EVALUATIONS
# =============================================================================

print("\n" + "="*100)
print("EVALUATION RESULTS")
print("="*100)

results = {'fp32': {}, 'ternary': {}}

for step in [1000, 2000, 5000]:
    print(f"\n{'='*100}")
    print(f"CHECKPOINT: {step} STEPS")
    print(f"{'='*100}")
    
    # FP32
    print(f"\n[FP32 @ {step}]")
    fp32_model.load_state_dict(checkpoints['fp32'][step]['model_state_dict'])
    
    print("   Standard evaluation...")
    fp32_standard = evaluate_standard(fp32_model, eval_loader, max_batches=100)
    print(f"      Loss: {fp32_standard['loss']:.4f}")
    print(f"      Perplexity: {fp32_standard['perplexity']:.2f}")
    print(f"      Accuracy: {fp32_standard['accuracy']*100:.2f}%")
    
    print("   Context benefit evaluation...")
    fp32_context = evaluate_context_benefit(fp32_model, eval_loader, tokenizer, max_batches=50)
    print(f"      Short context acc: {fp32_context['short_acc']*100:.2f}%")
    print(f"      Long context acc: {fp32_context['long_acc']*100:.2f}%")
    print(f"      Context benefit: {fp32_context['context_benefit']*100:.2f}%")
    
    print("   Shuffle effect evaluation...")
    fp32_shuffle = evaluate_shuffle_effect(fp32_model, eval_loader, tokenizer, max_batches=50)
    print(f"      Original context acc: {fp32_shuffle['original_acc']*100:.2f}%")
    print(f"      Shuffled context acc: {fp32_shuffle['shuffled_acc']*100:.2f}%")
    print(f"      Shuffle effect: {fp32_shuffle['shuffle_effect']*100:.2f}%")
    
    results['fp32'][step] = {
        'standard': fp32_standard,
        'context': fp32_context,
        'shuffle': fp32_shuffle,
    }
    
    # Ternary
    print(f"\n[Ternary @ {step}]")
    ternary_model.load_state_dict(checkpoints['ternary'][step]['model_state_dict'])
    
    print("   Standard evaluation...")
    ternary_standard = evaluate_standard(ternary_model, eval_loader, max_batches=100)
    print(f"      Loss: {ternary_standard['loss']:.4f}")
    print(f"      Perplexity: {ternary_standard['perplexity']:.2f}")
    print(f"      Accuracy: {ternary_standard['accuracy']*100:.2f}%")
    
    print("   Context benefit evaluation...")
    ternary_context = evaluate_context_benefit(ternary_model, eval_loader, tokenizer, max_batches=50)
    print(f"      Short context acc: {ternary_context['short_acc']*100:.2f}%")
    print(f"      Long context acc: {ternary_context['long_acc']*100:.2f}%")
    print(f"      Context benefit: {ternary_context['context_benefit']*100:.2f}%")
    
    print("   Shuffle effect evaluation...")
    ternary_shuffle = evaluate_shuffle_effect(ternary_model, eval_loader, tokenizer, max_batches=50)
    print(f"      Original context acc: {ternary_shuffle['original_acc']*100:.2f}%")
    print(f"      Shuffled context acc: {ternary_shuffle['shuffled_acc']*100:.2f}%")
    print(f"      Shuffle effect: {ternary_shuffle['shuffle_effect']*100:.2f}%")
    
    results['ternary'][step] = {
        'standard': ternary_standard,
        'context': ternary_context,
        'shuffle': ternary_shuffle,
    }

# =============================================================================
# COMPARISON TABLE
# =============================================================================

print("\n" + "="*100)
print("COMPREHENSIVE COMPARISON TABLE")
print("="*100)

print(f"\n{'Step':<8} {'Model':<10} {'Eval Loss':<12} {'PPL':<10} {'Accuracy':<12} {'Ctx Benefit':<12} {'Shuffle Eff':<12}")
print("-" * 80)

for step in [1000, 2000, 5000]:
    for model_name in ['fp32', 'ternary']:
        r = results[model_name][step]
        s = r['standard']
        c = r['context']
        sh = r['shuffle']
        print(f"{step:<8} {model_name.upper():<10} {s['loss']:<12.4f} {s['perplexity']:<10.2f} {s['accuracy']*100:<12.2f}% {c['context_benefit']*100:<12.2f}% {sh['shuffle_effect']*100:<12.2f}%")

print("\n" + "="*100)
print("RAW NUMERICAL RESULTS COMPLETE")
print("="*100)

# Save results
with open(out_dir / 'phase22_evaluation_results.json', 'w') as f:
    # Convert to serializable format
    save_results = {}
    for model_name in ['fp32', 'ternary']:
        save_results[model_name] = {}
        for step in [1000, 2000, 5000]:
            save_results[model_name][step] = {
                'standard': {k: float(v) if isinstance(v, (int, float)) else v for k, v in results[model_name][step]['standard'].items()},
                'context': {k: float(v) if isinstance(v, (int, float)) else v for k, v in results[model_name][step]['context'].items()},
                'shuffle': {k: float(v) if isinstance(v, (int, float)) else v for k, v in results[model_name][step]['shuffle'].items()},
            }
    json.dump(save_results, f, indent=2)

print(f"\n✅ Results saved to: {out_dir / 'phase22_evaluation_results.json'}")
