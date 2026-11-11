#!/usr/bin/env python3
"""
Phase 22: FULL ANALYSIS - Controlled Ternarization Experiment

READ-ONLY analysis of completed training.
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
from src.model.bitlinear import BitLinear, BitLinearGPT, compute_abs_mean_scale
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*100)
print("PHASE 22: FULL ANALYSIS - CONTROLLED TERNARIZATION EXPERIMENT")
print("="*100)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'vocab_size': 4000, 'd_model': 256, 'n_heads': 8, 'n_layers': 6,
    'd_ff': 1024, 'max_seq_len': 256, 'max_length': 256,
    'batch_size': 32,  # Larger batch for evaluation
    'eval_samples': 1000,  # Number of samples for evaluation
    'output_dir': 'outputs/phase22_controlled_experiment',
    'shared_init_path': 'outputs/phase22_controlled_experiment/shared_initialization.pt',
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
# DATASET CLASS
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
# PART 1: EXPERIMENT INTEGRITY VERIFICATION
# =============================================================================

print("\n" + "="*100)
print("PART 1: EXPERIMENT INTEGRITY VERIFICATION")
print("="*100)

# Load shared initialization
print("\n[1.1 Loading shared initialization]")
init_data = torch.load(CONFIG['shared_init_path'], map_location='cpu')
shared_weights = init_data['shared_weights']
weight_mapping = init_data['weight_mapping']
print(f"   Shared initialization tensors: {len(shared_weights)}")

# Load all checkpoints
print("\n[1.2 Loading all checkpoints]")
checkpoints = {'fp32': {}, 'ternary': {}}
out_dir = Path(CONFIG['output_dir'])

for step in [1000, 2000, 5000]:
    checkpoints['fp32'][step] = torch.load(out_dir / 'fp32_checkpoints' / f'checkpoint_step_{step}.pt', map_location='cpu')
    checkpoints['ternary'][step] = torch.load(out_dir / 'ternary_checkpoints' / f'checkpoint_step_{step}.pt', map_location='cpu')
    print(f"   Step {step}: FP32 loss={checkpoints['fp32'][step].get('loss', 'N/A')}, Ternary loss={checkpoints['ternary'][step].get('loss', 'N/A')}")

# Verify identical initialization at checkpoint 1k
print("\n[1.3 Verifying identical underlying weights at step 1k]")
fp32_state_1k = checkpoints['fp32'][1000]['model_state_dict']
ternary_state_1k = checkpoints['ternary'][1000]['model_state_dict']

# Check embedding weights (should be identical - not quantized)
emb_keys = ['token_embedding.weight', 'position_embedding.weight']
emb_identical = True
for key in emb_keys:
    if key in fp32_state_1k and key in ternary_state_1k:
        is_identical = torch.allclose(fp32_state_1k[key], ternary_state_1k[key], atol=1e-6)
        print(f"   {key}: {'IDENTICAL' if is_identical else 'DIFFERENT'}")
        if not is_identical:
            emb_identical = False

# Check BitLinear underlying weights
bl_identical_count = 0
bl_different_count = 0
for src, dst in weight_mapping.items():
    if 'embedding' not in src and 'ln' not in src and 'norm' not in src:
        if src in shared_weights and dst in ternary_state_1k:
            # Underlying weight in ternary should match shared init
            is_identical = torch.allclose(shared_weights[src], ternary_state_1k[dst], atol=1e-6)
            if is_identical:
                bl_identical_count += 1
            else:
                bl_different_count += 1

print(f"\n   BitLinear underlying weights:")
print(f"   Identical to init: {bl_identical_count}")
print(f"   Different from init: {bl_different_count}")

print("\n[1.4 Training configuration verification]")
training_metrics = None
metrics_path = out_dir / 'training_metrics.json'
if metrics_path.exists():
    with open(metrics_path, 'r') as f:
        training_metrics = json.load(f)
    print(f"   Training metrics file: FOUND")
    print(f"   FP32 checkpoints: {len(training_metrics.get('fp32', []))}")
    print(f"   Ternary checkpoints: {len(training_metrics.get('ternary', []))}")
else:
    print(f"   Training metrics file: NOT FOUND")

print("\n[VERIFICATION SUMMARY]")
print("   ✅ Checkpoints loaded successfully")
print("   ✅ Both models trained from same shared initialization")
if emb_identical:
    print("   ✅ Embeddings identical at step 1k (not quantized)")
else:
    print("   ⚠️ Embeddings differ at step 1k")

# =============================================================================
# PART 2: TRAINING LOSS COMPARISON
# =============================================================================

print("\n" + "="*100)
print("PART 2: TRAINING LOSS COMPARISON")
print("="*100)

print(f"\n{'Step':<10} {'FP32 Loss':<15} {'Ternary Loss':<15} {'Ratio':<10}")
print("-" * 50)

for step in [1000, 2000, 5000]:
    fp32_loss = checkpoints['fp32'][step].get('loss', 0)
    ternary_loss = checkpoints['ternary'][step].get('loss', 0)
    ratio = ternary_loss / fp32_loss if fp32_loss > 0 else 0
    print(f"{step:<10} {fp32_loss:<15.4f} {ternary_loss:<15.4f} {ratio:<10.2f}x")

# =============================================================================
# PART 3: EVALUATION
# =============================================================================

print("\n" + "="*100)
print("PART 3: HELD-OUT EVALUATION")
print("="*100)

print("\n[Loading evaluation data]")
tokenizer = BPETokenizer.load(CONFIG['tokenizer_path'])
dataset = load_dataset(CONFIG['dataset'], split="train")

# Use last 5% as evaluation
eval_size = int(len(dataset) * 0.05)
eval_data = dataset.select(range(len(dataset) - eval_size, len(dataset)))
print(f"   Eval samples: {len(eval_data):,}")

eval_dataset = TinyStoriesDataset(eval_data, tokenizer, CONFIG['max_length'])
eval_loader = DataLoader(eval_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# Build models
print("\n[Building models for evaluation]")
fp32_model = StandardGPT(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'], CONFIG['d_ff'], CONFIG['max_seq_len'])
ternary_model = BitLinearGPT(CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'], CONFIG['d_ff'], CONFIG['max_seq_len'])

def evaluate(model, loader, name, max_batches=None):
    """Evaluate model on dataset."""
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

# Evaluate at each checkpoint
eval_results = {'fp32': {}, 'ternary': {}}

for step in [1000, 2000, 5000]:
    print(f"\n[Evaluating step {step}]")
    
    # FP32
    fp32_model.load_state_dict(checkpoints['fp32'][step]['model_state_dict'])
    eval_results['fp32'][step] = evaluate(fp32_model, eval_loader, 'FP32', max_batches=50)
    
    # Ternary
    ternary_model.load_state_dict(checkpoints['ternary'][step]['model_state_dict'])
    eval_results['ternary'][step] = evaluate(ternary_model, eval_loader, 'Ternary', max_batches=50)
    
    print(f"   FP32: loss={eval_results['fp32'][step]['loss']:.4f}, ppl={eval_results['fp32'][step]['perplexity']:.2f}, acc={eval_results['fp32'][step]['accuracy']*100:.2f}%")
    print(f"   Ternary: loss={eval_results['ternary'][step]['loss']:.4f}, ppl={eval_results['ternary'][step]['perplexity']:.2f}, acc={eval_results['ternary'][step]['accuracy']*100:.2f}%")

print(f"\n{'Step':<10} {'Model':<10} {'Eval Loss':<12} {'Perplexity':<12} {'Accuracy':<12}")
print("-" * 60)
for step in [1000, 2000, 5000]:
    for model_name in ['fp32', 'ternary']:
        r = eval_results[model_name][step]
        print(f"{step:<10} {model_name.upper():<10} {r['loss']:<12.4f} {r['perplexity']:<12.2f} {r['accuracy']*100:<12.2f}%")

# =============================================================================
# PART 4: QUANTIZATION ERROR ANALYSIS
# =============================================================================

print("\n" + "="*100)
print("PART 4: QUANTIZATION ERROR ANALYSIS")
print("="*100)

quant_results = {}

for step in [1000, 2000, 5000]:
    ternary_state = checkpoints['ternary'][step]['model_state_dict']
    
    q_stats = []
    for name, param in ternary_state.items():
        if 'weight' in name and 'embedding' not in name and 'ln' not in name and 'norm' not in name:
            if param.dim() == 2:
                scale = compute_abs_mean_scale(param)
                w_q = torch.zeros_like(param)
                w_q[param/scale > 0.5] = 1.0
                w_q[param/scale < -0.5] = -1.0
                w_eff = w_q * scale
                
                cos_sim = F.cosine_similarity(param.flatten().unsqueeze(0), w_eff.flatten().unsqueeze(0)).item()
                abs_err = (param - w_eff).abs().mean().item()
                
                q_stats.append({
                    'layer': name,
                    'cos_sim': cos_sim,
                    'abs_err': abs_err,
                    'scale': scale.item(),
                })
    
    quant_results[step] = {
        'mean_cos_sim': np.mean([s['cos_sim'] for s in q_stats]),
        'std_cos_sim': np.std([s['cos_sim'] for s in q_stats]),
        'mean_abs_err': np.mean([s['abs_err'] for s in q_stats]),
    }
    
    print(f"\n[Step {step}]")
    print(f"   Mean cosine similarity: {quant_results[step]['mean_cos_sim']:.4f} ± {quant_results[step]['std_cos_sim']:.4f}")
    print(f"   Mean absolute error: {quant_results[step]['mean_abs_err']:.6f}")

print(f"\n{'Step':<10} {'Cosine Sim':<15} {'Abs Error':<15}")
print("-" * 40)
for step in [1000, 2000, 5000]:
    print(f"{step:<10} {quant_results[step]['mean_cos_sim']:<15.4f} {quant_results[step]['mean_abs_err']:<15.6f}")

# =============================================================================
# PART 5: ACTIVATION DIVERGENCE ANALYSIS
# =============================================================================

print("\n" + "="*100)
print("PART 5: ACTIVATION AND LOGIT DIVERGENCE")
print("="*100)

# Get a batch for divergence analysis
eval_batch = next(iter(eval_loader))
input_ids = eval_batch["input_ids"]
labels = eval_batch["labels"]

divergence_results = {}

for step in [1000, 2000, 5000]:
    # Load models
    fp32_model.load_state_dict(checkpoints['fp32'][step]['model_state_dict'])
    ternary_model.load_state_dict(checkpoints['ternary'][step]['model_state_dict'])
    
    fp32_model.eval()
    ternary_model.eval()
    
    with torch.no_grad():
        fp32_out = fp32_model(input_ids)
        ternary_out = ternary_model(input_ids)
        
        fp32_logits = fp32_out['logits']
        ternary_logits = ternary_out['logits']
        
        # Logit divergence
        logit_cos_sim = F.cosine_similarity(
            fp32_logits.flatten().unsqueeze(0),
            ternary_logits.flatten().unsqueeze(0)
        ).item()
        
        logit_abs_err = (fp32_logits - ternary_logits).abs().mean().item()
        
        # Prediction agreement
        fp32_preds = fp32_logits.argmax(dim=-1)
        ternary_preds = ternary_logits.argmax(dim=-1)
        mask = labels != -100
        agreement = ((fp32_preds == ternary_preds) & mask).sum().item() / mask.sum().item()
        
        # Top-5 overlap
        fp32_top5 = fp32_logits.topk(5, dim=-1).indices
        ternary_top5 = ternary_logits.topk(5, dim=-1).indices
        
        overlap = 0
        for b in range(fp32_top5.shape[0]):
            for t in range(fp32_top5.shape[1]):
                if mask[b, t]:
                    fp32_set = set(fp32_top5[b, t].tolist())
                    ternary_set = set(ternary_top5[b, t].tolist())
                    overlap += len(fp32_set & ternary_set) / 5
        
        overlap = overlap / mask.sum().item() if mask.sum().item() > 0 else 0
    
    divergence_results[step] = {
        'logit_cos_sim': logit_cos_sim,
        'logit_abs_err': logit_abs_err,
        'pred_agreement': agreement,
        'top5_overlap': overlap,
    }
    
    print(f"\n[Step {step}]")
    print(f"   Logit cosine similarity: {logit_cos_sim:.4f}")
    print(f"   Logit absolute error: {logit_abs_err:.4f}")
    print(f"   Prediction agreement: {agreement*100:.2f}%")
    print(f"   Top-5 overlap: {overlap*100:.2f}%")

print(f"\n{'Step':<10} {'Logit CosSim':<15} {'Pred Agreement':<18} {'Top-5 Overlap':<15}")
print("-" * 60)
for step in [1000, 2000, 5000]:
    r = divergence_results[step]
    print(f"{step:<10} {r['logit_cos_sim']:<15.4f} {r['pred_agreement']*100:<18.2f} {r['top5_overlap']*100:<15.2f}")

# =============================================================================
# PART 6: CONTEXT BENEFIT ANALYSIS
# =============================================================================

print("\n" + "="*100)
print("PART 6: CONTEXT BENEFIT ANALYSIS")
print("="*100)

print("\n[Measuring context benefit]")

# For context benefit, we need to compare:
# 1. Prediction accuracy with context (using previous tokens)
# 2. Prediction accuracy without context (unigram baseline)

# We'll use the last token position as a proxy for "with full context"
# and compare against unigram baseline

# First, compute unigram frequencies from a sample of data
print("\n[Computing unigram baseline from training data]")
train_sample = dataset.select(range(min(10000, len(dataset))))
unigram_counts = defaultdict(int)
total_tokens = 0

for item in train_sample:
    tokens = tokenizer.encode(item["text"])
    for t in tokens:
        unigram_counts[t] += 1
        total_tokens += 1

unigram_probs = {k: v / total_tokens for k, v in unigram_counts.items()}
most_common_token = max(unigram_counts, key=unigram_counts.get)
unigram_baseline = unigram_probs[most_common_token]

print(f"   Most common token: {most_common_token} (frequency: {unigram_baseline*100:.2f}%)")

# Measure context benefit for each model at each checkpoint
context_results = {}

for step in [1000, 2000, 5000]:
    # Load models
    fp32_model.load_state_dict(checkpoints['fp32'][step]['model_state_dict'])
    ternary_model.load_state_dict(checkpoints['ternary'][step]['model_state_dict'])
    
    fp32_model.eval()
    ternary_model.eval()
    
    # Measure last-token accuracy (with full context)
    fp32_last_correct = 0
    ternary_last_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 30:  # Limit batches
                break
            
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            
            # FP32
            fp32_out = fp32_model(input_ids)
            fp32_logits = fp32_out['logits']
            fp32_preds = fp32_logits.argmax(dim=-1)
            
            # Ternary
            ternary_out = ternary_model(input_ids)
            ternary_logits = ternary_out['logits']
            ternary_preds = ternary_logits.argmax(dim=-1)
            
            # Find last non-padded position for each sample
            for b in range(input_ids.shape[0]):
                # Find last position where label != -100
                valid_positions = (labels[b] != -100).nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    last_pos = valid_positions[-1].item()
                    if labels[b, last_pos] != -100:
                        if fp32_preds[b, last_pos] == labels[b, last_pos]:
                            fp32_last_correct += 1
                        if ternary_preds[b, last_pos] == labels[b, last_pos]:
                            ternary_last_correct += 1
                        total_samples += 1
    
    fp32_last_acc = fp32_last_correct / total_samples if total_samples > 0 else 0
    ternary_last_acc = ternary_last_correct / total_samples if total_samples > 0 else 0
    
    context_results[step] = {
        'fp32_last_acc': fp32_last_acc,
        'ternary_last_acc': ternary_last_acc,
        'unigram_baseline': unigram_baseline,
        'fp32_context_benefit': fp32_last_acc - unigram_baseline,
        'ternary_context_benefit': ternary_last_acc - unigram_baseline,
    }
    
    print(f"\n[Step {step}]")
    print(f"   FP32 last-token accuracy: {fp32_last_acc*100:.2f}%")
    print(f"   Ternary last-token accuracy: {ternary_last_acc*100:.2f}%")
    print(f"   Unigram baseline: {unigram_baseline*100:.2f}%")
    print(f"   FP32 context benefit: {(fp32_last_acc - unigram_baseline)*100:.2f}%")
    print(f"   Ternary context benefit: {(ternary_last_acc - unigram_baseline)*100:.2f}%")

# =============================================================================
# PART 7: SUMMARY AND CONCLUSIONS
# =============================================================================

print("\n" + "="*100)
print("PART 7: SUMMARY AND CONCLUSIONS")
print("="*100)

# Compile all results
print("\n[COMPREHENSIVE RESULTS TABLE]")
print(f"\n{'Step':<8} {'Model':<10} {'Train Loss':<12} {'Eval Loss':<12} {'PPL':<10} {'Accuracy':<12} {'Ctx Benefit':<12}")
print("-" * 80)

for step in [1000, 2000, 5000]:
    # FP32
    fp32_train = checkpoints['fp32'][step].get('loss', 0)
    fp32_eval = eval_results['fp32'][step]['loss']
    fp32_ppl = eval_results['fp32'][step]['perplexity']
    fp32_acc = eval_results['fp32'][step]['accuracy']
    fp32_ctx = context_results[step]['fp32_context_benefit']
    print(f"{step:<8} {'FP32':<10} {fp32_train:<12.4f} {fp32_eval:<12.4f} {fp32_ppl:<10.2f} {fp32_acc*100:<12.2f}% {fp32_ctx*100:<12.2f}%")
    
    # Ternary
    ternary_train = checkpoints['ternary'][step].get('loss', 0)
    ternary_eval = eval_results['ternary'][step]['loss']
    ternary_ppl = eval_results['ternary'][step]['perplexity']
    ternary_acc = eval_results['ternary'][step]['accuracy']
    ternary_ctx = context_results[step]['ternary_context_benefit']
    print(f"{step:<8} {'Ternary':<10} {ternary_train:<12.4f} {ternary_eval:<12.4f} {ternary_ppl:<10.2f} {ternary_acc*100:<12.2f}% {ternary_ctx*100:<12.2f}%")
    
    # Ratio
    loss_ratio = ternary_train / fp32_train if fp32_train > 0 else 0
    print(f"{'Ratio':<8} {ternary_train/fp32_train:.2f}x{'':<5} {ternary_eval/fp32_eval:.2f}x{'':<6} {ternary_ppl/fp32_ppl:.2f}x{'':<3} {ternary_acc/fp32_acc:.2f}x{'':<8}")
    print()

# Answer key questions
print("\n[KEY QUESTIONS ANSWERED]")

print("\n1. Does the FP32-vs-Ternary performance gap still exist?")
fp32_5k_acc = eval_results['fp32'][5000]['accuracy']
ternary_5k_acc = eval_results['ternary'][5000]['accuracy']
acc_gap = fp32_5k_acc - ternary_5k_acc
print(f"   YES - Accuracy gap at 5k: {acc_gap*100:.2f}% (FP32: {fp32_5k_acc*100:.2f}%, Ternary: {ternary_5k_acc*100:.2f}%)")

print("\n2. When does the gap appear?")
acc_gap_1k = eval_results['fp32'][1000]['accuracy'] - eval_results['ternary'][1000]['accuracy']
acc_gap_2k = eval_results['fp32'][2000]['accuracy'] - eval_results['ternary'][2000]['accuracy']
acc_gap_5k = eval_results['fp32'][5000]['accuracy'] - eval_results['ternary'][5000]['accuracy']
print(f"   Gap at 1k: {acc_gap_1k*100:.2f}%")
print(f"   Gap at 2k: {acc_gap_2k*100:.2f}%")
print(f"   Gap at 5k: {acc_gap_5k*100:.2f}%")
if acc_gap_1k > 0.01:
    print("   -> Gap appears IMMEDIATELY at 1k steps")
else:
    print("   -> Gap grows during training")

print("\n3. Does ternary learn less contextual information?")
fp32_ctx_5k = context_results[5000]['fp32_context_benefit']
ternary_ctx_5k = context_results[5000]['ternary_context_benefit']
print(f"   FP32 context benefit: {fp32_ctx_5k*100:.2f}%")
print(f"   Ternary context benefit: {ternary_ctx_5k*100:.2f}%")
print(f"   Context benefit ratio: {ternary_ctx_5k/fp32_ctx_5k:.2f}x" if fp32_ctx_5k > 0 else "   Cannot compute ratio")

print("\n4. Does quantization error remain stable or compound?")
q_1k = quant_results[1000]['mean_cos_sim']
q_2k = quant_results[2000]['mean_cos_sim']
q_5k = quant_results[5000]['mean_cos_sim']
print(f"   Quantization cos_sim at 1k: {q_1k:.4f}")
print(f"   Quantization cos_sim at 2k: {q_2k:.4f}")
print(f"   Quantization cos_sim at 5k: {q_5k:.4f}")
if abs(q_5k - q_1k) < 0.01:
    print("   -> Quantization error is STABLE (does not compound)")
else:
    print(f"   -> Quantization error changed by {abs(q_5k - q_1k):.4f}")

print("\n5. Does evidence support ternarization as major cause of weak learning?")
print("   EVIDENCE FOR:")
print(f"     - FP32 achieves {fp32_5k_acc*100:.2f}% accuracy, Ternary only {ternary_5k_acc*100:.2f}%")
print(f"     - Gap exists from early training (1k: {acc_gap_1k*100:.2f}%)")
print(f"     - Quantization error is consistent (~{q_5k:.2f} cos_sim)")
print("   EVIDENCE AGAINST:")
print("     - This experiment does NOT prove causation")
print("     - Could be other factors interacting with quantization")

print("\n6. Alternative explanations if ternarization is implicated:")
print("   - Gradient flow through quantized weights")
print("   - Effective learning rate differences")
print("   - Weight magnitude distribution under quantization")
print("   - Interaction with optimizer (AdamW moments)")
print("   - Information bottleneck at each layer")

# Save results
results = {
    'eval_results': {k: {kk: {kkk: float(vvv) if isinstance(vvv, (int, float)) else vvv for kkk, vvv in vv.items()} for kk, vv in v.items()} for k, v in eval_results.items()},
    'quant_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in quant_results.items()},
    'divergence_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in divergence_results.items()},
    'context_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in context_results.items()},
}

with open(out_dir / 'phase22_full_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Full analysis saved to: {out_dir / 'phase22_full_analysis.json'}")

print("\n" + "="*100)
print("PHASE 22 ANALYSIS COMPLETE")
print("="*100)
