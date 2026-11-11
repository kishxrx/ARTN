#!/usr/bin/env python3
"""
ADAFACTOR EPS2 ABLATION - NO CLIPPING
=====================================

Controlled experiment to test whether eps2 is the CAUSAL factor in Adafactor failure.

Tests:
- eps2=1e-3 (current failing config)
- eps2=1e-8 (proposed fix - matches AdamW default)

CRITICAL: 
- Update clipping DISABLED to see actual effect
- beta2=0.98 (kept identical, known to NOT be the cause)
- Keep everything else IDENTICAL

Logs:
- Train/val loss
- Gradient norm
- Raw update norm (NOT clipped)
- Parameter norm
- Update/parameter ratio
- Second-moment statistics
- Effective update scale

KEY QUESTION: Does changing ONLY eps2 materially change convergence?

If eps2=1e-8 improves convergence → eps2 is causal contributor
If both fail similarly → investigate factorization mechanism itself
"""

import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import numpy as np

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from src.tokenizer.bpe_tokenizer import BPETokenizer

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Model
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    
    # Training
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "total_steps": 2000,
    "warmup_steps": 200,
    "max_lr": 0.001,
    "min_lr": 3e-5,
    
    # Optimizer (IDENTICAL for both, except eps2)
    "beta1": 0.5,
    "beta2": 0.98,  # KEPT IDENTICAL (known NOT to be the cause)
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "eps1": 1e-30,
    # eps2 varies: 1e-3 vs 1e-8
    "clip_threshold": float('inf'),  # DISABLED
    
    # Data
    "train_samples": 50000,
    "val_samples": 5000,
    
    # Evaluation
    "eval_every": 100,
    "log_every": 50,
    
    # Seed
    "seed": 42,
}

EPS2_VALUES = [1e-3, 1e-8]

# =============================================================================
# MODEL (FP32 GPT) - Same as before
# =============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool(), float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class FP32GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return {"logits": logits, "loss": loss}


# =============================================================================
# ADAFACTOR WITH DETAILED STATISTICS (NO CLIPPING)
# =============================================================================

class AdafactorNoClip(torch.optim.Optimizer):
    """
    Adafactor optimizer WITHOUT update clipping.
    Includes NaN/Inf safety check.
    """
    
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, 
                 eps1=1e-30, eps2=1e-3, weight_decay=0.0):
        defaults = {
            'lr': lr, 'beta1': beta1, 'beta2': beta2,
            'eps1': eps1, 'eps2': eps2, 'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)
        self.stats = {}
    
    def get_stats(self):
        return self.stats.copy()
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Statistics collection
        total_grad_norm = 0.0
        total_update_norm = 0.0
        total_param_norm = 0.0
        total_row_var = 0.0
        total_col_var = 0.0
        total_global_mean = 0.0
        total_effective_lr = 0.0
        num_2d_params = 0
        num_1d_params = 0
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # NaN/Inf check
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    print(f"WARNING: NaN/Inf in gradient! Skipping update.")
                    continue
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    if grad.dim() == 2:
                        state['exp_avg_sq_row'] = torch.zeros(grad.shape[0], dtype=grad.dtype, device=grad.device)
                        state['exp_avg_sq_col'] = torch.zeros(grad.shape[1], dtype=grad.dtype, device=grad.device)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                    if group['beta1'] is not None:
                        state['exp_avg'] = torch.zeros_like(grad)
                
                state['step'] += 1
                step = state['step']
                
                beta1 = group['beta1']
                beta2 = group['beta2']
                eps1 = group['eps1']
                eps2 = group['eps2']
                lr = group['lr']
                weight_decay = group['weight_decay']
                
                # Gradient norm
                grad_norm = grad.norm().item()
                total_grad_norm += grad_norm ** 2
                
                # Parameter norm
                param_norm = p.norm().item()
                total_param_norm += param_norm ** 2
                
                # Apply weight decay
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                
                # Compute second moment estimate
                if grad.dim() == 2:
                    grad_sq = grad * grad
                    
                    # Row average
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_row.mul_(beta2).add_(grad_sq.mean(dim=1), alpha=1 - beta2)
                    
                    # Column average
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    exp_avg_sq_col.mul_(beta2).add_(grad_sq.mean(dim=0), alpha=1 - beta2)
                    
                    # Global mean
                    global_mean = grad_sq.mean().item()
                    total_global_mean += global_mean
                    num_2d_params += 1
                    
                    # Row/col variance stats
                    total_row_var += exp_avg_sq_row.var().item()
                    total_col_var += exp_avg_sq_col.var().item()
                    
                    # Factorized estimate
                    numerator = exp_avg_sq_row.unsqueeze(1) * exp_avg_sq_col.unsqueeze(0)
                    factored_var = numerator / (global_mean + eps1)
                    update = grad / (factored_var.sqrt() + eps2)
                    
                    # Effective LR calculation
                    effective_lr = lr / (factored_var.sqrt().mean().item() + eps2)
                    total_effective_lr += effective_lr
                    
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    update = grad / (exp_avg_sq.sqrt() + eps2)
                    num_1d_params += 1
                
                # NO CLIPPING - raw update
                update_norm = update.norm().item()
                total_update_norm += update_norm ** 2
                
                # NaN/Inf check on update
                if math.isnan(update_norm) or math.isinf(update_norm):
                    print(f"WARNING: NaN/Inf in update norm ({update_norm})! Skipping update.")
                    continue
                
                # First moment (momentum)
                if beta1 is not None:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    bias_correction = 1 - beta1 ** step
                    update = exp_avg / bias_correction
                
                # Apply update
                p.add_(update, alpha=-lr)
        
        # Store statistics
        self.stats = {
            'grad_norm': math.sqrt(total_grad_norm),
            'update_norm': math.sqrt(total_update_norm),
            'param_norm': math.sqrt(total_param_norm),
            'update_param_ratio': math.sqrt(total_update_norm) / (math.sqrt(total_param_norm) + 1e-10),
            'row_var': total_row_var / max(num_2d_params, 1),
            'col_var': total_col_var / max(num_2d_params, 1),
            'global_mean': total_global_mean / max(num_2d_params, 1),
            'effective_lr': total_effective_lr / max(num_2d_params, 1),
            'num_2d_params': num_2d_params,
            'num_1d_params': num_1d_params,
        }
        
        return loss


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def get_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def load_data():
    train_data = torch.load(BASE_DIR / 'outputs/tokenized_cache/train_tokenized.pt', weights_only=False)
    val_data = torch.load(BASE_DIR / 'outputs/tokenized_cache/val_tokenized.pt', weights_only=False)
    return train_data, val_data


def get_batch(data, batch_size, seq_len, device):
    indices = np.random.choice(len(data), size=batch_size, replace=True)
    sequences = []
    for idx in indices:
        story = data[idx]
        if len(story) >= seq_len:
            sequences.append(story[:seq_len])
        else:
            padded = story + [0] * (seq_len - len(story))
            sequences.append(padded)
    return torch.tensor(sequences, dtype=torch.long, device=device)


def evaluate(model, val_data, config, device, num_batches=10):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            input_ids = get_batch(val_data, config['batch_size'], config['max_seq_len'], device)
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            output = model(input_ids, targets=targets)
            total_loss += output['loss'].item()
    return total_loss / num_batches


def run_experiment(eps2, config, train_data, val_data, device, output_dir, initial_state=None):
    print(f"\n{'='*80}")
    print(f"RUNNING: eps2={eps2:.1e}")
    print(f"{'='*80}")
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create model
    model = FP32GPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
    ).to(device)
    
    if initial_state is not None:
        model.load_state_dict(initial_state)
        print(f"  Loaded initial model state (identical starting point)")
    
    # Create optimizer
    optimizer = AdafactorNoClip(
        model.parameters(),
        lr=config['max_lr'],
        beta1=config['beta1'],
        beta2=config['beta2'],
        eps1=config['eps1'],
        eps2=eps2,
        weight_decay=config['weight_decay'],
    )
    
    log = []
    step = 0
    
    while step < config['total_steps']:
        optimizer.zero_grad()
        for _ in range(config['gradient_accumulation_steps']):
            input_ids = get_batch(train_data, config['batch_size'], config['max_seq_len'], device)
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            output = model(input_ids, targets=targets)
            loss = output['loss'] / config['gradient_accumulation_steps']
            loss.backward()
        
        lr = get_lr(step, config['max_lr'], config['min_lr'], config['warmup_steps'], config['total_steps'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        stats = optimizer.get_stats()
        
        if step % config['log_every'] == 0:
            train_loss = loss.item() * config['gradient_accumulation_steps']
            log_entry = {
                'step': step,
                'train_loss': train_loss,
                'lr': lr,
                'grad_norm': stats['grad_norm'],
                'update_norm': stats['update_norm'],
                'param_norm': stats['param_norm'],
                'update_param_ratio': stats['update_param_ratio'],
                'row_var': stats['row_var'],
                'col_var': stats['col_var'],
                'global_mean': stats['global_mean'],
                'effective_lr': stats['effective_lr'],
            }
            log.append(log_entry)
            
            print(f"Step {step:5d} | Loss: {train_loss:.4f} | LR: {lr:.2e} | "
                  f"Grad: {stats['grad_norm']:.4f} | Upd: {stats['update_norm']:.4f} | "
                  f"Upd/Param: {stats['update_param_ratio']:.6f} | "
                  f"EffLR: {stats['effective_lr']:.4e}")
        
        if step > 0 and step % config['eval_every'] == 0:
            val_loss = evaluate(model, val_data, config, device)
            log[-1]['val_loss'] = val_loss
            print(f"         | Val Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
        
        step += 1
    
    val_loss = evaluate(model, val_data, config, device, num_batches=50)
    final_entry = {
        'step': step,
        'val_loss': val_loss,
        'val_perplexity': math.exp(val_loss),
    }
    log.append(final_entry)
    
    print(f"\nFINAL: Val Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
    
    results = {
        'eps2': eps2,
        'config': config,
        'log': log,
    }
    
    results_path = output_dir / f'eps2_{eps2:.1e}_results.json'.replace('e-0', 'e-').replace('e+0', 'e+')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return results


def compare_results(results_list, output_dir):
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Eps2':<12} {'Final Train Loss':<18} {'Final Val Loss':<18} {'Final Val PPL':<15}")
    print("-"*80)
    
    for results in results_list:
        eps2 = results['eps2']
        log = results['log']
        
        train_losses = [e['train_loss'] for e in log if 'train_loss' in e]
        final_train = train_losses[-1] if train_losses else None
        
        val_entry = [e for e in log if 'val_loss' in e and 'val_perplexity' in e]
        if val_entry:
            final_val = val_entry[-1]['val_loss']
            final_ppl = val_entry[-1]['val_perplexity']
        else:
            final_val = None
            final_ppl = None
        
        print(f"{eps2:.1e:<12} {final_train:<18.4f} {final_val:<18.4f} {final_ppl:<15.2f}")
    
    print(f"\n{'='*80}")
    print("UPDATE DYNAMICS COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Eps2':<12} {'Avg Grad Norm':<15} {'Avg Update Norm':<18} {'Avg Upd/Param':<15}")
    print("-"*80)
    
    for results in results_list:
        eps2 = results['eps2']
        log = results['log']
        
        grad_norms = [e['grad_norm'] for e in log if 'grad_norm' in e]
        update_norms = [e['update_norm'] for e in log if 'update_norm' in e]
        update_ratios = [e['update_param_ratio'] for e in log if 'update_param_ratio' in e]
        
        avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        avg_update = sum(update_norms) / len(update_norms) if update_norms else 0
        avg_ratio = sum(update_ratios) / len(update_ratios) if update_ratios else 0
        
        print(f"{eps2:.1e:<12} {avg_grad:<15.4f} {avg_update:<18.4f} {avg_ratio:<15.6f}")
    
    print(f"\n{'='*80}")
    print("SECOND-MOMENT STATISTICS")
    print(f"{'='*80}")
    
    print(f"\n{'Eps2':<12} {'Avg Row Var':<15} {'Avg Col Var':<15} {'Avg Global Mean':<18}")
    print("-"*80)
    
    for results in results_list:
        eps2 = results['eps2']
        log = results['log']
        
        row_vars = [e['row_var'] for e in log if 'row_var' in e]
        col_vars = [e['col_var'] for e in log if 'col_var' in e]
        global_means = [e['global_mean'] for e in log if 'global_mean' in e]
        
        avg_row = sum(row_vars) / len(row_vars) if row_vars else 0
        avg_col = sum(col_vars) / len(col_vars) if col_vars else 0
        avg_global = sum(global_means) / len(global_means) if global_means else 0
        
        print(f"{eps2:.1e:<12} {avg_row:<15.6f} {avg_col:<15.6f} {avg_global:<18.6e}")
    
    print(f"\n{'='*80}")
    print("EFFECTIVE LEARNING RATE")
    print(f"{'='*80}")
    
    print(f"\n{'Eps2':<12} {'Avg Effective LR':<18} {'Min Effective LR':<18} {'Max Effective LR':<18}")
    print("-"*80)
    
    for results in results_list:
        eps2 = results['eps2']
        log = results['log']
        
        effective_lrs = [e['effective_lr'] for e in log if 'effective_lr' in e]
        
        avg_eff = sum(effective_lrs) / len(effective_lrs) if effective_lrs else 0
        min_eff = min(effective_lrs) if effective_lrs else 0
        max_eff = max(effective_lrs) if effective_lrs else 0
        
        print(f"{eps2:.1e:<12} {avg_eff:<18.4e} {min_eff:<18.4e} {max_eff:<18.4e}")
    
    # Save comparison
    comparison = {
        'eps2_values': [r['eps2'] for r in results_list],
        'final_train_losses': [],
        'final_val_losses': [],
        'final_val_ppls': [],
        'avg_grad_norms': [],
        'avg_update_norms': [],
        'avg_update_ratios': [],
        'avg_effective_lrs': [],
    }
    
    for results in results_list:
        log = results['log']
        train_losses = [e['train_loss'] for e in log if 'train_loss' in e]
        val_entries = [e for e in log if 'val_loss' in e and 'val_perplexity' in e]
        grad_norms = [e['grad_norm'] for e in log if 'grad_norm' in e]
        update_norms = [e['update_norm'] for e in log if 'update_norm' in e]
        update_ratios = [e['update_param_ratio'] for e in log if 'update_param_ratio' in e]
        effective_lrs = [e['effective_lr'] for e in log if 'effective_lr' in e]
        
        comparison['final_train_losses'].append(train_losses[-1] if train_losses else None)
        comparison['final_val_losses'].append(val_entries[-1]['val_loss'] if val_entries else None)
        comparison['final_val_ppls'].append(val_entries[-1]['val_perplexity'] if val_entries else None)
        comparison['avg_grad_norms'].append(sum(grad_norms) / len(grad_norms) if grad_norms else 0)
        comparison['avg_update_norms'].append(sum(update_norms) / len(update_norms) if update_norms else 0)
        comparison['avg_update_ratios'].append(sum(update_ratios) / len(update_ratios) if update_ratios else 0)
        comparison['avg_effective_lrs'].append(sum(effective_lrs) / len(effective_lrs) if effective_lrs else 0)
    
    comparison_path = output_dir / 'comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to: {comparison_path}")


def main():
    print("="*80)
    print("ADAFACTOR EPS2 ABLATION - NO CLIPPING")
    print("="*80)
    print(f"\nTesting eps2 values: {EPS2_VALUES}")
    print(f"Total steps: {CONFIG['total_steps']}")
    print(f"CRITICAL: Update clipping DISABLED")
    print(f"beta2=0.98 (kept IDENTICAL - known NOT to be the cause)")
    print(f"Keep everything else IDENTICAL")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = BASE_DIR / 'outputs' / f'adafactor_eps2_ablation_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print("\nLoading data...")
    train_data, val_data = load_data()
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    
    print("\nCreating initial model state...")
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    initial_model = FP32GPT(
        vocab_size=CONFIG['vocab_size'],
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        d_ff=CONFIG['d_ff'],
        max_seq_len=CONFIG['max_seq_len'],
    )
    initial_state = {k: v.clone() for k, v in initial_model.state_dict().items()}
    print(f"  Initial state created with {len(initial_state)} parameter tensors")
    
    results_list = []
    for eps2 in EPS2_VALUES:
        results = run_experiment(eps2, CONFIG, train_data, val_data, device, output_dir, initial_state)
        results_list.append(results)
    
    compare_results(results_list, output_dir)
    
    print(f"\n{'='*80}")
    print("ABLATION COMPLETE")
    print(f"{'='*80}")
    
    print("\nCAUSAL QUESTION: Does changing ONLY eps2 change convergence?")
    
    val_losses = [r['log'][-1].get('val_loss', float('inf')) for r in results_list]
    
    print(f"\nFinal Val Losses:")
    for i, eps2 in enumerate(EPS2_VALUES):
        print(f"  eps2={eps2:.1e}: {val_losses[i]:.4f}")
    
    if len(val_losses) == 2:
        ratio = val_losses[0] / val_losses[1] if val_losses[1] > 0 else float('inf')
        print(f"\nVal Loss Ratio (eps2=1e-3 / eps2=1e-8): {ratio:.2f}x")
        
        if ratio > 2.0:
            print("\n✅ STRONG EVIDENCE: eps2=1e-8 dramatically improves convergence")
            print("   → eps2 is a CAUSAL factor in Adafactor failure")
        elif ratio > 1.5:
            print("\n⚠️  MODERATE EVIDENCE: eps2=1e-8 improves convergence")
            print("   → eps2 contributes to failure, but other factors may be involved")
        elif abs(ratio - 1.0) < 0.1:
            print("\n❌ WEAK EVIDENCE: eps2 change has MINIMAL effect")
            print("   → eps2 is NOT the primary cause")
            print("   → NEXT: Investigate factorization mechanism itself")
        else:
            print("\n⚠️  UNEXPECTED: Results need further analysis")


if __name__ == "__main__":
    main()
