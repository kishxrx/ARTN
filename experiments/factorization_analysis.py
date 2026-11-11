#!/usr/bin/env python3
"""
ADAFACTOR FACTORIZATION ANALYSIS
=================================

Controlled experiment to test whether the factored representation produces
different parameter trajectories compared to full second-moment.

Tests:
1. Adafactor (factored updates) - row × col / global_mean
2. Full second-moment optimizer (like Adam but without bias correction)

CRITICAL: Keep EVERYTHING identical except the update calculation.

Logs at fixed checkpoints:
- Loss
- Gradient norm
- Parameter norm
- Parameter delta norm
- Delta/parameter ratio
- Cosine similarity between update vectors
- Parameter checksum/hash

Also performs line-by-line inspection of Adafactor implementation.

SHORT RUN: 500 steps to see trajectory divergence quickly.
"""

import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

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
    "total_steps": 500,  # SHORT RUN
    "warmup_steps": 50,
    "max_lr": 0.001,
    "min_lr": 3e-5,
    
    # Optimizer (IDENTICAL for both)
    "beta1": 0.5,
    "beta2": 0.98,
    "weight_decay": 0.01,
    "eps": 1e-8,  # Same eps for both
    "clip_threshold": float('inf'),  # NO CLIPPING
    
    # Data
    "seed": 42,
    
    # Checkpoints
    "log_every": 10,
    "eval_every": 100,
}

# =============================================================================
# MODEL (FP32 GPT)
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
# OPTIMIZER 1: ADAFACTOR (FACTORED)
# =============================================================================

class AdafactorFactored(torch.optim.Optimizer):
    """
    Adafactor with FACTORED second moment: row × col / global_mean
    
    Line-by-line inspection:
    
    For 2D parameters (weight matrices):
    1. Compute row-wise exp_avg_sq: exp_avg_sq_row = β₂ * exp_avg_sq_row + (1-β₂) * grad_sq.mean(dim=1)
    2. Compute col-wise exp_avg_sq: exp_avg_sq_col = β₂ * exp_avg_sq_col + (1-β₂) * grad_sq.mean(dim=0)
    3. Compute global_mean: global_mean = grad_sq.mean()
    4. Factorized variance: factored_var = (exp_avg_sq_row × exp_avg_sq_col) / (global_mean + eps1)
    5. Update: update = grad / (sqrt(factored_var) + eps2)
    
    For 1D parameters (vectors):
    1. Compute full exp_avg_sq: exp_avg_sq = β₂ * exp_avg_sq + (1-β₂) * grad²
    2. Update: update = grad / (sqrt(exp_avg_sq) + eps)
    """
    
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, 
                 eps=1e-8, weight_decay=0.0):
        defaults = {
            'lr': lr, 'beta1': beta1, 'beta2': beta2,
            'eps': eps, 'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)
        self.stats = {}
        self.last_update = None
    
    def get_stats(self):
        return self.stats.copy()
    
    def get_last_update(self):
        return self.last_update
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        total_grad_norm = 0.0
        total_update_norm = 0.0
        total_param_norm = 0.0
        updates_list = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    continue
                
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
                eps = group['eps']
                lr = group['lr']
                weight_decay = group['weight_decay']
                
                grad_norm = grad.norm().item()
                total_grad_norm += grad_norm ** 2
                param_norm = p.norm().item()
                total_param_norm += param_norm ** 2
                
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                
                # FACTORED UPDATE (Adafactor)
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
                    
                    # Factorized variance: row × col / global_mean
                    numerator = exp_avg_sq_row.unsqueeze(1) * exp_avg_sq_col.unsqueeze(0)
                    factored_var = numerator / (global_mean + eps)
                    
                    # Update
                    update = grad / (factored_var.sqrt() + eps)
                else:
                    # Full second moment for 1D
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    update = grad / (exp_avg_sq.sqrt() + eps)
                
                update_norm = update.norm().item()
                total_update_norm += update_norm ** 2
                
                if math.isnan(update_norm) or math.isinf(update_norm):
                    continue
                
                # First moment (momentum)
                if beta1 is not None:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    bias_correction = 1 - beta1 ** step
                    update = exp_avg / bias_correction
                
                updates_list.append(update.clone().flatten())
                p.add_(update, alpha=-lr)
        
        if updates_list:
            self.last_update = torch.cat(updates_list)
        else:
            self.last_update = None
        
        self.stats = {
            'grad_norm': math.sqrt(total_grad_norm),
            'update_norm': math.sqrt(total_update_norm),
            'param_norm': math.sqrt(total_param_norm),
        }
        
        return loss


# =============================================================================
# OPTIMIZER 2: FULL SECOND MOMENT (UNFACTORED)
# =============================================================================

class FullSecondMoment(torch.optim.Optimizer):
    """
    Full second moment optimizer (like Adam but without bias correction).
    
    For ALL parameters (both 2D and 1D):
    1. Compute full exp_avg_sq: exp_avg_sq = β₂ * exp_avg_sq + (1-β₂) * grad²
    2. Update: update = grad / (sqrt(exp_avg_sq) + eps)
    
    NO factorization - uses full variance for every parameter.
    """
    
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, 
                 eps=1e-8, weight_decay=0.0):
        defaults = {
            'lr': lr, 'beta1': beta1, 'beta2': beta2,
            'eps': eps, 'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)
        self.stats = {}
        self.last_update = None
    
    def get_stats(self):
        return self.stats.copy()
    
    def get_last_update(self):
        return self.last_update
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        total_grad_norm = 0.0
        total_update_norm = 0.0
        total_param_norm = 0.0
        updates_list = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    continue
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                    if group['beta1'] is not None:
                        state['exp_avg'] = torch.zeros_like(grad)
                
                state['step'] += 1
                step = state['step']
                
                beta1 = group['beta1']
                beta2 = group['beta2']
                eps = group['eps']
                lr = group['lr']
                weight_decay = group['weight_decay']
                
                grad_norm = grad.norm().item()
                total_grad_norm += grad_norm ** 2
                param_norm = p.norm().item()
                total_param_norm += param_norm ** 2
                
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                
                # FULL SECOND MOMENT (NO factorization)
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = grad / (exp_avg_sq.sqrt() + eps)
                
                update_norm = update.norm().item()
                total_update_norm += update_norm ** 2
                
                if math.isnan(update_norm) or math.isinf(update_norm):
                    continue
                
                # First moment (momentum)
                if beta1 is not None:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    bias_correction = 1 - beta1 ** step
                    update = exp_avg / bias_correction
                
                updates_list.append(update.clone().flatten())
                p.add_(update, alpha=-lr)
        
        if updates_list:
            self.last_update = torch.cat(updates_list)
        else:
            self.last_update = None
        
        self.stats = {
            'grad_norm': math.sqrt(total_grad_norm),
            'update_norm': math.sqrt(total_update_norm),
            'param_norm': math.sqrt(total_param_norm),
        }
        
        return loss


# =============================================================================
# HELPER FUNCTIONS
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


def compute_parameter_hash(model):
    """Compute SHA256 hash of all parameters."""
    hasher = hashlib.sha256()
    for p in model.parameters():
        hasher.update(p.data.cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def compute_cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None
    vec1 = vec1.float()
    vec2 = vec2.float()
    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()


# =============================================================================
# RUN EXPERIMENT
# =============================================================================

def run_experiment(optimizer_class, optimizer_name, config, train_data, val_data, device, output_dir, initial_state=None):
    print(f"\n{'='*80}")
    print(f"RUNNING: {optimizer_name}")
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
        print(f"  Loaded initial model state")
    
    # Store initial parameters
    initial_params = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Create optimizer
    optimizer = optimizer_class(
        model.parameters(),
        lr=config['max_lr'],
        beta1=config['beta1'],
        beta2=config['beta2'],
        eps=config['eps'],
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
            
            # Compute parameter delta
            current_params = model.state_dict()
            total_delta_norm = 0.0
            total_param_norm = 0.0
            for k in initial_params:
                delta = current_params[k] - initial_params[k]
                total_delta_norm += delta.norm().item() ** 2
                total_param_norm += initial_params[k].norm().item() ** 2
            
            delta_norm = math.sqrt(total_delta_norm)
            param_norm = math.sqrt(total_param_norm)
            delta_ratio = delta_norm / (param_norm + 1e-10)
            
            # Parameter hash
            param_hash = compute_parameter_hash(model)
            
            log_entry = {
                'step': step,
                'train_loss': train_loss,
                'lr': lr,
                'grad_norm': stats['grad_norm'],
                'update_norm': stats['update_norm'],
                'param_norm': stats['param_norm'],
                'delta_norm': delta_norm,
                'delta_param_ratio': delta_ratio,
                'param_hash': param_hash,
            }
            log.append(log_entry)
            
            print(f"Step {step:4d} | Loss: {train_loss:.4f} | LR: {lr:.2e} | "
                  f"Grad: {stats['grad_norm']:.4f} | Upd: {stats['update_norm']:.4f} | "
                  f"Δ/Param: {delta_ratio:.6f} | Hash: {param_hash[:8]}...")
        
        step += 1
    
    # Save results
    results = {
        'optimizer': optimizer_name,
        'config': config,
        'log': log,
    }
    
    results_path = output_dir / f'{optimizer_name}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results, optimizer


def compare_results(results_list, output_dir):
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Optimizer':<25} {'Final Loss':<12} {'Final Δ/Param':<15} {'Final Hash':<20}")
    print("-"*80)
    
    for results in results_list:
        name = results['optimizer']
        log = results['log']
        
        final_loss = log[-1]['train_loss']
        final_delta = log[-1]['delta_param_ratio']
        final_hash = log[-1]['param_hash']
        
        print(f"{name:<25} {final_loss:<12.4f} {final_delta:<15.6f} {final_hash:<20}")
    
    # Compare update cosine similarities
    print(f"\n{'='*80}")
    print("UPDATE COSINE SIMILARITY (from logged updates)")
    print(f"{'='*80}")
    
    # This requires both optimizers, so we'll compute it differently
    # We'll just compare the final parameter hashes
    hashes = [r['log'][-1]['param_hash'] for r in results_list]
    
    if len(hashes) == 2:
        if hashes[0] == hashes[1]:
            print(f"\n⚠️  IDENTICAL parameter hashes: {hashes[0]}")
            print(f"   → Both optimizers produced IDENTICAL parameter trajectories!")
        else:
            print(f"\n✓ DIFFERENT parameter hashes:")
            print(f"   Factored:   {hashes[0]}")
            print(f"   Unfactored: {hashes[1]}")
            print(f"   → Optimizers produced DIFFERENT parameter trajectories")


def inspect_adafactor_implementation():
    """Line-by-line inspection of Adafactor implementation."""
    print(f"\n{'='*80}")
    print("ADAFACTOR IMPLEMENTATION INSPECTION")
    print(f"{'='*80}")
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ ADAFACTOR FACTORED UPDATE ALGORITHM (Line-by-Line)                          │
└─────────────────────────────────────────────────────────────────────────────┘

For 2D parameters (weight matrices W ∈ R^{m×n}):

Step 1: Compute gradient squared
─────────────────────────────────────────────────────────────────────────────
  grad_sq = grad * grad  # Element-wise square

Step 2: Update row-wise second moment
─────────────────────────────────────────────────────────────────────────────
  exp_avg_sq_row = β₂ * exp_avg_sq_row + (1-β₂) * grad_sq.mean(dim=1)
  
  This computes: R_i = E[g_{ij}²] averaged over columns for each row i

Step 3: Update column-wise second moment
─────────────────────────────────────────────────────────────────────────────
  exp_avg_sq_col = β₂ * exp_avg_sq_col + (1-β₂) * grad_sq.mean(dim=0)
  
  This computes: C_j = E[g_{ij}²] averaged over rows for each column j

Step 4: Compute global mean
─────────────────────────────────────────────────────────────────────────────
  global_mean = grad_sq.mean()
  
  This computes: G = E[g_{ij}²] averaged over ALL elements

Step 5: Factorized variance estimate
─────────────────────────────────────────────────────────────────────────────
  factored_var = (exp_avg_sq_row × exp_avg_sq_col) / (global_mean + ε₁)
  
  Shape: [m, 1] × [1, n] / scalar = [m, n]
  
  INTERPRETATION:
  - Numerator: R_i × C_j approximates the variance at position (i,j)
  - Denominator: Global mean normalizes to avoid double counting

Step 6: Compute update
─────────────────────────────────────────────────────────────────────────────
  update = grad / (sqrt(factored_var) + ε₂)
  
  This divides gradient by estimated RMS at each position.

Step 7: Apply update
─────────────────────────────────────────────────────────────────────────────
  W = W - lr * update

┌─────────────────────────────────────────────────────────────────────────────┐
│ FULL SECOND MOMENT ALGORITHM (Unfactored - like Adam)                       │
└─────────────────────────────────────────────────────────────────────────────┘

For ALL parameters:

Step 1: Compute gradient squared
─────────────────────────────────────────────────────────────────────────────
  grad_sq = grad * grad  # Element-wise square

Step 2: Update second moment
─────────────────────────────────────────────────────────────────────────────
  exp_avg_sq = β₂ * exp_avg_sq + (1-β₂) * grad_sq
  
  This computes: V_ij = E[g_{ij}²] for EACH position

Step 3: Compute update
─────────────────────────────────────────────────────────────────────────────
  update = grad / (sqrt(exp_avg_sq) + ε)
  
  This divides gradient by exact RMS at each position.

Step 4: Apply update
─────────────────────────────────────────────────────────────────────────────
  W = W - lr * update

┌─────────────────────────────────────────────────────────────────────────────┐
│ KEY DIFFERENCE                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

FACTORED (Adafactor):
  - Variance at (i,j) ≈ R_i × C_j / G
  - Storage: O(m + n) instead of O(m × n)
  - APPROXIMATION: Assumes variance factorizes nicely

UNFACTORED (Full):
  - Variance at (i,j) = V_ij (exact)
  - Storage: O(m × n)
  - EXACT: No approximation

┌─────────────────────────────────────────────────────────────────────────────┐
│ POTENTIAL ISSUE                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

If global_mean → 0 (dead neurons, sparse gradients):
  factored_var = (R_i × C_j) / tiny → EXPLODES
  
If global_mean is noisy (fast β₂ decay):
  factored_var becomes UNSTABLE

Full second moment does NOT have this global_mean division!
""")


def main():
    print("="*80)
    print("ADAFACTOR FACTORIZATION ANALYSIS")
    print("="*80)
    print(f"\nComparing:")
    print(f"  1. Adafactor (factored updates)")
    print(f"  2. Full second moment (unfactored updates)")
    print(f"\nTotal steps: {CONFIG['total_steps']} (SHORT RUN)")
    print(f"CRITICAL: Same model, data, seed, LR, betas, eps, weight decay")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = BASE_DIR / 'outputs' / f'factorization_analysis_{timestamp}'
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
    initial_hash = compute_parameter_hash(initial_model)
    print(f"  Initial parameter hash: {initial_hash}")
    
    # Run factored
    results1, opt1 = run_experiment(
        AdafactorFactored, "AdafactorFactored", 
        CONFIG, train_data, val_data, device, output_dir, initial_state
    )
    
    # Run unfactored
    results2, opt2 = run_experiment(
        FullSecondMoment, "FullSecondMoment",
        CONFIG, train_data, val_data, device, output_dir, initial_state
    )
    
    results_list = [results1, results2]
    
    # Compare
    compare_results(results_list, output_dir)
    
    # Inspect implementation
    inspect_adafactor_implementation()
    
    # Final comparison
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    hash1 = results1['log'][-1]['param_hash']
    hash2 = results2['log'][-1]['param_hash']
    
    print(f"\nFinal Parameter Hashes:")
    print(f"  Factored:   {hash1}")
    print(f"  Unfactored: {hash2}")
    
    if hash1 == hash2:
        print(f"\n⚠️  CRITICAL: IDENTICAL parameter trajectories!")
        print(f"   → The factorization DOES NOT affect the result")
        print(f"   → The failure mechanism must be ELSEWHERE")
    else:
        print(f"\n✓ DIFFERENT parameter trajectories")
        print(f"   → The factorization DOES affect the result")
        print(f"   → Next: Compare final losses to see if unfactored converges better")


if __name__ == "__main__":
    main()
