#!/usr/bin/env python3
"""
FP32 vs Ternary Comparison Evaluation

Compares FP32 and Ternary models at 5k steps using IDENTICAL evaluation methodology.

CRITICAL: This is READ-ONLY evaluation. No model training or modification.

Primary question: Does FP32 learn materially more contextual information than ternary
under otherwise identical conditions?

Report actual differences and effect sizes. No arbitrary thresholds.
"""

import torch
import torch.nn.functional as F
import sys
import json
import math
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.bitlinear import BitLinearGPT

# ============================================================
# FP32 Model Definition (same as training script)
# ============================================================

class StandardLinear(torch.nn.Linear):
    """Standard FP32 Linear layer - no quantization"""
    pass


class StandardMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = StandardLinear(d_model, d_model)
        self.k_proj = StandardLinear(d_model, d_model)
        self.v_proj = StandardLinear(d_model, d_model)
        self.o_proj = StandardLinear(d_model, d_model)
    
    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        if return_attention:
            return self.o_proj(context), attn_weights
        return self.o_proj(context)


class StandardFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = StandardLinear(d_model, d_ff)
        self.w2 = StandardLinear(d_ff, d_model)
        self.activation = torch.nn.GELU()
    
    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))


class StandardTransformerBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.attn = StandardMultiHeadAttention(d_model, n_heads)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.ff = StandardFeedForward(d_model, d_ff)
    
    def forward(self, x, mask=None, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.ln1(x), mask, return_attention=True)
            x = x + attn_out
            x = x + self.ff(self.ln2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.ln1(x), mask)
            x = x + self.ff(self.ln2(x))
            return x


class StandardGPT(torch.nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(max_seq_len, d_model)
        
        self.blocks = torch.nn.ModuleList([
            StandardTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.ln_final = torch.nn.LayerNorm(d_model)
        self.lm_head = StandardLinear(d_model, vocab_size)
    
    def forward(self, input_ids, targets=None, return_attention=False):
        batch_size, seq_len = input_ids.shape
        
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        
        attention_weights = []
        for block in self.blocks:
            if return_attention:
                x, attn_weights = block(x, mask, return_attention=True)
                attention_weights.append(attn_weights)
            else:
                x = block(x, mask)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                reduction='mean'
            )
        
        if return_attention:
            return logits, loss, attention_weights
        return logits, loss


# ============================================================
# Evaluation Functions
# ============================================================

def load_heldout_data(tokenizer, num_stories=500):
    """Load held-out evaluation data"""
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    stories = []
    for i, item in enumerate(dataset):
        if i >= 50000 and i < 50000 + num_stories:
            stories.append(item["text"])
    
    sequences = []
    for story in stories:
        tokens = tokenizer.encode(story)[:255]
        tokens = tokens + [tokenizer.eos_token_id]
        if len(tokens) >= 20:
            sequences.append(tokens)
    
    return sequences


def evaluate_basic_metrics(model, sequences, tokenizer, model_name="Model"):
    """Evaluate basic metrics: loss, accuracy"""
    print(f"\n[Evaluating {model_name} - Basic Metrics]")
    
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0
    
    model.eval()
    
    with torch.no_grad():
        for tokens in sequences:
            tokens_padded = tokens + [tokenizer.pad_token_id] * (256 - len(tokens))
            
            input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
            targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
            targets[targets == tokenizer.pad_token_id] = -100
            
            outputs = model(input_ids, targets)
            
            # Handle both dict (BitLinearGPT) and tuple (StandardGPT) returns
            if isinstance(outputs, dict):
                logits = outputs["logits"]
                loss = outputs["loss"]
            else:
                logits, loss = outputs
            
            valid_mask = targets != -100
            num_pred = valid_mask.sum().item()
            predictions = logits.argmax(dim=-1)
            correct = ((predictions == targets) & valid_mask).sum().item()
            
            total_loss += loss.item() * num_pred
            total_correct += correct
            total_predictions += num_pred
    
    avg_loss = total_loss / total_predictions
    avg_ce = total_loss / total_predictions
    accuracy = total_correct / total_predictions * 100
    
    print(f"  Cross-Entropy: {avg_ce:.4f}")
    print(f"  Top-1 Accuracy: {accuracy:.4f}%")
    print(f"  Total predictions: {total_predictions}")
    
    return {
        "cross_entropy": avg_ce,
        "accuracy": accuracy,
        "total_predictions": total_predictions,
    }


def evaluate_context_benefit(model, sequences, tokenizer, model_name="Model"):
    """Evaluate context benefit: does context help prediction?"""
    print(f"\n[Evaluating {model_name} - Context Benefit]")
    
    model.eval()
    
    losses_with_context = []
    losses_no_context = []
    
    with torch.no_grad():
        for tokens in sequences[:100]:  # Sample 100 sequences
            if len(tokens) < 10:
                continue
            
            # With context (normal)
            tokens_ctx = tokens[:50] if len(tokens) >= 50 else tokens
            tokens_padded = tokens_ctx + [tokenizer.pad_token_id] * (50 - len(tokens_ctx))
            
            input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
            targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
            targets[targets == tokenizer.pad_token_id] = -100
            
            outputs = model(input_ids, targets)
            loss_ctx = outputs["loss"] if isinstance(outputs, dict) else outputs[1]
            losses_with_context.append(loss_ctx.item())
            
            # No context (first token only predicts second)
            input_ids_noctx = torch.tensor([[tokens[0]]], dtype=torch.long)
            targets_noctx = torch.tensor([[tokens[1]]], dtype=torch.long)
            
            outputs_noctx = model(input_ids_noctx, targets_noctx)
            loss_noctx = outputs_noctx["loss"] if isinstance(outputs_noctx, dict) else outputs_noctx[1]
            losses_no_context.append(loss_noctx.item())
    
    avg_with_context = np.mean(losses_with_context)
    avg_no_context = np.mean(losses_no_context)
    context_benefit = avg_no_context - avg_with_context
    
    print(f"  Loss with context (ctx=50): {avg_with_context:.4f}")
    print(f"  Loss without context (ctx=1): {avg_no_context:.4f}")
    print(f"  Context benefit: {context_benefit:.4f}")
    
    return {
        "loss_with_context": avg_with_context,
        "loss_without_context": avg_no_context,
        "context_benefit": context_benefit,
    }


def evaluate_shuffle_sensitivity(model, sequences, tokenizer, model_name="Model"):
    """Evaluate shuffle sensitivity: does shuffling context hurt?"""
    print(f"\n[Evaluating {model_name} - Shuffle Sensitivity]")
    
    model.eval()
    
    losses_original = []
    losses_shuffled = []
    
    with torch.no_grad():
        for tokens in sequences[:100]:
            if len(tokens) < 20:
                continue
            
            # Original order
            tokens_padded = tokens[:50] + [tokenizer.pad_token_id] * (50 - len(tokens[:50]))
            input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
            targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
            targets[targets == tokenizer.pad_token_id] = -100
            
            outputs = model(input_ids, targets)
            loss_orig = outputs["loss"] if isinstance(outputs, dict) else outputs[1]
            losses_original.append(loss_orig.item())
            
            # Shuffled order
            tokens_shuffled = tokens[:50].copy()
            np.random.shuffle(tokens_shuffled)
            tokens_shuffled_padded = tokens_shuffled + [tokenizer.pad_token_id] * (50 - len(tokens_shuffled))
            input_ids_shuf = torch.tensor([tokens_shuffled_padded[:-1]], dtype=torch.long)
            targets_shuf = torch.tensor([tokens_shuffled_padded[1:]], dtype=torch.long)
            targets_shuf[targets_shuf == tokenizer.pad_token_id] = -100
            
            outputs_shuf = model(input_ids_shuf, targets_shuf)
            loss_shuf = outputs_shuf["loss"] if isinstance(outputs_shuf, dict) else outputs_shuf[1]
            losses_shuffled.append(loss_shuf.item())
    
    avg_original = np.mean(losses_original)
    avg_shuffled = np.mean(losses_shuffled)
    shuffle_effect = avg_shuffled - avg_original
    
    print(f"  Loss original order: {avg_original:.4f}")
    print(f"  Loss shuffled order: {avg_shuffled:.4f}")
    print(f"  Shuffle effect: {shuffle_effect:.4f}")
    
    return {
        "loss_original": avg_original,
        "loss_shuffled": avg_shuffled,
        "shuffle_effect": shuffle_effect,
    }


def evaluate_attention_patterns(model, sequences, tokenizer, model_name="Model"):
    """Evaluate attention patterns: entropy, specialization"""
    print(f"\n[Evaluating {model_name} - Attention Patterns]")
    
    model.eval()
    
    all_entropies = []
    all_prev_attn = []
    all_first_attn = []
    
    with torch.no_grad():
        for tokens in sequences[:50]:
            if len(tokens) < 20:
                continue
            
            tokens_padded = tokens[:50] + [tokenizer.pad_token_id] * (50 - len(tokens[:50]))
            input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
            
            # Get attention weights - only supported by StandardGPT
            # For BitLinearGPT, we skip attention analysis
            try:
                outputs = model(input_ids, return_attention=True)
                if isinstance(outputs, dict):
                    # BitLinearGPT doesn't support return_attention
                    print(f"  Note: {model_name} does not support attention extraction, skipping")
                    return {
                        "avg_entropy": float('nan'),
                        "avg_prev_attn": float('nan'),
                        "avg_first_attn": float('nan'),
                    }
                _, _, attention_weights = outputs
                
                # attention_weights: list of (batch, heads, seq, seq)
                for layer_attn in attention_weights:
                    # Average across batch
                    attn = layer_attn[0]  # (heads, seq, seq)
                    
                    # Compute entropy for each head
                    for h in range(attn.shape[0]):
                        for pos in range(attn.shape[1]):
                            probs = attn[h, pos, :pos+1].numpy()
                            if len(probs) > 0 and probs.sum() > 0:
                                probs = probs / probs.sum()
                                entropy = -np.sum(probs * np.log(probs + 1e-10))
                                all_entropies.append(entropy)
                    
                    # Previous token attention
                    for h in range(attn.shape[0]):
                        for pos in range(1, attn.shape[1]):
                            all_prev_attn.append(attn[h, pos, pos-1].item())
                    
                    # First token attention
                    for h in range(attn.shape[0]):
                        for pos in range(1, attn.shape[1]):
                            all_first_attn.append(attn[h, pos, 0].item())
            except TypeError:
                # Model doesn't support return_attention parameter
                print(f"  Note: {model_name} does not support attention extraction, skipping")
                return {
                    "avg_entropy": float('nan'),
                    "avg_prev_attn": float('nan'),
                    "avg_first_attn": float('nan'),
                }
    
    avg_entropy = np.mean(all_entropies)
    avg_prev_attn = np.mean(all_prev_attn)
    avg_first_attn = np.mean(all_first_attn)
    
    print(f"  Average attention entropy: {avg_entropy:.4f}")
    print(f"  Average previous-token attention: {avg_prev_attn:.4f}")
    print(f"  Average first-token attention: {avg_first_attn:.4f}")
    
    return {
        "avg_entropy": avg_entropy,
        "avg_prev_attn": avg_prev_attn,
        "avg_first_attn": avg_first_attn,
    }


def compute_effect_size(fp32_value, ternary_value, fp32_std=None, ternary_std=None):
    """Compute Cohen's d effect size"""
    if fp32_std is None or ternary_std is None:
        # If no std provided, just report absolute difference
        return abs(fp32_value - ternary_value)
    
    pooled_std = np.sqrt((fp32_std**2 + ternary_std**2) / 2)
    if pooled_std == 0:
        return 0.0
    return (fp32_value - ternary_value) / pooled_std


# ============================================================
# Main
# ============================================================

def main():
    print("="*100)
    print("FP32 vs TERNARY COMPARISON EVALUATION")
    print("="*100)
    print("""
This compares FP32 and Ternary models at 5k steps using IDENTICAL evaluation.

CRITICAL: READ-ONLY evaluation. No model modification.

Primary question: Does FP32 learn materially more contextual information than ternary
under otherwise identical conditions?

Report actual differences and effect sizes. No arbitrary thresholds.
""")
    
    # Load tokenizer
    tokenizer_path = Path("outputs/artn-optimized/tokenizer.json")
    tokenizer = BPETokenizer.load(str(tokenizer_path))
    
    print(f"\n[Tokenizer]")
    print(f"  Vocab size: {len(tokenizer.vocab)}")
    
    # Load held-out data
    print(f"\n[Loading held-out data...]")
    sequences = load_heldout_data(tokenizer, num_stories=500)
    print(f"  Held-out sequences: {len(sequences)}")
    
    # Load FP32 model
    print(f"\n[Loading FP32 model...]")
    fp32_checkpoint_path = Path("outputs/fp32-5k-comparison/checkpoints/checkpoint_step_5000.pt")
    
    if not fp32_checkpoint_path.exists():
        print(f"  ERROR: FP32 checkpoint not found at {fp32_checkpoint_path}")
        print(f"  Please run experiments/fp32_train_5k.py first")
        return
    
    fp32_model = StandardGPT(
        vocab_size=4000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
    )
    
    fp32_checkpoint = torch.load(fp32_checkpoint_path, map_location="cpu")
    fp32_model.load_state_dict(fp32_checkpoint["model_state_dict"])
    fp32_model.eval()
    
    print(f"  Loaded FP32 checkpoint: step {fp32_checkpoint['step']}")
    
    # Load Ternary model
    print(f"\n[Loading Ternary model...]")
    ternary_checkpoint_path = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_5000.pt")
    
    if not ternary_checkpoint_path.exists():
        print(f"  ERROR: Ternary 5k checkpoint not found at {ternary_checkpoint_path}")
        return
    
    ternary_model = BitLinearGPT(
        vocab_size=4000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
    )
    
    ternary_checkpoint = torch.load(ternary_checkpoint_path, map_location="cpu")
    ternary_model.load_state_dict(ternary_checkpoint["model_state_dict"])
    ternary_model.eval()
    
    print(f"  Loaded Ternary checkpoint: step {ternary_checkpoint.get('step', 'unknown')}")
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    print("\n" + "="*100)
    print("PART 1: BASIC METRICS")
    print("="*100)
    
    fp32_basic = evaluate_basic_metrics(fp32_model, sequences, tokenizer, "FP32")
    ternary_basic = evaluate_basic_metrics(ternary_model, sequences, tokenizer, "Ternary")
    
    print("\n[Comparison - Basic Metrics]")
    print(f"  {'Metric':<25} {'FP32':<15} {'Ternary':<15} {'Difference':<15}")
    print(f"  {'-'*70}")
    print(f"  {'Cross-Entropy':<25} {fp32_basic['cross_entropy']:<15.4f} {ternary_basic['cross_entropy']:<15.4f} {fp32_basic['cross_entropy'] - ternary_basic['cross_entropy']:+<15.4f}")
    print(f"  {'Top-1 Accuracy (%)':<25} {fp32_basic['accuracy']:<15.4f} {ternary_basic['accuracy']:<15.4f} {fp32_basic['accuracy'] - ternary_basic['accuracy']:+<15.4f}")
    
    print("\n" + "="*100)
    print("PART 2: CONTEXT BENEFIT")
    print("="*100)
    
    np.random.seed(42)
    fp32_context = evaluate_context_benefit(fp32_model, sequences, tokenizer, "FP32")
    np.random.seed(42)
    ternary_context = evaluate_context_benefit(ternary_model, sequences, tokenizer, "Ternary")
    
    print("\n[Comparison - Context Benefit]")
    print(f"  {'Metric':<30} {'FP32':<15} {'Ternary':<15} {'Difference':<15}")
    print(f"  {'-'*75}")
    print(f"  {'Loss with context':<30} {fp32_context['loss_with_context']:<15.4f} {ternary_context['loss_with_context']:<15.4f} {fp32_context['loss_with_context'] - ternary_context['loss_with_context']:+<15.4f}")
    print(f"  {'Loss without context':<30} {fp32_context['loss_without_context']:<15.4f} {ternary_context['loss_without_context']:<15.4f} {fp32_context['loss_without_context'] - ternary_context['loss_without_context']:+<15.4f}")
    print(f"  {'Context benefit':<30} {fp32_context['context_benefit']:<15.4f} {ternary_context['context_benefit']:<15.4f} {fp32_context['context_benefit'] - ternary_context['context_benefit']:+<15.4f}")
    
    print("\n" + "="*100)
    print("PART 3: SHUFFLE SENSITIVITY")
    print("="*100)
    
    np.random.seed(42)
    fp32_shuffle = evaluate_shuffle_sensitivity(fp32_model, sequences, tokenizer, "FP32")
    np.random.seed(42)
    ternary_shuffle = evaluate_shuffle_sensitivity(ternary_model, sequences, tokenizer, "Ternary")
    
    print("\n[Comparison - Shuffle Sensitivity]")
    print(f"  {'Metric':<30} {'FP32':<15} {'Ternary':<15} {'Difference':<15}")
    print(f"  {'-'*75}")
    print(f"  {'Loss original order':<30} {fp32_shuffle['loss_original']:<15.4f} {ternary_shuffle['loss_original']:<15.4f} {fp32_shuffle['loss_original'] - ternary_shuffle['loss_original']:+<15.4f}")
    print(f"  {'Loss shuffled order':<30} {fp32_shuffle['loss_shuffled']:<15.4f} {ternary_shuffle['loss_shuffled']:<15.4f} {fp32_shuffle['loss_shuffled'] - ternary_shuffle['loss_shuffled']:+<15.4f}")
    print(f"  {'Shuffle effect':<30} {fp32_shuffle['shuffle_effect']:<15.4f} {ternary_shuffle['shuffle_effect']:<15.4f} {fp32_shuffle['shuffle_effect'] - ternary_shuffle['shuffle_effect']:+<15.4f}")
    
    print("\n" + "="*100)
    print("PART 4: ATTENTION PATTERNS")
    print("="*100)
    
    np.random.seed(42)
    fp32_attn = evaluate_attention_patterns(fp32_model, sequences, tokenizer, "FP32")
    np.random.seed(42)
    ternary_attn = evaluate_attention_patterns(ternary_model, sequences, tokenizer, "Ternary")
    
    print("\n[Comparison - Attention Patterns]")
    print(f"  {'Metric':<35} {'FP32':<15} {'Ternary':<15} {'Difference':<15}")
    print(f"  {'-'*80}")
    print(f"  {'Average entropy':<35} {fp32_attn['avg_entropy']:<15.4f} {ternary_attn['avg_entropy']:<15.4f} {fp32_attn['avg_entropy'] - ternary_attn['avg_entropy']:+<15.4f}")
    print(f"  {'Average prev-token attention':<35} {fp32_attn['avg_prev_attn']:<15.4f} {ternary_attn['avg_prev_attn']:<15.4f} {fp32_attn['avg_prev_attn'] - ternary_attn['avg_prev_attn']:+<15.4f}")
    print(f"  {'Average first-token attention':<35} {fp32_attn['avg_first_attn']:<15.4f} {ternary_attn['avg_first_attn']:<15.4f} {fp32_attn['avg_first_attn'] - ternary_attn['avg_first_attn']:+<15.4f}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*100)
    print("SUMMARY: FP32 vs TERNARY at 5k STEPS")
    print("="*100)
    
    print(f"""
[Key Differences]

Cross-Entropy:
  FP32:    {fp32_basic['cross_entropy']:.4f}
  Ternary: {ternary_basic['cross_entropy']:.4f}
  Difference: {fp32_basic['cross_entropy'] - ternary_basic['cross_entropy']:+.4f}

Top-1 Accuracy:
  FP32:    {fp32_basic['accuracy']:.4f}%
  Ternary: {ternary_basic['accuracy']:.4f}%
  Difference: {fp32_basic['accuracy'] - ternary_basic['accuracy']:+.4f}%

Context Benefit:
  FP32:    {fp32_context['context_benefit']:.4f}
  Ternary: {ternary_context['context_benefit']:.4f}
  Difference: {fp32_context['context_benefit'] - ternary_context['context_benefit']:+.4f}

Shuffle Effect:
  FP32:    {fp32_shuffle['shuffle_effect']:.4f}
  Ternary: {ternary_shuffle['shuffle_effect']:.4f}
  Difference: {fp32_shuffle['shuffle_effect'] - ternary_shuffle['shuffle_effect']:+.4f}

Attention Entropy:
  FP32:    {fp32_attn['avg_entropy']:.4f}
  Ternary: {ternary_attn['avg_entropy']:.4f}
  Difference: {fp32_attn['avg_entropy'] - ternary_attn['avg_entropy']:+.4f}
""")
    
    # Save results
    results = {
        "fp32": {
            "basic": fp32_basic,
            "context": fp32_context,
            "shuffle": fp32_shuffle,
            "attention": fp32_attn,
        },
        "ternary": {
            "basic": ternary_basic,
            "context": ternary_context,
            "shuffle": ternary_shuffle,
            "attention": ternary_attn,
        },
        "differences": {
            "cross_entropy": fp32_basic['cross_entropy'] - ternary_basic['cross_entropy'],
            "accuracy": fp32_basic['accuracy'] - ternary_basic['accuracy'],
            "context_benefit": fp32_context['context_benefit'] - ternary_context['context_benefit'],
            "shuffle_effect": fp32_shuffle['shuffle_effect'] - ternary_shuffle['shuffle_effect'],
            "attention_entropy": fp32_attn['avg_entropy'] - ternary_attn['avg_entropy'],
        }
    }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, float) and (obj != obj):  # NaN check
            return None
        else:
            return obj
    
    results = convert_to_native(results)
    
    output_path = Path("experiments/results/fp32_vs_ternary_5k_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Results saved to {output_path}]")
    
    print("\n" + "="*100)
    print("EVALUATION COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
