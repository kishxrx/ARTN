#!/usr/bin/env python3
"""
FP32 vs Ternary Comparison Evaluation - CORRECTED METHODOLOGY

This script uses SAME-TARGET evaluation methodology:
- Context benefit: Predict SAME token with DIFFERENT context lengths
- Shuffle test: Predict SAME token with original vs shuffled context

CRITICAL: This is READ-ONLY evaluation. No model training or modification.
"""

import torch
import torch.nn.functional as F
import sys
import numpy as np
from pathlib import Path
from datasets import load_dataset
import json

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.bitlinear import BitLinearGPT

# FP32 Model Definition
import math

class StandardLinear(torch.nn.Linear):
    pass

class StandardMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = StandardLinear(d_model, d_model)
        self.k_proj = StandardLinear(d_model, d_model)
        self.v_proj = StandardLinear(d_model, d_model)
        self.o_proj = StandardLinear(d_model, d_model)
    
    def forward(self, x, mask=None):
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
    
    def forward(self, x, mask=None):
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
        self.max_seq_len = max_seq_len
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(max_seq_len, d_model)
        self.blocks = torch.nn.ModuleList([StandardTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_final = torch.nn.LayerNorm(d_model)
        self.lm_head = StandardLinear(d_model, vocab_size)
    
    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1), ignore_index=-100, reduction='mean')
        return logits, loss

print("="*100)
print("FP32 vs Ternary Comparison - CORRECTED METHODOLOGY")
print("="*100)

# Load tokenizer
tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')

# Load test data
dataset = load_dataset("roneneldan/TinyStories", split="train")
test_stories = [dataset[i]["text"] for i in range(50000, 50500)]

sequences = []
for story in test_stories:
    tokens = tokenizer.encode(story)[:100]
    tokens = tokens + [tokenizer.eos_token_id]
    if len(tokens) >= 50:
        sequences.append(tokens)

print(f"\n[Data]")
print(f"  Test sequences: {len(sequences)}")

# Load FP32 model
print(f"\n[Loading FP32 Model]")
fp32_ckpt = torch.load('outputs/fp32-5k-comparison/checkpoints/checkpoint_step_5000.pt', map_location='cpu')
fp32_model = StandardGPT(vocab_size=4000, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_seq_len=256)
fp32_model.load_state_dict(fp32_ckpt['model_state_dict'])
fp32_model.eval()
print(f"  Loaded FP32 checkpoint: step {fp32_ckpt['step']}")

# Load Ternary model
print(f"\n[Loading Ternary Model]")
ternary_ckpt = torch.load('outputs/artn-50k-best-config/checkpoints/checkpoint_step_5000.pt', map_location='cpu')
ternary_model = BitLinearGPT(vocab_size=4000, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_seq_len=256)
ternary_model.load_state_dict(ternary_ckpt['model_state_dict'])
ternary_model.eval()
print(f"  Loaded Ternary checkpoint: step {ternary_ckpt['step']}")

# ============================================================
# PART 1: BASIC METRICS
# ============================================================

print("\n" + "="*100)
print("PART 1: BASIC METRICS")
print("="*100)

def evaluate_basic_metrics(model, sequences, tokenizer, model_name="Model"):
    """Evaluate basic metrics: loss, accuracy"""
    print(f"\n[Evaluating {model_name} - Basic Metrics]")
    
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0
    
    with torch.no_grad():
        for tokens in sequences:
            if len(tokens) < 10:
                continue
            
            # Pad to 50 tokens
            tokens_padded = tokens[:50] + [tokenizer.pad_token_id] * (50 - len(tokens[:50]))
            
            input_ids = torch.tensor([tokens_padded[:-1]], dtype=torch.long)
            targets = torch.tensor([tokens_padded[1:]], dtype=torch.long)
            
            # Create mask for non-pad tokens
            mask = targets != tokenizer.pad_token_id
            
            outputs = model(input_ids, targets)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[1]
            
            total_loss += loss.item()
            
            predictions = logits.argmax(dim=-1)
            correct = (predictions == targets) & mask
            total_correct += correct.sum().item()
            total_predictions += mask.sum().item()
    
    avg_ce = total_loss / len(sequences)
    accuracy = 100.0 * total_correct / total_predictions
    
    print(f"  Cross-Entropy: {avg_ce:.4f}")
    print(f"  Top-1 Accuracy: {accuracy:.4f}%")
    print(f"  Total predictions: {total_predictions}")
    
    return {
        "cross_entropy": avg_ce,
        "accuracy": accuracy,
        "total_predictions": total_predictions,
    }

fp32_basic = evaluate_basic_metrics(fp32_model, sequences, tokenizer, "FP32")
ternary_basic = evaluate_basic_metrics(ternary_model, sequences, tokenizer, "Ternary")

# ============================================================
# PART 2: CONTEXT BENEFIT - CORRECTED
# ============================================================

print("\n" + "="*100)
print("PART 2: CONTEXT BENEFIT - CORRECTED METHODOLOGY")
print("="*100)

print("""
[CORRECTED Methodology]

To measure context benefit, we predict the SAME target token with DIFFERENT amounts of context.

Example: Predict token at position 20
  - Context=1:  Use tokens[19:20] to predict token[20]
  - Context=5:  Use tokens[15:20] to predict token[20]
  - Context=10: Use tokens[10:20] to predict token[20]
  - Context=20: Use tokens[0:20] to predict token[20]

This ensures we're testing if MORE context helps predict the SAME token.
""")

def evaluate_context_benefit_corrected(model, sequences, model_name="Model"):
    """Evaluate context benefit with CORRECT methodology"""
    print(f"\n[Evaluating {model_name} - Context Benefit]")
    
    model.eval()
    
    # Test predicting the same token with different context lengths
    target_position = 20  # Predict token at position 20
    context_lengths = [1, 5, 10, 20]
    
    results = {}
    
    for ctx_len in context_lengths:
        losses = []
        
        with torch.no_grad():
            for tokens in sequences:
                if len(tokens) < target_position + 1:
                    continue
                
                start_pos = max(0, target_position - ctx_len)
                input_tokens = tokens[start_pos:target_position]
                target_token = tokens[target_position]
                
                if len(input_tokens) == 0:
                    continue
                
                input_ids = torch.tensor([input_tokens], dtype=torch.long)
                
                outputs = model(input_ids)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                last_logits = logits[0, -1, :]
                
                loss = F.cross_entropy(last_logits.unsqueeze(0), torch.tensor([target_token]))
                losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        results[ctx_len] = avg_loss
        print(f"  Context length {ctx_len:2d}: loss = {avg_loss:.4f}")
    
    # Calculate context benefit
    benefit = results[1] - results[20]
    print(f"\n  Context Benefit (ctx=1 loss - ctx=20 loss): {benefit:+.4f}")
    
    return results, benefit

print("\n[FP32 - Context Benefit]")
fp32_context, fp32_context_benefit = evaluate_context_benefit_corrected(fp32_model, sequences, "FP32")

print("\n[Ternary - Context Benefit]")
ternary_context, ternary_context_benefit = evaluate_context_benefit_corrected(ternary_model, sequences, "Ternary")

# ============================================================
# PART 3: SHUFFLE TEST - CORRECTED
# ============================================================

print("\n" + "="*100)
print("PART 3: SHUFFLE TEST - CORRECTED METHODOLOGY")
print("="*100)

print("""
[CORRECTED Methodology]

To measure shuffle sensitivity, we predict the SAME target token with original vs shuffled context.

Example: Predict token at position 20
  - Original: Use tokens[0:20] (original order) to predict token[20]
  - Shuffled: Use shuffled(tokens[0:20]) to predict token[20]

This ensures we're testing if the model learned token order relationships.
""")

def evaluate_shuffle_corrected(model, sequences, model_name="Model"):
    """Evaluate shuffle sensitivity with CORRECT methodology"""
    print(f"\n[Evaluating {model_name} - Shuffle Test]")
    
    model.eval()
    
    context_len = 20
    
    losses_original = []
    losses_shuffled = []
    
    with torch.no_grad():
        for tokens in sequences[:100]:
            if len(tokens) < context_len + 1:
                continue
            
            input_orig = tokens[:context_len]
            target = tokens[context_len]
            
            # Original order
            input_ids = torch.tensor([input_orig], dtype=torch.long)
            outputs = model(input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            last_logits = logits[0, -1, :]
            loss = F.cross_entropy(last_logits.unsqueeze(0), torch.tensor([target]))
            losses_original.append(loss.item())
            
            # Shuffled order
            input_shuf = input_orig.copy()
            np.random.shuffle(input_shuf)
            input_ids_shuf = torch.tensor([input_shuf], dtype=torch.long)
            outputs_shuf = model(input_ids_shuf)
            logits_shuf = outputs_shuf["logits"] if isinstance(outputs_shuf, dict) else outputs_shuf[0]
            last_logits_shuf = logits_shuf[0, -1, :]
            loss_shuf = F.cross_entropy(last_logits_shuf.unsqueeze(0), torch.tensor([target]))
            losses_shuffled.append(loss_shuf.item())
    
    avg_orig = np.mean(losses_original)
    avg_shuf = np.mean(losses_shuffled)
    shuffle_effect = avg_shuf - avg_orig
    
    print(f"  Loss with original order: {avg_orig:.4f}")
    print(f"  Loss with shuffled order: {avg_shuf:.4f}")
    print(f"  Shuffle effect: {shuffle_effect:+.4f}")
    
    return {
        "loss_original": avg_orig,
        "loss_shuffled": avg_shuf,
        "shuffle_effect": shuffle_effect,
    }

np.random.seed(42)
fp32_shuffle = evaluate_shuffle_corrected(fp32_model, sequences, "FP32")

np.random.seed(42)
ternary_shuffle = evaluate_shuffle_corrected(ternary_model, sequences, "Ternary")

# ============================================================
# PART 4: COMPARISON TABLE
# ============================================================

print("\n" + "="*100)
print("PART 4: RESULTS SUMMARY")
print("="*100)

print("\n[Basic Metrics]")
print(f"  {'Metric':<20} {'FP32':<15} {'Ternary':<15} {'Difference':<15}")
print(f"  {'-'*65}")
print(f"  {'Cross-Entropy':<20} {fp32_basic['cross_entropy']:<15.4f} {ternary_basic['cross_entropy']:<15.4f} {fp32_basic['cross_entropy'] - ternary_basic['cross_entropy']:+.4f}")
print(f"  {'Top-1 Accuracy (%)':<20} {fp32_basic['accuracy']:<15.4f} {ternary_basic['accuracy']:<15.4f} {fp32_basic['accuracy'] - ternary_basic['accuracy']:+.4f}")

print("\n[Context Benefit]")
print(f"  {'Context Length':<20} {'FP32 Loss':<15} {'Ternary Loss':<15}")
print(f"  {'-'*50}")
for ctx_len in [1, 5, 10, 20]:
    print(f"  {ctx_len:<20} {fp32_context[ctx_len]:<15.4f} {ternary_context[ctx_len]:<15.4f}")

print(f"\n  {'Model':<15} {'Context Benefit':<20}")
print(f"  {'-'*35}")
print(f"  {'FP32':<15} {fp32_context_benefit:+.4f}")
print(f"  {'Ternary':<15} {ternary_context_benefit:+.4f}")
print(f"  {'Difference':<15} {fp32_context_benefit - ternary_context_benefit:+.4f}")

print("\n[Shuffle Test]")
print(f"  {'Metric':<25} {'FP32':<15} {'Ternary':<15}")
print(f"  {'-'*55}")
print(f"  {'Loss (original)':<25} {fp32_shuffle['loss_original']:<15.4f} {ternary_shuffle['loss_original']:<15.4f}")
print(f"  {'Loss (shuffled)':<25} {fp32_shuffle['loss_shuffled']:<15.4f} {ternary_shuffle['loss_shuffled']:<15.4f}")
print(f"  {'Shuffle effect':<25} {fp32_shuffle['shuffle_effect']:<15.4f} {ternary_shuffle['shuffle_effect']:<15.4f}")

# ============================================================
# PART 5: SAVE RESULTS
# ============================================================

results = {
    "methodology": "CORRECTED - Same target, different context",
    "fp32": {
        "basic": fp32_basic,
        "context_benefit": fp32_context_benefit,
        "context_by_length": fp32_context,
        "shuffle": fp32_shuffle,
    },
    "ternary": {
        "basic": ternary_basic,
        "context_benefit": ternary_context_benefit,
        "context_by_length": ternary_context,
        "shuffle": ternary_shuffle,
    },
    "differences": {
        "accuracy": fp32_basic['accuracy'] - ternary_basic['accuracy'],
        "context_benefit": fp32_context_benefit - ternary_context_benefit,
        "shuffle_effect": fp32_shuffle['shuffle_effect'] - ternary_shuffle['shuffle_effect'],
    }
}

# Convert numpy types
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

output_path = Path("experiments/results/fp32_vs_ternary_5k_comparison_corrected.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[Results saved to {output_path}]")

print("\n" + "="*100)
print("EVALUATION COMPLETE")
print("="*100)
