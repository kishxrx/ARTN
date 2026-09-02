#!/usr/bin/env python3
"""
READ-ONLY: Audit the context benefit and shuffle evaluation methodology

This diagnostic verifies whether the evaluation methodology is correct by:
1. Checking that context=50 vs context=1 compare the SAME targets
2. Identifying any methodology artifacts
3. Proposing corrected evaluation if needed

CRITICAL: This is READ-ONLY. No model training or modification.
"""

import torch
import torch.nn.functional as F
import sys
import numpy as np
from pathlib import Path
from datasets import load_dataset

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
print("READ-ONLY: CONTEXT BENEFIT EVALUATION METHODOLOGY AUDIT")
print("="*100)
print("""
This audit verifies the evaluation methodology used for context benefit.

CRITICAL QUESTIONS:
1. Do context=50 and context=1 compare the SAME targets?
2. Are we predicting the same tokens with different context?
3. Is the negative context benefit a methodology artifact?

READ-ONLY: No model training or modification.
""")

# Load tokenizer
tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')

# Load test data
dataset = load_dataset("roneneldan/TinyStories", split="train")
test_stories = [dataset[i]["text"] for i in range(50000, 50050)]

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

# ==============================================================================
# PART 1: ANALYZE THE OLD METHODOLOGY
# ==============================================================================

print("\n" + "="*100)
print("PART 1: ANALYZE OLD METHODOLOGY")
print("="*100)

print("""
[OLD Methodology]

Context=50:
  Input:  tokens[0:49] (49 tokens)
  Target: tokens[1:50] (49 tokens)
  Loss:   average over 49 predictions (positions 1,2,3,...,49)

Context=1:
  Input:  tokens[0:0] (1 token)
  Target: tokens[1:1] (1 token)
  Loss:   1 prediction (position 1 ONLY)

PROBLEM: These are predicting DIFFERENT tokens!
- Context=50 predicts tokens at positions 1-49
- Context=1 predicts token at position 1 ONLY

This is NOT a fair comparison of context benefit!
""")

# Demonstrate the problem
print("\n[Example: What Each Condition Predicts]")
example_tokens = sequences[0][:50]
print(f"  Sequence (first 10 tokens): {example_tokens[:10]}")
print(f"  Total tokens: {len(example_tokens)}")
print(f"\n  Context=50 would predict:")
print(f"    - Position 1: token {example_tokens[1]}")
print(f"    - Position 2: token {example_tokens[2]}")
print(f"    - ...")
print(f"    - Position 49: token {example_tokens[49]}")
print(f"    - Total: 49 predictions")
print(f"\n  Context=1 would predict:")
print(f"    - Position 1: token {example_tokens[1]}")
print(f"    - Total: 1 prediction")

print(f"\n  ❌ PROBLEM: Different number of predictions AND different positions!")

# ==============================================================================
# PART 2: CORRECT METHODOLOGY
# ==============================================================================

print("\n" + "="*100)
print("PART 2: CORRECT METHODOLOGY")
print("="*100)

print("""
[CORRECT Methodology]

To measure context benefit, we must:
1. Predict the SAME target token
2. With DIFFERENT amounts of context

Example: Predict token at position 10
  - Context=1:  Use tokens[9:10] to predict token[10]
  - Context=5:  Use tokens[5:10] to predict token[10]
  - Context=10: Use tokens[0:10] to predict token[10]

This way, we're testing if MORE context helps predict the SAME token.
""")

# Implement correct methodology
def evaluate_context_benefit_corrected(model, sequences, model_name="Model"):
    """Evaluate context benefit with CORRECT methodology"""
    print(f"\n[Evaluating {model_name} - CORRECTED Context Benefit]")
    
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
                
                # Context: tokens[target_position - ctx_len : target_position]
                # Target: tokens[target_position]
                start_pos = max(0, target_position - ctx_len)
                
                input_tokens = tokens[start_pos:target_position]
                target_token = tokens[target_position]
                
                # Ensure minimum context length of 1
                if len(input_tokens) == 0:
                    continue
                
                input_ids = torch.tensor([input_tokens], dtype=torch.long)
                
                # Handle both FP32 (tuple) and Ternary (dict) models
                outputs = model(input_ids)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs[0]
                
                # Get logits at last position
                last_logits = logits[0, -1, :]  # (vocab_size,)
                
                # Compute cross entropy loss for this single prediction
                loss = F.cross_entropy(last_logits.unsqueeze(0), torch.tensor([target_token]))
                losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        results[ctx_len] = avg_loss
        print(f"  Context length {ctx_len:2d}: loss = {avg_loss:.4f}")
    
    return results

print("\n[CORRECTED Context Benefit Test]")
print("="*80)
print("Testing: Predict token at position 20 with different context lengths")
print("="*80)

fp32_results = evaluate_context_benefit_corrected(fp32_model, sequences, "FP32")
ternary_results = evaluate_context_benefit_corrected(ternary_model, sequences, "Ternary")

# ==============================================================================
# PART 3: SHUFFLE TEST AUDIT
# ==============================================================================

print("\n" + "="*100)
print("PART 3: SHUFFLE TEST AUDIT")
print("="*100)

print("""
[Shuffle Test Methodology]

Original:
  Input:  tokens[0:49] (original order)
  Target: tokens[1:50]
  Loss:   average over 49 predictions

Shuffled:
  Input:  shuffled(tokens[0:49])
  Target: tokens[1:50] (SAME targets)
  Loss:   average over 49 predictions

PROBLEM: The targets are NOT properly aligned!

When we shuffle the input, we should also track which token each position predicts.
The current test shuffles input but keeps targets in original order.

This is NOT a valid test of whether the model learned token relationships.
""")

# Correct shuffle test
def evaluate_shuffle_corrected(model, sequences, model_name="Model"):
    """Evaluate shuffle sensitivity with CORRECT methodology"""
    print(f"\n[Evaluating {model_name} - CORRECTED Shuffle Test]")
    
    model.eval()
    
    # Test: predict the same tokens with original vs shuffled context
    context_len = 20
    
    losses_original = []
    losses_shuffled = []
    
    with torch.no_grad():
        for tokens in sequences[:100]:
            if len(tokens) < context_len + 1:
                continue
            
            # Original order
            input_orig = tokens[:context_len]
            target = tokens[context_len]
            
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
    print(f"  Shuffle effect: {shuffle_effect:.4f}")
    
    return {
        "loss_original": avg_orig,
        "loss_shuffled": avg_shuf,
        "shuffle_effect": shuffle_effect,
    }

print("\n[CORRECTED Shuffle Test]")
print("="*80)
print("Testing: Predict token at position 20 with original vs shuffled context (positions 0-19)")
print("="*80)

np.random.seed(42)
fp32_shuffle = evaluate_shuffle_corrected(fp32_model, sequences, "FP32")

np.random.seed(42)
ternary_shuffle = evaluate_shuffle_corrected(ternary_model, sequences, "Ternary")

# ==============================================================================
# PART 4: COMPARISON TABLE
# ==============================================================================

print("\n" + "="*100)
print("PART 4: CORRECTED RESULTS COMPARISON")
print("="*100)

print("\n[Context Benefit - CORRECTED]")
print("Predict token at position 20 with different context lengths")
print()
print(f"  {'Context Len':<15} {'FP32 Loss':<15} {'Ternary Loss':<15} {'Difference':<15}")
print(f"  {'-'*60}")
for ctx_len in [1, 5, 10, 20]:
    fp32_loss = fp32_results[ctx_len]
    ternary_loss = ternary_results[ctx_len]
    diff = fp32_loss - ternary_loss
    print(f"  {ctx_len:<15} {fp32_loss:<15.4f} {ternary_loss:<15.4f} {diff:+<15.4f}")

print("\n[Context Benefit Calculation]")
print("  Benefit = Loss(ctx=1) - Loss(ctx=20)")
fp32_benefit = fp32_results[1] - fp32_results[20]
ternary_benefit = ternary_results[1] - ternary_results[20]
print(f"  FP32:    {fp32_results[1]:.4f} - {fp32_results[20]:.4f} = {fp32_benefit:+.4f}")
print(f"  Ternary: {ternary_results[1]:.4f} - {ternary_results[20]:.4f} = {ternary_benefit:+.4f}")

print("\n[Shuffle Effect - CORRECTED]")
print(f"  {'Metric':<25} {'FP32':<15} {'Ternary':<15}")
print(f"  {'-'*55}")
print(f"  {'Loss (original)':<25} {fp32_shuffle['loss_original']:<15.4f} {ternary_shuffle['loss_original']:<15.4f}")
print(f"  {'Loss (shuffled)':<25} {fp32_shuffle['loss_shuffled']:<15.4f} {ternary_shuffle['loss_shuffled']:<15.4f}")
print(f"  {'Shuffle effect':<25} {fp32_shuffle['shuffle_effect']:<15.4f} {ternary_shuffle['shuffle_effect']:<15.4f}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*100)
print("AUDIT SUMMARY")
print("="*100)

print("""
[VERIFIED PROBLEMS WITH OLD METHODOLOGY]

1. Context Benefit Test:
   ❌ OLD: Compared loss over 49 predictions (ctx=50) vs 1 prediction (ctx=1)
   ❌ OLD: Different target positions (1-49 vs just 1)
   ❌ OLD: Negative "benefit" was comparing apples to oranges
   
   ✅ CORRECT: Predict SAME target with DIFFERENT context lengths
   ✅ CORRECT: Compare loss for token at position 20 with ctx=1,5,10,20

2. Shuffle Test:
   ❌ OLD: Shuffled input but kept targets in original order
   ❌ OLD: Misaligned input-target pairs after shuffling
   
   ✅ CORRECT: Predict same token with original vs shuffled context
   ✅ CORRECT: Proper alignment maintained

[VERDICT]

The old methodology had CRITICAL FLAWS that could produce misleading results.
The corrected methodology is needed for valid comparison.
""")

print("\n" + "="*100)
print("AUDIT COMPLETE")
print("="*100)
