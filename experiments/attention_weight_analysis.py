#!/usr/bin/env python3
"""
READ-ONLY Attention-Weight Analysis

Determines whether attention heads are learning meaningful contextual relationships.

CRITICAL: This is READ-ONLY. No training, no checkpoint modification.
- We extract attention weights via forward pass
- Checkpoints are only READ, never modified or saved
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
import random

# Add project to path
sys.path.insert(0, "/root/agent-sandbox/ARTN")

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.model.bitlinear import BitLinearGPT

print("="*100)
print("READ-ONLY ATTENTION-WEIGHT ANALYSIS")
print("="*100)
print("""
This diagnostic determines whether attention heads are learning meaningful
contextual/token-to-token relationships.

We will check:
1. Where each attention head attends for real language examples
2. Whether attention focuses on meaningful tokens or is diffuse/random
3. Compare attention patterns across all 6 layers and heads
4. Quantify attention to relevant tokens vs shuffled control
5. Identify whether some heads show contextual behavior

DO NOT assume attention is the root cause.
""")

device = "cpu"

# ==============================================================================
# LOAD TOKENIZER AND MODEL
# ==============================================================================

tokenizer_path = Path("outputs/artn-optimized/tokenizer.json")
tokenizer = BPETokenizer.load(str(tokenizer_path))

print(f"\n[Tokenizer]")
print(f"  Vocab size: {len(tokenizer.vocab)}")

# Load model
checkpoint_path = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")
model = BitLinearGPT(
    vocab_size=len(tokenizer.vocab),
    d_model=256,
    n_layers=6,
    n_heads=8,
    d_ff=1024,
    max_seq_len=256,
    dropout=0.0
)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"\n[Model]")
print(f"  Checkpoint: step {checkpoint.get('step', 'unknown')}")
print(f"  Layers: 6, Heads: 8")

# ==============================================================================
# LOAD TEST DATA
# ==============================================================================

print(f"\n[Loading test data...]")

dataset = load_dataset("roneneldan/TinyStories", split="train")
test_stories = []
for i, item in enumerate(dataset):
    if i >= 50000 and i < 50050:  # 50 stories
        test_stories.append(item["text"])

print(f"  Test stories: {len(test_stories)}")

# Tokenize
test_sequences = []
for story in test_stories:
    tokens = tokenizer.encode(story)[:128]  # Shorter sequences for analysis
    if len(tokens) >= 30:  # Need enough context
        test_sequences.append(tokens)

print(f"  Usable sequences: {len(test_sequences)}")

# ==============================================================================
# HELPER: EXTRACT ATTENTION WEIGHTS
# ==============================================================================

def get_attention_weights(model, input_ids):
    """
    Extract attention weights from all layers and heads.
    
    Returns: dict[layer_idx][head_idx] = attention_weights
    """
    attention_weights = defaultdict(lambda: defaultdict(list))
    
    # Hook to capture attention weights
    def make_attn_hook(layer_idx, head_idx):
        def hook(module, input, output):
            # For MultiHeadAttention, we need to capture the attention weights
            # This depends on the implementation
            pass
        return hook
    
    # We need to modify the forward pass to return attention weights
    # Let's do a manual forward pass
    
    batch_size, seq_len = input_ids.shape
    
    # Get embeddings
    positions = torch.arange(seq_len, device=input_ids.device)
    x = model.token_embedding(input_ids) + model.position_embedding(positions)
    
    # Causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
    
    all_attn_weights = {}
    
    for layer_idx, block in enumerate(model.blocks):
        # Pre-norm
        x_norm = block.ln1(x)
        
        # Get Q, K, V
        attn = block.attn
        
        # Reshape for multi-head attention
        B, T, C = x_norm.shape
        
        # Q, K, V projections
        q = attn.q_proj(x_norm)  # (B, T, d_model)
        k = attn.k_proj(x_norm)
        v = attn.v_proj(x_norm)
        
        # Reshape to (B, n_heads, T, head_dim)
        n_heads = 8
        head_dim = C // n_heads
        
        q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Apply causal mask
        scores = scores.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Store
        all_attn_weights[layer_idx] = attn_weights.detach().clone()
        
        # Continue forward pass
        attn_out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        attn_out = attn.o_proj(attn_out)
        
        # Residual
        x = x + attn_out
        
        # FFN
        x = x + block.ff(block.ln2(x))
    
    return all_attn_weights

# ==============================================================================
# PART 1: VISUALIZE ATTENTION FOR SAMPLE SEQUENCES
# ==============================================================================

print("\n" + "="*100)
print("PART 1: WHERE DO ATTENTION HEADS ATTEND?")
print("="*100)

# Take a sample sequence
sample_tokens = test_sequences[0][:50]  # First 50 tokens
input_ids = torch.tensor([sample_tokens], dtype=torch.long)

print(f"\n[Sample Sequence]")
print(f"  Length: {len(sample_tokens)} tokens")
print(f"  First 20 tokens: {sample_tokens[:20]}")

# Get attention weights
with torch.no_grad():
    attn_weights = get_attention_weights(model, input_ids)

print(f"\n[Attention Weight Shape]")
print(f"  Layers: {len(attn_weights)}")
print(f"  Per layer: {attn_weights[0].shape}")  # (1, n_heads, seq_len, seq_len)

# Analyze attention patterns
print(f"\n[Attention Pattern Summary by Layer]")

for layer_idx in range(6):
    weights = attn_weights[layer_idx][0]  # (n_heads, seq_len, seq_len)
    
    # For each head, compute statistics
    head_stats = []
    
    for head_idx in range(8):
        head_weights = weights[head_idx]  # (seq_len, seq_len)
        
        # Only look at positions with context (not first few tokens)
        # and ignore padding
        valid_positions = min(len(sample_tokens), 50)
        
        # Average attention to:
        # 1. Previous token (position t-1)
        # 2. First token (position 0) - often high in untrained models
        # 3. Self (position t)
        # 4. All other positions (diffuse)
        
        prev_token_attn = []
        first_token_attn = []
        self_attn = []
        
        for pos in range(1, valid_positions):
            attn_dist = head_weights[pos, :pos+1]  # Can only attend to previous positions
            
            if pos >= 1:
                prev_token_attn.append(attn_dist[pos-1].item())
            
            first_token_attn.append(attn_dist[0].item())
            self_attn.append(attn_dist[pos].item())
        
        avg_prev = np.mean(prev_token_attn) if prev_token_attn else 0
        avg_first = np.mean(first_token_attn)
        avg_self = np.mean(self_attn)
        
        # Compute entropy (lower = more focused, higher = more diffuse)
        entropy_vals = []
        for pos in range(1, valid_positions):
            attn_dist = head_weights[pos, :pos+1]
            # Normalize
            attn_dist = attn_dist / attn_dist.sum()
            # Entropy
            ent = -(attn_dist * torch.log(attn_dist + 1e-10)).sum().item()
            entropy_vals.append(ent)
        
        avg_entropy = np.mean(entropy_vals) if entropy_vals else 0
        max_entropy = np.log(valid_positions)  # Uniform distribution
        
        head_stats.append({
            'head_idx': head_idx,
            'prev_token': avg_prev,
            'first_token': avg_first,
            'self': avg_self,
            'entropy': avg_entropy,
            'normalized_entropy': avg_entropy / (max_entropy + 1e-10),
        })
    
    # Print summary
    print(f"\n  Layer {layer_idx}:")
    print(f"    {'Head':<6} {'Prev':<8} {'First':<8} {'Self':<8} {'Entropy':<10} {'Focus':<10}")
    
    for stats in head_stats:
        focus = "FOCUSED" if stats['normalized_entropy'] < 0.5 else "DIFFUSE"
        print(f"    {stats['head_idx']:<6} {stats['prev_token']:<8.4f} {stats['first_token']:<8.4f} {stats['self']:<8.4f} {stats['normalized_entropy']:<10.4f} {focus:<10}")

# ==============================================================================
# PART 2: ATTENTION TO RELEVANT PREVIOUS TOKENS
# ==============================================================================

print("\n" + "="*100)
print("PART 2: ATTENTION TO RELEVANT PREVIOUS TOKENS")
print("="*100)

print("""
[Question: Does attention focus on meaningful previous tokens?]

We define "relevant" tokens as:
- Recent tokens (within last 5 positions)
- Tokens that appear multiple times in context (repeated words)
- Subject/object tokens in simple patterns

We compare attention to relevant tokens vs random/shuffled baseline.
""")

# Analyze multiple sequences
relevant_attn_scores = []
random_attn_scores = []

for seq_idx, tokens in enumerate(test_sequences[:20]):  # 20 sequences
    input_ids = torch.tensor([tokens[:80]], dtype=torch.long)
    
    with torch.no_grad():
        attn_weights = get_attention_weights(model, input_ids)
    
    # For each position, identify "relevant" previous tokens
    # Here we use: repeated tokens in the previous 10 positions
    seq_len = min(len(tokens), 80)
    
    for layer_idx in [0, 2, 5]:  # Sample layers: early, middle, late
        weights = attn_weights[layer_idx][0]  # (n_heads, seq_len, seq_len)
        
        for pos in range(10, seq_len):  # Skip first few positions
            current_token = tokens[pos]
            
            # Find relevant tokens: same token appeared recently
            relevant_positions = []
            for prev_pos in range(max(0, pos-10), pos):
                if tokens[prev_pos] == current_token:
                    relevant_positions.append(prev_pos)
            
            # Also consider adjacent tokens as relevant
            if pos >= 1:
                if pos - 1 not in relevant_positions:
                    relevant_positions.append(pos - 1)  # Previous token
            
            # Compute attention to relevant positions
            for head_idx in range(8):
                attn_dist = weights[head_idx, pos, :pos]
                
                if relevant_positions:
                    relevant_attn = sum(attn_dist[rel_pos].item() for rel_pos in relevant_positions if rel_pos < pos)
                    relevant_attn_scores.append(relevant_attn)
                
                # Random baseline: attention to random previous positions
                random_pos = random.randint(0, pos-1)
                random_attn = attn_dist[random_pos].item()
                random_attn_scores.append(random_attn)

# Compare
avg_relevant = np.mean(relevant_attn_scores)
avg_random = np.mean(random_attn_scores)
ratio = avg_relevant / (avg_random + 1e-10)

print(f"\n[Attention to Relevant vs Random Tokens]")
print(f"  Average attention to relevant positions: {avg_relevant:.6f}")
print(f"  Average attention to random positions: {avg_random:.6f}")
print(f"  Ratio (relevant/random): {ratio:.2f}x")

if ratio > 1.5:
    print(f"\n  ✅ Attention DOES focus on relevant tokens more than random")
elif ratio > 1.1:
    print(f"\n  ⚠️  Attention slightly prefers relevant tokens")
else:
    print(f"\n  ❌ Attention does NOT focus on relevant tokens")

# ==============================================================================
# PART 3: ATTENTION PATTERN COMPARISON ACROSS LAYERS
# ==============================================================================

print("\n" + "="*100)
print("PART 3: ATTENTION PATTERNS ACROSS LAYERS")
print("="*100)

# Compute layer-wise statistics
layer_stats = []

for seq_idx, tokens in enumerate(test_sequences[:10]):
    input_ids = torch.tensor([tokens[:60]], dtype=torch.long)
    
    with torch.no_grad():
        attn_weights = get_attention_weights(model, input_ids)
    
    seq_len = min(len(tokens), 60)
    
    for layer_idx in range(6):
        weights = attn_weights[layer_idx][0]  # (n_heads, seq_len, seq_len)
        
        prev_attns = []
        first_attns = []
        entropies = []
        
        for head_idx in range(8):
            for pos in range(5, seq_len):
                attn_dist = weights[head_idx, pos, :pos]
                
                # Previous token attention
                prev_attns.append(attn_dist[pos-1].item())
                
                # First token attention
                first_attns.append(attn_dist[0].item())
                
                # Entropy
                attn_norm = attn_dist / attn_dist.sum()
                ent = -(attn_norm * torch.log(attn_norm + 1e-10)).sum().item()
                entropies.append(ent)
        
        layer_stats.append({
            'layer': layer_idx,
            'prev_attn': np.mean(prev_attns),
            'first_attn': np.mean(first_attns),
            'entropy': np.mean(entropies),
        })

# Aggregate by layer
print(f"\n[Layer-wise Attention Patterns]")

for layer_idx in range(6):
    layer_data = [s for s in layer_stats if s['layer'] == layer_idx]
    
    avg_prev = np.mean([s['prev_attn'] for s in layer_data])
    avg_first = np.mean([s['first_attn'] for s in layer_data])
    avg_entropy = np.mean([s['entropy'] for s in layer_data])
    
    # Pattern interpretation
    if avg_first > 0.3:
        pattern = "FIRST-TOKEN FOCUS"
    elif avg_prev > 0.2:
        pattern = "PREVIOUS-TOKEN FOCUS"
    elif avg_entropy > 2.5:
        pattern = "DIFFUSE"
    else:
        pattern = "MIXED"
    
    print(f"  Layer {layer_idx}: prev={avg_prev:.4f}, first={avg_first:.4f}, entropy={avg_entropy:.4f} [{pattern}]")

# ==============================================================================
# PART 4: SHUFFLED CONTEXT CONTROL
# ==============================================================================

print("\n" + "="*100)
print("PART 4: SHUFFLED CONTEXT CONTROL")
print("="*100)

print("""
[Question: Does attention prefer original order over shuffled context?]

If attention learned meaningful patterns:
  - Original order should get different attention than shuffled
  - Shuffling should disrupt attention patterns
""")

original_entropies = []
shuffled_entropies = []

for seq_idx, tokens in enumerate(test_sequences[:20]):
    # Original
    input_ids = torch.tensor([tokens[:60]], dtype=torch.long)
    
    with torch.no_grad():
        attn_orig = get_attention_weights(model, input_ids)
    
    # Shuffled context
    shuffled_tokens = tokens[:60].copy()
    random.shuffle(shuffled_tokens)
    input_ids_shuffled = torch.tensor([shuffled_tokens], dtype=torch.long)
    
    with torch.no_grad():
        attn_shuffled = get_attention_weights(model, input_ids_shuffled)
    
    # Compute entropy differences
    for layer_idx in range(6):
        orig_weights = attn_orig[layer_idx][0]
        shuffled_weights = attn_shuffled[layer_idx][0]
        
        for head_idx in range(8):
            for pos in range(10, min(60, len(tokens))):
                # Original entropy
                attn_dist = orig_weights[head_idx, pos, :pos]
                attn_norm = attn_dist / attn_dist.sum()
                ent_orig = -(attn_norm * torch.log(attn_norm + 1e-10)).sum().item()
                
                # Shuffled entropy
                attn_dist_shuf = shuffled_weights[head_idx, pos, :pos]
                attn_norm_shuf = attn_dist_shuf / attn_dist_shuf.sum()
                ent_shuf = -(attn_norm_shuf * torch.log(attn_norm_shuf + 1e-10)).sum().item()
                
                original_entropies.append(ent_orig)
                shuffled_entropies.append(ent_shuf)

avg_orig = np.mean(original_entropies)
avg_shuf = np.mean(shuffled_entropies)

print(f"\n[Entropy Comparison]")
print(f"  Original context entropy: {avg_orig:.4f}")
print(f"  Shuffled context entropy: {avg_shuf:.4f}")
print(f"  Difference: {avg_orig - avg_shuf:.4f}")

if abs(avg_orig - avg_shuf) < 0.1:
    print(f"\n  ❌ Attention patterns are SIMILAR for original vs shuffled")
    print(f"     This suggests attention may not be learning token relationships")
else:
    print(f"\n  ✅ Attention patterns DIFFER for original vs shuffled")
    print(f"     This suggests attention learned something about token order")

# ==============================================================================
# PART 5: HEAD DIVERSITY ANALYSIS
# ==============================================================================

print("\n" + "="*100)
print("PART 5: HEAD DIVERSITY - DO HEADS SPECIALIZE?")
print("="*100)

print("""
[Question: Do different heads learn different patterns?]

In trained transformers:
  - Some heads focus on previous token
  - Some heads focus on first token (sentence-level)
  - Some heads are diffuse
  - Diversity indicates learning
""")

# Compute head specialization
head_patterns = defaultdict(lambda: defaultdict(list))

for seq_idx, tokens in enumerate(test_sequences[:20]):
    input_ids = torch.tensor([tokens[:60]], dtype=torch.long)
    
    with torch.no_grad():
        attn_weights = get_attention_weights(model, input_ids)
    
    seq_len = min(len(tokens), 60)
    
    for layer_idx in range(6):
        weights = attn_weights[layer_idx][0]
        
        for head_idx in range(8):
            prev_attns = []
            first_attns = []
            
            for pos in range(5, seq_len):
                attn_dist = weights[head_idx, pos, :pos]
                prev_attns.append(attn_dist[pos-1].item())
                first_attns.append(attn_dist[0].item())
            
            head_patterns[layer_idx][head_idx].append({
                'prev': np.mean(prev_attns),
                'first': np.mean(first_attns),
            })

# Classify heads
print(f"\n[Head Specialization by Layer]")

specialized_count = 0
total_heads = 0

for layer_idx in range(6):
    print(f"\n  Layer {layer_idx}:")
    
    for head_idx in range(8):
        patterns = head_patterns[layer_idx][head_idx]
        
        avg_prev = np.mean([p['prev'] for p in patterns])
        avg_first = np.mean([p['first'] for p in patterns])
        
        # Classify
        if avg_first > 0.3:
            pattern_type = "FIRST-TOKEN"
            specialized_count += 1
        elif avg_prev > 0.2:
            pattern_type = "PREVIOUS-TOKEN"
            specialized_count += 1
        elif avg_prev < 0.05 and avg_first < 0.05:
            pattern_type = "DIFFUSE"
        else:
            pattern_type = "MIXED"
            specialized_count += 0.5
        
        total_heads += 1
        
        print(f"    Head {head_idx}: prev={avg_prev:.4f}, first={avg_first:.4f} [{pattern_type}]")

specialization_ratio = specialized_count / total_heads

print(f"\n[Specialization Summary]")
print(f"  Specialized heads: {specialized_count}/{total_heads} ({specialization_ratio*100:.1f}%)")

if specialization_ratio > 0.5:
    print(f"  ✅ Many heads show specialized patterns")
else:
    print(f"  ❌ Most heads do not show clear specialization")

# ==============================================================================
# SUMMARY AND CLASSIFICATION
# ==============================================================================

print("\n" + "="*100)
print("SUMMARY AND CLASSIFICATION")
print("="*100)

findings = []

# Finding 1: Attention to relevant tokens
if ratio > 1.5:
    findings.append(("Attention focuses on relevant tokens", "VERIFIED"))
elif ratio > 1.1:
    findings.append(("Attention slightly prefers relevant tokens", "VERIFIED"))
else:
    findings.append(("Attention focuses on relevant tokens", "NOT SUPPORTED"))

# Finding 2: Shuffled control
if abs(avg_orig - avg_shuf) < 0.1:
    findings.append(("Attention distinguishes original from shuffled context", "NOT SUPPORTED"))
else:
    findings.append(("Attention distinguishes original from shuffled context", "VERIFIED"))

# Finding 3: Head specialization
if specialization_ratio > 0.5:
    findings.append(("Attention heads show specialization", "VERIFIED"))
else:
    findings.append(("Attention heads show specialization", "NOT SUPPORTED"))

# Finding 4: First-token focus
first_token_avg = np.mean([s['first_attn'] for s in layer_stats])
if first_token_avg > 0.3:
    findings.append(("Strong first-token attention bias exists", "VERIFIED"))
else:
    findings.append(("Strong first-token attention bias exists", "NOT SUPPORTED"))

print("\n[CLASSIFICATION OF FINDINGS]")
print("-"*80)

for finding, status in findings:
    status_str = f"✅ {status}" if status == "VERIFIED" else f"❌ {status}" if status == "NOT SUPPORTED" else f"⚠️ {status}"
    print(f"  {finding}")
    print(f"    → {status_str}")
    print()

print("="*80)
print("CONCLUSION")
print("="*80)

print("""
This diagnostic analyzed what the model's attention is actually doing.

All findings are READ-ONLY - no checkpoints were modified.

Key questions:
1. Does attention focus on meaningful tokens?
2. Does attention learn contextual relationships?
3. Do heads specialize in different patterns?
4. Is attention just first-token bias (common untrained pattern)?
""")

print("\n" + "="*100)
print("DIAGNOSTIC COMPLETE")
print("="*100)
