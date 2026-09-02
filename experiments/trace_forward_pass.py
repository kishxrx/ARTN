#!/usr/bin/env python3
"""
Forward Pass and Training Objective Trace

Traces complete forward pass and training objective to identify where the model fails.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*70)
print("FORWARD PASS AND TRAINING OBJECTIVE TRACE")
print("="*70)

# ============================================================
# Load model and tokenizer
# ============================================================

tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
print(f"\nTokenizer loaded: vocab_size={tokenizer.vocab_size}")

# Load both models
trained_model = BitLinearGPT(4000, 256, 8, 6, 1024, 256)
untrained_model = BitLinearGPT(4000, 256, 8, 6, 1024, 256)

trained_ckpt = torch.load('outputs/artn-50k-best-config/final_model/model.pt', map_location='cpu')
trained_model.load_state_dict(trained_ckpt)

# Untrained model uses random initialization
trained_model.eval()
untrained_model.eval()

print(f"Loaded trained model and untrained model")

# ============================================================
# Create test batch from real TinyStories
# ============================================================

print(f"\n{'='*70}")
print("TEST BATCH CREATION")
print(f"{'='*70}")

# Use actual TinyStories-like sentences
stories = [
    "Once upon a time there was a little girl named Sarah.",
    "The little girl loved to play in the garden.",
    "One day she found a magic flower.",
]

# Tokenize
sequences = []
for story in stories:
    tokens = tokenizer.encode(story)
    if len(tokens) > 50:
        tokens = tokens[:50]
    sequences.append(tokens)

# Pad to same length
max_len = max(len(s) for s in sequences)
padded_sequences = []
for seq in sequences:
    padded = seq + [0] * (max_len - len(seq))
    padded_sequences.append(padded)

input_ids = torch.tensor(padded_sequences)
print(f"\nInput shape: {input_ids.shape}")
print(f"Max length: {max_len}")

# Print tokenized examples
print(f"\nTokenized sequences:")
for i, (seq, story) in enumerate(zip(sequences, stories)):
    print(f"  [{i}] {story[:40]}... → {len(seq)} tokens")
    # Print first few token pairs
    decoded = [tokenizer.decode([t]) for t in seq[:5]]
    print(f"      First 5 tokens: {decoded}")

# ============================================================
# TRACE STAGE 1: EMBEDDINGS
# ============================================================

print(f"\n{'='*70}")
print("STAGE 1: EMBEDDINGS")
print(f"{'='*70}")

with torch.no_grad():
    # Get embeddings from both models
    trained_embed = trained_model.token_embedding(input_ids)
    untrained_embed = untrained_model.token_embedding(input_ids)
    
    print(f"\n--- Tensor Statistics ---")
    print(f"Shape: {trained_embed.shape}")
    
    print(f"\nTrained embeddings:")
    print(f"  Mean: {trained_embed.mean().item():.6f}")
    print(f"  Std: {trained_embed.std().item():.6f}")
    print(f"  Min: {trained_embed.min().item():.6f}")
    print(f"  Max: {trained_embed.max().item():.6f}")
    print(f"  Zero %: {100 * (trained_embed == 0).float().mean().item():.2f}%")
    print(f"  NaN: {torch.isnan(trained_embed).any().item()}")
    print(f"  Inf: {torch.isinf(trained_embed).any().item()}")
    
    print(f"\nUntrained embeddings:")
    print(f"  Mean: {untrained_embed.mean().item():.6f}")
    print(f"  Std: {untrained_embed.std().item():.6f}")
    print(f"  Min: {untrained_embed.min().item():.6f}")
    print(f"  Max: {untrained_embed.max().item():.6f}")
    
    # Check if embeddings differ
    embed_diff = (trained_embed - untrained_embed).abs().mean().item()
    print(f"\nDifference between trained/untrained: {embed_diff:.6f}")
    
    # Check if different tokens have different embeddings
    token_0_embed = trained_model.token_embedding.weight[0]
    token_1_embed = trained_model.token_embedding.weight[1]
    token_diff = (token_0_embed - token_1_embed).abs().mean().item()
    print(f"Different token embeddings differ by: {token_diff:.6f}")
    
    # VERDICT
    if embed_diff > 0.01:
        print(f"\n✓ VERIFIED: Trained embeddings differ from untrained")
    else:
        print(f"\n✗ VERIFIED INCORRECT: Trained embeddings = untrained (not trained!)")

# ============================================================
# TRACE STAGE 2: POSITIONAL ENCODING
# ============================================================

print(f"\n{'='*70}")
print("STAGE 2: POSITIONAL ENCODING")
print(f"{'='*70}")

with torch.no_grad():
    # Add positional encoding
    seq_len = input_ids.size(1)
    
    # Check if model has position embedding
    if hasattr(trained_model, 'position_embedding'):
        pos_embed = trained_model.position_embedding.weight[:seq_len]
        trained_with_pos = trained_embed + pos_embed
        print(f"\nPosition embeddings shape: {pos_embed.shape}")
        print(f"Position embed mean: {pos_embed.mean().item():.6f}")
        print(f"Position embed std: {pos_embed.std().item():.6f}")
    else:
        print(f"\nNo learned position embedding, using sinusoidal or none")
        trained_with_pos = trained_embed
    
    print(f"\nAfter adding position:")
    print(f"  Mean: {trained_with_pos.mean().item():.6f}")
    print(f"  Std: {trained_with_pos.std().item():.6f}")

# ============================================================
# TRACE STAGE 3: TRANSFORMER LAYERS
# ============================================================

print(f"\n{'='*70}")
print("STAGE 3: TRANSFORMER LAYERS")
print(f"{'='*70}")

with torch.no_grad():
    # Hook to capture intermediate activations
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook
    
    # Register hooks on first layer
    trained_model.blocks[0].attn.q_proj.register_forward_hook(hook_fn('q_proj'))
    trained_model.blocks[0].attn.k_proj.register_forward_hook(hook_fn('k_proj'))
    trained_model.blocks[0].attn.v_proj.register_forward_hook(hook_fn('v_proj'))
    
    # Forward pass through first layer only
    x = trained_with_pos
    for i, block in enumerate(trained_model.blocks):
        if i == 0:
            # Detailed trace for first block
            print(f"\n--- Block {i} ---")
            
            # Self-attention
            residual = x
            x_norm = block.ln1(x)
            
            # Q/K/V projections
            Q = block.attn.q_proj(x_norm)
            K = block.attn.k_proj(x_norm)
            V = block.attn.v_proj(x_norm)
            
            print(f"\nQ/K/V projections:")
            print(f"  Q shape: {Q.shape}, mean: {Q.mean().item():.6f}, std: {Q.std().item():.6f}")
            print(f"  K shape: {K.shape}, mean: {K.mean().item():.6f}, std: {K.std().item():.6f}")
            print(f"  V shape: {V.shape}, mean: {V.mean().item():.6f}, std: {V.std().item():.6f}")
            
            # Q/K/V should be different
            qk_diff = (Q - K).abs().mean().item()
            qv_diff = (Q - V).abs().mean().item()
            print(f"  Q vs K difference: {qk_diff:.6f}")
            print(f"  Q vs V difference: {qv_diff:.6f}")
            
            if qk_diff < 0.001:
                print(f"  ✗ VERIFIED INCORRECT: Q ≈ K (attention broken)")
            else:
                print(f"  ✓ VERIFIED: Q ≠ K")
            
            # Reshape for attention
            batch_size, seq_len, d_model = Q.shape
            n_heads = block.attn.n_heads
            head_dim = d_model // n_heads
            
            Q = Q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
            
            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
            
            print(f"\nAttention scores:")
            print(f"  Shape: {scores.shape}")
            print(f"  Mean: {scores.mean().item():.6f}")
            print(f"  Std: {scores.std().item():.6f}")
            print(f"  Min: {scores.min().item():.6f}")
            print(f"  Max: {scores.max().item():.6f}")
            print(f"  Zero %: {100 * (scores == 0).float().mean().item():.2f}%")
            
            # Causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            print(f"\nAfter causal masking:")
            print(f"  -inf count: {(scores == float('-inf')).sum().item()} / {scores.numel()}")
            
            # Softmax
            attn_probs = F.softmax(scores, dim=-1)
            
            print(f"\nAttention probabilities:")
            print(f"  Shape: {attn_probs.shape}")
            print(f"  Sum per position: {attn_probs[0, 0, 0, :].sum().item():.6f} (should be ≈ 1.0)")
            print(f"  Mean: {attn_probs.mean().item():.6f}")
            print(f"  Min (non-zero): {attn_probs[attn_probs > 0].min().item():.6f}")
            print(f"  NaN: {torch.isnan(attn_probs).any().item()}")
            print(f"  Inf: {torch.isinf(attn_probs).any().item()}")
            
            # Check if attention varies across positions
            attn_std_per_pos = attn_probs.std(dim=-1).mean().item()
            print(f"  Std across last dim: {attn_std_per_pos:.6f}")
            
            if attn_std_per_pos < 0.01:
                print(f"  ✗ VERIFIED INCORRECT: Attention is uniform (not informative)")
            else:
                print(f"  ✓ VERIFIED: Attention varies (informative)")
            
            # Attention output
            attn_output = torch.matmul(attn_probs, V)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            attn_output = block.attn.o_proj(attn_output)
            
            print(f"\nAttention output:")
            print(f"  Shape: {attn_output.shape}")
            print(f"  Mean: {attn_output.mean().item():.6f}")
            print(f"  Std: {attn_output.std().item():.6f}")
            print(f"  Zero %: {100 * (attn_output == 0).float().mean().item():.2f}%")
            
            # Residual connection
            x = residual + attn_output
            
            print(f"\nAfter residual connection:")
            print(f"  Mean: {x.mean().item():.6f}")
            print(f"  Std: {x.std().item():.6f}")
            
            residual_change = (x - residual).abs().mean().item()
            print(f"  Change from residual: {residual_change:.6f}")
            
            if residual_change < 0.001:
                print(f"  ✗ VERIFIED INCORRECT: Residual connection not changing representation")
            else:
                print(f"  ✓ VERIFIED: Residual connection changes representation")
            
            # FFN
            x_norm = block.ln2(x)
            ffn_output = block.ff(x_norm)
            
            print(f"\nFFN output:")
            print(f"  Shape: {ffn_output.shape}")
            print(f"  Mean: {ffn_output.mean().item():.6f}")
            print(f"  Std: {ffn_output.std().item():.6f}")
            print(f"  Zero %: {100 * (ffn_output == 0).float().mean().item():.2f}%")
            
            # Final residual
            x = x + ffn_output
            
        else:
            # Just pass through other layers
            x = block(x)
    
    hidden_states = x
    
    print(f"\n--- Final Hidden States ---")
    print(f"Shape: {hidden_states.shape}")
    print(f"Mean: {hidden_states.mean().item():.6f}")
    print(f"Std: {hidden_states.std().item():.6f}")
    print(f"Min: {hidden_states.min().item():.6f}")
    print(f"Max: {hidden_states.max().item():.6f}")

# ============================================================
# TRACE STAGE 4: OUTPUT LOGITS
# ============================================================

print(f"\n{'='*70}")
print("STAGE 4: OUTPUT LOGITS")
print(f"{'='*70}")

with torch.no_grad():
    # Output projection
    logits = trained_model.lm_head(hidden_states)
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Mean: {logits.mean().item():.6f}")
    print(f"Std: {logits.std().item():.6f}")
    print(f"Min: {logits.min().item():.6f}")
    print(f"Max: {logits.max().item():.6f}")
    
    # Check if logits are uniform
    logits_flat = logits.view(-1)
    logits_std = logits_flat.std().item()
    print(f"Logits std (flat): {logits_std:.6f}")
    
    if logits_std < 0.1:
        print(f"✗ VERIFIED INCORRECT: Logits are nearly uniform (model not discriminative)")
    else:
        print(f"✓ VERIFIED: Logits have meaningful variance")
    
    # Check logit distribution for first position
    first_pos_logits = logits[0, 0, :]
    top5_vals, top5_indices = first_pos_logits.topk(5)
    
    print(f"\nTop-5 predictions for first position:")
    for i, (val, idx) in enumerate(zip(top5_vals, top5_indices)):
        token = tokenizer.decode([idx.item()])
        prob = torch.softmax(first_pos_logits, dim=0)[idx].item()
        print(f"  {i+1}. Token {idx.item()} ('{token}'): logit={val.item():.4f}, prob={prob:.4f}")
    
    # Check if predictions change with position
    second_pos_logits = logits[0, 1, :]
    first_second_diff = (first_pos_logits - second_pos_logits).abs().mean().item()
    print(f"\nLogit difference between position 0 and 1: {first_second_diff:.6f}")
    
    if first_second_diff < 0.01:
        print(f"✗ VERIFIED INCORRECT: Logits don't change across positions")
    else:
        print(f"✓ VERIFIED: Logits vary across positions")

# ============================================================
# TRACE STAGE 5: TRAINING OBJECTIVE VERIFICATION
# ============================================================

print(f"\n{'='*70}")
print("STAGE 5: TRAINING OBJECTIVE VERIFICATION")
print(f"{'='*70}")

print(f"\n--- Input/Target Construction ---")

# Use first sequence
seq = sequences[0]
print(f"Sequence length: {len(seq)}")
print(f"Decoded sequence: {tokenizer.decode(seq[:20])}...")

# Construct input and target
input_tokens = seq[:-1]
target_tokens = seq[1:]

print(f"\nInput tokens: {input_tokens[:10]}...")
print(f"Target tokens: {target_tokens[:10]}...")

# Verify alignment
print(f"\n--- Alignment Check ---")
for i in range(min(10, len(input_tokens))):
    input_token = input_tokens[i]
    target_token = target_tokens[i]
    
    input_str = tokenizer.decode([input_token])
    target_str = tokenizer.decode([target_token])
    
    # Verify target[i] is next token after input[i]
    print(f"Position {i}: input='{input_str}' (token {input_token}) → target='{target_str}' (token {target_token})")

# Verify padding handling
print(f"\n--- Padding Check ---")
for i, seq in enumerate(sequences):
    print(f"Sequence {i}: {len(seq)} tokens (padded to {max_len})")
    padding_count = max_len - len(seq)
    if padding_count > 0:
        print(f"  Has {padding_count} padding tokens at the end")

# ============================================================
# FINAL TEST: TRAINED vs UNTRAINED COMPARISON
# ============================================================

print(f"\n{'='*70}")
print("FINAL TEST: TRAINED vs UNTRAINED MODEL")
print(f"{'='*70}")

with torch.no_grad():
    # Forward pass through both models
    trained_output = trained_model(input_ids)
    untrained_output = untrained_model(input_ids)
    
    trained_logits = trained_output['logits']
    untrained_logits = untrained_output['logits']
    
    # Construct targets
    targets = torch.cat([input_ids[:, 1:], torch.zeros(input_ids.size(0), 1, dtype=torch.long)], dim=1)
    
    # Compute loss
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    trained_loss = criterion(
        trained_logits.reshape(-1, 4000),
        targets.reshape(-1)
    )
    
    untrained_loss = criterion(
        untrained_logits.reshape(-1, 4000),
        targets.reshape(-1)
    )
    
    # Compute accuracy
    trained_preds = trained_logits.argmax(dim=-1)
    untrained_preds = untrained_logits.argmax(dim=-1)
    
    trained_correct = (trained_preds == targets).float().mean().item()
    untrained_correct = (untrained_preds == targets).float().mean().item()
    
    # Compute top-5 accuracy
    trained_top5 = trained_logits.topk(5, dim=-1).indices
    untrained_top5 = untrained_logits.topk(5, dim=-1).indices
    
    trained_top5_correct = 0
    untrained_top5_correct = 0
    total = 0
    
    for i in range(targets.size(0)):
        for j in range(targets.size(1)):
            target = targets[i, j].item()
            if target in trained_top5[i, j].tolist():
                trained_top5_correct += 1
            if target in untrained_top5[i, j].tolist():
                untrained_top5_correct += 1
            total += 1
    
    trained_top5_acc = 100 * trained_top5_correct / total
    untrained_top5_acc = 100 * untrained_top5_correct / total
    
    # Compute correct token probability
    trained_probs = F.softmax(trained_logits, dim=-1)
    untrained_probs = F.softmax(untrained_logits, dim=-1)
    
    trained_correct_prob = 0
    untrained_correct_prob = 0
    
    for i in range(targets.size(0)):
        for j in range(targets.size(1)):
            target = targets[i, j].item()
            trained_correct_prob += trained_probs[i, j, target].item()
            untrained_correct_prob += untrained_probs[i, j, target].item()
    
    trained_correct_prob = 100 * trained_correct_prob / total
    untrained_correct_prob = 100 * untrained_correct_prob / total
    
    print(f"\n--- Loss ---")
    print(f"Trained model: {trained_loss.item():.4f}")
    print(f"Untrained model: {untrained_loss.item():.4f}")
    print(f"Improvement: {100 * (untrained_loss - trained_loss) / untrained_loss:.2f}%")
    
    print(f"\n--- Top-1 Accuracy ---")
    print(f"Trained model: {100 * trained_correct:.2f}%")
    print(f"Untrained model: {100 * untrained_correct:.2f}%")
    print(f"Improvement: {100 * (trained_correct - untrained_correct) / max(untrained_correct, 0.0001):.2f}%")
    
    print(f"\n--- Top-5 Accuracy ---")
    print(f"Trained model: {trained_top5_acc:.2f}%")
    print(f"Untrained model: {untrained_top5_acc:.2f}%")
    print(f"Improvement: {trained_top5_acc - untrained_top5_acc:.2f}%")
    
    print(f"\n--- Correct Token Probability ---")
    print(f"Trained model: {trained_correct_prob:.4f}%")
    print(f"Untrained model: {untrained_correct_prob:.4f}%")
    print(f"Improvement: {trained_correct_prob - untrained_correct_prob:.4f}%")
    
    # Compare hidden states
    print(f"\n--- Hidden States Comparison ---")
    print(f"Trained logits std: {trained_logits.std().item():.6f}")
    print(f"Untrained logits std: {untrained_logits.std().item():.6f}")
    
    logits_diff = (trained_logits - untrained_logits).abs().mean().item()
    print(f"Mean absolute difference: {logits_diff:.6f}")
    
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}")
    
    if trained_correct < 0.01:
        print(f"\n✗ CRITICAL: Trained model has near-zero accuracy (same as untrained)")
        print(f"   Loss improved by {100 * (untrained_loss - trained_loss) / untrained_loss:.2f}%")
        print(f"   But accuracy did NOT improve")
        print(f"\n   CONCLUSION: Model is optimizing loss but not learning to predict correctly")
    elif trained_correct > untrained_correct * 1.5:
        print(f"\n✓ Model shows improvement in accuracy")
        print(f"   Further investigation needed for generation quality")
    else:
        print(f"\n? Model shows minimal improvement")
        print(f"   Need to check if training is effective")

print(f"\n{'='*70}")
print("TRACE COMPLETE")
print(f"{'='*70}")
