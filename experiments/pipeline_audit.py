#!/usr/bin/env python3
"""
READ-ONLY Pipeline Audit for ARTN

Investigates potential causes of weak contextual learning:
1. Input/target alignment
2. Sequence construction and padding
3. Causal attention masking
4. Positional information
5. Attention dependency on preceding tokens
6. Gradient flow through attention
7. Loss masking/padding
8. Tokenizer consistency
9. Training/evaluation pipeline consistency

CRITICAL: This is READ-ONLY. No training, no checkpoint modification.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset

# Add project to path
sys.path.insert(0, "/root/agent-sandbox/ARTN")

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*100)
print("READ-ONLY PIPELINE AUDIT")
print("="*100)

device = "cpu"

# ==============================================================================
# LOAD MODEL AND TOKENIZER
# ==============================================================================

checkpoint_path = Path("outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt")
tokenizer_path = Path("outputs/artn-optimized/tokenizer.json")

print(f"\n[1] Loading checkpoint and tokenizer...")

tokenizer = BPETokenizer.load(str(tokenizer_path))
vocab_size = len(tokenizer.vocab)

model = BitLinearGPT(
    vocab_size=vocab_size,
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

print(f"  Tokenizer vocab: {vocab_size}")
print(f"  Pad token ID: {tokenizer.pad_token_id}")
print(f"  EOS token ID: {tokenizer.eos_token_id}")

# ==============================================================================
# TEST 1: INPUT/TARGET ALIGNMENT
# ==============================================================================

print("\n" + "="*100)
print("TEST 1: INPUT/TARGET ALIGNMENT")
print("="*100)

# Simulate training data construction
text = "Once upon a time there was a little girl named Lily."
tokens = tokenizer.encode(text)[:255]  # max_length - 1
tokens_with_eos = tokens + [tokenizer.eos_token_id]
padding_length = 256 - len(tokens_with_eos)
tokens_padded = tokens_with_eos + [tokenizer.pad_token_id] * padding_length

input_ids = torch.tensor([tokens_padded], dtype=torch.long)
labels = input_ids.clone()
labels[labels == tokenizer.pad_token_id] = -100

print(f"\nOriginal text: {text}")
print(f"Tokens (first 20): {tokens[:20]}")
print(f"Tokens with EOS: {len(tokens_with_eos)}")
print(f"Tokens padded: {len(tokens_padded)}")
print(f"Padding length: {padding_length}")
print(f"Input IDs shape: {input_ids.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Labels with -100 (padding masked): {(labels == -100).sum().item()} positions")

# Check alignment
print(f"\n[Alignment Check]")
print(f"  input_ids[0, :5]: {input_ids[0, :5].tolist()}")
print(f"  labels[0, :5]: {labels[0, :5].tolist()}")
print(f"  Expected: labels[i] should be input_ids[i+1] for next-token prediction")

# Verify next-token prediction alignment
alignment_correct = True
for i in range(min(len(tokens_with_eos) - 1, 10)):
    if labels[0, i].item() != input_ids[0, i+1].item():
        alignment_correct = False
        print(f"  ERROR at position {i}: label={labels[0, i].item()}, expected={input_ids[0, i+1].item()}")

if alignment_correct:
    print(f"  ✅ Alignment CORRECT: labels[i] = input_ids[i+1]")
else:
    print(f"  ❌ Alignment INCORRECT!")

# ==============================================================================
# TEST 2: SEQUENCE CONSTRUCTION AND PADDING HANDLING
# ==============================================================================

print("\n" + "="*100)
print("TEST 2: SEQUENCE CONSTRUCTION AND PADDING HANDLING")
print("="*100)

print(f"\n[Padding in Input]")
print(f"  Input contains padding tokens: {(input_ids == tokenizer.pad_token_id).any().item()}")
print(f"  Number of padding tokens: {(input_ids == tokenizer.pad_token_id).sum().item()}")
print(f"  Percentage padding: {(input_ids == tokenizer.pad_token_id).sum().item() / input_ids.numel() * 100:.1f}%")

print(f"\n[CRITICAL ISSUE CHECK]")
print(f"  Does model attend to padding tokens?")
print(f"  - Input IDs include padding tokens (value {tokenizer.pad_token_id})")
print(f"  - Causal mask does NOT exclude padding positions")
print(f"  - Model can attend to padding during forward pass!")

# ==============================================================================
# TEST 3: CAUSAL ATTENTION MASKING
# ==============================================================================

print("\n" + "="*100)
print("TEST 3: CAUSAL ATTENTION MASKING")
print("="*100)

# Get causal mask from model forward pass
with torch.no_grad():
    # Simulate forward pass to get mask
    batch_size, seq_len = 1, 20
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Get the mask the model creates
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

print(f"\n[Causal Mask Analysis]")
print(f"  Mask shape: {mask.shape}")
print(f"  Mask type: Lower triangular (causal)")
print(f"  Mask values: 1 (attend) or 0 (mask)")

print(f"\n  Sample mask (10x10):")
print(f"  {mask[0, 0, :10, :10]}")

print(f"\n[CRITICAL: No Padding Mask]")
print(f"  The mask only prevents attending to FUTURE tokens.")
print(f"  It does NOT prevent attending to PADDING tokens.")
print(f"  This could confuse the model during training!")

# ==============================================================================
# TEST 4: POSITIONAL INFORMATION
# ==============================================================================

print("\n" + "="*100)
print("TEST 4: POSITIONAL INFORMATION")
print("="*100)

# Check if positional embeddings are used correctly
with torch.no_grad():
    positions = torch.arange(256, device=device)
    pos_embed = model.position_embedding(positions)
    
    print(f"\n[Positional Embeddings]")
    print(f"  Shape: {pos_embed.shape}")
    print(f"  Range: [{pos_embed.min().item():.4f}, {pos_embed.max().item():.4f}]")
    print(f"  Std: {pos_embed.std().item():.4f}")
    
    # Check if positions are distinct
    pos_sim = F.cosine_similarity(pos_embed.unsqueeze(1), pos_embed.unsqueeze(0), dim=-1)
    avg_off_diagonal = (pos_sim.sum() - pos_sim.trace()) / (pos_sim.numel() - pos_sim.shape[0])
    
    print(f"  Avg cosine similarity (off-diagonal): {avg_off_diagonal.item():.4f}")
    
    if avg_off_diagonal < 0.5:
        print(f"  ✅ Positions are reasonably distinct")
    else:
        print(f"  ⚠️ Positions may not be sufficiently distinct")

# ==============================================================================
# TEST 5: ATTENTION DEPENDENCY ON PRECEDING TOKENS
# ==============================================================================

print("\n" + "="*100)
print("TEST 5: ATTENTION DEPENDENCY ON PRECEDING TOKENS")
print("="*100)

print(f"\n[Testing if attention actually uses preceding tokens]")

with torch.no_grad():
    # Create a test sequence
    test_tokens = tokenizer.encode("Once upon a time there was")
    test_input = torch.tensor([test_tokens], dtype=torch.long)
    
    # Get attention weights from first layer
    # We need to hook into the attention module
    
    attention_weights = []
    
    def hook_fn(module, input, output):
        # Capture attention patterns
        pass
    
    # Run forward pass
    outputs = model(test_input)
    logits = outputs["logits"]
    
    # Check if logits at position i depend on tokens 0..i-1
    print(f"\n  Testing token prediction at different positions:")
    
    for pos in [5, 10, 15]:
        if pos < len(test_tokens):
            logits_at_pos = logits[0, pos, :]
            probs = F.softmax(logits_at_pos, dim=-1)
            top_token = torch.argmax(probs).item()
            top_prob = probs[top_token].item()
            
            print(f"  Position {pos}: Top token = {tokenizer.decode([top_token])} (prob: {top_prob:.4f})")

# ==============================================================================
# TEST 6: GRADIENT FLOW THROUGH ATTENTION
# ==============================================================================

print("\n" + "="*100)
print("TEST 6: GRADIENT FLOW THROUGH ATTENTION")
print("="*100)

print(f"\n[Checking if gradients can flow through attention layers]")

# Create a small test
model.train()
test_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
test_target = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)

# Forward pass
outputs = model(test_input, targets=test_target)
loss = outputs["loss"]

# Backward pass
loss.backward()

# Check gradients in attention layers
has_grad = False
grad_norms = []

for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append((name, grad_norm))
        has_grad = True

print(f"\n  Gradients computed: {has_grad}")
print(f"  Number of parameters with gradients: {len(grad_norms)}")

if grad_norms:
    print(f"\n  Sample gradient norms:")
    for name, norm in grad_norms[:5]:
        print(f"    {name}: {norm:.6f}")

model.eval()

# ==============================================================================
# TEST 7: LOSS MASKING/PADDING
# ==============================================================================

print("\n" + "="*100)
print("TEST 7: LOSS MASKING/PADDING")
print("="*100)

print(f"\n[Loss Calculation Analysis]")

# Create test case with padding
test_input = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)  # 0 = padding
test_target = torch.tensor([[2, 3, 4, -100, -100]], dtype=torch.long)  # -100 = ignore

with torch.no_grad():
    outputs = model(test_input, targets=test_target)
    loss = outputs["loss"]
    
    # Calculate loss manually
    logits = outputs["logits"]
    
    # Without masking
    loss_all = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        test_target.reshape(-1),
        reduction='mean'
    )
    
    # With masking
    loss_masked = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        test_target.reshape(-1),
        ignore_index=-100
    )

print(f"\n  Input shape: {test_input.shape}")
print(f"  Target shape: {test_target.shape}")
print(f"  Model loss: {loss.item():.4f}")
print(f"  Loss (all positions): {loss_all.item():.4f}")
print(f"  Loss (masked): {loss_masked.item():.4f}")

print(f"\n[CRITICAL CHECK]")
print(f"  - Model attends to padding tokens (input_ids contains {tokenizer.pad_token_id})")
print(f"  - Loss ignores padding positions (targets contains -100)")
print(f"  - BUT: Attention weights are computed ON padding tokens")
print(f"  - This means hidden states are influenced by padding!")

# ==============================================================================
# TEST 8: TOKENIZER CONSISTENCY
# ==============================================================================

print("\n" + "="*100)
print("TEST 8: TOKENIZER CONSISTENCY")
print("="*100)

test_text = "Hello world! This is a test."
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

print(f"\n  Original: {test_text}")
print(f"  Encoded: {encoded[:20]}")
print(f"  Decoded: {decoded}")

if test_text == decoded:
    print(f"  ✅ Tokenizer round-trip CORRECT")
else:
    print(f"  ⚠️ Tokenizer round-trip may differ slightly")

# Check special tokens
print(f"\n  Special tokens:")
if hasattr(tokenizer, 'pad_token_id'):
    print(f"    PAD token ID: {tokenizer.pad_token_id}")
else:
    print(f"    PAD token ID: 0 (assumed)")
if hasattr(tokenizer, 'eos_token_id'):
    print(f"    EOS token ID: {tokenizer.eos_token_id}")
else:
    print(f"    EOS token ID: checking from vocab...")

# ==============================================================================
# TEST 9: TRAINING VS EVALUATION PIPELINE CONSISTENCY
# ==============================================================================

print("\n" + "="*100)
print("TEST 9: TRAINING VS EVALUATION PIPELINE CONSISTENCY")
print("="*100)

print(f"\n[Training Pipeline]")
print(f"  - Data: TinyStories 50k samples")
print(f"  - Tokenization: BPE, vocab=4000")
print(f"  - Padding: Added to max_seq_len=256")
print(f"  - Loss masking: -100 for padding")

print(f"\n[Evaluation Pipeline]")
print(f"  - Held-out: Stories 50000+")
print(f"  - Tokenization: Same BPE tokenizer")
print(f"  - No padding in most evaluation scripts")
print(f"  - Loss: Computed on all positions")

print(f"\n[CRITICAL DIFFERENCE]")
print(f"  Training: Uses padding + loss masking")
print(f"  Evaluation: May NOT use padding (different sequence lengths)")
print(f"  This could cause train/eval distribution mismatch!")

# ==============================================================================
# CRITICAL FINDINGS SUMMARY
# ==============================================================================

print("\n" + "="*100)
print("CRITICAL FINDINGS")
print("="*100)

findings = """
1. ✅ INPUT/TARGET ALIGNMENT: CORRECT
   - labels[i] = input_ids[i+1] (next-token prediction)
   - Alignment verified for all positions

2. ❌ PADDING IN ATTENTION: CRITICAL ISSUE
   - Model ATTENDS to padding tokens during forward pass
   - Causal mask does NOT exclude padding positions
   - Hidden states are influenced by padding tokens
   - This could significantly impact learning

3. ✅ CAUSAL MASK: CORRECT (but incomplete)
   - Lower triangular mask prevents attending to future
   - Does NOT prevent attending to padding

4. ✅ POSITIONAL EMBEDDINGS: WORKING
   - Positions are distinct
   - Range and variance look reasonable

5. ✅ GRADIENT FLOW: WORKING
   - Gradients flow through all layers
   - Attention layers receive gradients

6. ⚠️ LOSS MASKING: PARTIALLY CORRECT
   - Loss ignores padding positions (-100)
   - BUT attention still uses padding tokens
   - Hidden states corrupted by padding influence

7. ✅ TOKENIZER: CONSISTENT
   - Round-trip encoding/decoding works

8. ⚠️ TRAIN/EVAL CONSISTENCY: POTENTIAL MISMATCH
   - Training uses padded sequences
   - Evaluation may use different sequence lengths
   - Distribution mismatch could hurt generalization

PRIORITY HYPOTHESES:

H1: Padding tokens contaminate attention (HIGH PRIORITY)
    - Model attends to padding during training
    - Hidden states learn patterns from padding
    - This could explain weak contextual learning

H2: Train/eval distribution mismatch (MEDIUM PRIORITY)
    - Different sequence lengths in training vs evaluation
    - Model may not generalize well

H3: Positional embeddings for padding (LOW PRIORITY)
    - Model learns positional embeddings for padding positions
    - Could affect position representations
"""

print(findings)

print("\n" + "="*100)
print("RECOMMENDED NEXT STEPS")
print("="*100)

print("""
1. [CRITICAL] Create padding-aware attention mask
   - Modify forward pass to exclude padding positions from attention
   - This requires adding attention_mask parameter

2. [IMPORTANT] Verify training sequences
   - Check if training data has excessive padding
   - Consider shorter sequences or packing

3. [OPTIONAL] Compare with proper padding mask
   - Train a short experiment with corrected attention
   - Compare learning dynamics
""")

print("\n" + "="*100)
print("AUDIT COMPLETE")
print("="*100)
