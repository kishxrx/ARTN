import torch
import torch.nn.functional as F
import sys
import math
sys.path.insert(0, '.')

from src.model.bitlinear import BitLinearGPT
from datasets import load_dataset
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*70)
print("FORENSIC AUDIT: 50k CHECKPOINT - CURRENT vs NEXT-TOKEN LOSS")
print("="*70)

# Load 50k checkpoint
ckpt_path = 'outputs/artn-50k-best-config/checkpoints/checkpoint_step_50000.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"\nCheckpoint config:")
print(f"  beta1: {ckpt['config']['beta1']}")
print(f"  beta2: {ckpt['config']['beta2']}")
print(f"  total_steps: {ckpt['config']['total_steps']}")
print(f"  Loss at checkpoint: {ckpt['loss']:.4f}")

# Create model
model = BitLinearGPT(
    vocab_size=4000,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_seq_len=256
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f"\nModel loaded from 50k checkpoint")

# Load tokenizer and data
tokenizer = BPETokenizer.load('outputs/artn-50k-best-config/tokenizer.json')
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

# Load a small batch from TinyStories
ds = load_dataset('roneneldan/TinyStories', split='train')
print(f"\nDataset loaded: {len(ds):,} samples")

# Get a fixed batch
batch_texts = [ds[i]['text'] for i in range(4)]
print(f"\nSample texts (first 50 chars):")
for i, t in enumerate(batch_texts):
    print(f"  [{i}] {t[:50]}...")

# Tokenize - keep at 254 to allow for shift
batch_tokens = []
for text in batch_texts:
    tokens = tokenizer.encode(text)[:254]  # Leave room for EOS and shift
    tokens = tokens + [tokenizer.eos_token_id]
    # Pad to 256
    tokens = tokens + [tokenizer.pad_token_id] * (256 - len(tokens))
    batch_tokens.append(tokens)

batch = torch.tensor(batch_tokens, dtype=torch.long)
print(f"\nBatch shape: {batch.shape}")

# ===================================================================
# TEST 1: CURRENT-TOKEN PREDICTION (COPY TASK)
# ===================================================================
print("\n" + "="*70)
print("TEST 1: CURRENT-TOKEN PREDICTION (COPY TASK)")
print("="*70)
print("\nThis is what train_optimized.py does:")
print("  Input:  [t0, t1, t2, t3, ...]")
print("  Target: [t0, t1, t2, t3, ...] (same as input)")
print("  Position i predicts input[i] (can see itself via causal mask)")

# For copy task: input_ids = target
input_ids_copy = batch
targets_copy = batch.clone()

# Mask padding in targets
targets_copy[targets_copy == tokenizer.pad_token_id] = -100

# Forward pass
with torch.no_grad():
    outputs = model(input_ids_copy, targets_copy)
    logits = outputs['logits']
    loss_copy = outputs['loss']

print(f"\nResults:")
print(f"  Current-token CE Loss: {loss_copy.item():.4f}")
print(f"  Current-token Perplexity: {math.exp(loss_copy.item()):.2f}")

# Calculate accuracy
predictions = logits.argmax(dim=-1)
mask = targets_copy != -100
correct = (predictions == targets_copy) & mask
total_tokens = mask.sum().item()
correct_tokens = correct.sum().item()
accuracy_copy = correct_tokens / total_tokens if total_tokens > 0 else 0

print(f"  Current-token Accuracy: {accuracy_copy*100:.2f}%")
print(f"  Correct: {correct_tokens}/{total_tokens}")

# ===================================================================
# TEST 2: NEXT-TOKEN PREDICTION (CORRECT LANGUAGE MODELING)
# ===================================================================
print("\n" + "="*70)
print("TEST 2: NEXT-TOKEN PREDICTION (CORRECT LANGUAGE MODELING)")
print("="*70)
print("\nThis is what train_beta1_0.5.py does:")
print("  Input:  [t0, t1, t2, t3, ...] (batch[:, :-1])")
print("  Target: [t1, t2, t3, t4, ...] (batch[:, 1:])")
print("  Position i predicts input[i+1] (cannot see target)")

# For next-token: shift
# Model expects max_seq_len, so pad to 256 before shifting, then slice
input_ids_next = batch[:, :-1].contiguous()  # Shape: [4, 255]
targets_next = batch[:, 1:].contiguous()      # Shape: [4, 255]

# Create padded versions for model
input_ids_padded = torch.zeros(4, 256, dtype=torch.long)
input_ids_padded[:, :255] = input_ids_next
input_ids_padded[:, 255] = tokenizer.pad_token_id

targets_padded = torch.zeros(4, 256, dtype=torch.long) - 100  # Start with all masked
targets_padded[:, :255] = targets_next

# Mask padding in targets
pad_mask = targets_next == tokenizer.pad_token_id
targets_padded[:, :255][pad_mask] = -100

# Also mask where previous position had EOS
eos_mask = input_ids_next == tokenizer.eos_token_id
targets_padded[:, :255][eos_mask] = -100

# Forward pass
with torch.no_grad():
    outputs = model(input_ids_padded, targets_padded)
    logits = outputs['logits']
    loss_next = outputs['loss']

print(f"\nResults:")
print(f"  Next-token CE Loss: {loss_next.item():.4f}")
print(f"  Next-token Perplexity: {math.exp(loss_next.item()):.2f}")

# Calculate accuracy on the 255 positions
predictions = logits[:, :255].argmax(dim=-1)
mask = targets_padded[:, :255] != -100
correct = (predictions == targets_padded[:, :255]) & mask
total_tokens = mask.sum().item()
correct_tokens = correct.sum().item()
accuracy_next = correct_tokens / total_tokens if total_tokens > 0 else 0

print(f"  Next-token Accuracy: {accuracy_next*100:.2f}%")
print(f"  Correct: {correct_tokens}/{total_tokens}")

# ===================================================================
# COMPARISON
# ===================================================================
print("\n" + "="*70)
print("COMPARISON: WHICH TASK WAS THE MODEL TRAINED FOR?")
print("="*70)

print(f"\n{'Metric':<30} {'Current-Token':<20} {'Next-Token':<20}")
print("-" * 70)
print(f"{'CE Loss':<30} {loss_copy.item():<20.4f} {loss_next.item():<20.4f}")
print(f"{'Perplexity':<30} {math.exp(loss_copy.item()):<20.2f} {math.exp(loss_next.item()):<20.2f}")
print(f"{'Accuracy':<30} {accuracy_copy*100:<19.2f}% {accuracy_next*100:<19.2f}%")

# Training loss from checkpoint was ~7.12
print(f"\n{'Training Loss (from ckpt)':<30} {ckpt['loss']:<20.4f}")

# Determine which task the model was trained for
print("\n" + "="*70)
print("VERDICT")
print("="*70)

diff_copy = abs(loss_copy.item() - ckpt['loss'])
diff_next = abs(loss_next.item() - ckpt['loss'])

print(f"\nDifference from training loss:")
print(f"  Current-token loss diff: {diff_copy:.4f}")
print(f"  Next-token loss diff: {diff_next:.4f}")

if diff_next < diff_copy:
    print("\n>>> MODEL WAS TRAINED FOR NEXT-TOKEN PREDICTION <<<")
    print(f">>> Next-token loss ({loss_next.item():.2f}) is closer to training loss ({ckpt['loss']:.2f})")
else:
    print("\n>>> MODEL WAS TRAINED FOR CURRENT-TOKEN COPY TASK <<<")
    print(f">>> Current-token loss ({loss_copy.item():.2f}) is closer to training loss ({ckpt['loss']:.2f})")

print("\n" + "="*70)
print("CONCLUSIONS")
print("="*70)
print("""
1. train_beta1_0.5.py uses CORRECT shifting:
   - input_ids = batch[:, :-1]
   - targets = batch[:, 1:]
   
2. train_optimized.py uses NO shifting (copy task):
   - input_ids = batch["input_ids"]
   - targets = batch["labels"] = input_ids.clone()

3. Check which loss matches to determine training objective.
""")
