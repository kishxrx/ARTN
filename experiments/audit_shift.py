import torch, sys
sys.path.insert(0, '.')
from datasets import load_dataset
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*70)
print("AUDIT: Input/Label Shifting Analysis")
print("="*70)

tok = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
print(f"PAD token: {tok.pad_token_id}")
print(f"EOS token: {tok.eos_token_id}")

# Check how data is prepared
ds = load_dataset('roneneldan/TinyStories', split='train')
sample_text = ds[0]['text'][:100]  # First 100 chars
print(f"\nSample text: {sample_text}")

tokens = tok.encode(sample_text)
print(f"\nEncoded tokens: {tokens[:20]}...")

# Current implementation
tokens_padded = tokens[:255] + [tok.eos_token_id]
tokens_padded = tokens_padded + [tok.pad_token_id] * (256 - len(tokens_padded))

input_ids = torch.tensor(tokens_padded, dtype=torch.long)
labels = input_ids.clone()
labels[labels == tok.pad_token_id] = -100

print(f"\nInput IDs (first 10): {input_ids[:10].tolist()}")
print(f"Labels (first 10): {labels[:10].tolist()}")

print("\n" + "="*70)
print("PROBLEM ANALYSIS")
print("="*70)

print("\nFor causal language modeling, next-token prediction should be:")
print("  Position 0: Predict token at position 1 from tokens [0]")
print("  Position 1: Predict token at position 2 from tokens [0,1]")
print("  etc.")

print("\nBut current implementation has:")
print("  Position 0: Predict token at position 0 from tokens [0] (can see itself!)")
print("  Position 1: Predict token at position 1 from tokens [0,1] (can see itself!)")
print("  etc.")

print("\nWith causal mask, position i can see positions 0..i (inclusive).")
print("This means each position can SEE the token it's trying to predict!")

print("\nThis explains the 99-100% accuracy - the model learns to copy the input!")

print("\n" + "="*70)
print("CORRECT IMPLEMENTATION")
print("="*70)

print("\nFor next-token prediction, labels should be SHIFTED:")
print("  input_ids = [t0, t1, t2, t3, EOS, PAD, PAD, ...]")
print("  labels    = [t1, t2, t3, EOS, PAD, -100, -100, ...]")

print("\nOR with BOS token:")
print("  input_ids = [BOS, t0, t1, t2, t3, EOS, PAD, ...]")
print("  labels    = [t0, t1, t2, t3, EOS, PAD, -100, ...]")

print("\nBut tokenizer has no BOS token defined!")

# Check if tokenizer has BOS
try:
    print(f"\nBOS token ID: {tok.bos_token_id}")
except:
    print("\nBOS token ID: NOT DEFINED")

print("\n" + "="*70)
print("VERDICT: Training is NOT next-token prediction!")
print("="*70)
print("\nThe model is trained to predict the current token (with access to itself)")
print("This is why accuracy is ~100% - trivial copying task!")
print("\nLOSS still reveals the gap because:")
print("  - FP32: More confident predictions")
print("  - Ternary: Less confident predictions (noisier)")
print("\nBut NEITHER model is learning proper next-token prediction!")
