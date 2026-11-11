import torch, sys
sys.path.insert(0, '.')
from datasets import load_dataset
from src.tokenizer.bpe_tokenizer import BPETokenizer

tok = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
print(f"PAD={tok.pad_token_id}, EOS={tok.eos_token_id}")

ds = load_dataset('roneneldan/TinyStories', split='train')
text = ds[0]['text'][:50]
tokens = tok.encode(text)[:10]
print(f"Text: {text}")
print(f"Tokens: {tokens}")

# Current implementation
t = tokens[:9] + [tok.eos_token_id]
t = t + [tok.pad_token_id] * (10 - len(t))
x = torch.tensor(t)
y = x.clone()
y[y == tok.pad_token_id] = -100
print(f"Input:  {x.tolist()}")
print(f"Labels: {y.tolist()}")
print("\nPROBLEM: labels[i] = input[i] means model sees token it predicts!")
print("At pos i, model attends to 0..i (causal mask), including itself.")
print("SOLUTION: labels should be shifted: labels[i] = input[i+1]")
