import torch, sys, math
sys.path.insert(0, '.')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print('='*70)
print('PHASE 22: FAST EVAL')
print('='*70)

# FP32 model uses standard Linear layers
class StandardLinear(nn.Linear):
    pass

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.q_proj = StandardLinear(d_model, d_model)
        self.k_proj = StandardLinear(d_model, d_model)
        self.v_proj = StandardLinear(d_model, d_model)
        self.o_proj = StandardLinear(d_model, d_model)
    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None: scores = scores.masked_fill(mask == 0, float('-inf'))
        out = torch.matmul(F.softmax(scores, dim=-1), v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

class StandardFF(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = StandardLinear(d_model, d_ff)
        self.w2 = StandardLinear(d_ff, d_model)
    def forward(self, x): return self.w2(F.gelu(self.w1(x)))

class StandardBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StandardAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = StandardFF(d_model, d_ff)
    def forward(self, x, mask=None): return x + self.ff(self.ln2(x + self.attn(self.ln1(x), mask)))

class StandardGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([StandardBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = StandardLinear(d_model, vocab_size)
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        x = self.token_embedding(input_ids) + self.position_embedding(torch.arange(T, device=input_ids.device))
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks: x = block(x, mask)
        logits = self.lm_head(self.ln_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

# Dataset
class DS(Dataset):
    def __init__(self, d, t, l=256): self.d, self.t, self.l = d, t, l
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        s = self.t.encode(self.d[i]['text'])[:self.l-1] + [self.t.eos_token_id]
        s = s + [self.t.pad_token_id] * (self.l - len(s))
        x = torch.tensor(s, dtype=torch.long)
        y = x.clone(); y[y == self.t.pad_token_id] = -100
        return {'input_ids': x, 'labels': y}

# Load data
tok = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
ds = load_dataset('roneneldan/TinyStories', split='train')
ev = ds.select(range(int(len(ds)*0.95), len(ds)))
print(f'Eval samples: {len(ev):,}')

evl = DataLoader(DS(ev, tok), batch_size=64, shuffle=False)

# Models - FP32 uses StandardGPT, Ternary uses BitLinearGPT
fm = StandardGPT(4000, 256, 8, 6, 1024, 256)
tm = BitLinearGPT(4000, 256, 8, 6, 1024, 256)

# Eval function
def evl_fn(m, l, nb=10):
    m.eval(); tl, tc, tt = 0, 0, 0
    with torch.no_grad():
        for i, b in enumerate(l):
            if i >= nb: break
            o = m(b['input_ids'], b['labels'])
            lg, ls = o['logits'], o['loss']
            lb = b['labels']; pd = lg.argmax(-1); mk = lb != -100
            tc += ((pd == lb) & mk).sum().item(); tt += mk.sum().item(); tl += ls.item()
    return {'loss': tl/nb, 'ppl': math.exp(tl/nb), 'acc': tc/tt}

od = 'outputs/phase22_controlled_experiment'
print(f'\nStep   Model      Eval Loss   PPL        Accuracy')
print('-' * 55)

for step in [1000, 2000, 5000]:
    fp32_path = f'{od}/fp32_checkpoints/checkpoint_step_{step}.pt'
    fm.load_state_dict(torch.load(fp32_path, map_location='cpu')['model_state_dict'])
    r = evl_fn(fm, evl, nb=10)
    print(f"{step:<6} FP32       {r['loss']:<11.4f} {r['ppl']:<10.2f} {r['acc']*100:<8.2f}%")
    
    ternary_path = f'{od}/ternary_checkpoints/checkpoint_step_{step}.pt'
    tm.load_state_dict(torch.load(ternary_path, map_location='cpu')['model_state_dict'])
    r = evl_fn(tm, evl, nb=10)
    print(f"{step:<6} Ternary    {r['loss']:<11.4f} {r['ppl']:<10.2f} {r['acc']*100:<8.2f}%")
    print()

print('Done')
