import torch, sys, math
sys.path.insert(0, '/root/agent-sandbox/ARTN')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

print("="*60)
print("PHASE 22: MINIMAL EVAL (5 batches)")
print("="*60)

# Minimal FP32 model
class L(nn.Linear): pass
class A(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.d, self.h, self.dh = d, h, d//h
        self.q, self.k, self.v, self.o = L(d,d), L(d,d), L(d,d), L(d,d)
    def forward(self, x, m=None):
        B,T,C = x.shape
        q = self.q(x).view(B,T,self.h,self.dh).transpose(1,2)
        k = self.k(x).view(B,T,self.h,self.dh).transpose(1,2)
        v = self.v(x).view(B,T,self.h,self.dh).transpose(1,2)
        s = torch.matmul(q,k.transpose(-2,-1))/(self.dh**0.5)
        if m is not None: s = s.masked_fill(m==0, float('-inf'))
        return self.o(torch.matmul(F.softmax(s,-1),v).transpose(1,2).contiguous().view(B,T,C))
class FF(nn.Module):
    def __init__(self, d, f): super().__init__(); self.w1, self.w2 = L(d,f), L(f,d)
    def forward(self, x): return self.w2(F.gelu(self.w1(x)))
class BL(nn.Module):
    def __init__(self, d, h, f): super().__init__(); self.l1, self.a, self.l2, self.f = nn.LayerNorm(d), A(d,h), nn.LayerNorm(d), FF(d,f)
    def forward(self, x, m=None): return x + self.f(self.l2(x + self.a(self.l1(x), m)))
class GPT(nn.Module):
    def __init__(self, v, d, h, l, f, s):
        super().__init__(); self.te, self.pe = nn.Embedding(v,d), nn.Embedding(s,d)
        self.bl = nn.ModuleList([BL(d,h,f) for _ in range(l)])
        self.ln, self.lm = nn.LayerNorm(d), L(d,v)
    def forward(self, x, t=None):
        B,T = x.shape; h = self.te(x) + self.pe(torch.arange(T))
        m = torch.tril(torch.ones(T,T)).unsqueeze(0).unsqueeze(0)
        for b in self.bl: h = b(h,m)
        lg = self.lm(self.ln(h))
        loss = F.cross_entropy(lg.reshape(-1,lg.size(-1)), t.reshape(-1), ignore_index=-100) if t is not None else None
        return {'logits': lg, 'loss': loss}

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
tok = BPETokenizer.load('/root/agent-sandbox/ARTN/outputs/artn-optimized/tokenizer.json')
ds = load_dataset('roneneldan/TinyStories', split='train')
ev = ds.select(range(int(len(ds)*0.95), len(ds)))
evl = DataLoader(DS(ev, tok), batch_size=64, shuffle=False)

# Models
fm = GPT(4000, 256, 8, 6, 1024, 256)
tm = BitLinearGPT(4000, 256, 8, 6, 1024, 256)

def evl(m, l, nb=5):
    m.eval(); tl, tc, tt = 0, 0, 0
    with torch.no_grad():
        for i, b in enumerate(l):
            if i >= nb: break
            o = m(b['input_ids'], b['labels'])
            lg, ls = o['logits'], o['loss']
            lb = b['labels']; pd = lg.argmax(-1); mk = lb != -100
            tc += ((pd == lb) & mk).sum().item(); tt += mk.sum().item(); tl += ls.item()
    return {'loss': tl/nb, 'ppl': math.exp(tl/nb), 'acc': tc/tt}

# Results
od = '/root/agent-sandbox/ARTN/outputs/phase22_controlled_experiment'
print(f"\n{'Step':<6} {'Model':<10} {'Loss':<10} {'PPL':<10} {'Acc':<10}")
print('-' * 50)
for step in [1000, 2000, 5000]:
    for name, model in [('FP32', fm), ('Ternary', tm)]:
        path = f'{od}/{"fp32" if name=="FP32" else "ternary"}_checkpoints/checkpoint_step_{step}.pt'
        model.load_state_dict(torch.load(path, map_location='cpu')['model_state_dict'])
        r = evl(model, evl, nb=5)
        print(f"{step:<6} {name:<10} {r['loss']:<10.4f} {r['ppl']:<10.2f} {r['acc']*100:<8.2f}%")
    print()
print("Done")
