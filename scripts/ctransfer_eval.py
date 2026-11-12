#!/usr/bin/env python3
"""Construction-transfer evaluation harness (batched-by-length trie projection).

Scoring mathematics identical to the unbatched reference: q in Delta^21 over the
20 action lemmas + OTHER, computed as the exact trie leaf-probability sum, then
top-1 correctness = argmax(q[:20]) == subject index (CANON is the identity bijection).

  --selfcheck : verify batched vs unbatched numerical agreement on a small subset
  --arms      : a,b,c,b0=<ckpt>[,<ckpt>...] produce per-arm subject-level results
                with cluster bootstrap over the 20 subjects
"""
import argparse, json, math, random, sys
from pathlib import Path
from collections import defaultdict
import torch

BASE = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE))
from experiments import ctransfer_common as C
from experiments import ctransfer_train as T
from src.tokenizer.bpe_tokenizer import BPETokenizer

NSUB = len(C.SUBJECTS)


def _chain_tensors(seqs, internal):
    return T._seq_chains(seqs, internal, _CLASS_OF)


def batched_q(model, prefixes, seqs, internal, order):
    """prefixes: list of id-lists -> tensor (n,21). One forward per distinct length."""
    paths, rows, toks, mask, seq_verb = T._seq_chains(seqs, internal, _CLASS_OF)
    rg = defaultdict(list)
    for i, p in enumerate(prefixes):
        rg[len(p)].append((i, p))
    root = [None] * len(prefixes)
    for ln, g in rg.items():
        b = torch.tensor([x[1] for x in g], dtype=torch.long)
        pr = torch.softmax(model(b)['logits'][:, -1, :], dim=-1)
        for k, (i, _p) in enumerate(g):
            root[i] = pr[k]
    items = []
    for i, p in enumerate(prefixes):
        for pa in internal:
            items.append((i, len(p) + len(pa), p + list(pa)))
    ag = defaultdict(list)
    for it in items:
        ag[it[1]].append(it)
    aux = [dict() for _ in prefixes]
    for ln, g in ag.items():
        b = torch.tensor([it[2] for it in g], dtype=torch.long)
        pr = torch.softmax(model(b)['logits'][:, -1, :], dim=-1)
        for k, it in enumerate(g):
            aux[it[0]][tuple(it[2][len(prefixes[it[0]]):])] = pr[k]
    out = []
    for i in range(len(prefixes)):
        P = torch.empty((len(paths), root[i].shape[0]))
        P[0] = root[i]
        for j, p in enumerate(order):
            if j > 0:
                P[j] = aux[i][p]
        leaf = P[rows, toks].pow(mask).prod(dim=1)
        q = torch.zeros(NSUB + 1)
        q.index_add_(0, seq_verb, leaf)
        q[-1] = (1.0 - q[:-1].sum()).clamp_min(0.0)
        out.append(q / q.sum().clamp_min(1e-9))
    return torch.stack(out)


def unbatched_q(model, prefix, seqs, internal, order):
    """Reference implementation: one forward per prompt and per trie path."""
    root = torch.softmax(model(torch.tensor([prefix]))['logits'][0, -1, :], dim=-1)
    P = torch.empty((len(order), root.shape[0]))
    P[0] = root
    for j, p in enumerate(order):
        if j > 0:
            b = torch.tensor([prefix + list(p)])
            P[j] = torch.softmax(model(b)['logits'][0, -1, :], dim=-1)
    paths, rows, toks, mask, seq_verb = T._seq_chains(seqs, internal, _CLASS_OF)
    leaf = P[rows, toks].pow(mask).prod(dim=1)
    q = torch.zeros(NSUB + 1)
    q.index_add_(0, seq_verb, leaf)
    q[-1] = (1.0 - q[:-1].sum()).clamp_min(0.0)
    return q / q.sum().clamp_min(1e-9)


def cluster_ci(vec, B=10000, seed=0, alpha=0.05):
    """Cluster bootstrap over the 20 subjects (resample subjects with replacement)."""
    rng = random.Random(seed)
    n = len(vec)
    means = []
    for _ in range(B):
        s = sum(vec[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    return means[int(alpha / 2 * B)], means[int((1 - alpha / 2) * B) - 1]


def eval_arm(model, tok, frames, seqs, internal, order):
    prefixes, meta = [], []
    for fid in frames:
        for si, subj in enumerate(C.SUBJECTS):
            tmpl, _key = C.frame_prefix(fid, subj)
            prefixes.append(tok.encode(tmpl))
            meta.append((fid, si))
    qs = batched_q(model, prefixes, seqs, internal, order)
    per_frame = defaultdict(lambda: [0, 0])
    per_subj = defaultdict(list)
    for (fid, si), q in zip(meta, qs):
        hit = int(int(q[:NSUB].argmax()) == si)
        per_frame[fid][0] += hit
        per_frame[fid][1] += 1
        per_subj[si].append(hit)
    subj_vec = [sum(per_subj[i]) / len(per_subj[i]) for i in range(NSUB)]
    return {f: per_frame[f][0] / per_frame[f][1] for f in per_frame}, subj_vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--arms', default=None, help='A=path,B=path,...')
    ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    tok = BPETokenizer.load(str(C.TOKENIZER_PATH))
    C.measure_frames(tok)
    seqs, nodes, internal, class_of = C.build_trie(tok)
    globals()['_CLASS_OF'] = class_of
    order = [()] + [tuple(p) for p in sorted(internal, key=lambda x: (len(x), x))]
    heldout = C.HELDOUT
    seen = [f[0] for f in C.FRAMES if f[0] not in heldout]

    if a.selfcheck:
        ck = torch.load(str(T.INIT_CKPT), map_location='cpu', weights_only=False)
        m = T.make_model(); m.load_state_dict(ck['model_state_dict']); m.eval()
        probes = C.SUBJECTS[:5]
        pre = [tok.encode(C.frame_prefix(f, s)[0]) for f in heldout for s in probes]
        bq = batched_q(m, pre, seqs, internal, order)
        mx = 0.0
        for k, (f, s) in enumerate([(f, s) for f in heldout for s in probes]):
            uq = unbatched_q(m, pre[k], seqs, internal, order)
            mx = max(mx, float((bq[k] - uq).abs().max()))
        print('SELFCHECK: prompts=%d  max|batched-unbatched| = %.3e' % (len(pre), mx))
        print('  (float32 eps ~1.2e-7; agreement expected < 1e-6)')
        return

    arms = {}
    if a.arms:
        for part in a.arms.split(','):
            k, v = part.split('=')
            arms[k] = v
    elif a.ckpt:
        arms['X'] = a.ckpt

    res = {}
    for name, path in arms.items():
        ck = torch.load(path, map_location='cpu', weights_only=False)
        m = T.make_model(); m.load_state_dict(ck['model_state_dict']); m.eval()
        with torch.no_grad():
            hf, hv = eval_arm(m, tok, heldout, seqs, internal, order)
            sf, sv = eval_arm(m, tok, seen, seqs, internal, order)
        mean = sum(hv) / len(hv)
        lo, hi = cluster_ci(hv)
        res[name] = {'heldout_frames': hf, 'heldout_subject_mean': mean,
                     'heldout_CI95': [lo, hi], 'seen_frames_mean': sum(sf.values()) / len(sf),
                     'subject_vec': hv}
        print('%-4s heldout=%.3f  CI95=[%.3f,%.3f]  seen=%.3f  per-frame=%s'
              % (name, mean, lo, hi, res[name]['seen_frames_mean'],
                 {k: round(v, 3) for k, v in hf.items()}))

    if 'B' in res:
        for other in ['A', 'B0', 'C']:
            if other in res:
                d = [res['B']['subject_vec'][i] - res[other]['subject_vec'][i] for i in range(NSUB)]
                lo, hi = cluster_ci(d)
                print('paired B-%s = %+.3f  CI95=[%+.3f,%+.3f]  (excludes 0: %s)'
                      % (other, sum(d) / len(d), lo, hi, (lo > 0 or hi < 0)))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
