#!/usr/bin/env python3
"""Construction-Transfer Experiment trainer (arms A, C, B, B0).

Read-only w.r.t. all existing artifacts. Does not modify the ARTN blueprint.

  A  = L_LM                                     + constrained canonical exposure
  C  = L_LM                                     + diverse canonical exposure
  B  = L_LM + beta*(L_anchor + lambda*L_match)  + constrained exposure
  B0 = L_LM + beta*L_anchor                     + constrained exposure

Approved: 2000 steps, batch 8 x 256, AdamW 3e-4 -> 3e-5 cosine, warmup 100,
wd 0.01, betas (0.9,0.95), beta=0.5, lambda=1.0, eps=0.02.
Padding: masked per-token CE summed over non-PAD targets, divided by B*(T-1).
"""
import argparse, json, math, os, sys, time
from pathlib import Path
from collections import defaultdict
import random
import torch
import torch.nn.functional as F

BASE = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE))

from experiments.artn_model import ARTNGPT
from experiments import ctransfer_common as C
from src.tokenizer.bpe_tokenizer import BPETokenizer

OUT_ROOT = BASE / 'outputs' / 'ctransfer'
INIT_CKPT = OUT_ROOT / 'init' / 'ctransfer_init.pt'
VERB_NAMES = [v[0] for v in C.VERBS]
VERB_INDEX = {v: i for i, v in enumerate(VERB_NAMES)}

CFG = dict(d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_seq_len=256, vocab_size=4000,
           target_passes=4.0, ponder_loss_weight=0.01,
           steps=2000, batch=8, lr=3e-4, min_lr=3e-5, warmup=100, wd=0.01,
           beta1=0.9, beta2=0.95, beta=0.5, lam=1.0, eps=0.02, seed=42, log_every=25)

ARMS = {'A': 'constrained', 'B': 'constrained', 'B0': 'constrained', 'C': 'diverse'}


def make_model():
    m = ARTNGPT(vocab_size=CFG['vocab_size'], d_model=CFG['d_model'], n_heads=CFG['n_heads'],
                n_layers=CFG['n_layers'], d_ff=CFG['d_ff'], max_seq_len=CFG['max_seq_len'],
                target_passes=CFG['target_passes'], ponder_loss_weight=CFG['ponder_loss_weight'])
    return m


def ensure_init():
    """Create the single designated initialization checkpoint once; all arms load it."""
    if INIT_CKPT.exists():
        return INIT_CKPT
    INIT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(CFG['seed'])
    random.seed(CFG['seed'])
    m = make_model()
    torch.save({'step': 0, 'model_state_dict': m.state_dict(), 'config': CFG}, INIT_CKPT)
    print('created init checkpoint:', INIT_CKPT, flush=True)
    return INIT_CKPT


def pad_batch(records, T=256):
    ids = []
    for r in records:
        v = list(r['ids'][:T])
        v = v + [0] * (T - len(v))
        ids.append(v)
    return torch.tensor(ids, dtype=torch.long)


def build_targets(input_ids):
    """targets[i] = next token at position i; -1 at PAD and at the final position."""
    tgt = torch.full_like(input_ids, -1)
    tgt[:, :-1] = input_ids[:, 1:]
    tgt[tgt == 0] = -1
    return tgt


def lemma_q(probs_by_seq, seq_to_verb):
    q = torch.zeros(len(VERB_NAMES) + 1, dtype=torch.float32)
    for s, p in probs_by_seq.items():
        i = VERB_INDEX[seq_to_verb[s]]
        q[i] = q[i] + p
    q[-1] = (1.0 - q[:-1].sum()).clamp_min(0.0)
    return q


def aux_probs(model, records, internal_paths):
    """One batched forward per distinct sequence length; returns per-record
    {path: probs} for non-empty internal paths (root comes from the main forward)."""
    items = []
    for ri, r in enumerate(records):
        L = r['slot_pos'] + 1
        pref = list(r['ids'][:L])
        for pi in internal_paths:
            items.append((len(pref) + len(pi), ri, tuple(pi), pref + list(pi)))
    groups = defaultdict(list)
    for ln, ri, pi, ids in items:
        groups[ln].append((ri, pi, ids))
    out = [dict() for _ in records]
    n_pos = 0
    for ln, grp in sorted(groups.items()):
        b = torch.tensor([g[2] for g in grp], dtype=torch.long)
        n_pos += b.numel()
        logits = model(b)['logits'][:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        for k, (ri, pi, _ids) in enumerate(grp):
            out[ri][pi] = probs[k]
    return out, n_pos


SEQPATHS = None


def _seq_chains(seqs, internal_paths, class_of):
    global SEQPATHS
    if SEQPATHS is None:
        paths = [()] + [tuple(p) for p in sorted(internal_paths, key=lambda x: (len(x), x))]
        row_of = {p: i for i, p in enumerate(paths)}
        maxlen = max(len(s) for s in seqs)
        rows = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        toks = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        mask = torch.zeros((len(seqs), maxlen), dtype=torch.float32)
        for i, s in enumerate(seqs):
            path = ()
            for j, t in enumerate(s):
                rows[i, j] = row_of[path]
                toks[i, j] = t
                mask[i, j] = 1.0
                path = path + (t,)
        seq_verb = torch.tensor([VERB_INDEX[class_of[s]] for s in seqs], dtype=torch.long)
        SEQPATHS = (paths, rows, toks, mask, seq_verb)
    return SEQPATHS


def rel_losses(model, records, seqs, class_of, internal_paths, root_probs, aux):
    """Returns (L_anchor, L_match)."""
    eps = CFG['eps']
    paths, rows, toks, mask, seq_verb = _seq_chains(seqs, internal_paths, class_of)
    K = len(VERB_NAMES)
    qs = []
    anchors = []
    for ri, r in enumerate(records):
        P = torch.empty((len(paths), root_probs[ri].shape[0]))
        P[0] = root_probs[ri]
        for i, p in enumerate(paths):
            if i > 0:
                P[i] = aux[ri][p]
        g = P[rows, toks]
        leaf = g.pow(mask).prod(dim=1)
        q = torch.zeros(K + 1)
        q.index_add_(0, seq_verb, leaf)
        q[-1] = (1.0 - q[:-1].sum()).clamp_min(0.0)
        q = q / q.sum().clamp_min(1e-9)
        qs.append(q)
        t = torch.full((K + 1,), eps / K)
        t[VERB_INDEX[r['verb']]] = 1.0 - eps
        anchors.append((q * (q / t).log()).sum())
    L_anchor = torch.stack(anchors).mean()
    # L_match: JSD between constructions, canonical records only
    L_match = torch.zeros((), dtype=torch.float32)
    by_frame = defaultdict(list)
    for ri, r in enumerate(records):
        if r['canonical']:
            by_frame[r['frame']].append(qs[ri])
    frames = sorted(by_frame.keys())
    if len(frames) >= 2:
        jsds = []
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                q1 = torch.stack(by_frame[frames[i]]).mean(0)
                q2 = torch.stack(by_frame[frames[j]]).mean(0)
                m = 0.5 * (q1 + q2)
                jsd = 0.5 * (q1 * (q1 / m).log()).sum() + 0.5 * (q2 * (q2 / m).log()).sum()
                jsds.append(jsd)
        L_match = torch.stack(jsds).mean()
    return L_anchor, L_match


def build_schedule(corpus_type, steps, seed=42):
    alloc = C.allocation(corpus_type)
    frame_pool = [f for f, (n, _d) in alloc.items() if n > 0]
    dist_frames = [f for f, (_n, d) in alloc.items() if d > 0]
    sched = []
    pair_i = 0
    dist_i = 0
    for step in range(steps):
        si = step % len(C.SUBJECTS)
        nf = len(frame_pool)
        fa = frame_pool[pair_i % nf]
        fb = frame_pool[(pair_i + 1) % nf]
        if fb == fa:
            fb = frame_pool[(pair_i + 2) % nf]
        pair_i = (pair_i + 2) % max(nf, 3)
        d1 = dist_frames[dist_i % len(dist_frames)]
        d2 = dist_frames[(dist_i + 3) % len(dist_frames)]
        d3 = dist_frames[(dist_i + 5) % len(dist_frames)]
        d4 = dist_frames[(dist_i + 7) % len(dist_frames)]
        dist_i = (dist_i + 1) % len(dist_frames)
        sched.append({'subject': si,
                      'canonical': [(fa, step), (fa, step * 3 + 1), (fb, step * 5 + 2), (fb, step * 7 + 3)],
                      'distractor': [d1, d2, d3, d4]})
    return sched


def index_corpus(corpus_type, tok, seed=42):
    recs = C.build_corpus(corpus_type, tok, seed)
    idx = defaultdict(list)
    for r in recs:
        idx[(r['subject_idx'], r['frame'], r['canonical'])].append(r)
    return recs, idx


def make_batch(schedule_step, idx, tok):
    sid = schedule_step['subject']
    out = []
    ck = 0
    for (frame, k) in schedule_step['canonical']:
        pool = idx[(sid, frame, True)]
        out.append(pool[(k + ck) % len(pool)])
        ck += 1
    for j, frame in enumerate(schedule_step['distractor']):
        pool = idx[(sid, frame, False)]
        out.append(pool[(j * 3 + 1) % len(pool)] if pool else idx[(sid, frame, True)][0])
    assert len(out) == 8, len(out)
    return out


def lr_at(step):
    if step < CFG['warmup']:
        return CFG['lr'] * (step + 1) / CFG['warmup']
    prog = (step - CFG['warmup']) / (CFG['steps'] - CFG['warmup'])
    return CFG['min_lr'] + 0.5 * (CFG['lr'] - CFG['min_lr']) * (1 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['A', 'C', 'B', 'B0'])
    ap.add_argument('--steps', type=int, default=CFG['steps'])
    ap.add_argument('--log-every', type=int, default=CFG['log_every'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--make-init', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(CFG['seed'])
    random.seed(CFG['seed'])
    tok = BPETokenizer.load(str(C.TOKENIZER_PATH))
    C.measure_frames(tok)
    C.check_marginals()
    seqs, nodes, internal, class_of = C.build_trie(tok)
    print('trie: seqs=%d nodes=%d internal=%d' % (len(seqs), len(nodes), len(internal)), flush=True)
    assert len(seqs) == 79 and len(internal) == 45, 'trie does not match the verified structure'

    if args.make_init:
        ensure_init()
        return

    init_path = ensure_init()
    corpus_type = ARMS[args.arm]
    recs, idx = index_corpus(corpus_type, tok, CFG['seed'])
    sched = build_schedule(corpus_type, args.steps, CFG['seed'])

    out_dir = Path(args.out) if args.out else (OUT_ROOT / ('arm_' + args.arm))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / 'train_log.jsonl', 'a')

    ck = torch.load(init_path, map_location='cpu', weights_only=False)
    model = make_model()
    model.load_state_dict(ck['model_state_dict'])
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], betas=(CFG['beta1'], CFG['beta2']),
                            weight_decay=CFG['wd'])

    print('arm=%s corpus=%s steps=%d out=%s' % (args.arm, corpus_type, args.steps, out_dir), flush=True)
    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g['lr'] = lr
        batch_recs = make_batch(sched[step], idx, tok)
        ids = pad_batch(batch_recs)
        tgt = build_targets(ids)
        out = model(ids)
        logits = out['logits']
        T = ids.shape[1]
        ce_sum = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                 tgt[:, :-1].reshape(-1), ignore_index=-1, reduction='sum')
        L_LM = ce_sum / (ids.shape[0] * (T - 1))
        rem = out['halting_info']['remaining_history']
        exp_passes = 1.0 + sum(r.mean() for r in rem)
        L_ponder = CFG['ponder_loss_weight'] * (exp_passes - CFG['target_passes']) ** 2
        loss = L_LM + L_ponder
        L_anchor = torch.zeros(()); L_match = torch.zeros(()); n_aux = 0
        if args.arm in ('B', 'B0'):
            root_probs = [torch.softmax(logits[i, batch_recs[i]['slot_pos']], dim=-1)
                          for i in range(len(batch_recs))]
            aux, n_aux = aux_probs(model, batch_recs, internal)
            L_anchor, L_match = rel_losses(model, batch_recs, seqs, class_of, internal, root_probs, aux)
            loss = loss + CFG['beta'] * (L_anchor + (CFG['lam'] * L_match if args.arm == 'B' else 0.0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            rec = dict(step=step, lr=lr, loss=float(loss), L_LM=float(L_LM), L_ponder=float(L_ponder),
                       L_anchor=float(L_anchor), L_match=float(L_match), aux_positions=int(n_aux),
                       elapsed=round(time.time() - t0, 1))
            log_f.write(json.dumps(rec) + '\n'); log_f.flush()
            print('step %4d  loss=%.4f  LM=%.4f  anchor=%.4f  match=%.4f  aux_pos=%d  %.0fs'
                  % (step, rec['loss'], rec['L_LM'], rec['L_anchor'], rec['L_match'], n_aux, rec['elapsed']), flush=True)

    ckpt = out_dir / 'checkpoints' / 'artn' / ('step_%d.pt' % args.steps)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'step': args.steps, 'arm': args.arm, 'corpus_type': corpus_type,
                'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(),
                'config': CFG, 'init_checkpoint': str(init_path)}, ckpt)
    print('saved checkpoint:', ckpt, flush=True)
    log_f.close()


if __name__ == '__main__':
    main()
