#!/usr/bin/env python3
"""
Prompt-Identity Retention Probe (read-only).

Tests whether the ARTN shared recursive block preserves prompt-subject identity
at the point of generation, using token-length-matched prompts.

Design:
  Frame: "once upon a time there was a {X} who"
  Group A (1-token subject): subject at pos 7, shared suffix "who" at pos 8
  Group B (4-token subject): subject at pos 7-10, shared suffix "who" at pos 11

Measures (per recursive pass):
  - hidden state at subject span
  - hidden state at shared suffix position
  - cross-subject cosine similarity (subject pos + suffix pos)
  - within-prompt consecutive-pass cosine (reconciled with cross-prompt)
  - top-k next-token distribution + pairwise JS divergence (suffix position)

Artifacts: one small JSON. No generation, no training, no modifications.
Usage: cd /root/agent-sandbox/ARTN && python scripts/prompt_identity_probe.py
"""
import sys, os, json, math, itertools
import torch

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = torch.device('cpu')
BASE = '/root/agent-sandbox/ARTN'
CKPT = os.path.join(BASE, 'outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt')
TOK = os.path.join(BASE, 'outputs/artn-50k-best-config/tokenizer.json')
OUT = os.path.join(BASE, 'outputs/artn_recursive_20261025_002725/prompt_identity_probe.json')

FRAME = 'once upon a time there was a {x} who'

# Subjects. Group A = single BPE token; Group B = four BPE tokens (peacock-matched)
GROUP_A = {
    'dog': 'common', 'bird': 'common', 'fish': 'common', 'cat': 'common', 'horse': 'common',
    'dragon': 'mid', 'farmer': 'mid', 'queen': 'mid', 'truck': 'inanimate', 'spoon': 'inanimate',
    'king': 'target-fail', 'teacher': 'target-fail',
}
GROUP_B = {'peacock': 'target-fail', 'crocodile': 'rare-4tok', 'helicopter': 'rare-4tok', 'harmonica': 'rare-4tok'}


def js_divergence(p, q, eps=1e-12):
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a / b).log()).sum().item()
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def topk(logits, k=5):
    v, i = logits.topk(k)
    return [(int(t), float(s)) for t, s in zip(i.tolist(), v.tolist())]


def main():
    tok = BPETokenizer.load(TOK)
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = ARTNGPT(
        vocab_size=cfg['vocab_size'], d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], d_ff=cfg['d_ff'], max_seq_len=cfg['max_seq_len'],
        target_passes=cfg.get('target_passes', 4.0), ponder_loss_weight=cfg.get('ponder_loss_weight', 0.01),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Checkpoint step={ckpt['step']} val_loss={ckpt.get('val_loss'):.4f} max_passes={model.max_passes}")

    def run(subject):
        text = FRAME.format(x=subject)
        ids = tok.encode(text)
        subj_ids = tok.encode(subject)
        n = len(subj_ids)
        # subject span = [7, 7+n); shared suffix ("who") at 7+n
        subj_pos = list(range(7, 7 + n))
        suffix_pos = 7 + n
        inp = torch.tensor([ids], device=DEVICE)
        with torch.no_grad():
            x = model.token_embedding(inp) + model.position_embedding(torch.arange(inp.shape[1], device=DEVICE))
            per_pass = []
            for _ in range(model.max_passes):
                x = model.block(x)
                per_pass.append(x[0].clone())  # (T, d)
        subj_reps = [h[subj_pos].mean(0) for h in per_pass]           # per-pass subject span
        suffix_reps = [h[suffix_pos] for h in per_pass]               # per-pass suffix pos
        logit_dists = []
        for h in per_pass:
            logits = model.lm_head(model.ln_f(h))[suffix_pos]         # APPROXIMATE per-pass projection
            logit_dists.append(torch.softmax(logits, dim=-1))
        return {
            'n_subj_tokens': n, 'subj_pos': subj_pos, 'suffix_pos': suffix_pos,
            'subj_reps': [r.tolist() for r in subj_reps],
            'suffix_reps': [r.tolist() for r in suffix_reps],
            'logit_dists': [d.tolist() for d in logit_dists],
            'topk': [topk(model.lm_head(model.ln_f(h))[suffix_pos], 5) for h in per_pass],
        }

    results = {}
    for group, subjects in [('A_1token', GROUP_A), ('B_4token', GROUP_B)]:
        results[group] = {}
        for s, cat in subjects.items():
            results[group][s] = run(s)
            results[group][s]['category'] = cat
            print(f"  [{group}] {s}: subj_tokens={results[group][s]['n_subj_tokens']} suffix_pos={results[group][s]['suffix_pos']}")

    # ---- analysis ----
    analysis = {}
    for group, subjects in [('A_1token', GROUP_A), ('B_4token', GROUP_B)]:
        names = list(subjects.keys())
        per_pass = {
            'cross_subj_cos_at_subject': [],
            'cross_subj_cos_at_suffix': [],
            'within_prompt_consecutive_cos': {n: [] for n in names},
            'within_prompt_pass1_cos': {n: [] for n in names},
            'pairwise_js_at_suffix': [],
            'pairwise_topk_overlap_at_suffix': [],
        }
        npass = model.max_passes
        for p in range(npass):
            # cross-subject cosine at subject span
            sv = [torch.tensor(results[group][n]['subj_reps'][p]) for n in names]
            cs = []
            for a, b in itertools.combinations(range(len(names)), 2):
                cs.append(torch.nn.functional.cosine_similarity(sv[a], sv[b], dim=0).item())
            per_pass['cross_subj_cos_at_subject'].append(sum(cs) / len(cs))
            # cross-subject cosine at suffix pos
            fv = [torch.tensor(results[group][n]['suffix_reps'][p]) for n in names]
            cf = []
            for a, b in itertools.combinations(range(len(names)), 2):
                cf.append(torch.nn.functional.cosine_similarity(fv[a], fv[b], dim=0).item())
            per_pass['cross_subj_cos_at_suffix'].append(sum(cf) / len(cf))
            # within-prompt consecutive-pass cosine
            for n in names:
                cur = torch.tensor(results[group][n]['suffix_reps'][p])
                if p == 0:
                    per_pass['within_prompt_pass1_cos'][n].append(1.0)
                else:
                    prv = torch.tensor(results[group][n]['suffix_reps'][p - 1])
                    per_pass['within_prompt_consecutive_cos'][n].append(torch.nn.functional.cosine_similarity(cur, prv, dim=0).item())
                    p1 = torch.tensor(results[group][n]['suffix_reps'][0])
                    per_pass['within_prompt_pass1_cos'][n].append(torch.nn.functional.cosine_similarity(cur, p1, dim=0).item())
            # pairwise JS + top-k overlap at suffix distribution
            dists = [torch.tensor(results[group][n]['logit_dists'][p]) for n in names]
            js = []; ov = []
            for a, b in itertools.combinations(range(len(names)), 2):
                js.append(js_divergence(dists[a], dists[b]))
                ta = set(t for t, _ in results[group][names[a]]['topk'][p])
                tb = set(t for t, _ in results[group][names[b]]['topk'][p])
                ov.append(len(ta & tb) / 5.0)
            per_pass['pairwise_js_at_suffix'].append(sum(js) / len(js))
            per_pass['pairwise_topk_overlap_at_suffix'].append(sum(ov) / len(ov))
        analysis[group] = per_pass

    out = {
        'checkpoint': CKPT, 'step': ckpt['step'], 'frame': FRAME,
        'group_a_subjects': GROUP_A, 'group_b_subjects': GROUP_B,
        'results': results, 'analysis': analysis,
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved: {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")

    # ---- concise report ----
    print("\n=== CROSS-SUBJECT SIMILARITY BY PASS (mean pairwise cosine) ===")
    for group in ['A_1token', 'B_4token']:
        a = analysis[group]
        print(f"\n{group}:")
        print(f"  pass:            " + "  ".join(f"{i+1}" for i in range(len(a['cross_subj_cos_at_subject']))))
        print(f"  at SUBJECT span: " + "  ".join(f"{v:.3f}" for v in a['cross_subj_cos_at_subject']))
        print(f"  at SUFFIX pos:   " + "  ".join(f"{v:.3f}" for v in a['cross_subj_cos_at_suffix']))
        print(f"  next-tok JS:     " + "  ".join(f"{v:.4f}" for v in a['pairwise_js_at_suffix']))
        print(f"  top5 overlap:    " + "  ".join(f"{v:.2f}" for v in a['pairwise_topk_overlap_at_suffix']))

    print("\n=== WITHIN-PROMPT CONSECUTIVE-PASS COSINE (suffix pos) ===")
    for group in ['A_1token', 'B_4token']:
        print(f"\n{group}:")
        for n, vals in analysis[group]['within_prompt_consecutive_cos'].items():
            print(f"  {n:<10} consec={[round(v,3) for v in vals]}")

    print("\n=== TOP-5 NEXT TOKENS AT SUFFIX ('who' ...) PER SUBJECT (pass 6) ===")
    for group in ['A_1token', 'B_4token']:
        print(f"\n{group}:")
        for n in results[group]:
            tk = results[group][n]['topk'][-1]
            ws = [tok.decode([t]) for t, _ in tk]
            print(f"  {n:<10} -> {ws}")


if __name__ == '__main__':
    main()
