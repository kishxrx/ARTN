#!/usr/bin/env python3
"""
Full-Vocabulary Follow-up to the Subject-Specificity Probe (read-only, forward-only).

Re-measures the "the {X} can " cloze over the FULL vocabulary:
  - top-20 next-token predictions per subject
  - full-vocab rank + absolute probability of each candidate verb
  - total probability mass covered by the 16 candidates
  - full-distribution entropy vs candidate-restricted entropy
Compares ORIGINAL step-10000 vs TREATMENT B step-15000.

Artifacts: one small JSON. No training, generation, or modifications.
Usage: cd /root/agent-sandbox/ARTN && python scripts/full_vocab_followup.py
"""
import sys, os, json, math
import torch

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = torch.device('cpu')
BASE = '/root/agent-sandbox/ARTN'
TOK = os.path.join(BASE, 'outputs/artn-50k-best-config/tokenizer.json')
CKPTS = {
    'original_step10000': os.path.join(BASE, 'outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt'),
    'treatmentB_step15000': os.path.join(BASE, 'outputs/artn_treatment_b_20261105_154942/checkpoints/artn/step_15000.pt'),
}
OUT = os.path.join(BASE, 'outputs/artn_recursive_20261025_002725/full_vocab_followup.json')

FRAME = 'the {x} can '
CANDIDATES = ['fly','swim','run','sing','read','teach','jump','eat','walk','talk','sleep','play','cook','build','draw','help']
GROUP_A = ['dog','bird','fish','cat','horse','dragon','farmer','queen','truck','spoon','king','teacher']
GROUP_B = ['peacock','crocodile','helicopter','harmonica']


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    m = ARTNGPT(vocab_size=cfg['vocab_size'], d_model=cfg['d_model'], n_heads=cfg['n_heads'],
                n_layers=cfg['n_layers'], d_ff=cfg['d_ff'], max_seq_len=cfg['max_seq_len'],
                target_passes=cfg.get('target_passes', 4.0), ponder_loss_weight=cfg.get('ponder_loss_weight', 0.01))
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m, ckpt


def next_token_logprobs(model, ids):
    inp = torch.tensor([ids], device=DEVICE)
    with torch.no_grad():
        T = inp.shape[1]
        x = model.token_embedding(inp) + model.position_embedding(torch.arange(T, device=DEVICE))
        remaining = torch.ones(1, T, 1, device=DEVICE)
        output = torch.zeros(1, T, model.d_model, device=DEVICE)
        for p_idx in range(model.max_passes):
            x = model.block(x)
            h = torch.sigmoid(model.halting_gate(x))
            p = h * remaining
            if p_idx == model.max_passes - 1:
                p = remaining
            output = output + x * p
            remaining = remaining - p
        output = output + x * remaining
        logits = model.lm_head(model.ln_f(output))[:, -1, :]
        return torch.log_softmax(logits, dim=-1)[0]


def main():
    tok = BPETokenizer.load(TOK)
    cand_ids = {w: tok.encode(w)[0] for w in CANDIDATES}

    all_out = {'frame': FRAME, 'candidates': CANDIDATES, 'group_a': GROUP_A, 'group_b': GROUP_B, 'checkpoints': {}}

    for ck_name, ck_path in CKPTS.items():
        model, ckpt = load_model(ck_path)
        ck_out = {'step': ckpt['step'], 'group_a': {}, 'group_b': {}}
        print(f"\n{'='*100}\n=== {ck_name} (step {ckpt['step']}) ===")

        for group, subjects in [('group_a', GROUP_A), ('group_b', GROUP_B)]:
            print(f"\n--- {group} ---")
            for s in subjects:
                ids = tok.encode(FRAME.format(x=s))
                logp = next_token_logprobs(model, ids)
                prob = logp.exp()
                # full-vocab top-20
                tv, ti = prob.topk(20)
                top20 = [{'tok': int(t), 'word': tok.decode([int(t)]), 'p': float(p)} for t, p in zip(ti.tolist(), tv.tolist())]
                # candidate stats in full vocab
                sorted_idx = torch.argsort(prob, descending=True)
                rank_of = {}
                pos = 0
                for idx in sorted_idx.tolist():
                    rank_of[idx] = pos + 1
                    pos += 1
                    if pos >= 4000:
                        break
                cand_stats = {w: {'p': float(prob[cand_ids[w]]), 'vocab_rank': rank_of[cand_ids[w]]} for w in CANDIDATES}
                cand_mass = sum(float(prob[cand_ids[w]]) for w in CANDIDATES)
                # entropies
                full_ent = float(-(prob * logp).sum())  # over full vocab
                cand_probs = torch.tensor([float(prob[cand_ids[w]]) for w in CANDIDATES])
                cn = cand_probs / cand_probs.sum()
                cand_ent = float(-(cn * cn.log()).sum())
                ck_out[group][s] = {
                    'n_frame_tokens': len(ids), 'top20': top20, 'candidates': cand_stats,
                    'candidate_mass': cand_mass, 'full_vocab_entropy': full_ent,
                    'entropy_over_16_candidates': cand_ent,
                    'argmax_full': top20[0]['word'], 'argmax_full_p': top20[0]['p'],
                }
                top3 = ', '.join(f"{t['word']}:{t['p']:.3f}" for t in top20[:5])
                print(f"  {s:<11} mass16={cand_mass:.4f}  H_full={full_ent:.2f}  H_cand={cand_ent:.2f}  top5[full]={top3}")

        all_out['checkpoints'][ck_name] = ck_out

    with open(OUT, 'w') as f:
        json.dump(all_out, f, indent=1)
    print(f"\nSaved: {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")

    # ---- candidate full-vocab ranks & probabilities -------
    print("\n\n=== CANDIDATE VERB: full-vocab rank / absolute probability ===")
    for ck_name in CKPTS:
        ck = all_out['checkpoints'][ck_name]
        print(f"\n{ck_name}")
        for group in ['group_a', 'group_b']:
            print(f"  {group}:  {'subject':<11} " + ' '.join(f"{w:>9}" for w in ['fly','swim','run','teach','play','help']))
            for s in (GROUP_A if group == 'group_a' else GROUP_B):
                c = ck[group][s]['candidates']
                cells = ' '.join(f"{c[w]['p']:.2e}({c[w]['vocab_rank']:>4})" for w in ['fly','swim','run','teach','play','help'])
                print(f"  {'':<9} {s:<11} {cells}")

    # ---- reference: what IS the top-1 full-vocab token per subject ----
    print("\n=== TOP-1 FULL-VOCAB TOKEN PER SUBJECT ===")
    for ck_name in CKPTS:
        print(f"\n{ck_name}")
        for group in ['group_a', 'group_b']:
            for s in (GROUP_A if group == 'group_a' else GROUP_B):
                d = all_out['checkpoints'][ck_name][group][s]
                t = d['top20'][0]
                print(f"  {s:<11} -> {t['word']!r} p={t['p']:.4f}   (mass over 16 cands={d['candidate_mass']:.4f})")


if __name__ == '__main__':
    main()
