#!/usr/bin/env python3
"""
Subject-Specificity Cloze Probe (read-only, forward-only).

Question: does subject identity actually change the model's predicted next-token
distribution when a SUBJECT-SPECIFIC continuation is required?

Frame: "the {X} can "  (last position expects an ability verb)
Candidates: single-token verbs (verified).

Measures:
  - log P(candidate | subject frame) matrix
  - argmax candidate per subject
  - cross-subject JS divergence of the normalized candidate distribution
  - subject-specific contrasts (bird->fly vs fish->fly, etc.)
Compares ORIGINAL step-10000 vs TREATMENT B step-15000.

Artifacts: one small JSON. No training, no generation, no modifications.
Usage: cd /root/agent-sandbox/ARTN && python scripts/subject_specificity_probe.py
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
OUT = os.path.join(BASE, 'outputs/artn_recursive_20261025_002725/subject_specificity_probe.json')

FRAME = 'the {x} can '
CANDIDATES = ['fly','swim','run','sing','read','teach','jump','eat','walk','talk','sleep','play','cook','build','draw','help']

GROUP_A = ['dog','bird','fish','cat','horse','dragon','farmer','queen','truck','spoon','king','teacher']
GROUP_B = ['peacock','crocodile','helicopter','harmonica']

# Hand-designated appropriate verb(s) for utility check (labeling is explicit/human)
APPROPRIATE = {
    'bird':'fly', 'fish':'swim', 'dog':'run', 'cat':'play', 'horse':'run',
    'dragon':'fly', 'farmer':'build', 'queen':'read', 'truck':'run', 'spoon':'eat',
    'king':'read', 'teacher':'teach', 'peacock':'fly', 'crocodile':'swim',
    'helicopter':'fly', 'harmonica':'sing',
}
INAPPROPRIATE = {
    'bird':'swim', 'fish':'fly', 'dog':'fly', 'cat':'fly', 'horse':'fly',
    'dragon':'swim', 'farmer':'fly', 'queen':'fly', 'truck':'eat', 'spoon':'fly',
    'king':'fly', 'teacher':'fly', 'peacock':'swim', 'crocodile':'fly',
    'helicopter':'read', 'harmonica':'fly',
}


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


def js(p, q, eps=1e-12):
    p = [max(x, eps) for x in p]; q = [max(x, eps) for x in q]
    m = [0.5*(a+b) for a, b in zip(p, q)]
    kl = lambda a, b: sum(x*math.log(x/y) for x, y in zip(a, b))
    return 0.5*kl(p, m) + 0.5*kl(q, m)


def main():
    tok = BPETokenizer.load(TOK)
    cand_ids = {w: tok.encode(w)[0] for w in CANDIDATES}
    assert all(len(tok.encode(w)) == 1 for w in CANDIDATES), 'candidate not single-token'

    all_out = {'frame': FRAME, 'candidates': CANDIDATES, 'appropriate': APPROPRIATE,
               'inappropriate': INAPPROPRIATE, 'group_a': GROUP_A, 'group_b': GROUP_B, 'checkpoints': {}}

    for ck_name, ck_path in CKPTS.items():
        model, ckpt = load_model(ck_path)
        print(f"\n=== {ck_name} (step {ckpt['step']}, val {ckpt.get('val_loss'):.4f}) ===")
        ck_out = {'step': ckpt['step'], 'group_a': {}, 'group_b': {}}

        for group, subjects in [('group_a', GROUP_A), ('group_b', GROUP_B)]:
            logp_matrix = {}
            for s in subjects:
                ids = tok.encode(FRAME.format(x=s))
                logp = next_token_logprobs(model, ids)
                row = {w: float(logp[cand_ids[w]]) for w in CANDIDATES}
                logp_matrix[s] = row
                ck_out[group][s] = {'n_frame_tokens': len(ids), 'logp': row,
                                    'argmax': max(row, key=row.get)}
            # cross-subject JS of normalized candidate dist
            probs = {s: [math.exp(logp_matrix[s][w]) for w in CANDIDATES] for s in subjects}
            for s in subjects:
                tot = sum(probs[s]); probs[s] = [v/tot for v in probs[s]]
            js_vals = []
            for i in range(len(subjects)):
                for j in range(i+1, len(subjects)):
                    js_vals.append(js(probs[subjects[i]], probs[subjects[j]]))
            ck_out[group]['_mean_cross_subject_js'] = sum(js_vals)/len(js_vals)
            ck_out[group]['_max_cross_subject_js'] = max(js_vals)
            ck_out[group]['_min_cross_subject_js'] = min(js_vals)

        all_out['checkpoints'][ck_name] = ck_out

        # ---- report per checkpoint ----
        for group in ['group_a', 'group_b']:
            g = ck_out[group]
            print(f"\n  {group}: mean cross-subject JS of candidate dist = {g['_mean_cross_subject_js']:.5f}  (min {g['_min_cross_subject_js']:.5f}, max {g['_max_cross_subject_js']:.5f})")
            print(f"  {'subject':<11} {'argmax':<8} {'logP(appr)':<11} {'logP(inappr)':<12} {'delta':<8} {'appr_rank'}")
            for s in (GROUP_A if group == 'group_a' else GROUP_B):
                row = ck_out[group][s]['logp']
                ap = APPROPRIATE[s]; ip = INAPPROPRIATE[s]
                delta = row[ap] - row[ip]
                order = sorted(CANDIDATES, key=lambda w: -row[w])
                rank = order.index(ap) + 1
                print(f"  {s:<11} {ck_out[group][s]['argmax']:<8} {row[ap]:<11.3f} {row[ip]:<12.3f} {delta:<+8.3f} {rank}/16")

    with open(OUT, 'w') as f:
        json.dump(all_out, f, indent=1)
    print(f"\nSaved: {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")

    # ---- key contrasts ----
    print("\n=== SUBJECT-SPECIFIC CONTRASTS (logP(fly) and logP(swim) by subject) ===")
    for ck_name in CKPTS:
        print(f"\n{ck_name}:")
        for group, subjects in [('group_a', GROUP_A), ('group_b', GROUP_B)]:
            print(f"  {group}: {'subj':<11} {'fly':<9} {'swim':<9} {'run':<9} {'teach':<9} {'sing'}")
            for s in subjects:
                r = all_out['checkpoints'][ck_name][group][s]['logp']
                print(f"           {s:<11} {r['fly']:<9.3f} {r['swim']:<9.3f} {r['run']:<9.3f} {r['teach']:<9.3f} {r['sing']:.3f}")


if __name__ == '__main__':
    main()
