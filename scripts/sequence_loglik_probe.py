#!/usr/bin/env python3
"""
Full-Sequence Conditional Log-Likelihood Probe (read-only, forward-only).

Frame: "the {X} can " + continuation.
For each subject, three continuations: appropriate / generic / inappropriate.
Measures total sequence log-likelihood and length-normalized log-likelihood.
Also counts phrase occurrences in the original training corpus.

Compares ORIGINAL step-10000 vs TREATMENT B step-15000.
No training, no free-form generation, no modifications.
Usage: cd /root/agent-sandbox/ARTN && python scripts/sequence_loglik_probe.py
"""
import sys, os, json
import torch

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = torch.device('cpu')
BASE = '/root/agent-sandbox/ARTN'
TOK = os.path.join(BASE, 'outputs/artn-50k-best-config/tokenizer.json')
TRAIN = os.path.join(BASE, 'outputs/tokenized_cache/train_tokenized.pt')
CKPTS = {
    'original_step10000': os.path.join(BASE, 'outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt'),
    'treatmentB_step15000': os.path.join(BASE, 'outputs/artn_treatment_b_20261105_154942/checkpoints/artn/step_15000.pt'),
}
OUT = os.path.join(BASE, 'outputs/artn_recursive_20261025_002725/sequence_loglik_probe.json')

FRAME = 'the {x} can '
CONDITIONS = {
    'bird':     ('fly in the sky',      'go and play',   'swim in the water'),
    'fish':     ('swim in the water',   'go and play',   'fly in the sky'),
    'dragon':   ('fly over the hills',  'go and play',   'swim in the water'),
    'dog':      ('run in the park',     'go and play',   'fly in the sky'),
    'king':     ('help his people',     'go and play',   'fly in the sky'),
    'teacher':  ('teach the children',  'go and play',   'fly in the sky'),
    'peacock':  ('fly in the sky',      'go and play',   'swim in the water'),
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


def seq_logprob(model, full_ids, prompt_len):
    inp = torch.tensor([full_ids], device=DEVICE)
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
        logp = torch.log_softmax(model.lm_head(model.ln_f(output))[0], dim=-1)
    total = 0.0
    per_tok = []
    for i in range(prompt_len, len(full_ids)):
        lp = float(logp[i - 1, full_ids[i]])
        per_tok.append(lp)
        total += lp
    return total, per_tok


def main():
    tok = BPETokenizer.load(TOK)
    print('Loading + decoding training cache for occurrence check...', flush=True)
    train_raw = torch.load(TRAIN, weights_only=False)
    rows = list(train_raw)[:49988]
    corpus = '\n'.join(tok.decode(r.tolist() if hasattr(r, 'tolist') else list(r)) for r in rows).lower()

    def count_phrase(phrase):
        return corpus.count(phrase.lower())

    all_out = {'frame': FRAME, 'conditions': CONDITIONS, 'checkpoints': {}}

    for ck_name, ck_path in CKPTS.items():
        model, ckpt = load_model(ck_path)
        print("\n" + "=" * 96)
        print(f"=== {ck_name} (step {ckpt['step']}) ===")
        ck_out = {'step': ckpt['step'], 'subjects': {}}
        for subj, (appr, gen, inappr) in CONDITIONS.items():
            prompt = FRAME.format(x=subj)
            prompt_ids = tok.encode(prompt)
            subj_out = {'prompt': prompt, 'prompt_tokens': len(prompt_ids),
                        'n_subj_tokens': len(tok.encode(subj)), 'conds': {}}
            for label, cont in [('appropriate', appr), ('generic', gen), ('inappropriate', inappr)]:
                full_ids = tok.encode(prompt + cont)
                assert full_ids[:len(prompt_ids)] == prompt_ids, f'prefix mismatch {subj}/{label}'
                c_ids = full_ids[len(prompt_ids):]
                total, per_tok = seq_logprob(model, full_ids, len(prompt_ids))
                subj_out['conds'][label] = {
                    'text': cont, 'n_tokens': len(c_ids), 'token_ids': c_ids,
                    'total_loglik': total, 'len_norm_loglik': total / max(len(c_ids), 1),
                    'per_token_loglik': per_tok,
                    'train_occurrences_cont': count_phrase(cont),
                    'train_occurrences_full': count_phrase(prompt + cont),
                }
            ck_out['subjects'][subj] = subj_out
            a = subj_out['conds']['appropriate']
            g = subj_out['conds']['generic']
            i = subj_out['conds']['inappropriate']
            print(f"\n  {subj}  (prompt='{prompt}', {len(prompt_ids)} tok; subj={subj_out['n_subj_tokens']} tok)")
            for lbl, c in [('appropriate', a), ('generic', g), ('inappropriate', i)]:
                line = "    {l:<14} '{t}'  n={n}  total={tt:+.2f}  norm={nn:+.3f}  occ_cont={oc}  occ_full={of}".format(
                    l=lbl, t=c['text'], n=c['n_tokens'], tt=c['total_loglik'],
                    nn=c['len_norm_loglik'], oc=c['train_occurrences_cont'], of=c['train_occurrences_full'])
                print(line)
            print("    -> appr>gen: {}   appr>inappr: {}   gen>inappr: {}".format(
                a['len_norm_loglik'] > g['len_norm_loglik'],
                a['len_norm_loglik'] > i['len_norm_loglik'],
                g['len_norm_loglik'] > i['len_norm_loglik']))
        all_out['checkpoints'][ck_name] = ck_out

    with open(OUT, 'w') as f:
        json.dump(all_out, f, indent=1)
    print(f"\nSaved: {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")

    print('\n\n=== SUMMARY: length-normalized log-likelihood (higher = more likely) ===')
    for ck_name in CKPTS:
        print(f"\n{ck_name}")
        print("  {:<9} {:>13} {:>10} {:>15}".format('subject', 'appropriate', 'generic', 'inappropriate'))
        for subj in CONDITIONS:
            s = all_out['checkpoints'][ck_name]['subjects'][subj]['conds']
            print("  {:<9} {:>13.3f} {:>10.3f} {:>15.3f}".format(
                subj, s['appropriate']['len_norm_loglik'], s['generic']['len_norm_loglik'],
                s['inappropriate']['len_norm_loglik']))


if __name__ == '__main__':
    main()
