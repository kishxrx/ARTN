#!/usr/bin/env python3
"""
Grammatical-Context Transfer Probe (read-only, forward-only).

For subject->action relationships, compares the teacher-forced likelihood of the
TARGET action verb form vs a matched CONTROL action verb form in the SAME
construction, across familiar (past / progressive / present) and unseen
(modal / infinitive / future) constructions.

Delta = logP(target_form | prefix) - logP(control_form | prefix)
Reuses load_model / seq_logprob from sequence_loglik_probe.py.
Compares ORIGINAL step-10000 vs TREATMENT B step-15000.
No training, free-form generation or architecture changes.
"""
import sys, os, json, importlib.util
sys.path.insert(0, '/root/agent-sandbox/ARTN')

spec = importlib.util.spec_from_file_location('slp', '/root/agent-sandbox/ARTN/scripts/sequence_loglik_probe.py')
slp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slp)
tok = slp.BPETokenizer.load(slp.TOK)

OUT = '/root/agent-sandbox/ARTN/outputs/artn_recursive_20261025_002725/grammatical_context_transfer_probe.json'

FORMS = {
    'fly':   {'base': 'fly',   '3sg': 'flies',  'past': 'flew',     'prog': 'flying'},
    'swim':  {'base': 'swim',  '3sg': 'swims',  'past': 'swam',     'prog': 'swimming'},
    'run':   {'base': 'run',   '3sg': 'runs',   'past': 'ran',      'prog': 'running'},
    'help':  {'base': 'help',  '3sg': 'helps',  'past': 'helped',   'prog': 'helping'},
    'teach': {'base': 'teach', '3sg': 'teaches','past': 'taught',   'prog': 'teaching'},
    'sing':  {'base': 'sing',  '3sg': 'sings',  'past': 'sang',     'prog': 'singing'},
    'rule':  {'base': 'rule',  '3sg': 'rules',  'past': 'ruled',    'prog': 'ruling'},
    'play':  {'base': 'play',  '3sg': 'plays',  'past': 'played',   'prog': 'playing'},
}

# (label, prefix template, familiarity, form key)
CONSTRUCTIONS = [
    ('past',          'the {x} ',            'familiar',  'past'),
    ('progressive',   'the {x} was ',        'familiar',  'prog'),
    ('present-3sg',   'the {x} ',            'familiar',  '3sg'),
    ('infinitive',    'the {x} wanted to ',  'moderate',  'base'),
    ('modal-can',     'the {x} can ',        'unseen',    'base'),
    ('modal-could',   'the {x} could ',      'unseen',    'base'),
    ('future-will',   'the {x} will ',       'unseen',    'base'),
]

# subject, target action, control action (control 1: 'fly' absurd for some; control 2: 'play', plausible)
PAIRS = [
    ('bird',    'fly',   'swim', 'sing'),
    ('fish',    'swim',  'fly',  'play'),
    ('dog',     'run',   'fly',  'play'),
    ('teacher', 'teach', 'fly',  'help'),
    ('king',    'help',  'fly',  'rule'),
    ('peacock', 'fly',   'swim', 'play'),
]


def verb_logprob(model, prefix, verb):
    ids = tok.encode(prefix + verb)
    p_ids = tok.encode(prefix)
    c_ids = ids[len(p_ids):]
    total, per = slp.seq_logprob(model, ids, len(p_ids))
    return total, c_ids


def main():
    out = {'constructions': [c[0] for c in CONSTRUCTIONS], 'pairs': PAIRS, 'checkpoints': {}}
    for ck_name, ck_path in slp.CKPTS.items():
        model, ckpt = slp.load_model(ck_path)
        print('\n' + '=' * 92)
        print(f'=== {ck_name} (step {ckpt["step"]}) ===')
        ck_out = {'step': ckpt['step'], 'subjects': {}}
        for subj, tgt, ctl1, ctl2 in PAIRS:
            sub_out = {}
            for label, tmpl, fam, formkey in CONSTRUCTIONS:
                prefix = tmpl.format(x=subj)
                t_lp, t_ids = verb_logprob(model, prefix, FORMS[tgt][formkey])
                c1_lp, c1_ids = verb_logprob(model, prefix, FORMS[ctl1][formkey])
                c2_lp, c2_ids = verb_logprob(model, prefix, FORMS[ctl2][formkey])
                t_norm = t_lp / max(len(t_ids), 1)
                sub_out[label] = {
                    'familiarity': fam, 'prefix': prefix,
                    'target': FORMS[tgt][formkey], 'target_loglik': t_lp, 'target_norm': t_norm,
                    'control1': FORMS[ctl1][formkey], 'tgt_minus_c1_total': t_lp - c1_lp,
                    'tgt_minus_c1_norm': t_norm - c1_lp / max(len(c1_ids), 1),
                    'control2': FORMS[ctl2][formkey], 'tgt_minus_c2_total': t_lp - c2_lp,
                    'tgt_minus_c2_norm': t_norm - c2_lp / max(len(c2_ids), 1),
                }
            ck_out['subjects'][subj] = sub_out
        out['checkpoints'][ck_name] = ck_out

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nSaved: {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)')

    for ck_name in slp.CKPTS:
        print('\n' + '=' * 92)
        print(f'{ck_name}: Delta (logP target - logP control) by construction')
        print('  {:<9} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}'.format(
            'subject', 'past', 'prog', '3sg', 'infin', 'can', 'could', 'will'))
        for subj, tgt, ctl1, ctl2 in PAIRS:
            cells = []
            for label, _, _, _ in CONSTRUCTIONS:
                d = out['checkpoints'][ck_name]['subjects'][subj][label]
                cells.append('{:>11.2f}'.format(d['tgt_minus_c1_total']))
            print('  {:<9} {}'.format(f'{subj}->{tgt}', ' '.join(cells)))

        print('  (normalized per-token delta, control 2 = plausible alternative)')
        print('  {:<9} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}'.format(
            'subject', 'past', 'prog', '3sg', 'infin', 'can', 'could', 'will'))
        for subj, tgt, ctl1, ctl2 in PAIRS:
            cells = []
            for label, _, _, _ in CONSTRUCTIONS:
                d = out['checkpoints'][ck_name]['subjects'][subj][label]
                cells.append('{:>11.2f}'.format(d['tgt_minus_c2_norm']))
            print('  {:<9} {}'.format(f'{subj}->{tgt}', ' '.join(cells)))

    print('\n=== SUMMARY: mean delta by familiarity group ===')
    for ck_name in slp.CKPTS:
        print(f'\n{ck_name}')
        print('  {:<11} {:>12} {:>12} {:>12}'.format('relation', 'familiar', 'infinitive', 'unseen-modal'))
        for subj, tgt, ctl1, ctl2 in PAIRS:
            g = lambda name: [out['checkpoints'][ck_name]['subjects'][subj][l][name] for l, _, f, _ in CONSTRUCTIONS]
            fam = [out['checkpoints'][ck_name]['subjects'][subj][l]['tgt_minus_c2_norm'] for l, _, f, _ in CONSTRUCTIONS if f == 'familiar']
            inf = [out['checkpoints'][ck_name]['subjects'][subj][l]['tgt_minus_c2_norm'] for l, _, f, _ in CONSTRUCTIONS if f == 'moderate']
            uns = [out['checkpoints'][ck_name]['subjects'][subj][l]['tgt_minus_c2_norm'] for l, _, f, _ in CONSTRUCTIONS if f == 'unseen']
            avg = lambda v: sum(v) / len(v) if v else 0.0
            print('  {:<11} {:>12.2f} {:>12.2f} {:>12.2f}'.format(f'{subj}->{tgt}', avg(fam), avg(inf), avg(uns)))


if __name__ == '__main__':
    main()
