#!/usr/bin/env python3
"""
Relationship Form/Construction Breakdown (read-only; reuses the existing
relationship_exposure_analysis module).

Classifies each subject->action occurrence by grammatical construction:
  modal+base (can/could/will/would/may/might/must/shall/should + base),
  infinitive (to + base), progressive (-ing), past, present-3sg, present-base,
  distant co-occurrence (action present but outside the subject window).

Also counts the exact evaluation-style frame "<subject> can <base>."
Corpora: original, treatment_a, treatment_b. No model inference, no training.
"""
import sys, os, json, re, importlib.util
sys.path.insert(0, '/root/agent-sandbox/ARTN')

spec = importlib.util.spec_from_file_location('relmod', '/root/agent-sandbox/ARTN/scripts/relationship_exposure_analysis.py')
relmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relmod)

tok = relmod.BPETokenizer.load(relmod.TOK)

FORMS = {
    'fly':   {'base': 'fly',   '3sg': 'flies',  'past': 'flew', 'prog': 'flying'},
    'swim':  {'base': 'swim',  '3sg': 'swims',  'past': 'swam', 'prog': 'swimming'},
    'run':   {'base': 'run',   '3sg': 'runs',   'past': 'ran',  'prog': 'running'},
    'help':  {'base': 'help',  '3sg': 'helps',  'past': 'helped', 'prog': 'helping'},
    'teach': {'base': 'teach', '3sg': 'teaches', 'past': 'taught', 'prog': 'teaching'},
}
MODALS = ['can', 'could', 'will', 'would', 'may', 'might', 'must', 'shall', 'should']
REL = relmod.REL_ACTION
SUBJ = relmod.SUBJECT_FORMS
OUT = '/root/agent-sandbox/ARTN/outputs/artn_recursive_20261025_002725/relationship_form_breakdown.json'


def alt(ws):
    return '(?:' + '|'.join(re.escape(w) for w in ws) + ')'


VALID = ['modal+base', 'infinitive', 'progressive', 'past', 'present-3sg', 'present-base', 'distant-cooc', 'subject-only']


def make_classifier(subj, action):
    f = FORMS[action]
    subj_re = re.compile(r'\b' + alt(SUBJ[subj]) + r'\b', re.I)
    modal_re = re.compile(r'\b(?:' + '|'.join(MODALS) + r')\b', re.I)
    base_re = re.compile(r'\b' + re.escape(f['base']) + r'\b', re.I)
    to_re = re.compile(r'\bto\b\s+' + re.escape(f['base']) + r'\b', re.I)
    prog_re = re.compile(r'\b' + re.escape(f['prog']) + r'\b', re.I)
    past_re = re.compile(r'\b' + re.escape(f['past']) + r'\b', re.I)
    s3_re = re.compile(r'\b' + re.escape(f['3sg']) + r'\b', re.I)
    anyform_re = re.compile(r'\b(?:' + '|'.join(re.escape(v) for v in f.values()) + r')\b', re.I)
    frame_re = re.compile(r'\b' + alt(SUBJ[subj]) + r'\b\s+(?:can|could|would|will)\s+' + re.escape(f['base']) + r'\b', re.I)

    def run(sentence):
        labels = set()
        for m in subj_re.finditer(sentence):
            window = sentence[m.end():m.end() + 60]
            if modal_re.search(window) and base_re.search(window):
                labels.add('modal+base')
            elif to_re.search(window):
                labels.add('infinitive')
            elif prog_re.search(window):
                labels.add('progressive')
            elif past_re.search(window):
                labels.add('past')
            elif s3_re.search(window):
                labels.add('present-3sg')
            elif base_re.search(window):
                labels.add('present-base')
            elif anyform_re.search(sentence):
                labels.add('distant-cooc')
            else:
                labels.add('subject-only')
        if frame_re.search(sentence):
            labels.add('EXACT_FRAME')
        return labels
    return run


def main():
    out = {'corpora': {}, 'relationships': REL}
    for name, path in relmod.CORPORA.items():
        print(f'Decoding {name} ...', flush=True)
        texts = relmod.load_texts(path, tok)
        counts = {}
        for subj, action in REL.items():
            run = make_classifier(subj, action)
            c = {k: 0 for k in VALID}
            c['EXACT_FRAME'] = 0
            c['sentences_with_subject'] = 0
            subj_re = re.compile(r'\b' + alt(SUBJ[subj]) + r'\b', re.I)
            for text in texts:
                for s in relmod.split_sentences(text):
                    if subj_re.search(s):
                        c['sentences_with_subject'] += 1
                        for lb in run(s):
                            c[lb] += 1
            counts[subj] = c
        out['corpora'][name] = counts
        print(f'  {name} done', flush=True)

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'Saved: {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)\n')

    hdr = '  {:<10} {:>7} {:>7} {:>7} {:>6} {:>5} {:>6} {:>6} {:>8} {:>10}'
    for cname in ['original', 'treatment_a', 'treatment_b']:
        print(f'=== {cname} ===')
        print(hdr.format('relation', 'sents', 'modal', 'infin', 'prog', 'past', '3sg', 'base', 'distant', 'EXACT_FRAME'))
        for subj in REL:
            c = out['corpora'][cname][subj]
            print(hdr.format(f'{subj}->{REL[subj]}', c['sentences_with_subject'], c['modal+base'], c['infinitive'],
                             c['progressive'], c['past'], c['present-3sg'], c['present-base'], c['distant-cooc'], c['EXACT_FRAME']))
        print()

    print('=== MODAL+BASE (the evaluated construction) across corpora ===')
    print('  {:<12} {:>10} {:>10} {:>10}'.format('relation', 'original', 'treatA', 'treatB'))
    for subj in REL:
        o = out['corpora']['original'][subj]['modal+base']
        a = out['corpora']['treatment_a'][subj]['modal+base']
        b = out['corpora']['treatment_b'][subj]['modal+base']
        print('  {:<12} {:>10} {:>10} {:>10}'.format(f'{subj}->{REL[subj]}', o, a, b))

    print('\n=== PAST vs MODAL (original) ===')
    for subj in REL:
        c = out['corpora']['original'][subj]
        past, modal, prog = c['past'], c['modal+base'], c['progressive']
        ratio = 'inf' if modal == 0 else round(past / modal, 1)
        print(f"  {subj}->{REL[subj]:<7} past={past:<5} progressive={prog:<5} modal={modal:<4} past/modal={ratio}")


if __name__ == '__main__':
    main()
