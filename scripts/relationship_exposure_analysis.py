#!/usr/bin/env python3
"""
Relationship-Exposure Analysis (read-only, existing data only).

For target subject->action relationships, measures exposure in the ORIGINAL,
TREATMENT A and TREATMENT B training corpora (decoded from existing tokenized caches):
  - same-story and same-sentence co-occurrence
  - direct agentive evidence (subject is agent of the action) vs mere co-occurrence
  - contextual patterns (auxiliary/modal forms used)
  - positive vs negative/contradictory contexts
  - representative examples

No model calls, training, generation or modifications.
Usage: cd /root/agent-sandbox/ARTN && python scripts/relationship_exposure_analysis.py
"""
import sys, os, json, re
from collections import Counter
import torch

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from src.tokenizer.bpe_tokenizer import BPETokenizer

BASE = '/root/agent-sandbox/ARTN'
TOK = os.path.join(BASE, 'outputs/artn-50k-best-config/tokenizer.json')
CORPORA = {
    'original':   os.path.join(BASE, 'outputs/tokenized_cache/train_tokenized.pt'),
    'treatment_a': os.path.join(BASE, 'outputs/treatment_oversample/tokenized_cache/train_tokenized.pt'),
    'treatment_b': os.path.join(BASE, 'outputs/treatment_augment/tokenized_cache/train_tokenized.pt'),
}
OUT = os.path.join(BASE, 'outputs/artn_recursive_20261025_002725/relationship_exposure_analysis.json')

SUBJECT_FORMS = {
    'bird': ['bird', 'birds'], 'fish': ['fish', 'fishes'], 'dragon': ['dragon', 'dragons'],
    'dog': ['dog', 'dogs'], 'king': ['king', 'kings'], 'teacher': ['teacher', 'teachers'],
    'peacock': ['peacock', 'peacocks'],
}
ACTION_FORMS = {
    'fly': ['fly', 'flies', 'flying', 'flew', 'flys'],
    'swim': ['swim', 'swims', 'swimming', 'swam'],
    'run': ['run', 'runs', 'running', 'ran'],
    'help': ['help', 'helps', 'helping', 'helped'],
    'teach': ['teach', 'teaches', 'teaching', 'taught'],
}
REL_ACTION = {'bird': 'fly', 'fish': 'swim', 'dragon': 'fly', 'dog': 'run',
              'king': 'help', 'teacher': 'teach', 'peacock': 'fly'}
AUX = ['can', 'could', 'will', 'would', 'may', 'might', 'must', 'shall', 'should', 'to']


def alt(words):
    return '(?:' + '|'.join(re.escape(w) for w in words) + ')'


def load_texts(path, tok):
    raw = torch.load(path, weights_only=False)
    out = []
    for r in raw:
        ids = r.tolist() if hasattr(r, 'tolist') else list(r)
        ids = [i for i in ids if i != 0]  # strip padding
        out.append(tok.decode(ids))
    return out


def split_sentences(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def analyse(texts, rels):
    res = {}
    for subj, action in rels.items():
        s_pat = re.compile(r'\b' + alt(SUBJECT_FORMS[subj]) + r'\b', re.I)
        a_pat = re.compile(r'\b' + alt(ACTION_FORMS[action]) + r'\b', re.I)
        # direct agentive patterns
        modal_pat = re.compile(r'\b' + alt(SUBJECT_FORMS[subj]) + r'\b\s+(?:' + '|'.join(AUX) + r')\s+' + alt(ACTION_FORMS[action]) + r'\b', re.I)
        direct_pat = re.compile(r'\b' + alt(SUBJECT_FORMS[subj]) + r'\b\s+(?:\w+\s+){0,1}?' + alt(ACTION_FORMS[action]) + r'\b', re.I)
        neg_pat = re.compile(r"\bnot\b|n't\b|\bnever\b", re.I)

        st = {'stories_total': len(texts), 'stories_subject': 0, 'stories_action': 0, 'stories_both': 0,
              'sent_total': 0, 'sent_subject': 0, 'sent_action': 0, 'sent_both': 0,
              'direct_count': 0, 'modal_direct_count': 0, 'direct_positive': 0, 'direct_negative': 0,
              'context_patterns': Counter(), 'examples_direct': [], 'examples_cooccur': []}
        for text in texts:
            sents = split_sentences(text)
            st['sent_total'] += len(sents)
            has_s = bool(s_pat.search(text)); has_a = bool(a_pat.search(text))
            st['stories_subject'] += has_s; st['stories_action'] += has_a; st['stories_both'] += (has_s and has_a)
            for s in sents:
                hs = bool(s_pat.search(s)); ha = bool(a_pat.search(s))
                st['sent_subject'] += hs; st['sent_action'] += ha
                if hs and ha:
                    st['sent_both'] += 1
                    if len(st['examples_cooccur']) < 4:
                        st['examples_cooccur'].append(s.strip())
                    if direct_pat.search(s):
                        st['direct_count'] += 1
                        if modal_pat.search(s):
                            st['modal_direct_count'] += 1
                        if neg_pat.search(s):
                            st['direct_negative'] += 1
                        else:
                            st['direct_positive'] += 1
                        m = modal_pat.search(s)
                        if m:
                            st['context_patterns'][m.group(0).lower()] += 1
                        if len(st['examples_direct']) < 4:
                            st['examples_direct'].append(s.strip())
        st['context_patterns'] = dict(st['context_patterns'].most_common(10))
        res[subj] = st
    return res


def main():
    tok = BPETokenizer.load(TOK)
    all_out = {'corpora': {}, 'relationships': {k: v for k, v in REL_ACTION.items()}}
    for name, path in CORPORA.items():
        print(f'Decoding {name} ...', flush=True)
        texts = load_texts(path, tok)
        print(f'  {len(texts)} stories decoded', flush=True)
        all_out['corpora'][name] = analyse(texts, REL_ACTION)

    with open(OUT, 'w') as f:
        json.dump(all_out, f, indent=1)
    print(f'\nSaved: {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)')

    # ---- report ----
    for rel, action in REL_ACTION.items():
        print('\n' + '=' * 78)
        print(f'RELATIONSHIP: {rel} -> {action}')
        print('=' * 78)
        print('  {:<13} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7}'.format(
            'corpus', 'stories', 'st_subj', 'st_act', 'st_both', 'snt_both', 'direct', 'pos', 'neg'))
        for name in CORPORA:
            d = all_out['corpora'][name][rel]
            print('  {:<13} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7}'.format(
                name, d['stories_total'], d['stories_subject'], d['stories_action'], d['stories_both'],
                d['sent_both'], d['direct_count'], d['direct_positive'], d['direct_negative']))
        for name in CORPORA:
            d = all_out['corpora'][name][rel]
            if d['context_patterns']:
                print(f"  [{name}] patterns: {d['context_patterns']}")
        d = all_out['corpora']['original'][rel]
        print(f"  [original] direct examples:")
        for e in d['examples_direct'][:3]:
            print(f"      - {e[:150]}")

    # ---- deltas ----
    print('\n\n=== DELTAS vs ORIGINAL (direct agentive evidence) ===')
    print('  {:<10} {:>10} {:>12} {:>12}'.format('relation', 'original', 'treatA', 'treatB'))
    for rel in REL_ACTION:
        o = all_out['corpora']['original'][rel]['direct_count']
        a = all_out['corpora']['treatment_a'][rel]['direct_count']
        b = all_out['corpora']['treatment_b'][rel]['direct_count']
        print('  {:<10} {:>10} {:>12} {:>12}'.format(rel, o, a, b))


if __name__ == '__main__':
    main()
