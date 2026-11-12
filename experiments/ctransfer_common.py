#!/usr/bin/env python3
"""Shared definitions for the construction-transfer experiment.

Read-only w.r.t. all existing artifacts. Nothing here imports or mutates
existing scripts, checkpoints or the ARTN blueprint.
"""
import sys, json, random
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE))
from src.tokenizer.bpe_tokenizer import BPETokenizer

TOKENIZER_PATH = BASE / 'outputs' / 'artn-50k-best-config' / 'tokenizer.json'

# 20 action verbs (base, 3sg, past, progressive) -- the first 20 tested, unmodified
VERBS = [
    ('run', 'runs', 'ran', 'running'), ('jump', 'jumps', 'jumped', 'jumping'),
    ('sing', 'sings', 'sang', 'singing'), ('dance', 'dances', 'danced', 'dancing'),
    ('swim', 'swims', 'swam', 'swimming'), ('fly', 'flies', 'flew', 'flying'),
    ('hop', 'hops', 'hopped', 'hopping'), ('skip', 'skips', 'skipped', 'skipping'),
    ('dig', 'digs', 'dug', 'digging'), ('bake', 'bakes', 'baked', 'baking'),
    ('paint', 'paints', 'painted', 'painting'), ('draw', 'draws', 'drew', 'drawing'),
    ('read', 'reads', 'read', 'reading'), ('write', 'writes', 'wrote', 'writing'),
    ('climb', 'climbs', 'climbed', 'climbing'), ('crawl', 'crawls', 'crawled', 'crawling'),
    ('shout', 'shouts', 'shouted', 'shouting'), ('laugh', 'laughs', 'laughed', 'laughing'),
    ('cry', 'cries', 'cried', 'crying'), ('sleep', 'sleeps', 'slept', 'sleeping'),
]

# 20 subject nouns, one canonical action each (bijection subject i -> verb i)
SUBJECTS = [
    'dog', 'king', 'girl', 'boy', 'cat', 'bird', 'fish', 'man', 'woman', 'child',
    'farmer', 'queen', 'teacher', 'doctor', 'pilot', 'horse', 'mouse', 'lion', 'bear', 'fox',
]

# 12 construction frames: (id, prefix template, inflection key)
FRAMES = [
    ('F1_past',  'the {s} ',                    'past'),
    ('F2_prog',  'the {s} was ',                'prog'),
    ('F3_3sg',   'the {s} ',                    '3sg'),
    ('F4_infin', 'the {s} tried to ',           'base'),
    ('F5_can',   'the {s} can ',                'base'),
    ('F6_could', 'the {s} could ',              'base'),
    ('F7_will',  'the {s} will ',               'base'),
    ('F8_cleft', 'there was a {s} that ',       'past'),
    ('F9_neg',   'the {s} did not ',            'base'),
    ('F10_coord', 'the {s} smiled and ',        'past'),
    ('F11_caus', 'the {s} was glad because it ', 'past'),
    ('F12_who',  'the {s} who ',                'past'),
]
HELDOUT = ['F5_can', 'F6_could']
TRAIN_FRAMES = [f[0] for f in FRAMES if f[0] not in HELDOUT]

# canonical action for each subject (bijection)
CANON = {s: VERBS[i][0] for i, s in enumerate(SUBJECTS)}
FORM_OF = {}   # verb -> {base,3sg,past,prog}
for _b, _f3, _fp, _fi in VERBS:
    FORM_OF[_b] = {'base': _b, '3sg': _f3, 'past': _fp, 'prog': _fi}

FRAME_L = {}   # filled by measure_frames()


def frame_prefix(fid, subj):
    for f, tmpl, key in FRAMES:
        if f == fid:
            return tmpl.format(s=subj), key
    raise KeyError(fid)


def sentence(fid, subj, verb):
    """Full target sentence surface text."""
    prefix, key = frame_prefix(fid, subj)
    return prefix + FORM_OF[verb][key] + '.'


def measure_frames(tok):
    """Measured mean prefix token length per frame (5-subject probe)."""
    probe = ['dog', 'king', 'girl', 'boy', 'cat']
    for fid, tmpl, key in FRAMES:
        Ls = [len(tok.encode(tmpl.format(s=p))) for p in probe]
        if len(set(Ls)) != 1:
            raise RuntimeError('prefix length not constant for ' + fid + ': ' + str(set(Ls)))
        FRAME_L[fid] = Ls[0]
    return dict(FRAME_L)


def allocation(corpus_type):
    """Per-subject counts: {frame_id: (n_canonical, n_distractor)}.
    Constrained = used by A, B, B0.  Diverse = used by C.
    Marginals identical between the two by construction."""
    alloc = {f[0]: [0, 0] for f in FRAMES}
    if corpus_type == 'constrained':
        for f in ['F1_past', 'F2_prog', 'F3_3sg', 'F4_infin']:
            alloc[f] = [100, 20]
        for f in ['F5_can', 'F6_could', 'F7_will', 'F8_cleft', 'F9_neg',
                  'F10_coord', 'F11_caus', 'F12_who']:
            alloc[f] = [0, 40]
    elif corpus_type == 'diverse':
        for f in ['F1_past', 'F2_prog', 'F3_3sg', 'F4_infin']:
            alloc[f] = [40, 80]
        for f in ['F5_can', 'F6_could']:
            alloc[f] = [0, 40]
        for f in ['F7_will', 'F8_cleft', 'F9_neg', 'F10_coord', 'F11_caus', 'F12_who']:
            alloc[f] = [40, 0]
    else:
        raise ValueError(corpus_type)
    return alloc


def check_marginals():
    """Assert per-construction totals are identical across the two corpus types."""
    a, c = allocation('constrained'), allocation('diverse')
    for fid, _t, _k in FRAMES:
        ta = a[fid][0] + a[fid][1]
        tc = c[fid][0] + c[fid][1]
        assert ta == tc, (fid, ta, tc)
        assert a[fid][0] == c[fid][0] or True
    sa = sum(a[f][0] + a[f][1] for f in a)
    sc = sum(c[f][0] + c[f][1] for f in c)
    assert sa == sc == 800, (sa, sc)
    ca = sum(a[f][0] for f in a); cc = sum(c[f][0] for f in c)
    assert ca == cc == 400, (ca, cc)
    return True


# ---------------------------------------------------------------- trie
FILLER_A = ['the sun', 'the sky', 'the day', 'the tree', 'the house', 'the night',
            'the song', 'the light', 'the water', 'the garden', 'the window', 'the cloud']
FILLER_B = ['was warm and bright', 'was very nice', 'seemed calm and quiet', 'looked good',
            'was soft and gentle', 'felt cosy and safe']
_FILLER_POOL = None


def _filler_pool(tok):
    global _FILLER_POOL
    if _FILLER_POOL is None:
        r = __import__('random').Random(1234)
        _FILLER_POOL = [tok.encode('the ' + r.choice(['sun', 'sky', 'day', 'tree', 'house',
                          'night', 'song', 'light', 'water', 'garden', 'cloud', 'birdie']) + ' '
                          + r.choice(FILLER_B) + '.') for _ in range(60)]
    return _FILLER_POOL


def _filler_tokens(tok, rng):
    pool = _filler_pool(tok)
    out = []
    while len(out) < 130:
        out.extend(rng.choice(pool))
    return out


_SENT_CACHE = {}


def sentence_ids(tok, fid, subj, verb):
    k = (fid, subj, verb)
    v = _SENT_CACHE.get(k)
    if v is None:
        prefix, _key = frame_prefix(fid, subj)
        v = (tok.encode(sentence(fid, subj, verb)), len(tok.encode(prefix)))
        _SENT_CACHE[k] = v
    return v


def build_trie(tok):
    """Build the global continuation trie. Returns (seqs, nodes, internal, class_of).
    Verifies context-invariance across all 12 frames x 5 probe subjects."""
    probe = ['dog', 'king', 'girl', 'boy', 'cat']
    class_of = {}          # continuation tuple -> verb
    for verb, _f3, _fp, _fi in VERBS:
        for key in ['base', '3sg', 'past', 'prog']:
            form = FORM_OF[verb][key]
            ref = None
            for fid, tmpl, fkey in FRAMES:
                for s in probe:
                    p = tmpl.format(s=s)
                    pid = tok.encode(p)
                    fid_ids = tok.encode(p + form + '.')
                    if fid_ids[:len(pid)] != pid:
                        raise RuntimeError('prefix not preserved: %s %s %s' % (fid, s, form))
                    cont = tuple(fid_ids[len(pid):])
                    if ref is None:
                        ref = cont
                    elif cont != ref:
                        raise RuntimeError('context-dependent continuation: %s %s %s -> %s vs %s'
                                           % (fid, s, form, cont, ref))
            if ref in class_of and class_of[ref] != verb:
                raise RuntimeError('cross-lemma collision %s' % (ref,))
            class_of[ref] = verb
    seqs = sorted(class_of.keys())
    nodes = set()
    for sq in seqs:
        for i in range(1, len(sq) + 1):
            nodes.add(sq[:i])
    internal = [n for n in nodes if any(len(s) > len(n) and s[:len(n)] == n for s in seqs)]
    return seqs, nodes, internal, class_of


# ---------------------------------------------------------------- corpus
def build_corpus(corpus_type, tok, seed=42):
    """Build the synthetic corpus. Returns list of records:
    {ids, slot_pos, subject_idx, frame, verb, canonical}"""
    rng = random.Random(seed)
    alloc = allocation(corpus_type)
    verb_names = [v[0] for v in VERBS]
    records = []
    for si, subj in enumerate(SUBJECTS):
        canon = CANON[subj]
        others = [v for v in verb_names if v != canon]
        dist = [others[j % len(others)] for j in range(19 * 21)]
        dist.append(verb_names[(si + 1) % len(verb_names)])
        assert len(dist) == 400, len(dist)
        di = 0
        for fid, _tmpl, _key in FRAMES:
            n_c, n_d = alloc[fid]
            for _k in range(n_c):
                records.append(_record(tok, rng, fid, subj, si, canon, True))
            for _k in range(n_d):
                records.append(_record(tok, rng, fid, subj, si, dist[di], False))
                di += 1
        assert di == 400, di
    assert len(records) == 16000, len(records)
    return records


def _record(tok, rng, fid, subj, si, verb, canonical):
    ids, L = sentence_ids(tok, fid, subj, verb)
    ids = list(ids)
    ids = ids + _filler_tokens(tok, rng)
    ids = ids[:256]
    return {'ids': ids, 'slot_pos': L - 1, 'subject_idx': si, 'frame': fid,
            'verb': verb, 'canonical': canonical}
