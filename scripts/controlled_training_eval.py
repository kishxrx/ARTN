#!/usr/bin/env python3
"""
Controlled Training Evaluation: 4 ARTN checkpoints × 10 targeted prompts.

Checkpoints:
  - Original ARTN step_10000
  - Control step_15000
  - Treatment A step_15000
  - Treatment B step_15000

Prompts: 10 targeted (3 king, 3 teacher, 3 peacock, 1 familiar control)
Decoding: greedy, threshold=1.0, 100 tokens, deterministic (seed=42)

Output: outputs/controlled_training_eval.json

Usage: cd /root/agent-sandbox/ARTN && python scripts/controlled_training_eval.py
"""

import sys, os, json, time, re, torch
sys.path.insert(0, '/root/agent-sandbox/ARTN')

from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = torch.device('cpu')
SEED = 42
MAX_TOKENS = 100

BASE = '/root/agent-sandbox/ARTN'
TOKENIZER_PATH = os.path.join(BASE, 'outputs/artn-50k-best-config/tokenizer.json')

CHECKPOINTS = {
    "original": os.path.join(BASE, 'outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt'),
    "control": sorted(__import__('glob').glob(os.path.join(BASE, 'outputs/artn_control_*/checkpoints/artn/step_15000.pt')))[-1],
    "treatment_a": sorted(__import__('glob').glob(os.path.join(BASE, 'outputs/artn_treatment_a_*/checkpoints/artn/step_15000.pt')))[-1],
    "treatment_b": sorted(__import__('glob').glob(os.path.join(BASE, 'outputs/artn_treatment_b_*/checkpoints/artn/step_15000.pt')))[-1],
}

# 10 targeted prompts
PROMPTS = [
    # King (3 prompts — different contexts)
    ("king_1", "a king", ["king"]),
    ("king_2", "a king lived in a castle", ["king", "castle"]),
    ("king_3", "there was a king who was kind", ["king"]),
    # Teacher (3 prompts)
    ("teacher_1", "a teacher", ["teacher"]),
    ("teacher_2", "a teacher walked into the classroom", ["teacher", "classroom"]),
    ("teacher_3", "there was a teacher who helped", ["teacher"]),
    # Peacock (3 prompts)
    ("peacock_1", "a peacock", ["peacock"]),
    ("peacock_2", "a peacock walked in the zoo", ["peacock", "zoo"]),
    ("peacock_3", "there was a peacock at the lake", ["peacock", "lake"]),
    # Familiar control (1 prompt)
    ("dog_control", "a dog", ["dog"]),
]

# Held-out combination check: search original training data for these prompts
HELD_OUT_COMBOS = {
    "king_2": "king lived in a castle",
    "king_3": "king who was kind",
    "teacher_2": "teacher walked into the classroom",
    "teacher_3": "teacher who helped",
    "peacock_2": "peacock walked in the zoo",
    "peacock_3": "peacock at the lake",
}

MALFORMED_EXEMPT = {
    'shouldn', 'wouldn', 'couldn', 'didn', 'wasn', 'weren', 'hasn',
    'strength', 'lengths', 'through', 'thought', 'thoughts',
    'scratched', 'scratches', 'stretched', 'catches', 'watching',
    'children', 'grandmother', 'grandfather', 'grandparents',
    'angry', 'cycle', 'brightly', 'anything', 'knight', 'knights',
    'cyck',  # known peacock artifact
}

def generate_artn(model, tok, prompt, max_new_tokens=100, threshold=1.0):
    """Greedy ARTN generation, deterministic."""
    torch.manual_seed(SEED)
    input_ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    prompt_len = input_ids.shape[1]
    passes_log = []
    t0 = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            B, T = input_ids.shape
            tok_emb = model.token_embedding(input_ids)
            pos_emb = model.position_embedding(torch.arange(T, device=DEVICE))
            x = tok_emb + pos_emb

            remaining = torch.ones(B, T, 1, device=DEVICE)
            output = torch.zeros(B, T, model.d_model, device=DEVICE)
            passes_this = 0

            for pass_idx in range(model.max_passes):
                passes_this += 1
                x = model.block(x)
                h = torch.sigmoid(model.halting_gate(x))
                p = h * remaining
                if pass_idx == model.max_passes - 1:
                    p = remaining
                output = output + x * p
                remaining = remaining - p

            output = output + x * remaining
            x = model.ln_f(output)
            logits = model.lm_head(x)
            logits = logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            passes_log.append(passes_this)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    elapsed = time.time() - t0
    text = tok.decode(input_ids[0].tolist())
    return text, passes_log, elapsed


def compute_repetition_stats(text):
    words = text.split()
    if len(words) < 4:
        return {"repetition_rate": 0, "unique_word_ratio": 1.0}
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    bigram_counts = {}
    for bg in bigrams:
        bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
    repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
    repetition_rate = repeated_bigrams / max(len(bigram_counts), 1)
    unique_ratio = len(set(words)) / max(len(words), 1)
    return {
        "repetition_rate": round(repetition_rate, 3),
        "unique_word_ratio": round(unique_ratio, 3),
    }


def detect_malformed_words(text):
    words = text.split()
    malformed = []
    for w in words:
        clean = re.sub(r'[^\w]', '', w).lower()
        if len(clean) < 3:
            continue
        consonant_runs = re.findall(r'[bcdfghjklmnpqrstvwxyz]{4,}', clean)
        if consonant_runs and clean not in MALFORMED_EXEMPT:
            malformed.append(w)
    return malformed


def check_entity_retention(prompt, generated_text, entities):
    """Check if entities appear in generated text (excluding prompt), word-boundary."""
    gen_only = generated_text[len(prompt):].strip()
    retained = []
    for e in entities:
        if re.search(r'\b' + re.escape(e) + r'\b', gen_only.lower()):
            retained.append(e)
    return retained


# ============================================================
# Load tokenizer
# ============================================================
print("Loading tokenizer...", flush=True)
tok = BPETokenizer.load(TOKENIZER_PATH)
print(f"  vocab={tok.vocab_size}, pad_token_id={tok.pad_token_id}", flush=True)

# ============================================================
# Check held-out combinations in original training data
# ============================================================
print("\nChecking held-out combinations in training data...", flush=True)
train_data = torch.load(os.path.join(BASE, 'outputs/tokenized_cache/train_tokenized.pt'), weights_only=False)

held_out_results = {}
for prompt_id, combo in HELD_OUT_COMBOS.items():
    found_in_train = False
    found_in_treatment_b = False
    # Check a sample of training data (full scan too slow)
    for i, s in enumerate(train_data):
        if i >= 49988:
            break
        decoded = tok.decode(s).lower()
        if combo.lower() in decoded:
            found_in_train = True
            break
    # Check Treatment B augmented stories
    tb_stories_path = os.path.join(BASE, 'outputs/treatment_b_augmented_stories/all_500_stories.json')
    with open(tb_stories_path) as f:
        tb_stories = json.load(f)
    for concept in ['king', 'teacher', 'peacock']:
        for story in tb_stories[concept]:
            if combo.lower() in story.lower():
                found_in_treatment_b = True
                break
        if found_in_treatment_b:
            break
    held_out_results[prompt_id] = {
        "combination": combo,
        "found_in_original_training": found_in_train,
        "found_in_treatment_b": found_in_treatment_b,
    }
    status = "HELD OUT" if not found_in_train and not found_in_treatment_b else "NOT HELD OUT"
    if found_in_train:
        status += " (in original training)"
    if found_in_treatment_b:
        status += " (in Treatment B)"
    print(f"  {prompt_id}: '{combo}' → {status}", flush=True)

# ============================================================
# Load all 4 models
# ============================================================
models = {}
for name, ckpt_path in CHECKPOINTS.items():
    print(f"\nLoading {name}: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = ARTNGPT(
        vocab_size=cfg['vocab_size'], d_model=cfg['d_model'],
        n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
        d_ff=cfg['d_ff'], max_seq_len=cfg['max_seq_len'],
        target_passes=cfg.get('target_passes', 4.0),
        ponder_loss_weight=cfg.get('ponder_loss_weight', 0.01)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    models[name] = model
    print(f"  step={ckpt['step']}, val_loss={ckpt.get('val_loss', 'N/A')}, params={sum(p.numel() for p in model.parameters()):,}", flush=True)

# ============================================================
# Run evaluation
# ============================================================
print(f"\n{'=' * 80}", flush=True)
print(f"Running evaluation: 4 checkpoints × 10 prompts × 100 tokens = 40 runs", flush=True)
print(f"{'=' * 80}\n", flush=True)

all_results = []

for ckpt_name, model in models.items():
    print(f"\n--- {ckpt_name.upper()} ---", flush=True)
    for prompt_id, prompt_text, entities in PROMPTS:
        print(f"  {prompt_id}: {prompt_text}", flush=True)
        text, passes, elapsed = generate_artn(model, tok, prompt_text, MAX_TOKENS, threshold=1.0)

        gen_only = text[len(prompt_text):].strip()
        rep_stats = compute_repetition_stats(text)
        malformed = detect_malformed_words(text)
        retained = check_entity_retention(prompt_text, text, entities)
        avg_passes = sum(passes) / len(passes) if passes else 0

        result = {
            "checkpoint": ckpt_name,
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "entities": entities,
            "generated_text": text,
            "generated_only": gen_only,
            "repetition_rate": rep_stats["repetition_rate"],
            "unique_word_ratio": rep_stats["unique_word_ratio"],
            "malformed_words": malformed,
            "malformed_count": len(malformed),
            "retained_entities": retained,
            "entity_retention_count": len(retained),
            "entity_retention_total": len(entities),
            "avg_passes": round(avg_passes, 2),
            "elapsed_seconds": round(elapsed, 2),
            "tok_per_sec": round(MAX_TOKENS / elapsed, 2),
        }
        all_results.append(result)
        print(f"    → retention={len(retained)}/{len(entities)}, rep_rate={rep_stats['repetition_rate']}, malformed={len(malformed)}, passes={avg_passes:.1f}, {MAX_TOKENS/elapsed:.1f} tok/s", flush=True)

# ============================================================
# Save results
# ============================================================
output = {
    "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "checkpoints": {k: v for k, v in CHECKPOINTS.items()},
    "prompts": [{"id": p[0], "text": p[1], "entities": p[2]} for p in PROMPTS],
    "held_out_check": held_out_results,
    "results": all_results,
}

output_path = os.path.join(BASE, 'outputs', 'controlled_training_eval.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to: {output_path}", flush=True)

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 80}", flush=True)
print("SUMMARY", flush=True)
print(f"{'=' * 80}\n", flush=True)

# Per-checkpoint summary
for ckpt_name in CHECKPOINTS:
    ckpt_results = [r for r in all_results if r["checkpoint"] == ckpt_name]
    total_retained = sum(r["entity_retention_count"] for r in ckpt_results)
    total_entities = sum(r["entity_retention_total"] for r in ckpt_results)
    total_malformed = sum(r["malformed_count"] for r in ckpt_results)
    avg_rep = sum(r["repetition_rate"] for r in ckpt_results) / len(ckpt_results)
    avg_uniq = sum(r["unique_word_ratio"] for r in ckpt_results) / len(ckpt_results)
    avg_tok_s = sum(r["tok_per_sec"] for r in ckpt_results) / len(ckpt_results)

    print(f"{ckpt_name.upper()}:", flush=True)
    print(f"  Entity retention: {total_retained}/{total_entities} ({total_retained/total_entities*100:.1f}%)", flush=True)
    print(f"  Malformed words: {total_malformed}", flush=True)
    print(f"  Avg repetition rate: {avg_rep:.3f}", flush=True)
    print(f"  Avg unique word ratio: {avg_uniq:.3f}", flush=True)
    print(f"  Avg tok/s: {avg_tok_s:.1f}", flush=True)
    print(flush=True)

# Per-concept summary
print(f"\n{'=' * 80}", flush=True)
print("PER-CONCEPT COMPARISON", flush=True)
print(f"{'=' * 80}\n", flush=True)

for concept in ['king', 'teacher', 'peacock', 'dog']:
    concept_prompts = [p for p in PROMPTS if p[0].startswith(concept)]
    if not concept_prompts:
        continue
    print(f"\n{concept.upper()}:", flush=True)
    for ckpt_name in CHECKPOINTS:
        concept_results = [r for r in all_results
                          if r["checkpoint"] == ckpt_name
                          and r["prompt_id"].startswith(concept)]
        retained = sum(r["entity_retention_count"] for r in concept_results)
        total = sum(r["entity_retention_total"] for r in concept_results)
        malformed = sum(r["malformed_count"] for r in concept_results)
        print(f"  {ckpt_name}: retention={retained}/{total}, malformed={malformed}", flush=True)

# Print all generated texts
print(f"\n{'=' * 80}", flush=True)
print("ALL GENERATED OUTPUTS", flush=True)
print(f"{'=' * 80}\n", flush=True)

for prompt_id, prompt_text, entities in PROMPTS:
    print(f"\n--- {prompt_id}: '{prompt_text}' ---", flush=True)
    for ckpt_name in CHECKPOINTS:
        r = [x for x in all_results if x["checkpoint"] == ckpt_name and x["prompt_id"] == prompt_id][0]
        retained_str = f"retained={r['retained_entities']}" if r['retained_entities'] else "retained=none"
        mal_str = f"malformed={r['malformed_words']}" if r['malformed_words'] else "malformed=none"
        print(f"  [{ckpt_name}] {retained_str}, {mal_str}", flush=True)
        print(f"    {r['generated_only'][:200]}", flush=True)
        print(flush=True)

print(f"\n{'=' * 80}", flush=True)
print("EVALUATION COMPLETE", flush=True)
print(f"{'=' * 80}", flush=True)
