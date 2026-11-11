#!/usr/bin/env python3
"""
ARTN Generated-Only Repetition Penalty Evaluation

Runs only the 2 configs missing from broader_gen_eval.json:
  E. Greedy + sign-aware penalty 1.3, generated-only scope, t=1.0
  F. Greedy + sign-aware penalty 1.3, generated-only scope, t=0.9

Same 20 prompts, same 100 tokens, same seed. Reuses broader_gen_eval.py prompts
and metrics. Output saved separately for comparison.

Output: outputs/artn_recursive_20261025_002725/genonly_penalty_eval.json
"""
import sys, os, json, time, re, torch

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = torch.device('cpu')
SEED = 42
MAX_TOKENS = 100
OUTPUT_DIR = '/root/agent-sandbox/ARTN/outputs/artn_recursive_20261025_002725'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'genonly_penalty_eval.json')

PROMPTS = {
    "familiar": [
        "once upon a time there was a dog",
        "there was a little girl",
        "there lived a",
        "once there was a little boy",
        "a cat sat on the mat",
        "the sun was shining bright",
        "a little bird flew high",
        "there was a big tree",
        "a fish swam in the pond",
        "the children played outside",
    ],
    "less_common": [
        "a bison from wild forest",
        "a peacock",
        "a king",
        "a robot in the garden",
        "a dragon lived in a cave",
    ],
    "challenging": [
        "a king and a queen had three children",
        "the fox and the rabbit were friends but one day",
        "a storm came and the boat started to sink",
        "the teacher gave the students a surprise test",
        "a wizard and a knight went on a quest together",
    ],
}

CONFIGS = [
    {"label": "E_genonly_p1.3_t1.0", "threshold": 1.0, "rep_penalty": 1.3},
    {"label": "F_genonly_p1.3_t0.9", "threshold": 0.9, "rep_penalty": 1.3},
]

print("Loading ARTN model...")
ckpt = torch.load(
    os.path.join(OUTPUT_DIR, 'checkpoints/artn/step_10000.pt'),
    map_location='cpu', weights_only=False
)
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
tok = BPETokenizer.load(os.path.join(OUTPUT_DIR, '..', 'artn-50k-best-config', 'tokenizer.json'))
print(f"  Loaded: step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}")
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
print()


def generate(prompt, max_new_tokens=100, threshold=1.0, rep_penalty=1.0):
    """Generate with sign-aware rep penalty, GENERATED-ONLY token scope."""
    input_ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    prompt_len = input_ids.shape[1]  # track where generated tokens begin
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

                if threshold >= 1.0:
                    h = torch.sigmoid(model.halting_gate(x))
                    p = h * remaining
                    if pass_idx == model.max_passes - 1:
                        p = remaining
                    output = output + x * p
                    remaining = remaining - p
                    continue

                h = torch.sigmoid(model.halting_gate(x))
                p = h * remaining
                if pass_idx == model.max_passes - 1:
                    p = remaining
                output = output + x * p
                remaining = remaining - p
                remaining = remaining.clamp(min=0.0)

                cumulative_last = 1.0 - remaining[0, -1, 0].item()
                if cumulative_last >= threshold and pass_idx < model.max_passes - 1:
                    break

            output = output + x * remaining
            x = model.ln_f(output)
            logits = model.lm_head(x)
            logits = logits[:, -1, :]

            # Sign-aware repetition penalty — GENERATED-ONLY scope
            if rep_penalty != 1.0:
                # Only penalize tokens that appear in GENERATED text, not prompt
                generated_tokens = input_ids[0, prompt_len:].tolist()
                prev_tokens = set(generated_tokens)
                for tid in prev_tokens:
                    val = logits[0, tid].item()
                    if val < 0:
                        logits[0, tid] = val * rep_penalty
                    else:
                        logits[0, tid] = val / rep_penalty

            next_token = logits.argmax(dim=-1, keepdim=True)
            passes_log.append(passes_this)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    elapsed = time.time() - t0
    text = tok.decode(input_ids[0].tolist())
    return text, passes_log, elapsed


def compute_repetition_stats(text):
    words = text.split()
    if len(words) < 4:
        return {"repetition_rate": 0, "max_consecutive_bigram_repeat": 0, "unique_word_ratio": 1.0}
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    bigram_counts = {}
    for bg in bigrams:
        bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
    repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
    repetition_rate = repeated_bigrams / max(len(bigram_counts), 1)
    max_repeat = 0
    cur = 1
    for i in range(1, len(bigrams)):
        if bigrams[i] == bigrams[i-1]:
            cur += 1
            max_repeat = max(max_repeat, cur)
        else:
            cur = 1
    unique_ratio = len(set(words)) / max(len(words), 1)
    return {
        "repetition_rate": round(repetition_rate, 3),
        "max_consecutive_bigram_repeat": max_repeat,
        "unique_word_ratio": round(unique_ratio, 3),
    }


def detect_malformed_words(text):
    words = text.split()
    malformed = []
    exempt = {
        'shouldn', 'wouldn', 'couldn', 'didn', 'wasn', 'weren', 'hasn',
        'strength', 'lengths', 'through', 'thought', 'thoughts',
        'scratched', 'scratches', 'stretched', 'catches', 'watching',
        'children', 'grandmother', 'grandfather', 'grandparents',
        'angry', 'cycle',
    }
    for w in words:
        clean = re.sub(r'[^\w]', '', w).lower()
        if len(clean) < 3:
            continue
        consonant_runs = re.findall(r'[bcdfghjklmnpqrstvwxyz]{4,}', clean)
        if consonant_runs and clean not in exempt:
            malformed.append(w)
    return malformed


all_prompts = []
for cat in ["familiar", "less_common", "challenging"]:
    for p in PROMPTS[cat]:
        all_prompts.append((cat, p))

total_runs = len(all_prompts) * len(CONFIGS)
run_num = 0
results = {}

for cat, prompt in all_prompts:
    if prompt not in results:
        results[prompt] = {"category": cat, "configs": {}}

    for config in CONFIGS:
        run_num += 1
        label = config["label"]
        print(f"  [{run_num}/{total_runs}] {label}  '{prompt[:40]}'")

        torch.manual_seed(SEED)
        text, passes, elapsed = generate(
            prompt, MAX_TOKENS,
            threshold=config["threshold"],
            rep_penalty=config["rep_penalty"]
        )

        rep_stats = compute_repetition_stats(text)
        malformed = detect_malformed_words(text)
        tok_s = MAX_TOKENS / elapsed if elapsed > 0 else 0
        avg_passes = sum(passes) / len(passes) if passes else 0

        results[prompt]["configs"][label] = {
            "threshold": config["threshold"],
            "rep_penalty": config["rep_penalty"],
            "token_scope": "generated_only",
            "generated_text": text,
            "passes_log": passes,
            "avg_passes": round(avg_passes, 2),
            "elapsed_s": round(elapsed, 2),
            "tokens_per_sec": round(tok_s, 2),
            "repetition_stats": rep_stats,
            "malformed_words": malformed,
        }

summary = {}
for config in CONFIGS:
    label = config["label"]
    config_results = []
    for cat, prompt in all_prompts:
        r = results[prompt]["configs"][label]
        config_results.append({"prompt": prompt, "category": cat, **r})

    avg_rep = sum(r["repetition_stats"]["repetition_rate"] for r in config_results) / len(config_results)
    avg_uniq = sum(r["repetition_stats"]["unique_word_ratio"] for r in config_results) / len(config_results)
    avg_tok_s = sum(r["tokens_per_sec"] for r in config_results) / len(config_results)
    avg_passes = sum(r["avg_passes"] for r in config_results) / len(config_results)
    total_malformed = sum(len(r["malformed_words"]) for r in config_results)
    prompts_with_malformed = sum(1 for r in config_results if r["malformed_words"])

    cat_stats = {}
    for cat_name in ["familiar", "less_common", "challenging"]:
        cat_res = [r for r in config_results if r["category"] == cat_name]
        cat_stats[cat_name] = {
            "avg_rep": round(sum(r["repetition_stats"]["repetition_rate"] for r in cat_res) / len(cat_res), 3),
            "avg_uniq": round(sum(r["repetition_stats"]["unique_word_ratio"] for r in cat_res) / len(cat_res), 3),
            "avg_passes": round(sum(r["avg_passes"] for r in cat_res) / len(cat_res), 2),
            "malformed_count": sum(len(r["malformed_words"]) for r in cat_res),
        }

    summary[label] = {
        "config": config,
        "avg_repetition_rate": round(avg_rep, 3),
        "avg_unique_word_ratio": round(avg_uniq, 3),
        "avg_tokens_per_sec": round(avg_tok_s, 2),
        "avg_recursive_passes": round(avg_passes, 2),
        "total_malformed_words": total_malformed,
        "prompts_with_malformed": prompts_with_malformed,
        "per_category": cat_stats,
    }

output = {
    "experiment": "generated_only_penalty_evaluation",
    "date": "2026-11-02",
    "checkpoint": f"step_{ckpt['step']}",
    "val_loss": ckpt['val_loss'],
    "params": sum(p.numel() for p in model.parameters()),
    "max_tokens": MAX_TOKENS,
    "seed": SEED,
    "num_prompts": len(all_prompts),
    "num_configs": len(CONFIGS),
    "token_scope": "generated_only",
    "summary": summary,
    "results": results,
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 70}")
print("GENERATED-ONLY PENALTY EVALUATION COMPLETE")
print(f"{'=' * 70}")
print(f"\nOutput: {OUTPUT_FILE}")
print(f"\n--- Summary ---")
print(f"{'Config':<25} {'Avg Rep':<10} {'Avg Uniq':<10} {'Malformed':<10} {'Tok/s':<10} {'Avg Pass':<10}")
print("-" * 75)
for label in summary:
    s = summary[label]
    print(f"{label:<25} {s['avg_repetition_rate']:<10} {s['avg_unique_word_ratio']:<10} "
          f"{s['total_malformed_words']:<10} {s['avg_tokens_per_sec']:<10} {s['avg_recursive_passes']:<10}")
