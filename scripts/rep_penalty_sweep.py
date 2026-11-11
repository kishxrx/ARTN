#!/usr/bin/env python3
"""
Investigation 3: Repetition Penalty Sweep

Compare repetition penalties 1.1, 1.3, 1.5 against existing 1.2 and
unpenalized (1.0) baseline. Greedy decoding at both thresholds.

Does NOT modify gen_quality_eval.py. New script, new output.
Output: rep_penalty_sweep.json
"""
import sys, json, time, os, math
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = torch.device('cpu')
BASE = '/root/agent-sandbox/ARTN'

print("Loading ARTN model...")
ckpt = torch.load(
    f'{BASE}/outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt',
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
tok = BPETokenizer.load(f'{BASE}/outputs/artn-50k-best-config/tokenizer.json')
print(f"  Loaded: step={ckpt['step']}")
print()

PROMPTS = [
    "once upon a time there was a dog",
    "there was a little girl",
    "there lived a",
    "a king",
    "king",
    "a bison from wild forest",
    "a peacock",
    "The scientist mixed two chemicals",
    "In the year 3024, humanity",
    "Why did the chicken cross",
]

MAX_NEW_TOKENS = 50
SEED = 42

PENALTIES = [1.0, 1.1, 1.2, 1.3, 1.5]
THRESHOLDS = [1.0, 0.9]


def forward_with_halt(input_ids, threshold):
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
    return logits[:, -1, :], passes_this


def generate(prompt, max_new_tokens, threshold, rep_penalty=1.0):
    """Greedy generation with repetition penalty."""
    torch.manual_seed(SEED)
    input_ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    passes_log = []
    t0 = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            T = input_ids.shape[1]
            if T >= model.position_embedding.num_embeddings:
                break

            logits, passes = forward_with_halt(input_ids, threshold)
            passes_log.append(passes)

            # Apply repetition penalty
            if rep_penalty != 1.0:
                prev_tokens = input_ids[0].tolist()
                for pt in set(prev_tokens):
                    logits[0, pt] = logits[0, pt] / rep_penalty

            next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    elapsed = time.time() - t0
    full_text = tok.decode(input_ids[0].tolist())
    generated_text = full_text[len(prompt):].strip()
    return {
        "full_text": full_text,
        "generated_text": generated_text,
        "num_tokens": len(passes_log),
        "avg_passes": round(sum(passes_log) / len(passes_log), 2) if passes_log else 0,
        "elapsed_s": round(elapsed, 2),
        "tokens_per_sec": round(len(passes_log) / elapsed, 2) if elapsed > 0 else 0,
    }


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


# ============================================================
# Run
# ============================================================
print("=" * 70)
print("INVESTIGATION 3: Repetition Penalty Sweep")
print("=" * 70)

all_results = {
    "experiment": "rep_penalty_sweep",
    "checkpoint": f"{BASE}/outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt",
    "max_new_tokens": MAX_NEW_TOKENS,
    "seed": SEED,
    "penalties": PENALTIES,
    "thresholds": THRESHOLDS,
    "prompts": PROMPTS,
    "results": {},
}

total = len(PROMPTS) * len(PENALTIES) * len(THRESHOLDS)
run = 0

for threshold in THRESHOLDS:
    t_key = f"threshold_{threshold}"
    all_results["results"][t_key] = {}

    for penalty in PENALTIES:
        p_key = f"penalty_{penalty}"
        all_results["results"][t_key][p_key] = []
        print(f"\n--- Threshold={threshold}, Penalty={penalty} ---")

        for prompt in PROMPTS:
            run += 1
            result = generate(prompt, MAX_NEW_TOKENS, threshold, rep_penalty=penalty)
            result["repetition_stats"] = compute_repetition_stats(result["generated_text"])
            result["prompt"] = prompt
            result["penalty"] = penalty
            result["threshold"] = threshold
            all_results["results"][t_key][p_key].append(result)

            rep = result["repetition_stats"]
            print(f"  [{run}/{total}] '{prompt[:35]:<35}' rep={rep['repetition_rate']:.3f} "
                  f"unique={rep['unique_word_ratio']:.3f} max_rep={rep['max_consecutive_bigram_repeat']}")

# Save
output_path = f'{BASE}/outputs/artn_recursive_20261025_002725/rep_penalty_sweep.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n\nResults saved to: {output_path}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: Average metrics by penalty and threshold")
print("=" * 70)
print(f"\n{'Penalty':<10} {'t=1.0 rep':<10} {'t=0.9 rep':<10} {'t=1.0 uniq':<11} {'t=0.9 uniq':<11} {'t=1.0 maxRpt':<12} {'t=0.9 maxRpt'}")
print("-" * 85)
for penalty in PENALTIES:
    p_key = f"penalty_{penalty}"
    t1 = all_results["results"]["threshold_1.0"][p_key]
    t09 = all_results["results"]["threshold_0.9"][p_key]
    t1_rep = sum(r["repetition_stats"]["repetition_rate"] for r in t1) / len(t1)
    t09_rep = sum(r["repetition_stats"]["repetition_rate"] for r in t09) / len(t09)
    t1_uniq = sum(r["repetition_stats"]["unique_word_ratio"] for r in t1) / len(t1)
    t09_uniq = sum(r["repetition_stats"]["unique_word_ratio"] for r in t09) / len(t09)
    t1_max = sum(r["repetition_stats"]["max_consecutive_bigram_repeat"] for r in t1) / len(t1)
    t09_max = sum(r["repetition_stats"]["max_consecutive_bigram_repeat"] for r in t09) / len(t09)
    print(f"  {penalty:<10} {t1_rep:<10.3f} {t09_rep:<10.3f} {t1_uniq:<11.3f} {t09_uniq:<11.3f} {t1_max:<12.1f} {t09_max}")

print("\nDONE")
