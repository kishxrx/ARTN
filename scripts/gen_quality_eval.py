#!/usr/bin/env python3
"""
Experiment 1: Decoding Strategy Comparison

Tests whether repetition and poor quality are decoding artifacts
by comparing greedy, temperature sampling, top-k, and repetition penalty.

Uses existing ARTN checkpoint. No retraining.
Output: gen_quality_eval.json
"""
import sys, json, time, os, math
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = torch.device('cpu')
BASE = '/root/agent-sandbox/ARTN'

# Load model
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
print(f"  Loaded: step={ckpt['step']}, params={sum(p.numel() for p in model.parameters()):,}")
print()

# 10 prompts: 7 from interactive tests + 3 new harder ones
PROMPTS = [
    "once upon a time there was a dog",
    "there was a little girl",
    "there lived a",
    "a king",
    "king",
    "a bison from wild forest",
    "a peacock",
    # New harder prompts
    "The scientist mixed two chemicals",
    "In the year 3024, humanity",
    "Why did the chicken cross",
]

MAX_NEW_TOKENS = 50
SEED = 42

# Decoding strategies
STRATEGIES = [
    {"name": "greedy", "temp": 1.0, "top_k": 0, "rep_penalty": 1.0},
    {"name": "temp_0.7", "temp": 0.7, "top_k": 0, "rep_penalty": 1.0},
    {"name": "temp_0.9", "temp": 0.9, "top_k": 0, "rep_penalty": 1.0},
    {"name": "topk_40", "temp": 0.9, "top_k": 40, "rep_penalty": 1.0},
    {"name": "topk_10", "temp": 0.9, "top_k": 10, "rep_penalty": 1.0},
    {"name": "greedy_rep_penalty_1.2", "temp": 1.0, "top_k": 0, "rep_penalty": 1.2},
]

THRESHOLDS = [1.0, 0.9]


def compute_repetition_stats(text):
    """Compute repetition metrics for generated text (excluding prompt)."""
    words = text.split()
    if len(words) < 4:
        return {"repetition_rate": 0, "max_consecutive_bigram_repeat": 0, "unique_word_ratio": 1.0}

    # Bigram repetition: how many bigrams appear more than once
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    bigram_counts = {}
    for bg in bigrams:
        bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
    repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
    repetition_rate = repeated_bigrams / max(len(bigram_counts), 1)

    # Max consecutive repeated bigram
    max_repeat = 0
    cur = 1
    for i in range(1, len(bigrams)):
        if bigrams[i] == bigrams[i-1]:
            cur += 1
            max_repeat = max(max_repeat, cur)
        else:
            cur = 1

    # Unique word ratio
    unique_ratio = len(set(words)) / max(len(words), 1)

    return {
        "repetition_rate": round(repetition_rate, 3),
        "max_consecutive_bigram_repeat": max_repeat,
        "unique_word_ratio": round(unique_ratio, 3),
    }


def generate(prompt, max_new_tokens, threshold, strategy):
    """Generate text with specified decoding strategy and halting threshold."""
    torch.manual_seed(SEED)
    input_ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
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

            # Apply repetition penalty
            if strategy["rep_penalty"] != 1.0:
                for prev_token in input_ids[0].tolist():
                    logits[0, prev_token] /= strategy["rep_penalty"]

            # Apply temperature
            logits = logits / strategy["temp"]

            # Apply top-k
            if strategy["top_k"] > 0:
                top_k = min(strategy["top_k"], logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                min_val = values[0, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.full_like(logits, float('-inf')),
                    logits
                )

            # Greedy or sample
            if strategy["temp"] == 1.0 and strategy["top_k"] == 0 and strategy["rep_penalty"] == 1.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            passes_log.append(passes_this)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    elapsed = time.time() - t0
    full_text = tok.decode(input_ids[0].tolist())
    generated_text = full_text[len(prompt):].strip()
    return {
        "full_text": full_text,
        "generated_text": generated_text,
        "passes": passes_log,
        "avg_passes": round(sum(passes_log) / len(passes_log), 4) if passes_log else 0,
        "elapsed_s": round(elapsed, 2),
        "tokens_per_sec": round(max_new_tokens / elapsed, 2) if elapsed > 0 else 0,
        "repetition_stats": compute_repetition_stats(generated_text),
    }


# Run all experiments
print("=" * 70)
print("EXPERIMENT 1: Decoding Strategy Comparison")
print("=" * 70)

all_results = {
    "experiment": "decoding_strategy_comparison",
    "checkpoint": f"{BASE}/outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt",
    "model_params": sum(p.numel() for p in model.parameters()),
    "max_new_tokens": MAX_NEW_TOKENS,
    "seed": SEED,
    "prompts": PROMPTS,
    "strategies": [s["name"] for s in STRATEGIES],
    "thresholds": THRESHOLDS,
    "results": {},
}

total_runs = len(PROMPTS) * len(STRATEGIES) * len(THRESHOLDS)
run_num = 0

for threshold in THRESHOLDS:
    t_key = f"threshold_{threshold}"
    all_results["results"][t_key] = {}

    for strategy in STRATEGIES:
        s_name = strategy["name"]
        all_results["results"][t_key][s_name] = []
        print(f"\n--- Threshold={threshold}, Strategy={s_name} ---")

        for prompt in PROMPTS:
            run_num += 1
            result = generate(prompt, MAX_NEW_TOKENS, threshold, strategy)
            all_results["results"][t_key][s_name].append({
                "prompt": prompt,
                **result,
            })
            rep = result["repetition_stats"]
            print(f"  [{run_num}/{total_runs}] '{prompt[:35]:<35}' "
                  f"passes={result['avg_passes']:.2f} "
                  f"rep_rate={rep['repetition_rate']:.3f} "
                  f"max_repeat={rep['max_consecutive_bigram_repeat']} "
                  f"unique={rep['unique_word_ratio']:.3f}")

# Save results
output_path = f'{BASE}/outputs/artn_recursive_20261025_002725/gen_quality_eval.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n\nResults saved to: {output_path}")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY: Average repetition rate by strategy and threshold")
print("=" * 70)
print(f"\n{'Strategy':<25} {'t=1.0 rep_rate':<15} {'t=0.9 rep_rate':<15} {'t=1.0 unique':<13} {'t=0.9 unique':<13}")
print("-" * 81)
for strategy in STRATEGIES:
    s_name = strategy["name"]
    t1_reps = [r["repetition_stats"]["repetition_rate"] for r in all_results["results"]["threshold_1.0"][s_name]]
    t09_reps = [r["repetition_stats"]["repetition_rate"] for r in all_results["results"]["threshold_0.9"][s_name]]
    t1_uniq = [r["repetition_stats"]["unique_word_ratio"] for r in all_results["results"]["threshold_1.0"][s_name]]
    t09_uniq = [r["repetition_stats"]["unique_word_ratio"] for r in all_results["results"]["threshold_0.9"][s_name]]
    print(f"  {s_name:<23} {sum(t1_reps)/len(t1_reps):<15.3f} {sum(t09_reps)/len(t09_reps):<15.3f} {sum(t1_uniq)/len(t1_uniq):<13.3f} {sum(t09_uniq)/len(t09_uniq):<13.3f}")

print("\nDONE")
