#!/usr/bin/env python3
"""
Experiment 3: Long Generation Test

Generates 200 tokens for 5 prompts at t=1.0 and t=0.9 (greedy).
Measures where outputs diverge, repetition rate over time, and quality degradation.

Uses existing ARTN checkpoint. No retraining.
Output: long_gen_test.json
"""
import sys, json, time, os
import torch

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
print(f"  Loaded: step={ckpt['step']}, max_seq_len={cfg['max_seq_len']}")
print()

# 5 prompts: mix of common and uncommon
PROMPTS = [
    "once upon a time there was a dog",
    "there was a little girl",
    "a king",
    "a peacock",
    "The scientist mixed two chemicals",
]

MAX_NEW_TOKENS = 200  # up to max_seq_len=256
SEED = 42


def generate_long(prompt, max_new_tokens, threshold):
    """Generate up to max_new_tokens, logging per-token info."""
    torch.manual_seed(SEED)
    input_ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    token_info = []
    t0 = time.time()

    with torch.no_grad():
        for gen_step in range(max_new_tokens):
            B, T = input_ids.shape

            # Safety: stop if we exceed max_seq_len
            if T >= model.position_embedding.num_embeddings:
                break

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
            next_token = logits.argmax(dim=-1, keepdim=True)

            # Log top-5 token probabilities
            top5_vals, top5_ids = torch.topk(logits[0], 5)
            top5_probs = torch.softmax(top5_vals, dim=-1)

            token_info.append({
                "step": gen_step,
                "context_len": T,
                "passes": passes_this,
                "token_id": next_token.item(),
                "token": tok.decode([next_token.item()]),
                "top5": [{"id": top5_ids[i].item(), "token": tok.decode([top5_ids[i].item()]),
                          "prob": round(top5_probs[i].item(), 4)} for i in range(5)],
            })
            input_ids = torch.cat([input_ids, next_token], dim=1)

    elapsed = time.time() - t0
    full_text = tok.decode(input_ids[0].tolist())
    generated_text = full_text[len(prompt):].strip()
    return {
        "full_text": full_text,
        "generated_text": generated_text,
        "num_tokens": len(token_info),
        "elapsed_s": round(elapsed, 2),
        "tokens_per_sec": round(len(token_info) / elapsed, 2) if elapsed > 0 else 0,
        "token_info": token_info,
    }


def measure_repetition_windows(text, window=20):
    """Measure repetition rate in sliding windows of 20 words."""
    words = text.split()
    if len(words) < window:
        return [{"start": 0, "rep_rate": 0, "unique_ratio": 1.0}]

    results = []
    for i in range(0, len(words) - window + 1, window):
        chunk = words[i:i+window]
        bigrams = [(chunk[j], chunk[j+1]) for j in range(len(chunk)-1)]
        bigram_counts = {}
        for bg in bigrams:
            bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
        repeated = sum(1 for c in bigram_counts.values() if c > 1)
        rep_rate = repeated / max(len(bigram_counts), 1)
        unique = len(set(chunk)) / max(len(chunk), 1)
        results.append({
            "start_word": i,
            "end_word": i + window,
            "rep_rate": round(rep_rate, 3),
            "unique_ratio": round(unique, 3),
        })
    return results


def find_first_divergence(text_a, text_b):
    """Find first word where two texts diverge."""
    words_a = text_a.split()
    words_b = text_b.split()
    for i in range(min(len(words_a), len(words_b))):
        if words_a[i] != words_b[i]:
            return i
    if len(words_a) != len(words_b):
        return min(len(words_a), len(words_b))
    return -1  # identical


# Run experiments
print("=" * 70)
print("EXPERIMENT 3: Long Generation Test (200 tokens)")
print("=" * 70)

all_results = {
    "experiment": "long_generation_test",
    "checkpoint": f"{BASE}/outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt",
    "max_new_tokens": MAX_NEW_TOKENS,
    "seed": SEED,
    "max_seq_len": cfg['max_seq_len'],
    "results": {},
}

for prompt in PROMPTS:
    print(f"\n--- Prompt: '{prompt}' ---")

    t1_result = generate_long(prompt, MAX_NEW_TOKENS, 1.0)
    t09_result = generate_long(prompt, MAX_NEW_TOKENS, 0.9)

    # Find divergence point
    div_word = find_first_divergence(t1_result["generated_text"], t09_result["generated_text"])

    # Repetition windows
    t1_rep_windows = measure_repetition_windows(t1_result["generated_text"])
    t09_rep_windows = measure_repetition_windows(t09_result["generated_text"])

    # Average passes
    t1_passes = [t["passes"] for t in t1_result["token_info"]]
    t09_passes = [t["passes"] for t in t09_result["token_info"]]

    all_results["results"][prompt] = {
        "t_1.0": {
            "generated_text": t1_result["generated_text"],
            "num_tokens": t1_result["num_tokens"],
            "avg_passes": round(sum(t1_passes) / len(t1_passes), 2) if t1_passes else 0,
            "tokens_per_sec": t1_result["tokens_per_sec"],
            "repetition_windows": t1_rep_windows,
            "token_info": t1_result["token_info"],
        },
        "t_0.9": {
            "generated_text": t09_result["generated_text"],
            "num_tokens": t09_result["num_tokens"],
            "avg_passes": round(sum(t09_passes) / len(t09_passes), 2) if t09_passes else 0,
            "tokens_per_sec": t09_result["tokens_per_sec"],
            "repetition_windows": t09_rep_windows,
            "token_info": t09_result["token_info"],
        },
        "divergence_word": div_word,
        "identical": div_word == -1,
    }

    print(f"  t=1.0: {t1_result['num_tokens']} tokens, avg_passes={sum(t1_passes)/len(t1_passes):.2f}, "
          f"speed={t1_result['tokens_per_sec']:.2f} tok/s")
    print(f"  t=0.9: {t09_result['num_tokens']} tokens, avg_passes={sum(t09_passes)/len(t09_passes):.2f}, "
          f"speed={t09_result['tokens_per_sec']:.2f} tok/s")
    print(f"  Divergence at word: {div_word if div_word >= 0 else 'identical'}")
    print(f"  t=1.0 text (first 200 chars): {t1_result['generated_text'][:200]}")
    print(f"  t=0.9 text (first 200 chars): {t09_result['generated_text'][:200]}")

# Save
output_path = f'{BASE}/outputs/artn_recursive_20261025_002725/long_gen_test.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n\nResults saved to: {output_path}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Prompt':<35} {'Div@word':<10} {'t=1.0 passes':<12} {'t=0.9 passes':<12} {'t=1.0 rep(first20)':<18} {'t=0.9 rep(first20)'}")
print("-" * 110)
for prompt in PROMPTS:
    r = all_results["results"][prompt]
    t1_rep = r["t_1.0"]["repetition_windows"][0]["rep_rate"] if r["t_1.0"]["repetition_windows"] else 0
    t09_rep = r["t_0.9"]["repetition_windows"][0]["rep_rate"] if r["t_0.9"]["repetition_windows"] else 0
    div = r["divergence_word"] if r["divergence_word"] >= 0 else "identical"
    print(f"  {prompt:<33} {str(div):<10} {r['t_1.0']['avg_passes']:<12} {r['t_0.9']['avg_passes']:<12} {t1_rep:<18} {t09_rep}")

print("\nDONE")
