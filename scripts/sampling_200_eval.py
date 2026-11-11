#!/usr/bin/env python3
"""
Investigation 1: 200-Token Sampling vs Greedy Comparison

Compare temperature 0.9 sampling with greedy decoding over 200 tokens.
Identical prompts and thresholds (1.0 and 0.9) as long_gen_test.
Preserve raw outputs; evaluate repetition and narrative coherence.

Output: sampling_200_eval.json
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
print(f"  Loaded: step={ckpt['step']}, params={sum(p.numel() for p in model.parameters()):,}")
print()

PROMPTS = [
    "once upon a time there was a dog",
    "there was a little girl",
    "a king",
    "a peacock",
    "The scientist mixed two chemicals",
]

MAX_NEW_TOKENS = 200
SEED = 42
TEMP = 0.9


def forward_with_halt(input_ids, threshold):
    """Single forward pass with halting; returns logits and passes used."""
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


def generate(prompt, max_new_tokens, threshold, mode="greedy", temp=0.9):
    """Generate text. mode='greedy' or 'sample'."""
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

            if mode == "greedy":
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temp
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

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
        "passes_log": passes_log,
    }


def compute_repetition_stats(text):
    """Repetition metrics for generated text."""
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


def compute_coherence_score(text):
    """
    Simple heuristic coherence metrics:
    - avg sentence length (words per sentence)
    - sentence count
    - fraction of complete sentences (end with . ! ?)
    """
    import re
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return {"avg_sentence_len": 0, "sentence_count": 0, "complete_fraction": 0}
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    # Count how many sentences end with punctuation
    punct_count = len(re.findall(r'[.!?]', text))
    word_count = len(text.split())
    return {
        "avg_sentence_len": round(avg_len, 1),
        "sentence_count": len(sentences),
        "complete_fraction": round(punct_count / max(len(sentences), 1), 3),
    }


# ============================================================
# Run experiments
# ============================================================
print("=" * 70)
print("INVESTIGATION 1: 200-Token Sampling vs Greedy")
print("=" * 70)

all_results = {
    "experiment": "sampling_vs_greedy_200",
    "checkpoint": f"{BASE}/outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt",
    "max_new_tokens": MAX_NEW_TOKENS,
    "seed": SEED,
    "sampling_temperature": TEMP,
    "prompts": PROMPTS,
    "results": {},
}

configs = [
    ("greedy", 1.0), ("sample_temp09", 1.0),
    ("greedy", 0.9), ("sample_temp09", 0.9),
]

total = len(PROMPTS) * len(configs)
run = 0

for prompt in PROMPTS:
    all_results["results"][prompt] = {}
    for mode, threshold in configs:
        run += 1
        label = f"{mode}_t{threshold}"
        print(f"\n  [{run}/{total}] prompt='{prompt[:30]}' mode={mode} threshold={threshold}")

        result = generate(prompt, MAX_NEW_TOKENS, threshold, mode=mode, temp=TEMP)
        result["repetition_stats"] = compute_repetition_stats(result["generated_text"])
        result["coherence_stats"] = compute_coherence_score(result["generated_text"])
        result["mode"] = mode
        result["threshold"] = threshold

        all_results["results"][prompt][label] = result
        rep = result["repetition_stats"]
        coh = result["coherence_stats"]
        print(f"    tokens={result['num_tokens']} passes={result['avg_passes']:.2f} "
              f"rep_rate={rep['repetition_rate']:.3f} unique={rep['unique_word_ratio']:.3f} "
              f"max_repeat={rep['max_consecutive_bigram_repeat']} "
              f"sent_len={coh['avg_sentence_len']} sents={coh['sentence_count']}")

# Save
output_path = f'{BASE}/outputs/artn_recursive_20261025_002725/sampling_200_eval.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n\nResults saved to: {output_path}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: Greedy vs Sampling (200 tokens)")
print("=" * 70)
print(f"\n{'Prompt':<30} {'Mode':<16} {'Rep':<6} {'Uniq':<6} {'MaxRpt':<7} {'AvgSent':<8} {'Passes':<7}")
print("-" * 85)
for prompt in PROMPTS:
    for label in ["greedy_t1.0", "sample_temp09_t1.0", "greedy_t0.9", "sample_temp09_t0.9"]:
        if label in all_results["results"][prompt]:
            r = all_results["results"][prompt][label]
            rep = r["repetition_stats"]
            coh = r["coherence_stats"]
            mode_str = label.replace("_t", " t=")
            print(f"  {prompt[:28]:<30} {mode_str:<16} {rep['repetition_rate']:<6} "
                  f"{rep['unique_word_ratio']:<6} {rep['max_consecutive_bigram_repeat']:<7} "
                  f"{coh['avg_sentence_len']:<8} {r['avg_passes']:<7}")

print("\nDONE")
