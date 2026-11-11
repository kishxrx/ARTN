#!/usr/bin/env python3
"""
Corrected repetition penalty: sign-aware (HuggingFace standard).
Positive logits: divide by penalty. Negative logits: multiply by penalty.
Apply once per unique previously-encountered token.

Also tests: prompt+generated vs generated-only token sets.

Reruns the 2000-step comparison (10 prompts x 2 penalties x 2 thresholds)
plus a second pass with generated-only token sets.

Output: corrected_rep_penalty_eval.json
"""
import sys, json, time, os
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

BASE = '/root/agent-sandbox/ARTN'
DEVICE = torch.device('cpu')

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
print(f"Model loaded: step={ckpt['step']}")


# ============================================================
# Unit tests for sign-aware penalty
# ============================================================
def test_sign_aware_penalty():
    """Verify positive and negative logit transformations."""
    tests_passed = 0
    tests_total = 0

    # Test 1: positive logit divided
    tests_total += 1
    logits = torch.tensor([[5.0, 3.0, -2.0, 0.0]])
    penalty = 1.3
    prev_tokens = [0]  # logit=5.0 (positive)
    result = apply_sign_aware(logits.clone(), prev_tokens, penalty)
    expected = 5.0 / 1.3  # ~3.846
    actual = result[0, 0].item()
    assert abs(actual - expected) < 1e-6, f"Positive: expected {expected}, got {actual}"
    tests_passed += 1

    # Test 2: negative logit multiplied
    tests_total += 1
    logits = torch.tensor([[5.0, 3.0, -2.0, 0.0]])
    prev_tokens = [2]  # logit=-2.0 (negative)
    result = apply_sign_aware(logits.clone(), prev_tokens, penalty)
    expected = -2.0 * 1.3  # -2.6
    actual = result[0, 2].item()
    assert abs(actual - expected) < 1e-6, f"Negative: expected {expected}, got {actual}"
    tests_passed += 1

    # Test 3: zero logit unchanged (neither positive nor negative)
    tests_total += 1
    logits = torch.tensor([[5.0, 3.0, -2.0, 0.0]])
    prev_tokens = [3]  # logit=0.0
    result = apply_sign_aware(logits.clone(), prev_tokens, penalty)
    # 0.0 is not < 0, so it gets divided: 0.0/1.3 = 0.0 (same result)
    actual = result[0, 3].item()
    assert actual == 0.0, f"Zero: expected 0.0, got {actual}"
    tests_passed += 1

    # Test 4: multiple tokens, mixed signs
    tests_total += 1
    logits = torch.tensor([[4.0, -1.0, 2.0]])
    prev_tokens = [0, 1, 2]
    result = apply_sign_aware(logits.clone(), prev_tokens, penalty)
    assert abs(result[0, 0].item() - 4.0/1.3) < 1e-6, f"Mixed pos: {result[0,0].item()}"
    assert abs(result[0, 1].item() - (-1.0*1.3)) < 1e-6, f"Mixed neg: {result[0,1].item()}"
    assert abs(result[0, 2].item() - 2.0/1.3) < 1e-6, f"Mixed pos2: {result[0,2].item()}"
    tests_passed += 1

    # Test 5: deduplication — same token appears twice, penalty applied once
    tests_total += 1
    logits = torch.tensor([[5.0, 3.0]])
    prev_tokens = [0, 0, 0]  # token 0 three times
    result = apply_sign_aware(logits.clone(), prev_tokens, penalty)
    expected = 5.0 / 1.3  # divided ONCE, not three times
    actual = result[0, 0].item()
    assert abs(actual - expected) < 1e-6, f"Dedup: expected {expected}, got {actual}"
    tests_passed += 1

    # Test 6: penalty=1.0 is a no-op
    tests_total += 1
    logits = torch.tensor([[5.0, -3.0, 0.0]])
    prev_tokens = [0, 1, 2]
    result = apply_sign_aware(logits.clone(), prev_tokens, 1.0)
    assert abs(result[0, 0].item() - 5.0) < 1e-6, f"Noop pos: {result[0,0].item()}"
    assert abs(result[0, 1].item() - (-3.0)) < 1e-6, f"Noop neg: {result[0,1].item()}"
    assert abs(result[0, 2].item() - 0.0) < 1e-6, f"Noop zero: {result[0,2].item()}"
    tests_passed += 1

    print(f"\n  Unit tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def apply_sign_aware(logits, prev_token_ids, penalty):
    """HuggingFace standard: multiply if negative, divide if positive. Once per unique token."""
    penalized = logits.clone()
    for tid in set(prev_token_ids):
        val = penalized[0, tid].item()
        if val < 0:
            penalized[0, tid] = val * penalty
        else:
            penalized[0, tid] = val / penalty
    return penalized


def apply_divide_only(logits, prev_token_ids, penalty):
    """Original rep_penalty_sweep.py: always divide. Once per unique token."""
    penalized = logits.clone()
    for tid in set(prev_token_ids):
        penalized[0, tid] = penalized[0, tid] / penalty
    return penalized


# ============================================================
# Generation
# ============================================================
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
PENALTIES = [1.2, 1.3]
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


def generate(prompt, max_new_tokens, threshold, penalty, penalty_mode, token_scope):
    """
    penalty_mode: 'sign_aware' or 'divide_only'
    token_scope: 'prompt_plus_generated' or 'generated_only'
    """
    torch.manual_seed(SEED)
    input_ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    prompt_len = input_ids.shape[1]
    passes_log = []
    disagreement_log = []  # track if divide-only would have chosen differently

    with torch.no_grad():
        for gen_step in range(max_new_tokens):
            T = input_ids.shape[1]
            if T >= model.position_embedding.num_embeddings:
                break

            logits, passes = forward_with_halt(input_ids, threshold)
            passes_log.append(passes)

            # Determine which tokens to penalize
            if token_scope == 'prompt_plus_generated':
                prev_tokens = input_ids[0].tolist()
            else:  # generated_only
                prev_tokens = input_ids[0, prompt_len:].tolist()

            # Apply penalty
            if penalty == 1.0:
                penalized = logits
            elif penalty_mode == 'sign_aware':
                penalized = apply_sign_aware(logits, prev_tokens, penalty)
            else:  # divide_only
                penalized = apply_divide_only(logits, prev_tokens, penalty)

            # Also compute divide-only for disagreement tracking
            if penalty_mode == 'sign_aware' and penalty != 1.0:
                penalized_divide = apply_divide_only(logits, prev_tokens, penalty)
                divide_argmax = penalized_divide.argmax(dim=-1).item()
                sign_argmax = penalized.argmax(dim=-1).item()
                disagreement = divide_argmax != sign_argmax
            else:
                disagreement = False

            next_token = penalized.argmax(dim=-1, keepdim=True)

            disagreement_log.append({
                "gen_step": gen_step,
                "disagreement": disagreement,
            })

            input_ids = torch.cat([input_ids, next_token], dim=1)

    full_text = tok.decode(input_ids[0].tolist())
    generated_text = full_text[len(prompt):].strip()
    return {
        "full_text": full_text,
        "generated_text": generated_text,
        "num_tokens": len(passes_log),
        "avg_passes": round(sum(passes_log) / len(passes_log), 2) if passes_log else 0,
        "disagreement_log": disagreement_log,
        "disagreements": sum(1 for d in disagreement_log if d["disagreement"]),
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
print("\n" + "=" * 70)
print("CORRECTED REP PENALTY EVALUATION")
print("=" * 70)

# Unit tests first
print("\n--- Unit Tests ---")
if not test_sign_aware_penalty():
    print("UNIT TESTS FAILED. Aborting.")
    sys.exit(1)

all_results = {
    "experiment": "corrected_rep_penalty_eval",
    "checkpoint": f"{BASE}/outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt",
    "max_new_tokens": MAX_NEW_TOKENS,
    "seed": SEED,
    "unit_tests_passed": True,
    "configs": [],
    "results": {},
}

# Configuration matrix
configs = []
for penalty_mode in ['sign_aware', 'divide_only']:
    for token_scope in ['prompt_plus_generated', 'generated_only']:
        for penalty in PENALTIES:
            for threshold in THRESHOLDS:
                config_name = f"{penalty_mode}_{token_scope}_p{penalty}_t{threshold}"
                configs.append({
                    "name": config_name,
                    "penalty_mode": penalty_mode,
                    "token_scope": token_scope,
                    "penalty": penalty,
                    "threshold": threshold,
                })

all_results["configs"] = [c["name"] for c in configs]

total = len(configs) * len(PROMPTS)
run = 0

for config in configs:
    config_name = config["name"]
    all_results["results"][config_name] = []
    print(f"\n--- {config_name} ---")

    for prompt in PROMPTS:
        run += 1
        result = generate(
            prompt, MAX_NEW_TOKENS, config["threshold"], config["penalty"],
            config["penalty_mode"], config["token_scope"]
        )
        result["repetition_stats"] = compute_repetition_stats(result["generated_text"])
        result["prompt"] = prompt
        result["config"] = config_name
        all_results["results"][config_name].append(result)

        rep = result["repetition_stats"]
        print(f"  [{run}/{total}] '{prompt[:35]:<35}' rep={rep['repetition_rate']:.3f} "
              f"uniq={rep['unique_word_ratio']:.3f} "
              f"disagree={result['disagreements']}/{result['num_tokens']}")

# Save
output_path = f'{BASE}/outputs/artn_recursive_20261025_002725/corrected_rep_penalty_eval.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n\nResults saved to: {output_path}")

# ============================================================
# Summary tables
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY 1: Sign-aware vs Divide-only (prompt+generated)")
print("=" * 70)
print(f"\n{'Penalty':<8} {'Thresh':<7} {'Mode':<12} {'Avg rep':<9} {'Avg uniq':<9} {'Disagreements'}")
print("-" * 60)

for penalty in PENALTIES:
    for threshold in THRESHOLDS:
        for mode in ['sign_aware', 'divide_only']:
            config_name = f"{mode}_prompt_plus_generated_p{penalty}_t{threshold}"
            recs = all_results["results"][config_name]
            avg_rep = sum(r["repetition_stats"]["repetition_rate"] for r in recs) / len(recs)
            avg_uniq = sum(r["repetition_stats"]["unique_word_ratio"] for r in recs) / len(recs)
            total_disagree = sum(r["disagreements"] for r in recs)
            print(f"  {penalty:<8} {threshold:<7} {mode:<12} {avg_rep:<9.3f} {avg_uniq:<9.3f} {total_disagree}")

print("\n" + "=" * 70)
print("SUMMARY 2: Prompt+generated vs Generated-only (sign-aware)")
print("=" * 70)
print(f"\n{'Penalty':<8} {'Thresh':<7} {'Scope':<22} {'Avg rep':<9} {'Avg uniq':<9}")
print("-" * 60)

for penalty in PENALTIES:
    for threshold in THRESHOLDS:
        for scope in ['prompt_plus_generated', 'generated_only']:
            config_name = f"sign_aware_{scope}_p{penalty}_t{threshold}"
            recs = all_results["results"][config_name]
            avg_rep = sum(r["repetition_stats"]["repetition_rate"] for r in recs) / len(recs)
            avg_uniq = sum(r["repetition_stats"]["unique_word_ratio"] for r in recs) / len(recs)
            print(f"  {penalty:<8} {threshold:<7} {scope:<22} {avg_rep:<9.3f} {avg_uniq:<9.3f}")

# Show peacock output for key configs
print("\n" + "=" * 70)
print("'a peacock' output comparison")
print("=" * 70)
for penalty in PENALTIES:
    for threshold in THRESHOLDS:
        for scope in ['prompt_plus_generated', 'generated_only']:
            config_name = f"sign_aware_{scope}_p{penalty}_t{threshold}"
            for r in all_results["results"][config_name]:
                if r["prompt"] == "a peacock":
                    print(f"\n  {config_name}:")
                    print(f"    text: {r['generated_text'][:150]}")
                    print(f"    rep={r['repetition_stats']['repetition_rate']} uniq={r['repetition_stats']['unique_word_ratio']}")

print("\nDONE")
