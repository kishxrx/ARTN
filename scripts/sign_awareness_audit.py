#!/usr/bin/env python3
"""
Sign-awareness audit: check whether the divide-only rep penalty in rep_penalty_sweep.py
differs from the standard HuggingFace sign-aware penalty for any token selection
on the 10 test prompts at penalty 1.2 and 1.3, threshold 1.0 and 0.9.

For each generation step, we:
  1. Compute raw logits
  2. Get the set of previously-seen token IDs (prompt + generated so far)
  3. Apply BOTH penalties (divide-only vs sign-aware) and check if argmax differs
  4. Count cases where the wrong sign handling would change the selected token

Output: sign_awareness_audit.json
"""
import sys, json, os
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
print(f"Model loaded: step={ckpt['step']}, vocab={cfg['vocab_size']}")

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


def apply_divide_only(logits, prev_token_ids, penalty):
    """rep_penalty_sweep.py implementation: always divide."""
    penalized = logits.clone()
    for tid in set(prev_token_ids):
        penalized[0, tid] = penalized[0, tid] / penalty
    return penalized


def apply_sign_aware(logits, prev_token_ids, penalty):
    """HuggingFace implementation: multiply if negative, divide if positive."""
    penalized = logits.clone()
    for tid in set(prev_token_ids):
        val = penalized[0, tid].item()
        if val < 0:
            penalized[0, tid] = val * penalty
        else:
            penalized[0, tid] = val / penalty
    return penalized


def generate_with_audit(prompt, threshold, penalty):
    """Generate with BOTH penalties, track where they disagree."""
    torch.manual_seed(SEED)
    input_ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    audit_log = []
    diverged = False
    divergence_step = -1

    with torch.no_grad():
        for gen_step in range(MAX_NEW_TOKENS):
            T = input_ids.shape[1]
            if T >= model.position_embedding.num_embeddings:
                break

            logits, passes = forward_with_halt(input_ids, threshold)
            prev_tokens = input_ids[0].tolist()
            prev_token_set = set(prev_tokens)

            # Apply both penalties
            logits_divide = apply_divide_only(logits, prev_tokens, penalty)
            logits_sign = apply_sign_aware(logits, prev_tokens, penalty)

            # Get argmax for each
            tok_divide = logits_divide.argmax(dim=-1).item()
            tok_sign = logits_sign.argmax(dim=-1).item()

            # Check if previously-seen tokens have negative raw logits
            neg_logit_tokens = []
            for tid in prev_token_set:
                val = logits[0, tid].item()
                if val < 0:
                    neg_logit_tokens.append((tid, tok.decode([tid]), round(val, 4)))

            # Check disagreement
            disagreement = tok_divide != tok_sign

            if disagreement and not diverged:
                diverged = True
                divergence_step = gen_step

            audit_log.append({
                "gen_step": gen_step,
                "context_len": T,
                "passes": passes,
                "neg_logit_prev_tokens": neg_logit_tokens,
                "num_neg_logits": len(neg_logit_tokens),
                "divide_argmax_id": tok_divide,
                "divide_argmax_tok": tok.decode([tok_divide]),
                "sign_aware_argmax_id": tok_sign,
                "sign_aware_argmax_tok": tok.decode([tok_sign]),
                "disagreement": disagreement,
            })

            # Use the DIVIDE-ONLY result (matching what the sweep actually did)
            next_token_id = tok_divide
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=DEVICE)], dim=1)

    return {
        "audit_log": audit_log,
        "diverged": diverged,
        "divergence_step": divergence_step,
    }


# ============================================================
# Run audit
# ============================================================
print("\n" + "=" * 70)
print("SIGN-AWARENESS AUDIT")
print("=" * 70)

all_results = {
    "experiment": "sign_awareness_audit",
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
            result = generate_with_audit(prompt, threshold, penalty)
            all_results["results"][t_key][p_key].append({
                "prompt": prompt,
                **result,
            })

            # Count disagreements
            disagreements = sum(1 for a in result["audit_log"] if a["disagreement"])
            total_neg = sum(a["num_neg_logits"] for a in result["audit_log"])
            print(f"  [{run}/{total}] '{prompt[:35]:<35}' disagreements={disagreements}/{len(result['audit_log'])} "
                  f"neg_logit_steps={total_neg} diverged={result['diverged']} "
                  f"div_step={result['divergence_step']}")

# Save
output_path = f'{BASE}/outputs/artn_recursive_20261025_002725/sign_awareness_audit.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n\nResults saved to: {output_path}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

total_disagree = 0
total_steps = 0
total_neg_steps = 0

for threshold in THRESHOLDS:
    for penalty in PENALTIES:
        for prompt_result in all_results["results"][f"threshold_{threshold}"][f"penalty_{penalty}"]:
            for step in prompt_result["audit_log"]:
                total_steps += 1
                if step["disagreement"]:
                    total_disagree += 1
                if step["num_neg_logits"] > 0:
                    total_neg_steps += 1

print(f"\nTotal steps audited: {total_steps}")
print(f"Steps with negative prev-token logits: {total_neg_steps} ({total_neg_steps}/{total_steps} = {total_neg_steps/total_steps*100:.1f}%)")
print(f"Steps where divide-only != sign-aware: {total_disagree} ({total_disagree}/{total_steps} = {total_disagree/total_steps*100:.1f}%)")

# Show the actual disagreement cases
print("\n=== Disagreement cases ===")
found = 0
for threshold in THRESHOLDS:
    for penalty in PENALTIES:
        for pr in all_results["results"][f"threshold_{threshold}"][f"penalty_{penalty}"]:
            for step in pr["audit_log"]:
                if step["disagreement"]:
                    found += 1
                    print(f"\n  [{found}] threshold={threshold} penalty={penalty} prompt='{pr['prompt'][:25]}' step={step['gen_step']}")
                    print(f"      divide-only:   {step['divide_argmax_tok']} (id={step['divide_argmax_id']})")
                    print(f"      sign-aware:     {step['sign_aware_argmax_tok']} (id={step['sign_aware_argmax_id']})")
                    print(f"      neg prev tokens: {step['neg_logit_prev_tokens']}")
                    print(f"      passes: {step['passes']}")

if found == 0:
    print("  NONE — divide-only and sign-aware produce identical token selections for all tested cases.")

print("\nDONE")
