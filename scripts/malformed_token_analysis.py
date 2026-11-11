#!/usr/bin/env python3
"""
Investigation 2: Token-Level Analysis of Malformed Words

At positions where malformed words appear, capture:
- Top-5 token IDs, decoded strings, probabilities
- Entropy of the full distribution
- Top-two probability margin
- Predictions across all 6 recursive passes (per-pass logits)

Output: malformed_token_analysis.json
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

# Prompts known to produce malformed words
PROMPTS = [
    "a peacock",
    "a king",
    "king",
]

MAX_NEW_TOKENS = 50
SEED = 42


def forward_all_passes(input_ids, threshold):
    """
    Forward pass that logs per-pass logits at the last position.
    Returns: final logits, passes used, per_pass_logits list
    """
    B, T = input_ids.shape
    tok_emb = model.token_embedding(input_ids)
    pos_emb = model.position_embedding(torch.arange(T, device=DEVICE))
    x = tok_emb + pos_emb

    remaining = torch.ones(B, T, 1, device=DEVICE)
    output = torch.zeros(B, T, model.d_model, device=DEVICE)
    passes_this = 0
    per_pass_logits = []

    for pass_idx in range(model.max_passes):
        passes_this += 1
        x = model.block(x)

        # Log per-pass logits at the LAST position
        x_ln_temp = model.ln_f(output + x * remaining)  # current accumulated output
        pass_logits = model.lm_head(x_ln_temp[:, -1, :])
        per_pass_logits.append(pass_logits.squeeze(0).clone())

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
    return logits[:, -1, :], passes_this, per_pass_logits


def compute_entropy(logits):
    """Shannon entropy of softmax distribution (in bits)."""
    probs = F.softmax(logits, dim=-1)
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log2(probs)).item()
    return entropy


def get_top5(logits):
    """Top-5 token IDs, strings, and probabilities."""
    if logits.dim() > 1:
        logits = logits.squeeze(0)
    probs = F.softmax(logits, dim=-1)
    top5_vals, top5_ids = torch.topk(logits, 5)
    top5_probs = probs[top5_ids]
    results = []
    for i in range(5):
        tid = top5_ids[i].item()
        results.append({
            "token_id": tid,
            "token": tok.decode([tid]),
            "prob": round(top5_probs[i].item(), 6),
        })
    return results


def get_margin(logits):
    """Top-two probability margin."""
    if logits.dim() > 1:
        logits = logits.squeeze(0)
    probs = F.softmax(logits, dim=-1)
    top2 = torch.topk(probs, 2).values
    return round((top2[0] - top2[1]).item(), 6)


def is_malformed(token_str):
    """Heuristic: check if a decoded token looks like a malformed word fragment."""
    # Clean the token (remove BPE end-of-word marker)
    clean = token_str.replace('</w>', '').strip()
    # Known good patterns: common English words, common suffixes
    # Malformed: contains unusual consonant clusters like 'cyck', 'yck', 'cyc'
    # We'll flag tokens that are not common English words
    # For this analysis, we'll just flag everything and let the analysis show the distribution
    return False  # We'll mark all positions and inspect manually


def generate_with_analysis(prompt, max_new_tokens, threshold):
    """Generate with full per-token analysis."""
    torch.manual_seed(SEED)
    input_ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    token_analysis = []

    with torch.no_grad():
        for gen_step in range(max_new_tokens):
            T = input_ids.shape[1]
            if T >= model.position_embedding.num_embeddings:
                break

            logits, passes_used, per_pass_logits = forward_all_passes(input_ids, threshold)

            # Final token
            next_token_id = logits.argmax().item()
            next_token_str = tok.decode([next_token_id])

            # Final distribution analysis
            entropy = compute_entropy(logits)
            top5 = get_top5(logits)
            margin = get_margin(logits)

            # Per-pass analysis
            pass_analysis = []
            for p_idx, p_logits in enumerate(per_pass_logits):
                p_entropy = compute_entropy(p_logits)
                p_top5 = get_top5(p_logits)
                p_margin = get_margin(p_logits)
                p_argmax_id = p_logits.argmax().item()
                pass_analysis.append({
                    "pass": p_idx + 1,
                    "argmax_token_id": p_argmax_id,
                    "argmax_token": tok.decode([p_argmax_id]),
                    "entropy_bits": round(p_entropy, 4),
                    "margin": p_margin,
                    "top5": p_top5,
                })

            token_analysis.append({
                "gen_step": gen_step,
                "context_length": T,
                "passes_used": passes_used,
                "next_token_id": next_token_id,
                "next_token": next_token_str,
                "entropy_bits": round(entropy, 4),
                "margin": margin,
                "top5": top5,
                "per_pass": pass_analysis,
            })

            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=DEVICE)], dim=1)

    full_text = tok.decode(input_ids[0].tolist())
    generated_text = full_text[len(prompt):].strip()
    return {
        "prompt": prompt,
        "threshold": threshold,
        "full_text": full_text,
        "generated_text": generated_text,
        "num_tokens": len(token_analysis),
        "token_analysis": token_analysis,
    }


# ============================================================
# Run analysis
# ============================================================
print("=" * 70)
print("INVESTIGATION 2: Token-Level Malformed Word Analysis")
print("=" * 70)

all_results = {
    "experiment": "malformed_token_analysis",
    "checkpoint": f"{BASE}/outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt",
    "max_new_tokens": MAX_NEW_TOKENS,
    "seed": SEED,
    "vocab_size": cfg['vocab_size'],
    "results": {},
}

for prompt in PROMPTS:
    all_results["results"][prompt] = {}
    for threshold in [1.0, 0.9]:
        print(f"\n--- Prompt: '{prompt}', threshold={threshold} ---")
        result = generate_with_analysis(prompt, MAX_NEW_TOKENS, threshold)
        all_results["results"][prompt][f"threshold_{threshold}"] = result

        # Print key positions
        print(f"  Generated: {result['generated_text'][:120]}")
        print(f"  Token-by-token (first 15):")
        for ta in result["token_analysis"][:15]:
            t = ta["next_token"].replace('</w>', '')
            print(f"    step {ta['gen_step']:>2} tok='{t}' ent={ta['entropy_bits']:.2f} margin={ta['margin']:.4f} passes={ta['passes_used']} top1_prob={ta['top5'][0]['prob']:.4f}")

# Save
output_path = f'{BASE}/outputs/artn_recursive_20261025_002725/malformed_token_analysis.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n\nResults saved to: {output_path}")

# Summary: entropy and margin at each position
print("\n" + "=" * 70)
print("SUMMARY: Entropy and Margin at Each Position")
print("=" * 70)
for prompt in PROMPTS:
    for threshold in [1.0, 0.9]:
        key = f"threshold_{threshold}"
        if key in all_results["results"][prompt]:
            r = all_results["results"][prompt][key]
            print(f"\n  Prompt: '{prompt}' threshold={threshold}")
            print(f"  {'Step':<5} {'Token':<10} {'Entropy':<9} {'Margin':<8} {'Passes':<7} {'Top1 prob':<9} {'Top2 prob':<9}")
            print("  " + "-" * 65)
            for ta in r["token_analysis"][:20]:
                t = ta["next_token"].replace('</w>', '')
                p1 = ta["top5"][0]["prob"]
                p2 = ta["top5"][1]["prob"] if len(ta["top5"]) > 1 else 0
                print(f"  {ta['gen_step']:<5} {t:<10} {ta['entropy_bits']:<9} {ta['margin']:<8} {ta['passes_used']:<7} {p1:<9} {p2}")

print("\nDONE")
