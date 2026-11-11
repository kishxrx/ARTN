#!/usr/bin/env python3
"""
Experiment 2: Halt Probability Analysis

Logs per-position halt probabilities at each pass (1-6) at t=1.0.
Compares halt distributions for common vs uncommon prompts.

Uses existing ARTN checkpoint. No retraining.
Output: halt_analysis.json
"""
import sys, json, os
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
print(f"  Loaded: step={ckpt['step']}")
print()

# 7 interactive prompts — classified as common/uncommon based on test results
PROMPTS = [
    {"prompt": "once upon a time there was a dog", "category": "common", "diverged": False},
    {"prompt": "there was a little girl", "category": "common", "diverged": False},
    {"prompt": "there lived a", "category": "common", "diverged": False},
    {"prompt": "a king", "category": "uncommon", "diverged": True},
    {"prompt": "king", "category": "uncommon", "diverged": True},
    {"prompt": "a bison from wild forest", "category": "common", "diverged": False},
    {"prompt": "a peacock", "category": "uncommon", "diverged": True},
]

MAX_GEN_TOKENS = 50


def analyze_halt(prompt, max_gen_tokens=50):
    """Generate with full 6 passes, logging halt probabilities per position per pass."""
    input_ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    halt_log = []  # list of {token_pos, pass, halt_prob, cumulative_halt, remaining}

    with torch.no_grad():
        for gen_step in range(max_gen_tokens):
            B, T = input_ids.shape
            tok_emb = model.token_embedding(input_ids)
            pos_emb = model.position_embedding(torch.arange(T, device=DEVICE))
            x = tok_emb + pos_emb

            remaining = torch.ones(B, T, 1, device=DEVICE)
            output = torch.zeros(B, T, model.d_model, device=DEVICE)

            # Run all 6 passes, log halt for last position
            last_pos_halts = []
            for pass_idx in range(model.max_passes):
                x = model.block(x)
                h = torch.sigmoid(model.halting_gate(x))  # (B, T, 1)
                p = h * remaining
                if pass_idx == model.max_passes - 1:
                    p = remaining

                # Log last position halt info
                last_h = h[0, -1, 0].item()
                last_p = p[0, -1, 0].item()
                last_remaining = remaining[0, -1, 0].item()
                cumulative = 1.0 - last_remaining
                last_pos_halts.append({
                    "pass": pass_idx + 1,
                    "halt_prob": round(last_h, 4),
                    "halt_weight": round(last_p, 4),
                    "remaining": round(last_remaining, 4),
                    "cumulative_halt": round(cumulative, 4),
                })

                output = output + x * p
                remaining = remaining - p

            output = output + x * remaining
            x = model.ln_f(output)
            logits = model.lm_head(x)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            halt_log.append({
                "gen_step": gen_step,
                "context_length": T,
                "last_pos_halts": last_pos_halts,
            })
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return halt_log


# Run analysis
print("=" * 70)
print("EXPERIMENT 2: Halt Probability Analysis")
print("=" * 70)

all_results = {
    "experiment": "halt_probability_analysis",
    "checkpoint": f"{BASE}/outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt",
    "max_gen_tokens": MAX_GEN_TOKENS,
    "prompts": PROMPTS,
    "results": {},
}

for prompt_info in PROMPTS:
    prompt = prompt_info["prompt"]
    category = prompt_info["category"]
    print(f"\n--- '{prompt}' (category: {category}) ---")

    halt_log = analyze_halt(prompt, MAX_GEN_TOKENS)
    all_results["results"][prompt] = {
        "category": category,
        "diverged_at_t09": prompt_info["diverged"],
        "halt_log": halt_log,
    }

    # Summarize: at which pass does cumulative halt cross 0.9?
    pass_at_09 = []
    pass_at_05 = []
    for step in halt_log:
        for p_info in step["last_pos_halts"]:
            if p_info["cumulative_halt"] >= 0.9:
                pass_at_09.append(p_info["pass"])
                break
        for p_info in step["last_pos_halts"]:
            if p_info["cumulative_halt"] >= 0.5:
                pass_at_05.append(p_info["pass"])
                break

    avg_pass_09 = sum(pass_at_09) / len(pass_at_09) if pass_at_09 else 6.0
    avg_pass_05 = sum(pass_at_05) / len(pass_at_05) if pass_at_05 else 6.0

    # Halt probability at pass 1 (how confident is early halt)
    pass1_halt_probs = [s["last_pos_halts"][0]["halt_prob"] for s in halt_log]
    avg_pass1_halt = sum(pass1_halt_probs) / len(pass1_halt_probs) if pass1_halt_probs else 0

    print(f"  Avg pass to reach 0.9 cumulative: {avg_pass_09:.2f}")
    print(f"  Avg pass to reach 0.5 cumulative: {avg_pass_05:.2f}")
    print(f"  Avg halt prob at pass 1: {avg_pass1_halt:.4f}")
    print(f"  Pass-1 halt distribution: min={min(pass1_halt_probs):.3f} max={max(pass1_halt_probs):.3f}")

    all_results["results"][prompt]["summary"] = {
        "avg_pass_to_reach_0.9": round(avg_pass_09, 2),
        "avg_pass_to_reach_0.5": round(avg_pass_05, 2),
        "avg_halt_prob_pass1": round(avg_pass1_halt, 4),
        "pass1_halt_min": round(min(pass1_halt_probs), 4),
        "pass1_halt_max": round(max(pass1_halt_probs), 4),
    }

# Save
output_path = f'{BASE}/outputs/artn_recursive_20261025_002725/halt_analysis.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n\nResults saved to: {output_path}")

# Summary comparison
print("\n" + "=" * 70)
print("SUMMARY: Common vs Uncommon prompts")
print("=" * 70)
print(f"\n{'Prompt':<35} {'Category':<10} {'Pass→0.9':<10} {'Pass→0.5':<10} {'Halt@p1':<10} {'Diverged?'}")
print("-" * 85)
for prompt_info in PROMPTS:
    p = prompt_info["prompt"]
    s = all_results["results"][p]["summary"]
    print(f"  {p:<33} {prompt_info['category']:<10} {s['avg_pass_to_reach_0.9']:<10} {s['avg_pass_to_reach_0.5']:<10} {s['avg_halt_prob_pass1']:<10} {prompt_info['diverged']}")

# Aggregate
common = [all_results["results"][p["prompt"]]["summary"] for p in PROMPTS if p["category"] == "common"]
uncommon = [all_results["results"][p["prompt"]]["summary"] for p in PROMPTS if p["category"] == "uncommon"]
print(f"\n  Common avg:    pass→0.9={sum(s['avg_pass_to_reach_0.9'] for s in common)/len(common):.2f}  "
      f"pass→0.5={sum(s['avg_pass_to_reach_0.5'] for s in common)/len(common):.2f}  "
      f"halt@p1={sum(s['avg_halt_prob_pass1'] for s in common)/len(common):.4f}")
print(f"  Uncommon avg:  pass→0.9={sum(s['avg_pass_to_reach_0.9'] for s in uncommon)/len(uncommon):.2f}  "
      f"pass→0.5={sum(s['avg_pass_to_reach_0.5'] for s in uncommon)/len(uncommon):.2f}  "
      f"halt@p1={sum(s['avg_halt_prob_pass1'] for s in uncommon)/len(uncommon):.4f}")

print("\nDONE")
