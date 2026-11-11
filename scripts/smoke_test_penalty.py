#!/usr/bin/env python3
"""
Smoke test for the updated artn_interactive_gen.py.
Runs the 7 original interactive-test prompts in both modes:
  1. greedy (unpenalized, original behavior)
  2. penalty (sign-aware rep penalty 1.3, prompt+generated scope)

At both thresholds 1.0 and 0.9, 50 tokens each.
Non-interactive — feeds prompts programmatically, prints outputs.
"""
import sys, torch, time

sys.path.insert(0, '/root/agent-sandbox/ARTN')

# Import the generate function from artn_interactive_gen
# We can't import directly (it has a loop), so we replicate the model loading
# and call the same generate logic

from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = torch.device('cpu')

print("Loading ARTN model...")
ckpt = torch.load(
    'outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt',
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
tok = BPETokenizer.load('outputs/artn-50k-best-config/tokenizer.json')
print(f"  Loaded: step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}")
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
print()


def generate(prompt, max_new_tokens=50, threshold=1.0, rep_penalty=1.0):
    """Same generate as artn_interactive_gen.py — sign-aware rep penalty."""
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

            # Sign-aware repetition penalty (prompt+generated tokens)
            if rep_penalty != 1.0:
                prev_tokens = set(input_ids[0].tolist())
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


# ============================================================
# 7 original prompts
# ============================================================
PROMPTS = [
    "once upon a time there was a dog",
    "there was a little girl",
    "there lived a",
    "a king",
    "king",
    "a bison from wild forest",
    "a peacock",
]

N_TOKENS = 50

print("=" * 70)
print("SMOKE TEST: Greedy vs Penalty 1.3 (sign-aware, prompt+generated)")
print("=" * 70)

for prompt in PROMPTS:
    print(f"\n{'=' * 70}")
    print(f"Prompt: '{prompt}'  ({N_TOKENS} tokens)")
    print(f"{'=' * 70}")

    # Greedy (unpenalized) — original behavior
    g_t1, g_p1, g_t1_time = generate(prompt, N_TOKENS, threshold=1.0, rep_penalty=1.0)
    g_t09, g_p09, g_t09_time = generate(prompt, N_TOKENS, threshold=0.9, rep_penalty=1.0)

    # Penalty 1.3 (sign-aware, prompt+generated)
    p_t1, p_p1, p_t1_time = generate(prompt, N_TOKENS, threshold=1.0, rep_penalty=1.3)
    p_t09, p_p09, p_t09_time = generate(prompt, N_TOKENS, threshold=0.9, rep_penalty=1.3)

    print(f"\n--- GREEDY (no penalty) ---")
    print(f"  [t=1.0] passes={sum(g_p1)/len(g_p1):.2f} time={g_t1_time:.2f}s")
    print(f"          {g_t1}")
    print(f"  [t=0.9] passes={sum(g_p09)/len(g_p09):.2f} time={g_t09_time:.2f}s")
    print(f"          {g_t09}")
    print(f"  Identical: {'YES' if g_t1 == g_t09 else 'NO'}")

    print(f"\n--- PENALTY 1.3 (sign-aware) ---")
    print(f"  [t=1.0] passes={sum(p_p1)/len(p_p1):.2f} time={p_t1_time:.2f}s")
    print(f"          {p_t1}")
    print(f"  [t=0.9] passes={sum(p_p09)/len(p_p09):.2f} time={p_t09_time:.2f}s")
    print(f"          {p_t09}")
    print(f"  Identical: {'YES' if p_t1 == p_t09 else 'NO'}")

    # Cross-comparison
    print(f"\n--- Comparison ---")
    print(f"  Greedy t=1.0 vs Penalty t=1.0 identical: {'YES' if g_t1 == p_t1 else 'NO'}")
    print(f"  Greedy t=0.9 vs Penalty t=0.9 identical: {'YES' if g_t09 == p_t09 else 'NO'}")

print("\n\nSMOKE TEST COMPLETE")
