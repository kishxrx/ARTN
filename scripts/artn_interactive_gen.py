#!/usr/bin/env python3
"""
ARTN Interactive Generation Tester

Type any prompt and see ARTN generate text at:
  - threshold=1.0 (full 6 passes)
  - threshold=0.9 (early-exit, ~3.5 passes)

Decoding modes (type to switch):
  /greedy    — unpenalized greedy (original default, deterministic)
  /penalty   — greedy + sign-aware rep penalty 1.3 (prompt+generated scope)

Uses the existing trained checkpoint. Greedy decoding (deterministic).
Type 'quit' or Ctrl+C to exit.
"""
import sys, torch, time

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = torch.device('cpu')

# Load model
print("Loading ARTN model...")
ckpt = torch.load(
    'outputs/artn_recursive_20261025_002725/checkpoints/artn/step_10000.pt',
    map_location='cpu', weights_only=False
)
cfg = ckpt['config']
model = ARTNGPT(
    vocab_size=cfg['vocab_size'],
    d_model=cfg['d_model'],
    n_heads=cfg['n_heads'],
    n_layers=cfg['n_layers'],
    d_ff=cfg['d_ff'],
    max_seq_len=cfg['max_seq_len'],
    target_passes=cfg.get('target_passes', 4.0),
    ponder_loss_weight=cfg.get('ponder_loss_weight', 0.01)
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  Loaded: step={ckpt['step']}, val_loss={ckpt['val_loss']:.4f}")
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# Load tokenizer
tok = BPETokenizer.load('outputs/artn-50k-best-config/tokenizer.json')
print(f"  Tokenizer: vocab={tok.vocab_size}, pad_id={tok.pad_token_id}")
print()


def generate(prompt, max_new_tokens=50, threshold=1.0, rep_penalty=1.0):
    """Generate text with ARTN at given threshold. Returns (text, passes_log, elapsed_s).

    rep_penalty: 1.0 = unpenalized greedy (original behavior).
                 >1.0 = sign-aware repetition penalty (HuggingFace standard):
                        positive logits divided, negative logits multiplied.
                        Applied once per unique previously-encountered token
                        (prompt tokens + generated tokens).
    """
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

            # Apply sign-aware repetition penalty (prompt+generated tokens)
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


# Interactive loop
DECODING_MODE = "greedy"  # "greedy" or "penalty"
REP_PENALTY = 1.3  # used only when mode is "penalty"

print("=" * 60)
print("ARTN Interactive Generation")
print("  Type any prompt, press Enter to generate.")
print("  Type 'quit' or Ctrl+C to exit.")
print("  Default: 50 new tokens, compare t=1.0 vs t=0.9")
print("  Commands: /greedy (unpenalized) | /penalty (rep penalty 1.3)")
print(f"  Active mode: {DECODING_MODE}")
print("=" * 60)

while True:
    try:
        prompt = input("\nPrompt> ").strip()
        if not prompt:
            continue
        if prompt.lower() in ('quit', 'exit', 'q'):
            break

        # Mode switching
        if prompt == '/greedy':
            DECODING_MODE = "greedy"
            print(f"  Decoding mode: greedy (unpenalized, rep_penalty=1.0)")
            continue
        if prompt == '/penalty':
            DECODING_MODE = "penalty"
            print(f"  Decoding mode: penalty (sign-aware rep_penalty={REP_PENALTY}, prompt+generated scope)")
            continue

        # Determine active penalty
        active_penalty = REP_PENALTY if DECODING_MODE == "penalty" else 1.0
        mode_label = f"penalty={active_penalty}" if active_penalty > 1.0 else "greedy (no penalty)"

        # Allow custom token count: "prompt |100" → 100 tokens
        n_tokens = 50
        if '|' in prompt:
            parts = prompt.rsplit('|', 1)
            try:
                n_tokens = int(parts[1].strip())
                prompt = parts[0].strip()
            except ValueError:
                pass

        print(f"\nPrompt: '{prompt}'  ({n_tokens} new tokens)")
        print(f"Decoding: {mode_label}")
        print("-" * 60)

        text_full, passes_full, t_full = generate(prompt, n_tokens, threshold=1.0, rep_penalty=active_penalty)
        text_09, passes_09, t_09 = generate(prompt, n_tokens, threshold=0.9, rep_penalty=active_penalty)

        print(f"\n[threshold=1.0]  passes: {passes_full}")
        print(f"  avg={sum(passes_full)/len(passes_full):.2f}  time={t_full:.2f}s  speed={n_tokens/t_full:.2f} tok/s")
        print(f"  Output: {text_full}")

        print(f"\n[threshold=0.9]  passes: {passes_09}")
        print(f"  avg={sum(passes_09)/len(passes_09):.2f}  time={t_09:.2f}s  speed={n_tokens/t_09:.2f} tok/s")
        print(f"  Output: {text_09}")

        print(f"\n  Identical: {'YES' if text_full == text_09 else 'NO'}")

    except KeyboardInterrupt:
        print("\n\nExiting.")
        break
    except Exception as e:
        print(f"Error: {e}")
