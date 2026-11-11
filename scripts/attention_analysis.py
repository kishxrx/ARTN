#!/usr/bin/env python3
"""
ARTN Attention Weight Extraction

Extracts attention weights from the ARTN model during generation to measure
how much the generating (last) position attends to prompt tokens vs generated
tokens, across all 6 recursive passes and 8 attention heads.

Compares on-theme (succeeding) and off-theme (failing) prompts.
Measures at generation steps 1, 5, 10, 25, 50, 75, 100.

Output: outputs/artn_recursive_20261025_002725/attention_analysis.json
"""
import sys, os, json, time, math, torch
import torch.nn.functional as F

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = torch.device('cpu')
SEED = 42
OUTPUT_DIR = '/root/agent-sandbox/ARTN/outputs/artn_recursive_20261025_002725'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'attention_analysis.json')

# Representative prompts
ON_THEME = [
    "once upon a time there was a dog",
    "a fish swam in the pond",
    "a dragon lived in a cave",
]
OFF_THEME = [
    "a peacock",
    "a king",
    "the teacher gave the students a surprise test",
]

# Generation steps at which to capture attention
CAPTURE_STEPS = [1, 5, 10, 25, 50, 75, 100]

# ============================================================
# Load model
# ============================================================
print("Loading ARTN model...")
ckpt = torch.load(os.path.join(OUTPUT_DIR, 'checkpoints/artn/step_10000.pt'),
                 map_location='cpu', weights_only=False)
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
tok = BPETokenizer.load(os.path.join(OUTPUT_DIR, '..', 'artn-50k-best-config', 'tokenizer.json'))

n_heads = cfg['n_heads']
d_head = cfg['d_model'] // n_heads
max_passes = cfg['n_layers']
print(f"  step={ckpt['step']}, params={sum(p.numel() for p in model.parameters()):,}")
print(f"  n_heads={n_heads}, d_head={d_head}, max_passes={max_passes}")


# ============================================================
# Attention extraction forward pass
# ============================================================
def extract_attention(input_ids, threshold=1.0):
    """
    Run ARTN forward pass with attention extraction at each recursive pass.
    Returns:
      - attentions: list of (n_heads, T, T) tensors, one per pass
      - halt_weights: list of (B, T, 1) tensors
      - output_logits: (B, vocab) for last position
      - passes_executed: int
    """
    B, T = input_ids.shape
    tok_emb = model.token_embedding(input_ids)
    pos_emb = model.position_embedding(torch.arange(T, device=DEVICE))
    x = tok_emb + pos_emb

    remaining = torch.ones(B, T, 1, device=DEVICE)
    output = torch.zeros(B, T, model.d_model, device=DEVICE)
    attentions = []
    halt_weights_list = []
    passes_executed = 0

    for pass_idx in range(max_passes):
        passes_executed += 1

        # Replicate BitLinearBlock.forward with attention extraction
        # Block: x = x + attn(ln1(x)); x = x + ff(ln2(x))
        x_norm = model.block.ln1(x)

        # Extract attention weights manually
        # BitLinearAttention.forward
        B2, T2, C2 = x_norm.shape
        q = model.block.attn.q_proj(x_norm).view(B2, T2, n_heads, d_head).transpose(1, 2)
        k = model.block.attn.k_proj(x_norm).view(B2, T2, n_heads, d_head).transpose(1, 2)
        v = model.block.attn.v_proj(x_norm).view(B2, T2, n_heads, d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d_head))
        # Causal mask
        causal_mask = torch.triu(torch.ones(T2, T2, device=DEVICE), diagonal=1).bool()
        att = att.masked_fill(causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B2, T2, C2)
        y = model.block.attn.o_proj(y)

        attentions.append(att.clone())  # (B, n_heads, T, T)

        # Residual
        x = x + y

        # FF
        x_norm2 = model.block.ln2(x)
        x = x + model.block.ff(x_norm2)

        # Halting
        h = torch.sigmoid(model.halting_gate(x))
        p = h * remaining
        if pass_idx == max_passes - 1:
            p = remaining
        output = output + x * p
        halt_weights_list.append(p.clone())
        remaining = remaining - p

        # Early exit check
        if threshold < 1.0:
            remaining = remaining.clamp(min=0.0)
            cumulative_last = 1.0 - remaining[0, -1, 0].item()
            if cumulative_last >= threshold and pass_idx < max_passes - 1:
                # Assign remaining mass to current pass
                output = output + x * remaining
                remaining = torch.zeros_like(remaining)
                break

    # Final projection
    x = model.ln_f(output)
    logits = model.lm_head(x)
    last_logits = logits[:, -1, :]

    return {
        'attentions': attentions,
        'halt_weights': halt_weights_list,
        'passes_executed': passes_executed,
        'last_logits': last_logits,
        'seq_len': T,
    }


# ============================================================
# Run attention analysis
# ============================================================
torch.manual_seed(SEED)

results = {}

all_prompts = [("on_theme", p) for p in ON_THEME] + [("off_theme", p) for p in OFF_THEME]

for theme_label, prompt in all_prompts:
    print(f"\n--- Prompt: '{prompt}' ({theme_label}) ---")

    prompt_ids = tok.encode(prompt)
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device=DEVICE)

    prompt_attn_data = {
        "prompt": prompt,
        "theme": theme_label,
        "prompt_token_ids": prompt_ids,
        "prompt_token_count": prompt_len,
        "capture_steps": {},
    }

    # Decode prompt tokens for reference
    prompt_tokens_decoded = [tok.decode([t]) for t in prompt_ids]
    prompt_attn_data["prompt_tokens"] = prompt_tokens_decoded

    for gen_step in range(1, 101):
        # Extract attention at this step
        attn_data = extract_attention(input_ids, threshold=1.0)

        if gen_step in CAPTURE_STEPS:
            step_info = {
                "gen_step": gen_step,
                "seq_len": attn_data['seq_len'],
                "passes_executed": attn_data['passes_executed'],
                "attention_by_pass": [],
            }

            for pass_idx, att in enumerate(attn_data['attentions']):
                # att shape: (1, n_heads, T, T)
                # We want attention from LAST position (row -1) to all positions
                last_pos_attn = att[0, :, -1, :]  # (n_heads, T)

                # Split into prompt tokens (0:prompt_len) and generated tokens (prompt_len:)
                prompt_attn = last_pos_attn[:, :prompt_len]  # (n_heads, prompt_len)
                generated_attn = last_pos_attn[:, prompt_len:]  # (n_heads, gen_len)

                # Sum attention to prompt vs generated per head
                prompt_attn_sum = prompt_attn.sum(dim=-1)  # (n_heads,)
                gen_attn_sum = generated_attn.sum(dim=-1)  # (n_heads,)

                # Per-head stats
                head_stats = []
                for h in range(n_heads):
                    head_stats.append({
                        "head": h,
                        "prompt_attn_sum": round(prompt_attn_sum[h].item(), 6),
                        "generated_attn_sum": round(gen_attn_sum[h].item(), 6),
                        "prompt_attn_pct": round(prompt_attn_sum[h].item() * 100, 4),
                        # Top 3 attended positions
                        "top3_positions": [],
                    })

                    # Find top-3 attended positions for this head
                    attn_row = last_pos_attn[h].cpu()
                    top3_vals, top3_idx = torch.topk(attn_row, min(3, len(attn_row)))
                    for v, i in zip(top3_vals, top3_idx):
                        pos = i.item()
                        is_prompt = pos < prompt_len
                        token_str = tok.decode([input_ids[0, pos].item()])
                        head_stats[-1]["top3_positions"].append({
                            "position": pos,
                            "token": token_str,
                            "is_prompt": is_prompt,
                            "attention": round(v.item(), 6),
                        })

                # Average across heads
                avg_prompt_attn = prompt_attn.mean(dim=-1).mean().item()
                avg_gen_attn = generated_attn.mean(dim=-1).mean().item() if generated_attn.shape[1] > 0 else 0
                total_attn = avg_prompt_attn + avg_gen_attn
                prompt_pct = avg_prompt_attn / total_attn * 100 if total_attn > 0 else 0

                step_info["attention_by_pass"].append({
                    "pass": pass_idx,
                    "avg_prompt_attn": round(avg_prompt_attn, 6),
                    "avg_generated_attn": round(avg_gen_attn, 6),
                    "prompt_attn_pct": round(prompt_pct, 4),
                    "head_stats": head_stats,
                })

            # Next token prediction
            next_token_id = attn_data['last_logits'].argmax(dim=-1).item()
            next_token = tok.decode([next_token_id])
            step_info["next_token"] = next_token
            step_info["next_token_id"] = next_token_id

            # Top-5 predictions
            top5_vals, top5_ids = torch.topk(attn_data['last_logits'][0], 5)
            step_info["top5_predictions"] = [
                {"token": tok.decode([tid.item()]), "id": tid.item(), "logit": round(v.item(), 4)}
                for v, tid in zip(top5_vals, top5_ids)
            ]

            prompt_attn_data["capture_steps"][str(gen_step)] = step_info

        # Generate next token (greedy)
        with torch.no_grad():
            x = model.token_embedding(input_ids)
            x = x + model.position_embedding(torch.arange(input_ids.shape[1], device=DEVICE))
            remaining = torch.ones(1, input_ids.shape[1], 1, device=DEVICE)
            output = torch.zeros(1, input_ids.shape[1], model.d_model, device=DEVICE)
            for pass_idx in range(max_passes):
                x = model.block(x)
                h = torch.sigmoid(model.halting_gate(x))
                p = h * remaining
                if pass_idx == max_passes - 1:
                    p = remaining
                output = output + x * p
                remaining = remaining - p
            x = model.ln_f(output)
            logits = model.lm_head(x)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    results[prompt] = prompt_attn_data
    print(f"  Completed. Captured at steps: {CAPTURE_STEPS}")

# ============================================================
# Save
# ============================================================
output = {
    "experiment": "attention_weight_extraction",
    "date": "2026-11-02",
    "checkpoint": f"step_{ckpt['step']}",
    "model_config": {
        "n_heads": n_heads,
        "d_head": d_head,
        "max_passes": max_passes,
        "d_model": cfg['d_model'],
    },
    "capture_steps": CAPTURE_STEPS,
    "on_theme_prompts": ON_THEME,
    "off_theme_prompts": OFF_THEME,
    "results": results,
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 70}")
print("ATTENTION ANALYSIS COMPLETE")
print(f"{'=' * 70}")
print(f"Output: {OUTPUT_FILE}")

# Print summary
print(f"\n--- Summary: Avg prompt attention % at step 10 (across passes) ---\n")
print(f"{'Prompt':<45} {'Theme':<10} {'Pass1':<10} {'Pass3':<10} {'Pass6':<10}")
print("-" * 85)
for prompt in ON_THEME + OFF_THEME:
    theme = "on" if prompt in ON_THEME else "off"
    step10 = results[prompt]["capture_steps"].get("10", {})
    row = f"{prompt[:43]:<45} {theme:<10}"
    for p in [0, 2, 5]:
        if p < len(step10.get("attention_by_pass", [])):
            pct = step10["attention_by_pass"][p]["prompt_attn_pct"]
            row += f"{pct:<10.2f}"
        else:
            row += f"{'N/A':<10}"
    print(row)
