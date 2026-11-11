#!/usr/bin/env python3
"""
ARTN Per-Pass Hidden State Analysis

Extracts hidden states at key prompt positions across all 6 recursive passes.
Measures cosine similarity, representation norms, and next-token distribution
evolution across passes.

Also performs controlled input substitutions to test whether changing the
prompt entity changes representations and predictions.

Output: outputs/artn_recursive_20261025_002725/hidden_state_analysis.json
"""
import sys, os, json, time, math, torch
import torch.nn.functional as F

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from experiments.artn_model import ARTNGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = torch.device('cpu')
SEED = 42
OUTPUT_DIR = '/root/agent-sandbox/ARTN/outputs/artn_recursive_20261025_002725'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'hidden_state_analysis.json')

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
d_model = cfg['d_model']
print(f"  step={ckpt['step']}, d_model={d_model}, max_passes={max_passes}")


# ============================================================
# Forward pass with per-pass hidden state extraction
# ============================================================
def forward_with_hidden_states(input_ids):
    """
    Full 6-pass forward with hidden state extraction at each pass.
    Returns:
      - per_pass_hidden: list of (B, T, d_model) — block output at each pass
      - per_pass_attn_input: list of (B, T, d_model) — ln1(x) at each pass (pre-attention)
      - final_hidden: (B, T, d_model) — after ln_f, weighted by halt probs
      - logits: (B, T, vocab) — final logits
      - halt_weights: list of (B, T, 1)
    """
    B, T = input_ids.shape
    tok_emb = model.token_embedding(input_ids)
    pos_emb = model.position_embedding(torch.arange(T, device=DEVICE))
    x = tok_emb + pos_emb

    per_pass_hidden = []
    per_pass_attn_input = []
    halt_weights_list = []

    remaining = torch.ones(B, T, 1, device=DEVICE)
    output = torch.zeros(B, T, d_model, device=DEVICE)

    for pass_idx in range(max_passes):
        x_norm = model.block.ln1(x)
        per_pass_attn_input.append(x_norm.clone())

        # Attention
        B2, T2, C2 = x_norm.shape
        q = model.block.attn.q_proj(x_norm).view(B2, T2, n_heads, d_head).transpose(1, 2)
        k = model.block.attn.k_proj(x_norm).view(B2, T2, n_heads, d_head).transpose(1, 2)
        v = model.block.attn.v_proj(x_norm).view(B2, T2, n_heads, d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d_head))
        causal_mask = torch.triu(torch.ones(T2, T2, device=DEVICE), diagonal=1).bool()
        att = att.masked_fill(causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B2, T2, C2)
        y = model.block.attn.o_proj(y)
        x = x + y

        # FF
        x_norm2 = model.block.ln2(x)
        x = x + model.block.ff(x_norm2)

        per_pass_hidden.append(x.clone())

        # Halting
        h = torch.sigmoid(model.halting_gate(x))
        p = h * remaining
        if pass_idx == max_passes - 1:
            p = remaining
        output = output + x * p
        halt_weights_list.append(p.clone())
        remaining = remaining - p

    x_final = model.ln_f(output)
    logits = model.lm_head(x_final)

    return {
        'per_pass_hidden': per_pass_hidden,
        'per_pass_attn_input': per_pass_attn_input,
        'final_hidden': x_final,
        'logits': logits,
        'halt_weights': halt_weights_list,
    }


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()


def get_logits_at_last(logits):
    """Get logits for last position."""
    return logits[0, -1, :]


def topk_from_logits(logits_last, k=10):
    """Get top-k tokens from logits."""
    vals, ids = torch.topk(logits_last, k)
    return [(tok.decode([tid.item()]), tid.item(), round(v.item(), 4)) for v, tid in zip(vals, ids)]


# ============================================================
# Prompts and key positions
# ============================================================
PROMPTS = {
    "dog": {"prompt": "once upon a time there was a dog", "entity_word": "dog", "entity_pos": 7},
    "fish": {"prompt": "a fish swam in the pond", "entity_word": "fish", "entity_pos": 1},
    "dragon": {"prompt": "a dragon lived in a cave", "entity_word": "dragon", "entity_pos": 1},
    "king": {"prompt": "a king", "entity_word": "king", "entity_pos": 1},
    "peacock": {"prompt": "a peacock", "entity_word": "peacock", "entity_pos": 1},
    "teacher": {"prompt": "the teacher gave the students a surprise test",
                "entity_word": "teacher", "entity_pos": 1},
}

# Substitution pairs: replace entity token to test if representations change
SUBSTITUTIONS = {
    "king_vs_dog": {
        "original": "a king",
        "substituted": "a dog",
        "entity_pos": 1,
    },
    "peacock_vs_dog": {
        "original": "a peacock",
        "substituted": "a dog",
        "entity_pos": 1,
    },
    "peacock_vs_dragon": {
        "original": "a peacock",
        "substituted": "a dragon",
        "entity_pos": 1,
    },
}

# ============================================================
# Run analysis
# ============================================================
torch.manual_seed(SEED)
results = {}

# --- Part 1: Per-pass hidden states for each prompt ---
print("\n=== Part 1: Per-pass hidden states ===\n")
for name, info in PROMPTS.items():
    prompt = info["prompt"]
    entity_pos = info["entity_pos"]
    entity_word = info["entity_word"]

    print(f"  Processing: '{prompt}' (entity='{entity_word}' at pos {entity_pos})")

    input_ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    fwd = forward_with_hidden_states(input_ids)

    # Extract hidden states at entity position across passes
    pass_states = []
    pass_norms = []
    for pass_idx in range(max_passes):
        state = fwd['per_pass_hidden'][pass_idx][0, entity_pos, :].clone()
        pass_states.append(state)
        pass_norms.append(round(torch.norm(state).item(), 4))

    # Cosine similarity between consecutive passes
    consecutive_cos = []
    for i in range(1, max_passes):
        cs = cosine_sim(pass_states[i], pass_states[i-1])
        consecutive_cos.append(round(cs, 6))

    # Cosine similarity between pass 1 and all later passes
    p1_cos = []
    for i in range(max_passes):
        cs = cosine_sim(pass_states[i], pass_states[0])
        p1_cos.append(round(cs, 6))

    # Last-position hidden states across passes
    last_pass_states = []
    last_pass_norms = []
    for pass_idx in range(max_passes):
        state = fwd['per_pass_hidden'][pass_idx][0, -1, :].clone()
        last_pass_states.append(state)
        last_pass_norms.append(round(torch.norm(state).item(), 4))

    # Next-token distribution at last position, per pass
    # We need to project each pass's hidden state through ln_f and lm_head
    # But ln_f and lm_head are applied to the WEIGHTED output, not individual passes
    # We can approximate by projecting each pass's hidden state through ln_f + lm_head
    per_pass_logits_last = []
    for pass_idx in range(max_passes):
        h = fwd['per_pass_hidden'][pass_idx][0, -1, :].clone()
        # Apply ln_f + lm_head to get logits
        h_normed = model.ln_f(h.unsqueeze(0).unsqueeze(0))
        pass_logits = model.lm_head(h_normed)[0, 0, :]
        per_pass_logits_last.append(pass_logits)

    # Top-10 predictions at each pass for last position
    per_pass_topk = []
    for pass_idx in range(max_passes):
        topk = topk_from_logits(per_pass_logits_last[pass_idx], 10)
        per_pass_topk.append(topk)

    # Probability distribution: entropy and max prob per pass
    per_pass_entropy = []
    per_pass_max_prob = []
    for pass_idx in range(max_passes):
        probs = F.softmax(per_pass_logits_last[pass_idx], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_prob = probs.max().item()
        per_pass_entropy.append(round(entropy, 4))
        per_pass_max_prob.append(round(max_prob, 6))

    # Entity token logit and probability across passes (at last position)
    entity_token_id = tok.encode(entity_word)
    if len(entity_token_id) == 1:
        eid = entity_token_id[0]
        entity_logit_per_pass = []
        entity_prob_per_pass = []
        for pass_idx in range(max_passes):
            lg = per_pass_logits_last[pass_idx][eid].item()
            probs = F.softmax(per_pass_logits_last[pass_idx], dim=-1)
            p = probs[eid].item()
            entity_logit_per_pass.append(round(lg, 4))
            entity_prob_per_pass.append(round(p, 6))
    else:
        entity_logit_per_pass = None
        entity_prob_per_pass = None
        eid = None

    results[name] = {
        "prompt": prompt,
        "entity_word": entity_word,
        "entity_pos": entity_pos,
        "entity_token_id": eid,
        "prompt_tokens": [tok.decode([t]) for t in tok.encode(prompt)],
        "pass_norms_at_entity": pass_norms,
        "consecutive_cosine_at_entity": consecutive_cos,
        "pass1_cosine_at_entity": p1_cos,
        "last_pos_pass_norms": last_pass_norms,
        "per_pass_topk_last": per_pass_topk,
        "per_pass_entropy_last": per_pass_entropy,
        "per_pass_max_prob_last": per_pass_max_prob,
        "entity_logit_per_pass": entity_logit_per_pass,
        "entity_prob_per_pass": entity_prob_per_pass,
    }


# --- Part 2: Controlled substitutions ---
print("\n=== Part 2: Controlled substitutions ===\n")
sub_results = {}

for sub_name, sub_info in SUBSTITUTIONS.items():
    orig_prompt = sub_info["original"]
    sub_prompt = sub_info["substituted"]
    entity_pos = sub_info["entity_pos"]

    print(f"  Substitution: '{orig_prompt}' → '{sub_prompt}'")

    # Original
    input_ids_orig = torch.tensor([tok.encode(orig_prompt)], device=DEVICE)
    fwd_orig = forward_with_hidden_states(input_ids_orig)

    # Substituted
    input_ids_sub = torch.tensor([tok.encode(sub_prompt)], device=DEVICE)
    fwd_sub = forward_with_hidden_states(input_ids_sub)

    # Hidden states at entity position across passes
    orig_states = [fwd_orig['per_pass_hidden'][p][0, entity_pos, :].clone() for p in range(max_passes)]
    sub_states = [fwd_sub['per_pass_hidden'][p][0, entity_pos, :].clone() for p in range(max_passes)]

    # Cosine similarity between orig and sub at entity position, per pass
    orig_sub_cos = []
    for p in range(max_passes):
        cs = cosine_sim(orig_states[p], sub_states[p])
        orig_sub_cos.append(round(cs, 6))

    # Last-position logits: compare top-5 per pass
    orig_last_logits = []
    sub_last_logits = []
    for p in range(max_passes):
        h_o = model.ln_f(orig_states[p].unsqueeze(0).unsqueeze(0))
        lg_o = model.lm_head(h_o)[0, 0, :]
        orig_last_logits.append(lg_o)

        h_s = model.ln_f(sub_states[p].unsqueeze(0).unsqueeze(0))
        lg_s = model.lm_head(h_s)[0, 0, :]
        sub_last_logits.append(lg_s)

    # Top-5 per pass for both
    orig_topk = [topk_from_logits(lg, 5) for lg in orig_last_logits]
    sub_topk = [topk_from_logits(lg, 5) for lg in sub_last_logits]

    # KL divergence between orig and sub next-token distributions, per pass
    kl_divs = []
    for p in range(max_passes):
        p_o = F.softmax(orig_last_logits[p], dim=-1)
        p_s = F.softmax(sub_last_logits[p], dim=-1)
        kl = F.kl_div(torch.log(p_o + 1e-10), p_s, reduction='sum').item()
        kl_divs.append(round(kl, 4))

    # Logit for original entity word at last position per pass
    orig_entity = tok.encode(orig_prompt.split()[entity_pos] if entity_pos < len(orig_prompt.split()) else orig_prompt)
    sub_entity = tok.encode(sub_prompt.split()[entity_pos] if entity_pos < len(sub_prompt.split()) else sub_prompt)

    sub_results[sub_name] = {
        "original_prompt": orig_prompt,
        "substituted_prompt": sub_prompt,
        "entity_pos": entity_pos,
        "orig_sub_cosine_per_pass": orig_sub_cos,
        "orig_topk_per_pass": orig_topk,
        "sub_topk_per_pass": sub_topk,
        "kl_divergence_per_pass": kl_divs,
    }

results["substitutions"] = sub_results


# ============================================================
# Save
# ============================================================
output = {
    "experiment": "per_pass_hidden_state_analysis",
    "date": "2026-11-02",
    "checkpoint": f"step_{ckpt['step']}",
    "model_config": {
        "d_model": d_model,
        "n_heads": n_heads,
        "max_passes": max_passes,
    },
    "prompts_analyzed": list(PROMPTS.keys()),
    "substitutions": list(SUBSTITUTIONS.keys()),
    "results": results,
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 70}")
print("HIDDEN STATE ANALYSIS COMPLETE")
print(f"{'=' * 70}")
print(f"Output: {OUTPUT_FILE}")

# Print summary
print(f"\n--- Summary: Entity position cosine sim (pass 1 → pass 6) ---\n")
print(f"{'Prompt':<35} {'P1→P2':<10} {'P1→P3':<10} {'P1→P4':<10} {'P1→P5':<10} {'P1→P6':<10}")
print("-" * 85)
for name in PROMPTS:
    r = results[name]
    cos = r['pass1_cosine_at_entity']
    print(f"{name:<35} {cos[1]:<10.4f} {cos[2]:<10.4f} {cos[3]:<10.4f} {cos[4]:<10.4f} {cos[5]:<10.4f}")

print(f"\n--- Entity token probability at last position per pass ---\n")
print(f"{'Prompt':<35} {'P1':<12} {'P2':<12} {'P3':<12} {'P4':<12} {'P5':<12} {'P6':<12}")
print("-" * 99)
for name in PROMPTS:
    r = results[name]
    if r['entity_prob_per_pass']:
        probs = r['entity_prob_per_pass']
        print(f"{name:<35} {probs[0]:<12.6f} {probs[1]:<12.6f} {probs[2]:<12.6f} {probs[3]:<12.6f} {probs[4]:<12.6f} {probs[5]:<12.6f}")
    else:
        print(f"{name:<35} (multi-token entity)")

print(f"\n--- Next-token entropy at last position per pass ---\n")
print(f"{'Prompt':<35} {'P1':<10} {'P2':<10} {'P3':<10} {'P4':<10} {'P5':<10} {'P6':<10}")
print("-" * 85)
for name in PROMPTS:
    r = results[name]
    ent = r['per_pass_entropy_last']
    print(f"{name:<35} {ent[0]:<10.4f} {ent[1]:<10.4f} {ent[2]:<10.4f} {ent[3]:<10.4f} {ent[4]:<10.4f} {ent[5]:<10.4f}")
