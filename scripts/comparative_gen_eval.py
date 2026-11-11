#!/usr/bin/env python3
"""
ARTN vs BitLinear vs FP32 Comparative Generation

Generates text from 3 models on the same 20 prompts, greedy decoding, 100 tokens.
Compares prompt adherence, entity retention, repetition, malformed words.

Models:
  - ARTN (1 shared block, 6 passes, 2.9M params, checkpoint step_10000)
  - BitLinear (6 separate blocks, 6.85M params, checkpoint step_9500)
  - FP32 (6 separate blocks, weight tying, 5.83M params, checkpoint step_9500)

Also runs ARTN with config F (gen-only penalty 1.3, t=0.9) for comparison.

Output: outputs/artn_recursive_20261025_002725/comparative_gen_eval.json
"""
import sys, os, json, time, re, torch

sys.path.insert(0, '/root/agent-sandbox/ARTN')

from experiments.artn_model import ARTNGPT
from experiments.bitlinear_adamw_controlled import BitLinearGPT
from experiments.fp32_baseline_controlled import FP32GPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

DEVICE = torch.device('cpu')
SEED = 42
MAX_TOKENS = 100
OUTPUT_DIR = '/root/agent-sandbox/ARTN/outputs/artn_recursive_20261025_002725'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'comparative_gen_eval.json')

PROMPTS = {
    "familiar": [
        "once upon a time there was a dog",
        "there was a little girl",
        "there lived a",
        "once there was a little boy",
        "a cat sat on the mat",
        "the sun was shining bright",
        "a little bird flew high",
        "there was a big tree",
        "a fish swam in the pond",
        "the children played outside",
    ],
    "less_common": [
        "a bison from wild forest",
        "a peacock",
        "a king",
        "a robot in the garden",
        "a dragon lived in a cave",
    ],
    "challenging": [
        "a king and a queen had three children",
        "the fox and the rabbit were friends but one day",
        "a storm came and the boat started to sink",
        "the teacher gave the students a surprise test",
        "a wizard and a knight went on a quest together",
    ],
}

# ============================================================
# Load tokenizer
# ============================================================
print("Loading tokenizer...")
tok = BPETokenizer.load(os.path.join(OUTPUT_DIR, '..', 'artn-50k-best-config', 'tokenizer.json'))
print(f"  vocab={tok.vocab_size}, pad_id={tok.pad_token_id}")

# ============================================================
# Load ARTN
# ============================================================
print("\nLoading ARTN...")
artn_ckpt = torch.load(os.path.join(OUTPUT_DIR, 'checkpoints/artn/step_10000.pt'),
                       map_location='cpu', weights_only=False)
artn_cfg = artn_ckpt['config']
artn_model = ARTNGPT(
    vocab_size=artn_cfg['vocab_size'], d_model=artn_cfg['d_model'],
    n_heads=artn_cfg['n_heads'], n_layers=artn_cfg['n_layers'],
    d_ff=artn_cfg['d_ff'], max_seq_len=artn_cfg['max_seq_len'],
    target_passes=artn_cfg.get('target_passes', 4.0),
    ponder_loss_weight=artn_cfg.get('ponder_loss_weight', 0.01)
)
artn_model.load_state_dict(artn_ckpt['model_state_dict'])
artn_model.eval()
print(f"  step={artn_ckpt['step']}, val_loss={artn_ckpt['val_loss']:.4f}")
print(f"  params={sum(p.numel() for p in artn_model.parameters()):,}")

# ============================================================
# Load BitLinear
# ============================================================
print("\nLoading BitLinear...")
bl_ckpt = torch.load(
    '/root/agent-sandbox/ARTN/outputs/bitlinear_adamw_20260910_143646/checkpoints/bitlinear_adamw/step_9500.pt',
    map_location='cpu', weights_only=False
)
bl_cfg = bl_ckpt['config']
bl_model = BitLinearGPT(
    vocab_size=bl_cfg['vocab_size'], d_model=bl_cfg['d_model'],
    n_heads=bl_cfg['n_heads'], n_layers=bl_cfg['n_layers'],
    d_ff=bl_cfg['d_ff'], max_seq_len=bl_cfg['max_seq_len']
)
bl_model.load_state_dict(bl_ckpt['model_state_dict'])
bl_model.eval()
print(f"  step={bl_ckpt['step']}, val_loss={bl_ckpt.get('val_loss', 'N/A')}")
print(f"  params={sum(p.numel() for p in bl_model.parameters()):,}")

# ============================================================
# Load FP32
# ============================================================
print("\nLoading FP32...")
fp32_ckpt = torch.load(
    '/root/agent-sandbox/ARTN/outputs/fp32_baseline_20260908_142259/checkpoints/adamw/step_9500.pt',
    map_location='cpu', weights_only=False
)
fp32_cfg = fp32_ckpt['config']
fp32_model = FP32GPT(
    vocab_size=fp32_cfg['vocab_size'], d_model=fp32_cfg['d_model'],
    n_heads=fp32_cfg['n_heads'], n_layers=fp32_cfg['n_layers'],
    d_ff=fp32_cfg['d_ff'], max_seq_len=fp32_cfg['max_seq_len']
)
fp32_model.load_state_dict(fp32_ckpt['model_state_dict'])
fp32_model.eval()
print(f"  step={fp32_ckpt['step']}, val_loss={fp32_ckpt.get('val_loss', 'N/A')}")
print(f"  params={sum(p.numel() for p in fp32_model.parameters()):,}")


# ============================================================
# Generation functions
# ============================================================
def generate_artn(prompt, max_new_tokens=100, threshold=1.0, rep_penalty=1.0):
    """ARTN greedy generation with optional sign-aware rep penalty (gen-only scope)."""
    input_ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    prompt_len = input_ids.shape[1]
    passes_log = []
    t0 = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            B, T = input_ids.shape
            tok_emb = artn_model.token_embedding(input_ids)
            pos_emb = artn_model.position_embedding(torch.arange(T, device=DEVICE))
            x = tok_emb + pos_emb

            remaining = torch.ones(B, T, 1, device=DEVICE)
            output = torch.zeros(B, T, artn_model.d_model, device=DEVICE)
            passes_this = 0

            for pass_idx in range(artn_model.max_passes):
                passes_this += 1
                x = artn_model.block(x)

                if threshold >= 1.0:
                    h = torch.sigmoid(artn_model.halting_gate(x))
                    p = h * remaining
                    if pass_idx == artn_model.max_passes - 1:
                        p = remaining
                    output = output + x * p
                    remaining = remaining - p
                    continue

                h = torch.sigmoid(artn_model.halting_gate(x))
                p = h * remaining
                if pass_idx == artn_model.max_passes - 1:
                    p = remaining
                output = output + x * p
                remaining = remaining - p
                remaining = remaining.clamp(min=0.0)

                cumulative_last = 1.0 - remaining[0, -1, 0].item()
                if cumulative_last >= threshold and pass_idx < artn_model.max_passes - 1:
                    break

            output = output + x * remaining
            x = artn_model.ln_f(output)
            logits = artn_model.lm_head(x)
            logits = logits[:, -1, :]

            if rep_penalty != 1.0:
                generated_tokens = input_ids[0, prompt_len:].tolist()
                prev_tokens = set(generated_tokens)
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


def generate_standard(model, prompt, max_new_tokens=100):
    """Standard 6-block model greedy generation (BitLinear or FP32)."""
    input_ids = torch.tensor([tok.encode(prompt)], device=DEVICE)
    t0 = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            B, T = input_ids.shape
            tok_emb = model.token_embedding(input_ids)
            pos_emb = model.position_embedding(torch.arange(T, device=DEVICE))
            x = tok_emb + pos_emb

            for block in model.blocks:
                x = block(x)

            x = model.ln_f(x)
            logits = model.lm_head(x)
            logits = logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    elapsed = time.time() - t0
    text = tok.decode(input_ids[0].tolist())
    return text, None, elapsed


# ============================================================
# Metrics (same as broader_gen_eval.py)
# ============================================================
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


MALFORMED_EXEMPT = {
    'shouldn', 'wouldn', 'couldn', 'didn', 'wasn', 'weren', 'hasn',
    'strength', 'lengths', 'through', 'thought', 'thoughts',
    'scratched', 'scratches', 'stretched', 'catches', 'watching',
    'children', 'grandmother', 'grandfather', 'grandparents',
    'angry', 'cycle', 'brightly', 'anything', 'knight', 'knights',
}

def detect_malformed_words(text):
    words = text.split()
    malformed = []
    for w in words:
        clean = re.sub(r'[^\w]', '', w).lower()
        if len(clean) < 3:
            continue
        consonant_runs = re.findall(r'[bcdfghjklmnpqrstvwxyz]{4,}', clean)
        if consonant_runs and clean not in MALFORMED_EXEMPT:
            malformed.append(w)
    return malformed


def check_entity_retention(prompt, generated_text, entities):
    """Check if entities appear in generated text (excluding prompt), word-boundary."""
    gen_only = generated_text[len(prompt):].strip()
    retained = []
    for e in entities:
        if re.search(r'\b' + re.escape(e) + r'\b', gen_only.lower()):
            retained.append(e)
    return retained


# Entity definitions (same as Phase 60.19)
ENTITY_CHECKS = {
    "once upon a time there was a dog": ["dog"],
    "there was a little girl": ["girl"],
    "once there was a little boy": ["boy"],
    "a cat sat on the mat": ["cat", "mat"],
    "the sun was shining bright": ["sun"],
    "a little bird flew high": ["bird"],
    "there was a big tree": ["tree"],
    "a fish swam in the pond": ["fish", "pond"],
    "the children played outside": ["children"],
    "a bison from wild forest": ["bison", "forest"],
    "a peacock": ["peacock"],
    "a king": ["king"],
    "a robot in the garden": ["robot", "garden"],
    "a dragon lived in a cave": ["dragon", "cave"],
    "a king and a queen had three children": ["king", "queen", "children"],
    "the fox and the rabbit were friends but one day": ["fox", "rabbit"],
    "a storm came and the boat started to sink": ["storm", "boat", "sink"],
    "the teacher gave the students a surprise test": ["teacher", "students", "test"],
    "a wizard and a knight went on a quest together": ["wizard", "knight", "quest"],
}

# ============================================================
# Run evaluation
# ============================================================
CONFIGS = [
    ("ARTN_greedy_t1.0", "artn_greedy", {"threshold": 1.0, "rep_penalty": 1.0}),
    ("ARTN_configF_t0.9_penalty1.3", "artn_penalty", {"threshold": 0.9, "rep_penalty": 1.3}),
    ("BitLinear_greedy", "bitlinear", {}),
    ("FP32_greedy", "fp32", {}),
]

all_prompts = []
for cat in ["familiar", "less_common", "challenging"]:
    for p in PROMPTS[cat]:
        all_prompts.append((cat, p))

total_runs = len(all_prompts) * len(CONFIGS)
run_num = 0
results = {}

for cat, prompt in all_prompts:
    if prompt not in results:
        results[prompt] = {"category": cat, "configs": {}}

    for label, model_type, params in CONFIGS:
        run_num += 1
        print(f"  [{run_num}/{total_runs}] {label:<35} '{prompt[:40]}'", flush=True)

        torch.manual_seed(SEED)

        if model_type == "artn_greedy":
            text, passes, elapsed = generate_artn(
                prompt, MAX_TOKENS, threshold=params["threshold"], rep_penalty=1.0)
        elif model_type == "artn_penalty":
            text, passes, elapsed = generate_artn(
                prompt, MAX_TOKENS, threshold=params["threshold"], rep_penalty=params["rep_penalty"])
        elif model_type == "bitlinear":
            text, _, elapsed = generate_standard(bl_model, prompt, MAX_TOKENS)
            passes = None
        elif model_type == "fp32":
            text, _, elapsed = generate_standard(fp32_model, prompt, MAX_TOKENS)
            passes = None

        rep_stats = compute_repetition_stats(text)
        malformed = detect_malformed_words(text)
        tok_s = MAX_TOKENS / elapsed if elapsed > 0 else 0
        entities = ENTITY_CHECKS.get(prompt, [])
        retained_entities = check_entity_retention(prompt, text, entities)

        results[prompt]["configs"][label] = {
            "model_type": model_type,
            "threshold": params.get("threshold", 1.0),
            "rep_penalty": params.get("rep_penalty", 1.0),
            "generated_text": text,
            "passes_log": passes,
            "avg_passes": round(sum(passes)/len(passes), 2) if passes else None,
            "elapsed_s": round(elapsed, 2),
            "tokens_per_sec": round(tok_s, 2),
            "repetition_stats": rep_stats,
            "malformed_words": malformed,
            "entities_checked": entities,
            "entities_retained": retained_entities,
            "entity_retention_count": len(retained_entities),
            "entity_retention_total": len(entities),
        }

# ============================================================
# Summary
# ============================================================
summary = {}
for label, model_type, params in CONFIGS:
    config_results = [results[p]["configs"][label] for cat, p in all_prompts]

    avg_rep = sum(r["repetition_stats"]["repetition_rate"] for r in config_results) / len(config_results)
    avg_uniq = sum(r["repetition_stats"]["unique_word_ratio"] for r in config_results) / len(config_results)
    avg_tok_s = sum(r["tokens_per_sec"] for r in config_results) / len(config_results)
    total_malformed = sum(len(r["malformed_words"]) for r in config_results)

    # Entity retention (only prompts with entities)
    entity_results = [r for r in config_results if r["entity_retention_total"] > 0]
    total_retained = sum(r["entity_retention_count"] for r in entity_results)
    total_entities = sum(r["entity_retention_total"] for r in entity_results)
    retention_pct = total_retained / total_entities * 100 if total_entities > 0 else 0

    avg_passes = None
    passes_results = [r["avg_passes"] for r in config_results if r["avg_passes"] is not None]
    if passes_results:
        avg_passes = round(sum(passes_results) / len(passes_results), 2)

    summary[label] = {
        "model_type": model_type,
        "avg_repetition_rate": round(avg_rep, 3),
        "avg_unique_word_ratio": round(avg_uniq, 3),
        "avg_tokens_per_sec": round(avg_tok_s, 2),
        "avg_recursive_passes": avg_passes,
        "total_malformed_words": total_malformed,
        "entity_retention": f"{total_retained}/{total_entities}",
        "entity_retention_pct": round(retention_pct, 1),
    }

# ============================================================
# Save
# ============================================================
output = {
    "experiment": "comparative_generation_evaluation",
    "date": "2026-11-02",
    "models": {
        "ARTN": {"checkpoint": f"step_{artn_ckpt['step']}", "val_loss": artn_ckpt['val_loss'],
                  "params": sum(p.numel() for p in artn_model.parameters())},
        "BitLinear": {"checkpoint": f"step_{bl_ckpt['step']}", "val_loss": bl_ckpt.get('val_loss', 'N/A'),
                      "params": sum(p.numel() for p in bl_model.parameters())},
        "FP32": {"checkpoint": f"step_{fp32_ckpt['step']}", "val_loss": fp32_ckpt.get('val_loss', 'N/A'),
                 "params": sum(p.numel() for p in fp32_model.parameters())},
    },
    "max_tokens": MAX_TOKENS,
    "seed": SEED,
    "num_prompts": len(all_prompts),
    "summary": summary,
    "results": results,
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'=' * 80}")
print("COMPARATIVE GENERATION EVALUATION COMPLETE")
print(f"{'=' * 80}")
print(f"\nOutput: {OUTPUT_FILE}")
print(f"\n--- Summary ---")
print(f"{'Config':<35} {'Avg Rep':<10} {'Avg Uniq':<10} {'Malformed':<10} {'Ent.Ret':<12} {'Tok/s':<10} {'Passes':<8}")
print("-" * 95)
for label, _, _ in CONFIGS:
    s = summary[label]
    print(f"{label:<35} {s['avg_repetition_rate']:<10} {s['avg_unique_word_ratio']:<10} "
          f"{s['total_malformed_words']:<10} {s['entity_retention']+'('+str(s['entity_retention_pct'])+'%)':<12} "
          f"{s['avg_tokens_per_sec']:<10} {str(s['avg_recursive_passes']):<8}")
