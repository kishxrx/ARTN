#!/usr/bin/env python3
"""
TRUE GENERALIZATION TEST ON REAL TINYSTORIES
=============================================

Tests whether FP32 and BitLinear models can generalize to UNSEEN TinyStories stories.

Uses existing trained models from Phase 46-47:
- FP32 + AdamW: step_10000.pt
- BitLinear + AdamW: step_10000.pt
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from pathlib import Path
import numpy as np

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from src.tokenizer.bpe_tokenizer import BPETokenizer

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "seq_len": 256,
    "batch_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    "fp32_checkpoint": str(BASE_DIR / "outputs/fp32_baseline_20260908_142259/checkpoints/adamw/step_10000.pt"),
    "bitlinear_checkpoint": str(BASE_DIR / "outputs/bitlinear_adamw_20260910_143646/checkpoints/bitlinear_adamw/step_10000.pt"),
    "tokenizer_path": str(BASE_DIR / "outputs/artn-50k-best-config/tokenizer.json"),
    
    "train_data_path": str(BASE_DIR / "outputs/tokenized_cache/train_tokenized.pt"),
    "val_data_path": str(BASE_DIR / "outputs/tokenized_cache/val_tokenized.pt"),
    
    "test_a_stories": 100,
    "test_b_stories": 100,
    
    "prompts": [
        "Once upon a time there was a little girl",
        "There was a boy who loved to play",
        "One day, a happy dog went to the park",
        "The cat sat on the mat and looked",
        "In a small village, there lived a kind",
    ],
    
    "generation_max_tokens": 50,
}

# =============================================================================
# FP32 MODEL (copied from training script)
# =============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool(), float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class FP32GPT(nn.Module):
    def __init__(self, vocab_size=4000, d_model=256, n_heads=4, n_layers=6, d_ff=512, max_seq_len=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -256:]
            logits = self(idx_cond)["logits"]
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, idx_next], dim=1)
        return input_ids


# =============================================================================
# BITLINEAR MODEL
# =============================================================================

from src.model.bitlinear import BitLinear

class BitLinearAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = BitLinear(d_model, d_model)
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        self.o_proj = BitLinear(d_model, d_model)
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool(), float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class BitLinearFF(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = BitLinear(d_model, d_ff)
        self.w2 = BitLinear(d_ff, d_model)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class BitLinearBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BitLinearAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = BitLinearFF(d_model, d_ff)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class BitLinearGPT(nn.Module):
    def __init__(self, vocab_size=4000, d_model=256, n_heads=4, n_layers=6, d_ff=512, max_seq_len=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([BitLinearBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = BitLinear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -256:]
            logits = self(idx_cond)["logits"]
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, idx_next], dim=1)
        return input_ids


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_data():
    print("Loading tokenized data...")
    train_data = torch.load(CONFIG["train_data_path"], weights_only=False)
    val_data = torch.load(CONFIG["val_data_path"], weights_only=False)
    print(f"  Train stories: {len(train_data)}")
    print(f"  Val stories: {len(val_data)}")
    return train_data, val_data

def create_test_a(train_data, num_stories, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(train_data), size=min(num_stories, len(train_data)), replace=False)
    stories = [train_data[i] for i in indices]
    return stories, indices.tolist()

def create_test_b(val_data, train_data, num_stories, seed=42):
    np.random.seed(seed)
    train_signatures = set()
    for story in train_data:
        sig = tuple(story[:10]) if len(story) >= 10 else tuple(story)
        train_signatures.add(sig)
    
    selected_indices = []
    for i, story in enumerate(val_data):
        if len(selected_indices) >= num_stories:
            break
        sig = tuple(story[:10]) if len(story) >= 10 else tuple(story)
        if sig not in train_signatures:
            selected_indices.append(i)
    
    stories = [val_data[i] for i in selected_indices[:num_stories]]
    return stories, selected_indices

def prepare_sequences(stories, seq_len):
    sequences = []
    for story in stories:
        if len(story) >= seq_len:
            sequences.append(story[:seq_len])
        else:
            padded = story + [0] * (seq_len - len(story))
            sequences.append(padded)
    return torch.tensor(sequences, dtype=torch.long)

def analyze_overlap(test_stories, train_data, test_name):
    print(f"\n{test_name} Overlap Analysis:")
    print("-" * 60)
    
    train_signatures = set()
    train_full_sequences = set()
    for story in train_data:
        sig = tuple(story[:10]) if len(story) >= 10 else tuple(story)
        train_signatures.add(sig)
        train_full_sequences.add(tuple(story))
    
    story_overlap = 0
    sequence_overlap = 0
    
    for story in test_stories:
        sig = tuple(story[:10]) if len(story) >= 10 else tuple(story)
        if sig in train_signatures:
            story_overlap += 1
        if tuple(story) in train_full_sequences:
            sequence_overlap += 1
    
    total = len(test_stories)
    print(f"  Total test stories: {total}")
    print(f"  Story overlap (first 10 tokens): {story_overlap} ({story_overlap/total*100:.2f}%)")
    print(f"  Exact sequence overlap: {sequence_overlap} ({sequence_overlap/total*100:.2f}%)")
    
    return {"total_stories": total, "story_overlap": story_overlap, "story_overlap_pct": story_overlap / total * 100}

def analyze_vocabulary(test_stories, train_data, test_name):
    print(f"\n{test_name} Vocabulary Analysis:")
    print("-" * 60)
    
    train_tokens = set()
    for story in train_data:
        train_tokens.update(story)
    
    test_tokens = set()
    for story in test_stories:
        test_tokens.update(story)
    
    oov_tokens = test_tokens - train_tokens
    
    test_token_list = []
    for story in test_stories:
        test_token_list.extend(story)
    
    oov_occurrences = sum(1 for t in test_token_list if t not in train_tokens)
    total_occurrences = len(test_token_list)
    
    print(f"  Train vocabulary size: {len(train_tokens)}")
    print(f"  Test vocabulary size: {len(test_tokens)}")
    print(f"  OOV tokens (type-level): {len(oov_tokens)}")
    print(f"  OOV occurrences (token-level): {oov_occurrences}/{total_occurrences} ({oov_occurrences/total_occurrences*100:.2f}%)")
    
    return {"train_vocab_size": len(train_tokens), "test_vocab_size": len(test_tokens), "oov_rate_pct": oov_occurrences / total_occurrences * 100}

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, sequences, device, model_name):
    print(f"\nEvaluating {model_name}...")
    print("-" * 60)
    
    model.eval()
    model.to(device)
    sequences = sequences.to(device)
    
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_tokens = 0
    
    batch_size = CONFIG["batch_size"]
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            output = model(batch)
            logits = output['logits']
            
            targets = batch[:, 1:]
            logits_for_loss = logits[:, :-1, :]
            
            loss = F.cross_entropy(logits_for_loss.reshape(-1, logits_for_loss.size(-1)), targets.reshape(-1), reduction='sum')
            total_loss += loss.item()
            
            predictions = logits_for_loss.argmax(dim=-1)
            correct_top1 = (predictions == targets).sum().item()
            total_correct_top1 += correct_top1
            
            _, top5_preds = logits_for_loss.topk(5, dim=-1)
            correct_top5 = (top5_preds == targets.unsqueeze(-1)).sum().item()
            total_correct_top5 += correct_top5
            
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    top1_acc = total_correct_top1 / total_tokens
    top5_acc = total_correct_top5 / total_tokens
    
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc*100:.2f}%")
    
    return {"loss": avg_loss, "perplexity": perplexity, "top1_accuracy": top1_acc, "top5_accuracy": top5_acc}

def generate_text(model, tokenizer, prompt, device, max_tokens=50):
    model.eval()
    model.to(device)
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = model.generate(tokens, max_new_tokens=max_tokens, temperature=1.0)
    generated_tokens = generated[0].tolist()
    text = tokenizer.decode(generated_tokens)
    return text

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("TRUE GENERALIZATION TEST ON REAL TINYSTORIES")
    print("="*80)
    
    device = CONFIG["device"]
    print(f"\nDevice: {device}")
    
    print(f"\nLoading tokenizer from: {CONFIG['tokenizer_path']}")
    tokenizer = BPETokenizer.load(CONFIG["tokenizer_path"])
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    
    train_data, val_data = load_data()
    
    print("\n" + "="*80)
    print("CREATING TEST A: SEEN TRAINING STORIES")
    print("="*80)
    test_a_stories, test_a_indices = create_test_a(train_data, CONFIG["test_a_stories"])
    test_a_sequences = prepare_sequences(test_a_stories, CONFIG["seq_len"])
    print(f"  Test A stories: {len(test_a_stories)}")
    
    print("\n" + "="*80)
    print("CREATING TEST B: UNSEEN VALIDATION STORIES")
    print("="*80)
    test_b_stories, test_b_indices = create_test_b(val_data, train_data, CONFIG["test_b_stories"])
    test_b_sequences = prepare_sequences(test_b_stories, CONFIG["seq_len"])
    print(f"  Test B stories: {len(test_b_stories)}")
    
    print("\n" + "="*80)
    print("OVERLAP ANALYSIS")
    print("="*80)
    overlap_a = analyze_overlap(test_a_stories, train_data, "Test A")
    overlap_b = analyze_overlap(test_b_stories, train_data, "Test B")
    vocab_a = analyze_vocabulary(test_a_stories, train_data, "Test A")
    vocab_b = analyze_vocabulary(test_b_stories, train_data, "Test B")
    
    print("\n" + "="*80)
    print("LOADING FP32 MODEL")
    print("="*80)
    print(f"Checkpoint: {CONFIG['fp32_checkpoint']}")
    fp32_model = FP32GPT(vocab_size=4000, d_model=256, n_heads=4, n_layers=6, d_ff=1024, max_seq_len=256)
    checkpoint = torch.load(CONFIG["fp32_checkpoint"], map_location=device, weights_only=False)
    fp32_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded from step: {checkpoint.get('step', 'unknown')}")
    
    print("\n" + "="*80)
    print("LOADING BITLINEAR MODEL")
    print("="*80)
    print(f"Checkpoint: {CONFIG['bitlinear_checkpoint']}")
    bitlinear_model = BitLinearGPT(vocab_size=4000, d_model=256, n_heads=4, n_layers=6, d_ff=1024, max_seq_len=256)
    checkpoint = torch.load(CONFIG["bitlinear_checkpoint"], map_location=device, weights_only=False)
    bitlinear_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded from step: {checkpoint.get('step', 'unknown')}")
    
    print("\n" + "="*80)
    print("FP32 MODEL EVALUATION")
    print("="*80)
    fp32_test_a = evaluate_model(fp32_model, test_a_sequences, device, "FP32 on Test A")
    fp32_test_b = evaluate_model(fp32_model, test_b_sequences, device, "FP32 on Test B")
    
    print("\n" + "="*80)
    print("BITLINEAR MODEL EVALUATION")
    print("="*80)
    bitlinear_test_a = evaluate_model(bitlinear_model, test_a_sequences, device, "BitLinear on Test A")
    bitlinear_test_b = evaluate_model(bitlinear_model, test_b_sequences, device, "BitLinear on Test B")
    
    print("\n" + "="*80)
    print("GENERATION QUALITY TEST")
    print("="*80)
    generations = {}
    for prompt in CONFIG["prompts"]:
        print(f"\nPrompt: {prompt}")
        print("-" * 60)
        fp32_gen = generate_text(fp32_model, tokenizer, prompt, device, CONFIG["generation_max_tokens"])
        print(f"FP32: {fp32_gen[:200]}...")
        bitlinear_gen = generate_text(bitlinear_model, tokenizer, prompt, device, CONFIG["generation_max_tokens"])
        print(f"BitLinear: {bitlinear_gen[:200]}...")
        generations[prompt] = {"fp32": fp32_gen, "bitlinear": bitlinear_gen}
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n" + "-"*80)
    print("TEST A (Seen Training Stories - Memorization Baseline)")
    print("-"*80)
    print(f"{'Model':<15} {'Loss':<12} {'PPL':<12} {'Top-1 Acc':<12} {'Top-5 Acc':<12}")
    print("-"*80)
    print(f"{'FP32':<15} {fp32_test_a['loss']:<12.4f} {fp32_test_a['perplexity']:<12.2f} {fp32_test_a['top1_accuracy']*100:<11.2f}% {fp32_test_a['top5_accuracy']*100:<11.2f}%")
    print(f"{'BitLinear':<15} {bitlinear_test_a['loss']:<12.4f} {bitlinear_test_a['perplexity']:<12.2f} {bitlinear_test_a['top1_accuracy']*100:<11.2f}% {bitlinear_test_a['top5_accuracy']*100:<11.2f}%")
    
    print("\n" + "-"*80)
    print("TEST B (Unseen Validation Stories - True Generalization)")
    print("-"*80)
    print(f"{'Model':<15} {'Loss':<12} {'PPL':<12} {'Top-1 Acc':<12} {'Top-5 Acc':<12}")
    print("-"*80)
    print(f"{'FP32':<15} {fp32_test_b['loss']:<12.4f} {fp32_test_b['perplexity']:<12.2f} {fp32_test_b['top1_accuracy']*100:<11.2f}% {fp32_test_b['top5_accuracy']*100:<11.2f}%")
    print(f"{'BitLinear':<15} {bitlinear_test_b['loss']:<12.4f} {bitlinear_test_b['perplexity']:<12.2f} {bitlinear_test_b['top1_accuracy']*100:<11.2f}% {bitlinear_test_b['top5_accuracy']*100:<11.2f}%")
    
    print("\n" + "-"*80)
    print("GENERALIZATION GAP (Test B - Test A)")
    print("-"*80)
    fp32_gap_loss = fp32_test_b['loss'] - fp32_test_a['loss']
    bitlinear_gap_loss = bitlinear_test_b['loss'] - bitlinear_test_a['loss']
    print(f"FP32 Loss Gap: {fp32_gap_loss:+.4f}")
    print(f"BitLinear Loss Gap: {bitlinear_gap_loss:+.4f}")
    
    results = {
        "overlap_analysis": {"test_a": overlap_a, "test_b": overlap_b},
        "vocabulary_analysis": {"test_a": vocab_a, "test_b": vocab_b},
        "results": {
            "fp32": {"test_a": fp32_test_a, "test_b": fp32_test_b},
            "bitlinear": {"test_a": bitlinear_test_a, "test_b": bitlinear_test_b},
        },
        "generations": generations,
    }
    
    output_path = BASE_DIR / "outputs" / "generalization_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
