#!/usr/bin/env python3
"""
FP32 Control Experiment

Identical to ternary experiment except uses standard FP32 Linear layers instead of BitLinear.
Compares learning capacity between FP32 and ternary quantization.

Key measurements:
- Loss/perplexity
- Top-1 / Top-5 accuracy
- Correct token probability
- Gradient norm
- Generation quality
- Repetition/diversity

All conditions identical to ternary experiment except:
- Linear layers are FP32 (not BitLinear)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
import math
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer.bpe_tokenizer import BPETokenizer

# ============================================================
# Configuration (IDENTICAL to ternary experiment)
# ============================================================

CONFIG = {
    "vocab_size": 4000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    "batch_size": 8,
    "learning_rate": 0.003,
    "max_steps": 5000,
    "seed": 42,
    "checkpoint_steps": [0, 500, 1000, 2000, 3000, 4000, 5000],
    "eval_interval": 100,
    "generation_interval": 500,
}

DEVICE = torch.device('cpu')

# Fixed evaluation prompts (same as ternary experiment)
EVAL_PROMPTS = [
    "Once upon a time",
    "The little girl",
    "A boy named",
    "One day",
    "There was a",
]

# ============================================================
# FP32 Model Implementation
# ============================================================

class StandardLinear(nn.Linear):
    """Standard FP32 Linear layer - no quantization"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        # Standard PyTorch Linear - FP32 weights
    
    def forward(self, x):
        # Standard matrix multiplication - FP32
        return super().forward(x)


class StandardMultiHeadAttention(nn.Module):
    """Standard multi-head attention with FP32 linear layers"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Standard Linear layers
        self.q_proj = StandardLinear(d_model, d_model)
        self.k_proj = StandardLinear(d_model, d_model)
        self.v_proj = StandardLinear(d_model, d_model)
        self.o_proj = StandardLinear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax
        attn_probs = F.softmax(scores, dim=-1)
        
        # Attention output
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        return self.o_proj(attn_output)


class StandardFFN(nn.Module):
    """Standard FFN with FP32 layers"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = StandardLinear(d_model, d_ff)
        self.fc2 = StandardLinear(d_ff, d_model)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class StandardTransformerBlock(nn.Module):
    """Standard transformer block with FP32 layers"""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StandardMultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = StandardFFN(d_model, d_ff)
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class StandardGPT(nn.Module):
    """Standard GPT model with FP32 layers (NO quantization)"""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = StandardLinear(d_model, vocab_size)
        
        print(f"StandardGPT initialized:")
        print(f"  Layers: {n_layers} x FP32 transformer blocks")
        print(f"  All Linear layers use FP32 weights (NO quantization)")
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding.weight[:seq_len]
        x = token_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final LayerNorm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.lm_head(x)
        
        return {'logits': logits}
    
    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        """Generate tokens autoregressively"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to max_seq_len
                if input_ids.size(1) >= self.token_embedding.num_embeddings:
                    break
                
                # Forward pass
                output = self(input_ids)
                logits = output['logits']
                
                # Get next token
                next_token_logits = logits[:, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                # Append
                input_ids = torch.cat([input_ids, next_token], dim=1)
            
            return input_ids

# ============================================================
# Evaluation functions
# ============================================================

def compute_metrics(model, eval_input_ids, targets, tokenizer):
    """Compute evaluation metrics"""
    model.eval()
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    with torch.no_grad():
        output = model(eval_input_ids)
        logits = output['logits']
        
        # Loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        perplexity = math.exp(loss.item()) if loss.item() < 10 else float('inf')
        
        # Top-1 accuracy
        preds = logits.argmax(dim=-1)
        top1_correct = (preds == targets).float().mean().item()
        top1_acc = 100 * top1_correct
        
        # Top-5 accuracy
        top5 = logits.topk(5, dim=-1).indices
        top5_correct = 0
        total = 0
        for i in range(targets.size(0)):
            for j in range(targets.size(1)):
                if targets[i, j].item() in top5[i, j].tolist():
                    top5_correct += 1
                total += 1
        top5_acc = 100 * top5_correct / total
        
        # Correct token probability
        probs = F.softmax(logits, dim=-1)
        correct_prob_sum = 0
        for i in range(targets.size(0)):
            for j in range(targets.size(1)):
                correct_prob_sum += probs[i, j, targets[i, j].item()].item()
        correct_prob = 100 * correct_prob_sum / total
    
    return {
        'loss': loss.item(),
        'perplexity': perplexity,
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'correct_prob': correct_prob,
    }


def generate_samples(model, tokenizer, prompts, max_tokens=40, temperature=1.0):
    """Generate samples from prompts"""
    model.eval()
    
    generations = []
    
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature)
        
        generated = tokenizer.decode(output_ids[0].tolist())
        
        # Compute diversity metrics
        words = generated.split()
        if len(words) > 1:
            unique_words = len(set(words))
            word_diversity = unique_words / len(words)
            
            bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            unique_bigrams = len(bigram_counts)
            total_bigrams = len(bigrams) if bigrams else 1
            bigram_rep = 1.0 - (unique_bigrams / total_bigrams)
        else:
            word_diversity = 0.0
            bigram_rep = 0.0
        
        generations.append({
            'prompt': prompt,
            'generated': generated,
            'word_diversity': word_diversity,
            'bigram_rep': bigram_rep,
        })
    
    return generations


def save_checkpoint(model, optimizer, step, loss, output_dir, is_best=False):
    """Save checkpoint"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save numbered checkpoint
    checkpoint_path = output_dir / f"step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = output_dir / "best_checkpoint.pt"
        torch.save(checkpoint, best_path)
        print(f"  Best checkpoint saved: {best_path}")

# ============================================================
# Training function
# ============================================================

def train():
    """Run FP32 training"""
    
    print("="*70)
    print("FP32 CONTROL EXPERIMENT")
    print("="*70)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Set seed
    torch.manual_seed(CONFIG['seed'])
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiments/results/fp32_control_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Config saved: {config_path}")
    
    # Load tokenizer (same as ternary experiment)
    tokenizer_path = "outputs/artn-optimized/tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"\nTokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # Create FP32 model
    print(f"\nInitializing FP32 model...")
    model = StandardGPT(
        vocab_size=CONFIG['vocab_size'],
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        d_ff=CONFIG['d_ff'],
        max_seq_len=CONFIG['max_seq_len'],
    )
    
    # Verify NO quantization
    print(f"\nVerification: Checking for quantization layers...")
    has_quantization = False
    for name, module in model.named_modules():
        if 'BitLinear' in type(module).__name__ or 'quantize' in name or 'ternary' in name:
            has_quantization = True
            print(f"  WARNING: Found quantization layer: {name} ({type(module).__name__})")
    
    if not has_quantization:
        print(f"  ✓ VERIFIED: No quantization layers found - pure FP32 model")
    else:
        print(f"  ✗ ERROR: Quantization detected - aborting")
        return
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer (same as ternary)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    # Load training data (same as ternary experiment)
    print(f"\nLoading training data...")
    
    # Use same 20 deterministic stories with same seed
    torch.manual_seed(CONFIG['seed'])
    
    training_sentences = [
        "Once upon a time there was a little girl named Sarah.",
        "The little girl loved to play in the garden every day.",
        "One day she found a magic flower in the garden.",
        "The flower glowed with beautiful colors.",
        "Sarah picked the flower and made a wish.",
        "Suddenly a fairy appeared before her eyes.",
        "The fairy granted Sarah three special wishes.",
        "Sarah wished for happiness and peace.",
        "The fairy smiled and disappeared into the light.",
        "From that day on Sarah was always happy.",
        "There was once a brave little boy named Tom.",
        "Tom loved to explore the forest behind his house.",
        "One day he found a secret path in the woods.",
        "The path led to a hidden treasure chest.",
        "Tom opened the chest and found golden coins.",
        "He shared the treasure with his family.",
        "Everyone was proud of the brave boy.",
        "Tom learned that adventures are exciting.",
        "He promised to have more adventures.",
        "The end of the story made everyone smile.",
    ]
    
    print(f"Training sentences: {len(training_sentences)}")
    
    # Tokenize
    train_sequences = []
    for sentence in training_sentences:
        tokens = tokenizer.encode(sentence)
        train_sequences.append(tokens)
    
    max_len = max(len(s) for s in train_sequences)
    print(f"Max seq length: {max_len}")
    
    # Pad sequences
    padded_sequences = []
    for seq in train_sequences:
        padded = seq + [0] * (max_len - len(seq))
        padded_sequences.append(padded)
    
    train_data = torch.tensor(padded_sequences)
    print(f"Training data shape: {train_data.shape}")
    
    # Create evaluation data (same as ternary)
    eval_sentences = [
        "Once upon a time there was a little girl named Sarah.",
        "The little girl loved to play in the garden every day.",
        "One day she found a magic flower in the garden.",
        "The flower glowed with beautiful colors.",
        "Sarah picked the flower and made a wish.",
        "Suddenly a fairy appeared before her eyes.",
        "The fairy granted Sarah three special wishes.",
        "Sarah wished for happiness and peace.",
        "The fairy smiled and disappeared into the light.",
        "From that day on Sarah was always happy.",
    ]
    
    eval_sequences = []
    for sentence in eval_sentences:
        tokens = tokenizer.encode(sentence)
        eval_sequences.append(tokens)
    
    eval_max_len = max(len(s) for s in eval_sequences)
    eval_padded = []
    for seq in eval_sequences:
        padded = seq + [0] * (eval_max_len - len(seq))
        eval_padded.append(padded)
    
    eval_input_ids = torch.tensor(eval_padded)
    eval_targets = torch.cat([eval_input_ids[:, 1:], torch.zeros(eval_input_ids.size(0), 1, dtype=torch.long)], dim=1)
    
    print(f"Evaluation data shape: {eval_input_ids.shape}")
    
    # Training loop
    print(f"\nStarting training...")
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    best_loss = float('inf')
    
    results = {
        'steps': [],
        'train_loss': [],
        'eval_loss': [],
        'eval_ppl': [],
        'top1_acc': [],
        'top5_acc': [],
        'correct_prob': [],
        'grad_norm': [],
        'generations': [],
    }
    
    start_time = time.time()
    
    for step in range(CONFIG['max_steps'] + 1):
        # Checkpoint at specific steps
        if step in CONFIG['checkpoint_steps']:
            model.eval()
            
            # Evaluate
            metrics = compute_metrics(model, eval_input_ids, eval_targets, tokenizer)
            
            # Save checkpoint
            is_best = metrics['loss'] < best_loss
            if is_best:
                best_loss = metrics['loss']
            
            save_checkpoint(model, optimizer, step, metrics['loss'], output_dir, is_best=is_best)
            
            print(f"\nStep {step}:")
            print(f"  Eval Loss: {metrics['loss']:.4f}")
            print(f"  Eval PPL: {metrics['perplexity']:.1f}")
            print(f"  Top-1 Acc: {metrics['top1_acc']:.2f}%")
            print(f"  Top-5 Acc: {metrics['top5_acc']:.2f}%")
            print(f"  Best Loss: {best_loss:.4f}")
        
        # Generate samples at intervals
        if step in CONFIG['checkpoint_steps']:
            generations = generate_samples(model, tokenizer, EVAL_PROMPTS[:3])
            
            print(f"\nGenerations:")
            for gen in generations:
                print(f"  Prompt: {gen['prompt']}")
                print(f"    Generated: {gen['generated'][:60]}...")
                print(f"    Diversity: {gen['word_diversity']:.3f}")
            
            results['generations'].append({
                'step': step,
                'samples': generations,
            })
        
        if step == CONFIG['max_steps']:
            break
        
        # Training step
        model.train()
        
        # Sample batch
        batch_indices = torch.randint(0, train_data.size(0), (CONFIG['batch_size'],))
        batch = train_data[batch_indices]
        
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Forward pass
        output = model(input_ids)
        logits = output['logits']
        
        # Loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Optimizer step
        optimizer.step()
        
        # Evaluate at intervals
        if (step + 1) % CONFIG['eval_interval'] == 0:
            model.eval()
            
            metrics = compute_metrics(model, eval_input_ids, eval_targets, tokenizer)
            
            print(f"\nStep {step+1}: Loss={loss.item():.4f}, Eval Loss={metrics['loss']:.4f}, "
                  f"PPL={metrics['perplexity']:.1f}, Top-1={metrics['top1_acc']:.2f}%, "
                  f"Grad={grad_norm:.4f}")
            
            results['steps'].append(step + 1)
            results['train_loss'].append(loss.item())
            results['eval_loss'].append(metrics['loss'])
            results['eval_ppl'].append(metrics['perplexity'])
            results['top1_acc'].append(metrics['top1_acc'])
            results['top5_acc'].append(metrics['top5_acc'])
            results['correct_prob'].append(metrics['correct_prob'])
            results['grad_norm'].append(grad_norm)
    
    # Final checkpoint
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': CONFIG['max_steps'],
        'loss': best_loss,
    }, output_dir / "final_checkpoint.pt")
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        # Convert to JSON-serializable
        results_save = {
            'config': CONFIG,
            'steps': results['steps'],
            'train_loss': results['train_loss'],
            'eval_loss': results['eval_loss'],
            'eval_ppl': results['eval_ppl'],
            'top1_acc': results['top1_acc'],
            'top5_acc': results['top5_acc'],
            'correct_prob': results['correct_prob'],
            'grad_norm': results['grad_norm'],
            'generations': [
                {
                    'step': g['step'],
                    'samples': [
                        {
                            'prompt': s['prompt'],
                            'generated': s['generated'],
                            'word_diversity': s['word_diversity'],
                            'bigram_rep': s['bigram_rep'],
                        }
                        for s in g['samples']
                    ]
                }
                for g in results['generations']
            ],
            'best_loss': best_loss,
            'training_time': time.time() - start_time,
        }
        json.dump(results_save, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Training time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    train()
