#!/usr/bin/env python3
"""
ARTN Beta1 Experiment - Testing beta1=0.5

Purpose: Find optimal beta1 value for ternary quantization
- Beta1=0.9 was too conservative (slow learning)
- Beta1=None caused collapse
- Testing beta1=0.5 as middle ground

Usage:
    python scripts/train_beta1_0.5.py --total-steps 10000
"""

import sys
import time
import math
import torch
import argparse
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.adafactor import Adafactor


# =============================================================================
# Configuration - BETA1=0.5 EXPERIMENT
# =============================================================================

CONFIG = {
    # Model architecture (SAME as before)
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "max_seq_len": 256,
    
    # Training parameters (SAME as before)
    "batch_size": 4,
    "gradient_accumulation_steps": 2,  # Effective batch = 8 (batch_size 4 × 2)
    "max_lr": 0.001,  # Higher LR (Microsoft's value)
    "min_lr": 0.00003,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    # EXPERIMENT: Testing beta1=0.5
    "beta1": 0.5,  # <-- CHANGED FROM 0.9
    "beta2": 0.98,
    
    # Short experiment
    "total_steps": 10000,
    "warmup_steps": 1000,  # SAME as original
    "checkpoint_every": 1000,
    "eval_every": 500,
    
    # Data
    "max_samples": 50000,
    "dataset": "roneneldan/TinyStories",
    
    # Output - separate folder
    "output_dir": "outputs/beta1-0.5-batch8-lr001",  # New folder for this test
}


class CosineWarmupScheduler:
    def __init__(self, warmup_steps, total_steps, max_lr, min_lr=1e-5):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
    
    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


class TinyStoriesDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        tokens = self.tokenizer.encode(text)[:self.max_length]
        tokens = tokens + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


def compute_entropy_and_top_token(model, input_ids, vocab_size=4000):
    """Compute entropy and top token probability for health check."""
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits'][0, -1, :]
        probs = torch.softmax(logits, dim=0)
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = math.log(vocab_size)
        entropy_pct = (entropy / max_entropy) * 100
        
        top_prob, top_idx = torch.max(probs, dim=0)
        return entropy_pct, top_prob.item() * 100


def generate_sample(model, tokenizer, prompt="Once upon a time", max_new_tokens=50, temperature=0.9):
    """Generate a sample for manual inspection."""
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated)
            logits = outputs['logits'][0, -1, :] / temperature
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    return tokenizer.decode(generated[0].tolist())


def main():
    parser = argparse.ArgumentParser(description='Train ARTN with beta1=0.5')
    parser.add_argument('--total-steps', type=int, default=None, help='Override total steps')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=None, help='Override gradient accumulation')
    args = parser.parse_args()
    
    config = CONFIG.copy()
    if args.total_steps:
        config['total_steps'] = args.total_steps
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.gradient_accumulation_steps:
        config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    
    print("=" * 70)
    print("ARTN BETA1 EXPERIMENT - Testing beta1=0.5")
    print("=" * 70)
    print(f"Config: beta1={config['beta1']}, total_steps={config['total_steps']}")
    print(f"Output: {config['output_dir']}")
    print("=" * 70)
    
    # Setup directories
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    
    # Load tokenizer (use existing)
    tokenizer_path = Path('outputs/artn-optimized/tokenizer.json')
    if tokenizer_path.exists():
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        print(f"[Tokenizer] Loaded from {tokenizer_path}")
    else:
        print("[Tokenizer] Training new tokenizer...")
        ds = load_dataset(config['dataset'], split='train')
        texts = [ds[i]['text'] for i in range(min(len(ds), config['max_samples']))]
        tokenizer = BPETokenizer(vocab_size=4000)
        tokenizer.train(texts)
        tokenizer.save(str(output_dir / 'tokenizer.json'))
        print(f"[Tokenizer] Saved to {output_dir / 'tokenizer.json'}")
    
    # Create model
    model = BitLinearGPT(
        vocab_size=4000,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
    )
    print(f"[Model] {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = Adafactor(
        model.parameters(),
        lr=config['max_lr'],
        beta1=config['beta1'],
        beta2=config['beta2'],
        weight_decay=config['weight_decay'],
    )
    print(f"[Optimizer] Adafactor (lr={config['max_lr']}, beta1={config['beta1']})")
    
    # LR scheduler
    scheduler = CosineWarmupScheduler(
        warmup_steps=config['warmup_steps'],
        total_steps=config['total_steps'],
        max_lr=config['max_lr'],
        min_lr=config['min_lr'],
    )
    
    # Load dataset
    print(f"[Dataset] Loading TinyStories...")
    ds = load_dataset(config['dataset'], split='train')
    train_data = ds.select(range(min(len(ds), config['max_samples'])))
    train_dataset = TinyStoriesDataset(train_data, tokenizer, config['max_seq_len'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    print(f"[DataLoader] {len(train_dataset)} samples, batch_size={config['batch_size']}")
    
    # Training loop
    model.train()
    global_step = 0
    accum_loss = 0.0
    
    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    
    start_time = time.time()
    
    while global_step < config['total_steps']:
        for batch in train_loader:
            if global_step >= config['total_steps']:
                break
            
            # Get learning rate
            lr = scheduler.get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            outputs = model(input_ids)
            logits = outputs['logits']
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 4000),
                targets.reshape(-1),
            )
            
            # Scale loss for accumulation
            scaled_loss = loss / config['gradient_accumulation_steps']
            scaled_loss.backward()
            accum_loss += loss.item()
            
            # Optimizer step after accumulation
            if (global_step + 1) % config['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            # Log progress
            if global_step % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed if elapsed > 0 else 0
                eta_sec = (config['total_steps'] - global_step) / steps_per_sec if steps_per_sec > 0 else 0
                eta_h = int(eta_sec // 3600)
                eta_m = int((eta_sec % 3600) // 60)
                
                avg_loss = accum_loss / 100 if global_step > 0 else 0
                
                # Health check
                test_input = torch.tensor([tokenizer.encode("Once upon a time")])
                entropy, top_prob = compute_entropy_and_top_token(model, test_input)
                
                status = "HEALTHY" if entropy > 30 and top_prob < 50 else "WARNING"
                
                print(f"Step {global_step:>6} | Loss {avg_loss:.4f} | LR {lr:.6f} | "
                      f"Entropy {entropy:.1f}% | TopProb {top_prob:.1f}% | "
                      f"ETA {eta_h}h{eta_m}m | {status}")
                
                accum_loss = 0.0
                model.train()
            
            # Checkpoint
            if global_step % config['checkpoint_every'] == 0:
                ckpt_path = output_dir / 'checkpoints' / f'checkpoint_step_{global_step}.pt'
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss if 'avg_loss' in dir() else 0,
                    'config': config,
                }, ckpt_path)
                
                # Generate sample
                sample = generate_sample(model, tokenizer)
                print(f"\n[Sample @ step {global_step}]\n{sample}\n")
                
                # Log to file
                with open(output_dir / 'monitor_log.txt', 'a') as f:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{ts} | Step {global_step} | Loss {avg_loss:.4f} | "
                           f"Entropy {entropy:.1f}% | TopProb {top_prob:.1f}% | STATUS: {status}\n")
    
    # Save final model
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    final_path = output_dir / 'final_model.pt'
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, final_path)
    print(f"Model saved to: {final_path}")
    
    # Final sample
    print("\n[Final Sample]")
    print(generate_sample(model, tokenizer))
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    print(f"\nTotal time: {hours}h {mins}m")


if __name__ == '__main__':
    main()
