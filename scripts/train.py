"""

ARTN Training Script.

Usage:
    python train.py                        # Train from scratch
    python train.py --resume checkpoints/  # Resume from checkpoint
    python train.py --preset minimal       # Use preset config
"""

import os
import sys
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.train_config import get_config, PRESETS
from src.model.bitlinear import BitLinearGPT
from src.training.adafactor import Adafactor
from src.training.checkpoint import CheckpointManager, save_final_model
from src.tokenizer.bpe_tokenizer import BPETokenizer


# =============================================================================
# Dataset Preparation
# =============================================================================

def prepare_dataset(config: dict, tokenizer=None):
    """Load and prepare TinyStories dataset."""
    
    print(f"\n[Dataset] Loading TinyStories...")
    
    # Load dataset
    dataset = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["split_train"],
    )
    
    # Limit samples if specified
    if config.get("train_samples"):
        dataset = dataset.select(range(min(config["train_samples"], len(dataset))))
    
    print(f"[Dataset] Loaded {len(dataset)} training samples")
    
    return dataset


class TinyStoriesDataset(torch.utils.data.Dataset):
    """TinyStories dataset for training."""
    
    def __init__(
        self,
        data,
        tokenizer,
        max_length: int = 256,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)[:self.max_length]
        
        # Pad to max_length
        padding_length = self.max_length - len(tokens)
        tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
        
        # Create input_ids and labels
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


# =============================================================================
# Training Functions
# =============================================================================

def create_model(config: dict) -> BitLinearGPT:
    """Create BitLinear model."""
    
    model = BitLinearGPT(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=config["model"]["dropout"],
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] Created BitLinearGPT with {num_params:,} parameters")
    
    return model


def create_optimizer(model, config: dict):
    """Create Adafactor optimizer."""
    
    optimizer = Adafactor(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        beta1=config.get("beta1", 0.9),
        beta2=config.get("beta2", 0.999),
    )
    
    print(f"[Optimizer] Adafactor (lr={config['learning_rate']})")
    
    return optimizer


def train_step(model, batch, optimizer, config, device):
    """Single training step."""
    
    model.train()
    
    # Use full batch
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    
    # Forward pass
    outputs = model(input_ids, targets=labels)
    loss = outputs["loss"]
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()


def evaluate(model, dataloader, device, max_batches=10):
    """Evaluate model on validation set."""
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if num_batches >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, targets=labels)
            total_loss += outputs["loss"].item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


# =============================================================================
# Progress Monitoring
# =============================================================================

class ProgressTracker:
    """Track and display training progress."""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.step_times = []
        self.losses = []
    
    def update(self, step: int, loss: float):
        """Update progress."""
        current_time = time.time()
        
        if len(self.step_times) > 0:
            step_time = current_time - self.step_times[-1][0]
        else:
            step_time = current_time - self.start_time
        
        self.step_times.append((current_time, step_time))
        self.losses.append(loss)
        
        # Calculate ETA
        avg_step_time = sum(s[1] for s in self.step_times[-100:]) / min(len(self.step_times), 100)
        remaining_steps = self.total_steps - step
        eta_seconds = avg_step_time * remaining_steps
        eta = self._format_time(eta_seconds)
        
        # Calculate elapsed time
        elapsed = self._format_time(current_time - self.start_time)
        
        # Progress bar
        progress = step / self.total_steps
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Recent loss trend
        if len(self.losses) > 10:
            recent_avg = sum(self.losses[-10:]) / 10
            loss_trend = " ↓" if self.losses[-1] < recent_avg else " ↑"
        else:
            loss_trend = ""
        
        print(f"\r[{bar}] Step {step}/{self.total_steps} | "
              f"Loss: {loss:.4f}{loss_trend} | "
              f"Elapsed: {elapsed} | "
              f"ETA: {eta}   ", end="")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def print_summary(self):
        """Print final summary."""
        total_time = time.time() - self.start_time
        avg_loss = sum(self.losses) / len(self.losses)
        
        print(f"\n\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time:  {self._format_time(total_time)}")
        print(f"Avg loss:    {avg_loss:.4f}")
        print(f"Total steps: {len(self.losses)}")
        print(f"{'='*60}")


# =============================================================================
# Signal Handler (for graceful interrupt)
# =============================================================================

interrupted = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global interrupted
    print("\n\n[Interrupt] Training stopped by user. Saving checkpoint...")
    interrupted = True


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: dict, resume_from: str = None):
    """Main training function."""
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Device
    device = torch.device(config["hardware"]["device"])
    print(f"\n[Device] Using {device}")
    
    # Create directories
    output_dir = Path(config["output"]["output_dir"]) / config["output"]["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    config["checkpoint_dir"] = str(output_dir / "checkpoints")
    
    # Create or load tokenizer
    # For simplicity, we use a character-level tokenizer or pre-trained
    # In production, train BPE on TinyStories
    print("\n[Tokenizer] Loading BPE tokenizer...")
    
    try:
        # Try to load existing tokenizer
        tokenizer = BPETokenizer.load(output_dir / "tokenizer.json")
        print(f"[Tokenizer] Loaded from {output_dir}")
    except:
        # Create new tokenizer
        print("[Tokenizer] Training new BPE tokenizer on dataset...")
        
        # Load full dataset for tokenizer training
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        texts = [item["text"] for item in dataset.select(range(min(100000, len(dataset))))]
        
        # Join texts into single corpus
        corpus = "\n".join(texts)
        
        tokenizer = BPETokenizer(
            vocab_size=config["model"]["vocab_size"],
            min_frequency=config["tokenizer"]["min_frequency"],
        )
        tokenizer.train(corpus)
        
        # Save tokenizer
        tokenizer.save(output_dir / "tokenizer.json")
        print(f"[Tokenizer] Saved to {output_dir}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        config["checkpoint_dir"],
        keep_last_n=config["keep_last_n_checkpoints"],
    )
    
    # Resume from checkpoint
    start_step = 0
    if resume_from:
        checkpoint = checkpoint_manager.load(Path(resume_from))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]
        print(f"\n[Resume] Starting from step {start_step}")
    elif checkpoint_manager.has_checkpoint():
        checkpoint = checkpoint_manager.load_latest()
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]
        print(f"\n[Resume] Found checkpoint, starting from step {start_step}")
    
    # Load dataset
    raw_dataset = prepare_dataset(config)
    
    # Create PyTorch dataset
    train_dataset = TinyStoriesDataset(
        raw_dataset,
        tokenizer,
        max_length=config["max_seq_len"],
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )
    
    print(f"\n[Data] {len(train_dataset)} samples, batch_size={config['batch_size']}")
    print(f"[Data] Effective batch: {config['batch_size'] * config['gradient_accumulation_steps']}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"TRAINING START")
    print(f"{'='*60}")
    print(f"Max steps:    {config['max_steps']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Device:       {device}")
    print(f"{'='*60}\n")
    
    progress = ProgressTracker(config["max_steps"])
    data_iter = iter(train_loader)
    
    step = start_step
    
    while step < config["max_steps"] and not interrupted:
        step += 1
        
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # Train step
        loss = train_step(model, batch, optimizer, config, device)
        
        # Update progress
        progress.update(step, loss)
        
        # Checkpoint
        if step % config["checkpoint_every"] == 0 or interrupted:
            checkpoint_manager.save(
                model, optimizer, step, loss,
                config=config,
            )
        
        # Log
        if step % config["log_every"] == 0:
            # Save loss to file
            with open(output_dir / "training_log.txt", "a") as f:
                f.write(f"{step}\t{loss:.6f}\t{time.time()}\n")
    
    # Save final model
    if not interrupted:
        save_final_model(model, output_dir / "final_model", config)
    
    progress.print_summary()
    
    return model


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ARTN")
    
    parser.add_argument("--preset", type=str, default="short",
                        choices=list(PRESETS.keys()),
                        help="Training preset")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max steps")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override learning rate")
    
    args = parser.parse_args()
    
    # Get config
    config = get_config(args.preset)
    
    # Apply overrides
    if args.max_steps:
        config["max_steps"] = args.max_steps
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    
    print(f"\n[Config] Preset: {args.preset}")
    print(f"[Config] Max steps: {config['max_steps']}")
    print(f"[Config] Learning rate: {config['learning_rate']}")
    
    # Train
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
