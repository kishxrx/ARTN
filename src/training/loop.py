"""
Training loop for ARTN.

Implements memory-efficient training with:
- Gradient accumulation (simulate larger batch sizes)
- Learning rate scheduling
- Checkpointing
- Metrics logging
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.training.dataset import TokenDataset


class Trainer:
    """
    Training loop with gradient accumulation.
    
    Args:
        model: Model to train
        optimizer: Optimizer (Adafactor recommended)
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        micro_batch_size: Batch size per forward pass (limited by memory)
        accumulation_steps: Number of batches to accumulate before optimizer step
        max_seq_len: Maximum sequence length
        lr_scheduler: Optional learning rate scheduler
        checkpoint_dir: Directory to save checkpoints
        log_interval: Steps between logging
        eval_interval: Steps between evaluations
        device: Device to train on (default: cpu)
    
    Effective batch size = micro_batch_size × accumulation_steps
    
    Example:
        micro_batch_size = 4
        accumulation_steps = 8
        → Effective batch size = 32
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_dataset: TokenDataset,
        val_dataset: Optional[TokenDataset] = None,
        micro_batch_size: int = 4,
        accumulation_steps: int = 8,
        max_seq_len: int = 256,
        lr_scheduler: Optional[Any] = None,
        checkpoint_dir: str | Path = "checkpoints",
        log_interval: int = 10,
        eval_interval: int = 100,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.micro_batch_size = micro_batch_size
        self.accumulation_steps = accumulation_steps
        self.max_seq_len = max_seq_len
        self.lr_scheduler = lr_scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.device = device
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=micro_batch_size,
            shuffle=True,
            num_workers=0,  # Keep 0 for CPU-only training
            pin_memory=False,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=micro_batch_size,
                shuffle=False,
                num_workers=0,
            )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Metrics
        self.history: list[Dict[str, Any]] = []
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_step(self, batch: torch.Tensor) -> float:
        """
        Single training step.
        
        Args:
            batch: Input tokens of shape (batch, seq_len)
            
        Returns:
            Loss value
        """
        batch = batch.to(self.device)
        targets = batch[:, 1:]  # Shift by one for next-token prediction
        
        # Create causal input (exclude last token)
        inputs = batch[:, :-1]
        
        # Forward pass
        outputs = self.model(inputs, targets=targets)
        loss = outputs['loss']
        
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        return loss.item()
    
    def train_batch(self) -> float:
        """
        Train one effective batch (accumulation_steps micro-batches).
        
        Returns:
            Average loss over accumulation steps
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        batch_count = 0
        
        # Accumulate gradients
        for i, batch in enumerate(self.train_loader):
            if i >= self.accumulation_steps:
                break
            
            loss = self.train_step(batch)
            total_loss += loss
            batch_count += 1
        
        # Clip gradients (stability)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Learning rate scheduling
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        self.global_step += 1
        
        return total_loss / batch_count if batch_count > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate on validation set.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            batch = batch.to(self.device)
            targets = batch[:, 1:]
            inputs = batch[:, :-1]
            
            outputs = self.model(inputs, targets=targets)
            total_loss += outputs['loss'].item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, filename: str = "checkpoint.pt") -> None:
        """Save model checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str = "checkpoint.pt") -> None:
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.global_step = checkpoint['global_step']
            self.epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.history = checkpoint.get('history', [])
    
    def log_metrics(self, train_loss: float, val_loss: Optional[float] = None) -> None:
        """Log training metrics."""
        metrics = {
            'step': self.global_step,
            'epoch': self.epoch,
            'train_loss': train_loss,
        }
        if val_loss is not None:
            metrics['val_loss'] = val_loss
        
        self.history.append(metrics)
        
        # Save to file
        metrics_path = self.checkpoint_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train(
        self,
        num_epochs: int,
        validate_every: int = 100,
        save_every: int = 500,
    ) -> Dict[str, Any]:
        """
        Run training loop.
        
        Args:
            num_epochs: Number of epochs to train
            validate_every: Steps between validation
            save_every: Steps between checkpoints
            
        Returns:
            Training history
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Effective batch size: {self.micro_batch_size} × {self.accumulation_steps} = {self.micro_batch_size * self.accumulation_steps}")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Reset loader each epoch
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.micro_batch_size,
                shuffle=True,
                num_workers=0,
            )
            
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx in range(len(self.train_loader) // self.accumulation_steps):
                # Train one batch
                loss = self.train_batch()
                epoch_loss += loss
                num_batches += 1
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    elapsed = time.time() - start_time
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"Step {self.global_step} | Loss: {loss:.4f} | LR: {lr:.2e} | Time: {elapsed:.1f}s")
                
                # Validation
                if self.val_loader and self.global_step % validate_every == 0:
                    val_loss = self.evaluate()
                    
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint("best_model.pt")
                    
                    print(f"  Validation Loss: {val_loss:.4f} (Best: {self.best_loss:.4f})")
                    self.log_metrics(loss, val_loss)
                else:
                    if self.global_step % self.log_interval == 0:
                        self.log_metrics(loss)
                
                # Checkpoint
                if self.global_step % save_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
            
            # End of epoch
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"\nEpoch {epoch + 1}/{num_epochs} | Avg Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.1f}s\n")
        
        # Final save
        self.save_checkpoint("final_model.pt")
        
        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Training complete in {total_time:.1f}s")
        print(f"Final loss: {self.history[-1]['train_loss']:.4f}")
        
        return {'history': self.history, 'total_time': total_time}


def overfit_small_dataset(
    model: nn.Module,
    dataset: TokenDataset,
    num_steps: int = 100,
    lr: float = 3e-4,
    device: str = "cpu",
) -> float:
    """
    Overfit a small dataset to verify the training loop.
    
    This is a sanity check: if loss doesn't go to near zero,
    something is wrong with the model or training.
    
    Args:
        model: Model to test
        dataset: Small dataset (few samples)
        num_steps: Number of training steps
        lr: Learning rate
        device: Device
    
    Returns:
        Final loss value
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model.train()
    
    loss = None
    for step in range(num_steps):
        # Get batch
        batch = next(iter(loader)).to(device)
        targets = batch[:, 1:]
        inputs = batch[:, :-1]
        
        # Forward
        outputs = model(inputs, targets=targets)
        loss = outputs['loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    final_loss = loss.item() if loss is not None else float('inf')
    print(f"\nFinal loss: {final_loss:.4f}")
    return final_loss
