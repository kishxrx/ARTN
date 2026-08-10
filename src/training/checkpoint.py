"""
Checkpoint Management for ARTN Training.

Features:
- Save checkpoints (model, optimizer, step, loss, config)
- Resume from checkpoint
- Keep last N checkpoints (cleanup old ones)
- Automatic recovery from interruptions
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class CheckpointManager:
    """
    Manages training checkpoints with save, load, and cleanup.
    
    Usage:
        manager = CheckpointManager("checkpoints", keep_last_n=3)
        
        # Save checkpoint
        manager.save(model, optimizer, step=1000, loss=0.5)
        
        # Resume from checkpoint
        checkpoint = manager.load_latest()
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step']
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        keep_last_n: int = 3,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        
        # Log file for training history
        self.log_file = self.checkpoint_dir / "training_log.json"
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        loss: float,
        config: Optional[Dict] = None,
        extra: Optional[Dict] = None,
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current training step
            loss: Current loss value
            config: Training configuration (optional)
            extra: Extra data to save (optional)
        
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint dict
        checkpoint = {
            "step": step,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        
        if config:
            checkpoint["config"] = config
        
        if extra:
            checkpoint["extra"] = extra
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Update training log
        self._update_log(step, loss)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"[Checkpoint] Saved at step {step} (loss={loss:.4f})")
        
        return checkpoint_path
    
    def load(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint dict
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"[Checkpoint] Loaded from {checkpoint_path.name}")
        
        return checkpoint
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Checkpoint dict or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        
        if not checkpoints:
            print("[Checkpoint] No checkpoints found")
            return None
        
        # Sort by step number
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
        latest = checkpoints[-1]
        
        return self.load(latest)
    
    def load_best(self) -> Optional[Dict[str, Any]]:
        """
        Load the checkpoint with lowest loss.
        
        Returns:
            Checkpoint dict or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        
        if not checkpoints:
            return None
        
        # Find checkpoint with lowest loss
        best_checkpoint = None
        best_loss = float('inf')
        
        for ckpt_path in checkpoints:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if ckpt['loss'] < best_loss:
                best_loss = ckpt['loss']
                best_checkpoint = ckpt_path
        
        if best_checkpoint:
            return self.load(best_checkpoint)
        
        return None
    
    def get_latest_step(self) -> Optional[int]:
        """Get the step number of the latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        
        if not checkpoints:
            return None
        
        steps = [int(p.stem.split("_")[-1]) for p in checkpoints]
        return max(steps)
    
    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists."""
        return len(list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))) > 0
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        
        if len(checkpoints) <= self.keep_last_n:
            return
        
        # Sort by step number
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
        
        # Remove all but last N
        to_remove = checkpoints[:-self.keep_last_n]
        
        for ckpt_path in to_remove:
            ckpt_path.unlink()
            print(f"[Checkpoint] Removed old: {ckpt_path.name}")
    
    def _update_log(self, step: int, loss: float):
        """Update training log with checkpoint info."""
        log_data = []
        
        # Load existing log
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            except:
                log_data = []
        
        # Append new entry
        log_data.append({
            "step": step,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Save log
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def get_training_history(self) -> list:
        """Get training history from log."""
        if not self.log_file.exists():
            return []
        
        with open(self.log_file, 'r') as f:
            return json.load(f)


def save_final_model(
    model: torch.nn.Module,
    output_dir: str,
    config: Dict,
):
    """
    Save final model for deployment.
    
    Args:
        model: Trained model
        output_dir: Output directory
        config: Model configuration
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save config
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"[Model] Saved to {output_path}")


if __name__ == "__main__":
    # Test checkpoint manager
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, keep_last_n=2)
        
        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Save checkpoints
        for step in [100, 200, 300]:
            loss = 1.0 - step * 0.001
            manager.save(model, optimizer, step, loss)
        
        # Check cleanup
        checkpoints = list(Path(tmpdir).glob("checkpoint_step_*.pt"))
        print(f"\nCheckpoints after cleanup: {len(checkpoints)}")
        assert len(checkpoints) == 2, "Should keep only 2 checkpoints"
        
        # Load latest
        ckpt = manager.load_latest()
        print(f"Latest checkpoint: step={ckpt['step']}, loss={ckpt['loss']:.4f}")
        
        print("\n[TEST PASSED]")
