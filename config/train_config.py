"""

ARTN Training Configuration.

Hardware: Lenovo E41-25, Dual-Core AMD CPU, 4GB RAM, CPU-only
Model: ~2.83M parameters (BitLinear Ternary)
Dataset: TinyStories
"""

# =============================================================================
# MODEL CONFIGURATION (per blueprint)
# =============================================================================
MODEL_CONFIG = {
    "vocab_size": 4000,        # BPE vocabulary
    "d_model": 256,            # Model dimension
    "n_heads": 8,              # Attention heads
    "n_layers": 6,             # Transformer layers (recursive loop)
    "d_ff": 1024,              # Feed-forward dimension
    "max_seq_len": 256,        # Maximum sequence length
    "dropout": 0.1,            # Dropout rate
}

# =============================================================================
# TRAINING CONFIGURATION (optimized for 4GB RAM, dual-core CPU)
# =============================================================================
TRAINING_CONFIG = {
    # Batch settings
    "batch_size": 4,                    # Fits in 4GB RAM
    "gradient_accumulation_steps": 4,  # Effective batch = 16
    
    # Sequence settings
    "max_seq_len": 256,
    
    # Optimizer (Adafactor - memory efficient)
    "optimizer": "adafactor",
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "relative_step": False,
    "scale_parameter": True,
    "warmup_init": True,
    
    # Learning rate schedule
    "warmup_steps": 500,
    "lr_scheduler": "cosine",
    "min_lr": 1e-5,
    
    # Training duration
    "max_steps": 10000,                 # ~2-3 hours on dual-core CPU
    "num_epochs": None,                 # Use max_steps instead
    
    # Gradient settings
    "max_grad_norm": 1.0,                # Gradient clipping
    "fp16": False,                       # No mixed precision (CPU)
    
    # Logging
    "log_every": 100,                     # Log every N steps
    "eval_every": 500,                    # Evaluate every N steps
    "sample_every": 1000,                 # Generate sample every N steps
    
    # Checkpointing
    "checkpoint_every": 500,             # Save checkpoint every N steps
    "checkpoint_dir": "checkpoints",
    "keep_last_n_checkpoints": 3,        # Keep last N checkpoints
    
    # Resume training
    "resume_from": None,                 # Path to checkpoint to resume from
}

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
DATASET_CONFIG = {
    "name": "roneneldan/TinyStories",
    "subset": None,
    "split_train": "train",
    "split_valid": "validation",
    
    # Use subset for faster training (optional)
    "train_samples": None,               # None = full dataset, or set limit
    "valid_samples": 1000,               # Validation set size
    
    # Preprocessing
    "max_length": 256,
    "truncate": True,
    "padding": "max_length",
}

# =============================================================================
# TOKENIZER CONFIGURATION
# =============================================================================
TOKENIZER_CONFIG = {
    "type": "bpe",                       # BPE tokenizer
    "vocab_size": 4000,
    "min_frequency": 2,
    "special_tokens": {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    },
}

# =============================================================================
# HARDWARE CONFIGURATION
# =============================================================================
HARDWARE_CONFIG = {
    "device": "cpu",
    "num_workers": 2,                    # Data loading workers (dual-core)
    "pin_memory": False,                 # No GPU
    "prefetch_factor": 2,
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
OUTPUT_CONFIG = {
    "output_dir": "outputs",
    "experiment_name": "artn-core-v1",
    "log_dir": "logs",
    "tensorboard": False,                # No tensorboard (minimal setup)
    "wandb": False,                      # No wandb
}

# =============================================================================
# QUICK PRESETS
# =============================================================================
PRESETS = {
    "quick": {
        # 3 min test - just verify everything works
        "max_steps": 200,
        "checkpoint_every": 100,
        "eval_every": 50,
        "train_samples": 5000,
    },
    "minimal": {
        # Quick test - 30 min
        "max_steps": 1000,
        "checkpoint_every": 200,
        "eval_every": 100,
        "train_samples": 10000,
    },
    "short": {
        # 1-2 hours
        "max_steps": 5000,
        "checkpoint_every": 500,
        "eval_every": 250,
        "train_samples": None,
    },
    "full": {
        # 3-5 hours
        "max_steps": 10000,
        "checkpoint_every": 500,
        "eval_every": 500,
        "train_samples": None,
    },
    "extended": {
        # 10+ hours
        "max_steps": 50000,
        "checkpoint_every": 1000,
        "eval_every": 1000,
        "train_samples": None,
    },
}


def get_config(preset: str = "short") -> dict:
    """Get training config with preset applied."""
    config = TRAINING_CONFIG.copy()
    config["model"] = MODEL_CONFIG.copy()
    config["dataset"] = DATASET_CONFIG.copy()
    config["tokenizer"] = TOKENIZER_CONFIG.copy()
    config["hardware"] = HARDWARE_CONFIG.copy()
    config["output"] = OUTPUT_CONFIG.copy()
    
    if preset in PRESETS:
        for key, value in PRESETS[preset].items():
            config[key] = value
    
    return config


if __name__ == "__main__":
    # Print config summary
    print("=" * 60)
    print("ARTN Training Configuration")
    print("=" * 60)
    print(f"\nModel parameters: ~2.83M")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Effective batch: {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Max steps: {TRAINING_CONFIG['max_steps']}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Device: {HARDWARE_CONFIG['device']}")
    print(f"\nPresets: {list(PRESETS.keys())}")
    print("=" * 60)
