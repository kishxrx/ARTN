# ARTN - Adaptive Recursive Ternary Networks

> Building Transformer Language Models Under Extreme Hardware Constraints

## Problem

The fundamental compute/memory wall on low-end hardware. Modern transformer language models demand substantial GPU memory and compute resources, leaving developers with consumer-grade or older hardware unable to experiment with and train these architectures.

## Constraints

**Target Hardware:** Lenovo E41-25
- Dual-Core AMD CPU
- 4GB RAM
- CPU-only (no GPU)

## Architecture

ARTN synthesizes five key innovations to enable transformer training under extreme hardware constraints:

| Component | Innovation | Impact |
|-----------|-----------|--------|
| **Weight RAM** | BitLinear 1.58-bit ($\{-1, 0, 1\}$) | ~85% parameter memory reduction |
| **Layer RAM** | Recursive Parameter Tying | Single layer, looped 6x |
| **Optimizer RAM** | Adafactor | ~60% optimizer state reduction |
| **Activation RAM** | Gradient Checkpointing | OOM elimination |
| **Compute Cycles** | Adaptive Halting Gate | Dynamic forward-pass reduction |

## Experiments

A rigorous ablation study measuring each optimization's isolated impact:

1. **Stage 001: Baseline** - Float32 GPT Decoder
2. **Stage 002: Recursive** - Parameter Tying
3. **Stage 003: Halting** - Adaptive Exit Gate
4. **Stage 004: INT8** - 8-bit Quantization Bridge
5. **Stage 005: Ternary** - 1.58-bit BitLinear

## Results

| Metric | Baseline | ARTN (Target) |
|--------|----------|---------------|
| Peak RAM (MB) | TBD | TBD |
| Checkpoint Size (MB) | TBD | TBD |
| Train Speed (steps/sec) | TBD | TBD |
| Inference Speed (tokens/sec) | TBD | TBD |
| Validation Loss | TBD | TBD |

*Results will be populated after experiment completion.*

## Future Work

**Era 2: ARTN-Chem** - Domain adaptation for SMILES molecular representations, training on chemical structure constraints from PubChem/ZINC databases.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ARTN.git
cd ARTN

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Usage

```bash
# Run tests
pytest

# Lint code
ruff check .

# Format code
black .

# Run training (Stage 3)
python -m src.training.loop --config experiments/001_baseline/config.yaml
```

## License

MIT

## Citation

```bibtex
@article{artn2024,
  title={Adaptive Recursive Ternary Networks: Building Transformer Language Models Under Extreme Hardware Constraints},
  author={ARTN Project},
  year={2024},
  note={Work in progress}
}
```
