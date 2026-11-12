# ARTN - Adaptive Recursive Ternary Networks

> Building Transformer Language Models Under Extreme Hardware Constraints

**Status:** Stages 1-9 complete — ARTN trained, benchmarked and generating coherent text on CPU-only hardware.

## Result at a glance

| | |
|---|---|
| **Architecture** | BitLinear 1.58-bit ternary, one shared block run 6x with a learned halting gate |
| **Parameters** | **2,904,065** — 50.2% fewer than the FP32 baseline |
| **Training** | 10,000 steps in **14.35 hours**, CPU-only |
| **Validation loss** | **2.0346** (perplexity 7.65) |
| **Fair non-padding NLL** | **2.904** (FP32 2.423 · BitLinear 2.743) |
| **Stability** | 0 NaN/Inf losses; halting converged to 4.00 expected passes |
| **Hardware** | Dual-core AMD CPU, 4 GB RAM, no GPU |

## Problem

The fundamental compute/memory wall on low-end hardware. Modern transformer language models demand substantial GPU memory and compute resources, leaving developers with consumer-grade or older hardware unable to experiment with and train these architectures.

## Constraints

**Target Hardware:** Lenovo E41-25
- Dual-Core AMD CPU
- 4GB RAM
- CPU-only (no GPU)

## Architecture

ARTN combines five techniques to train a transformer under extreme hardware constraints. Every figure below is measured from the released checkpoint.

| Constraint | Technique | Measured effect |
|---|---|---|
| **Weight memory** | BitLinear 1.58-bit `{-1, 0, 1}` with straight-through estimation | 2,904,065 params vs 5,828,608 FP32 — **50.2% fewer** |
| **Depth memory** | Recursive parameter tying — one shared block, 6 passes | **789,760** params in the shared block |
| **Compute cycles** | Learned adaptive halting gate | **3.57** average passes at threshold 0.9 (vs 6.0) |
| **Training stability** | AdamW + cosine schedule + gradient clipping | converged CPU-only, 0 NaN/Inf |
| **Hardware envelope** | — | full training in 14.35 h on 4 GB RAM |

## Experiments

A rigorous ablation study measuring each optimization's isolated impact:

1. **Stage 001: Baseline** - Float32 GPT Decoder
2. **Stage 002: Recursive** - Parameter Tying
3. **Stage 003: Halting** - Adaptive Exit Gate
4. **Stage 004: INT8** - 8-bit Quantization Bridge
5. **Stage 005: Ternary** - 1.58-bit BitLinear

## Results

### Training

| Metric | Value |
|---|---|
| Steps | 10,000 |
| Wall-clock | 14.35 h (51,653 s) |
| Validation loss | **2.0346** |
| Validation perplexity | 7.65 |
| Train loss | 2.0021 |
| Total parameters | 2,904,065 |
| Shared block parameters | 789,760 |
| Halting gate parameters | 257 |
| NaN / Inf losses | 0 |
| Expected passes (learned) | 4.0015 (target 4.0) |
| Ternary sparsity | 35.4% zeros, ~32.3% each +/-1 |

### Model comparison

| Model | Parameters | Val loss | Perplexity | Fair non-padding NLL |
|---|---|---|---|---|
| FP32 baseline | 5,828,608 | 1.69 | 5.44 | 2.423 |
| BitLinear GPT | 6,852,682 | 1.92 | 6.81 | 2.743 |
| **ARTN** (1 shared block) | **2,904,065** | **2.03** | **7.65** | **2.904** |

ARTN reaches comparable quality with **50.2% fewer parameters than FP32**, using one shared block instead of six unique ones.

### CPU inference benchmark

500 tokens, greedy decoding, seed 42, RSS measured via `psutil` (`scripts/stage9_benchmark.py`):

| Configuration | Throughput | Avg passes | Peak RSS |
|---|---|---|---|
| FP32 baseline | 22.62 tok/s | 6.00 | 278.4 MB |
| BitLinear GPT | 5.45 tok/s | 6.00 | 354.6 MB |
| ARTN, threshold 1.0 | 5.29 tok/s | 6.00 | 282.3 MB |
| **ARTN, threshold 0.9** | **8.42 tok/s** | **3.57** | **276.4 MB** |

The adaptive halting gate lifts throughput from 5.29 to **8.42 tokens/sec** at threshold 0.9 while lowering peak memory, by executing 3.57 passes on average instead of 6 — **1.59x faster** at a cost of +0.0024 NLL. FP32 uses PyTorch's native optimized matmul; ternary layers require per-layer scaling on CPU, which is what ARTN's halting recovers.

The halting gate averages **3.3-3.6 passes** at threshold 0.9 depending on the prompt set — 3.30 on the evaluation batches, 3.57 during this generation benchmark.

## Example output

ARTN checkpoint `step_10000` — sign-aware repetition penalty 1.3, greedy decoding, seed 42.

> **Prompt:** `once there was a little boy`
>
> **Output:** once there was a little boy who liked to play with his toys. One day, he saw something shiny in the box. He wanted to open it and see what it was. The little boy looked around and saw that it had many colors. It was so big that it could not touch it. But then the boy heard the noise and came out of the room. His mom said, "Please help me find it." She picked up the toy box and put it on the shelf. The little boy was very happy and thanked the boy for helping him. From that day on,

Reproducible from `outputs/artn_recursive_20261025_002725/broader_gen_eval.json` (config `C_penalty1.3_t1.0`, 20 prompts x 4 decoding configurations).

## Evaluation notes

Results are measured on TinyStories — a synthetic children's-story corpus — after 10,000 steps, and characterise that setting rather than general-domain text.

Generation quality is decoding-sensitive. Greedy decoding without a repetition penalty drifts into repetition; the sign-aware penalty of 1.3 used above removes it and produces the prose shown. Reproducing the example therefore means reproducing the decoding configuration, not just the checkpoint.

Prompt adherence for rare subjects remains the open problem, and is the focus of the ongoing work described below.

## Investigation

This repository is also a complete research record. Two documents hold the full account:

| Document | Contents |
|---|---|
| [`ROOT_CAUSE_INVESTIGATION_SUMMARY.md`](experiments/results/ROOT_CAUSE_INVESTIGATION_SUMMARY.md) | Phases 1-60.27 — root-cause analysis, optimizer forensics, architecture development, generation-quality and context-retention studies. Measured findings with evidence files for every claim. |
| [`TRAINING_HISTORY.md`](TRAINING_HISTORY.md) | Complete training history — configurations, loss trajectories, checkpoints and results for every run. |

Supporting material: `experiments/results/` (per-phase reports), `outputs/` (per-experiment JSON results), `scripts/` (evaluation and diagnostic tooling).

## Repository layout

```
src/                            Model, training, tokenizer, inference
experiments/                    Experiment scripts, model definitions, results/
scripts/                        Evaluation, benchmarking and diagnostic tooling
outputs/                        Per-experiment results, checkpoints and logs
config/  docs/  tests/           Configuration, documentation, test suite
experiments/bitnet_references/  Third-party BitNet reference clones
                                (reference material only, not built or imported)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/kishxrx/artn.git
cd artn

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
# Run the test suite
pytest

# Lint and format
ruff check .
black .

# Train from a preset (presets are defined in config/train_config.py)
python scripts/train.py --preset short

# Reproduce the ARTN run reported above (writes outputs/artn_recursive_<timestamp>/)
python experiments/artn_train.py

# Interactive generation from a trained checkpoint
python scripts/artn_interactive_gen.py

# Evaluation and benchmarking
python scripts/fair_eval.py              # fair non-padding NLL
python scripts/stage9_benchmark.py       # CPU throughput + peak RSS
python scripts/artn_threshold_sweep.py   # halting threshold sweep
```

## Future Work

**Era 2: ARTN-Chem** — domain adaptation for SMILES molecular representations, training on chemical structure constraints from PubChem/ZINC databases.

## License

MIT

## Citation

```bibtex
@article{artn2026,
  title  = {Adaptive Recursive Ternary Networks: Training Transformer Language Models Under Extreme Hardware Constraints},
  author = {ARTN Project},
  year   = {2026},
  note   = {Active research project}
}
```

---

## Ongoing research

ARTN is an active project and the model continues to improve. Current work focuses on generation quality and prompt adherence — reducing repetition, sustaining coherence over longer contexts, and following the prompt more faithfully across a wider range of subjects. Alongside that: further halting-threshold tuning, training-efficiency work, and an expanding evaluation suite.
