# ARTN — Adaptive Recursive Ternary Networks

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c.svg)
![CPU-only](https://img.shields.io/badge/Runs%20on-CPU--only-orange.svg)

> **A working language model trained from scratch on a 2-core CPU with 4 GB RAM — no GPU, no cluster, no excuses.**

> Building Transformer Language Models Under Extreme Hardware Constraints

**Status:** Stages 1–9 complete — ARTN trained, benchmarked and generating coherent text on CPU-only hardware.

## Table of contents

- [Result at a glance](#result-at-a-glance)
- [Why this matters](#why-this-matters)
- [The journey](#the-journey)
- [Problem](#problem)
- [Constraints](#constraints)
- [Architecture](#architecture)
- [Experiments](#experiments)
- [Results](#results)
- [Example output](#example-output)
- [Evaluation notes](#evaluation-notes)
- [Investigation](#investigation)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Usage](#usage)
- [Research Programme](#research-programme)
- [Active Work](#active-work)
- [License](#license)
- [Citation](#citation)

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

## Why this matters

Modern transformer language models are built on a simple assumption: you have GPUs. Lots of them. The papers, the open-weight releases, the training recipes — all assume A100s or better. If you don't have that hardware, you're a spectator.

ARTN challenges that assumption. Three architectural innovations — ternary weights, recursive parameter tying, and a learned halting gate — are stacked into a single model that trains end-to-end on a 4 GB laptop CPU and generates coherent text. Not a toy demo. Not a fine-tune. A model trained from scratch in 14 hours, producing stories with narrative structure, dialogue, and subject-specific vocabulary.

The techniques are not specific to this hardware. Ternary quantization, parameter sharing, and adaptive computation are general principles. If they work at 4 GB on a dual-core CPU, they work on edge devices, on IoT, on mobile — anywhere the memory wall is real.

## The journey

ARTN did not work on the first try. It did not work on the fifth.

**8 training attempts · 5 failed · 211,000 total steps · 390+ hours · 3.5 months**
**62 investigation phases · 11,000+ lines of documentation**

| # | Date | Steps | Hours | Result |
|---|---|---|---|---|
| 1 | Aug 1 | 5,000 | 4 | Pipeline verified |
| 2 | Aug 3–6 | 50,000 | 79 | COLLAPSED — "time time time..." |
| 3 | Aug 6 | 2,000 | 1 | Stopped — wrong accumulation |
| 4 | Aug 6–10 | 50,000 | 83 | Completed — healthy but gibberish |
| 5 | Aug 13–16 | 50,000 | 82 | Completed — healthy but gibberish |
| 6 | Aug 19 | 5,000 | — | FP32 + AdamW — first success (val 1.69) |
| 7 | Sep 10–11 | 10,000 | — | BitLinear + AdamW — ternary works (val 1.92) |
| **8** | **Oct 25** | **10,000** | **14.35** | **ARTN — the model that works (val 2.03)** |

244 hours of failed training across three 50K-step runs before the root cause was found: a bug in the optimizer that silently overwrote variance-scaled updates with momentum-only updates — making it behave like SGD with momentum instead of Adafactor. The model trained, the loss dropped, the entropy looked healthy. The output was gibberish.

The fix: switch to AdamW. Then build ARTN on top.

What this taught: that loss going down does not mean the model is learning. That entropy can be healthy while the output is gibberish. That an optimizer can compute the right gradients and then silently throw them away. That the difference between a working model and a broken one can be a single line of code — and finding it can take weeks. Every failure is documented in the [Investigation](#investigation) section below, with evidence files for every claim.

## Problem

The fundamental compute/memory wall on low-end hardware. Modern transformer language models demand substantial GPU memory and compute resources, leaving developers with consumer-grade or older hardware unable to experiment with and train these architectures.

## Constraints

**Target Hardware:** Lenovo E41-25
- Dual-Core AMD CPU
- 4GB RAM
- CPU-only (no GPU)

## Architecture

ARTN stacks three innovations into a single architecture. Each one attacks a different resource bottleneck. Every figure below is measured from the released checkpoint.

```
 Input tokens
     │
     ▼
┌─────────────┐
│  Embeddings  │   (token + position, not shared)
└──────┬──────┘
       │
       ▼
┌──────────────────┐    pass 1
│  BitLinearBlock   │◄── shared weights (ONE block, used 6× max)
│  {-1, 0, 1}       │
└──────┬───────────┘
       │            ┌──────────────┐
       ├───────────►│ Halting Gate  │  "has this position converged?"
       │            │ sigmoid(W·x)  │  learned per-token, per-pass
       │            └──────┬───────┘
       │                   │
      no (p < θ)           │ yes (p ≥ θ)
       │                   │
       ▼                   ▼
┌──────────────────┐    stop updating this position
│  BitLinearBlock   │◄── same weights, pass 2
│  (same weights)   │
└──────┬───────────┘
       │              ... repeat up to max_passes = 6
       ▼
┌─────────────┐
│   ln_f +     │
│   lm_head    │
└──────┬──────┘
       │
       ▼
   logits → next token
```

| Constraint | Technique | Measured effect |
|---|---|---|
| **Weight memory** | BitLinear 1.58-bit `{-1, 0, 1}` with straight-through estimation | 2,904,065 params vs 5,828,608 FP32 — **50.2% fewer** |
| **Depth memory** | Recursive parameter tying — one shared block, 6 passes | **789,760** params in the shared block (instead of 6 × 789,760) |
| **Compute cycles** | Learned adaptive halting gate | **3.57** average passes at threshold 0.9 (vs 6.0) |
| **Training stability** | AdamW + cosine schedule + gradient clipping | converged CPU-only, 0 NaN/Inf |
| **Hardware envelope** | — | full training in 14.35 h on 4 GB RAM |

**How the three fit together.** Ternary weights cut the model's parameter footprint in half — each weight is one of three values instead of a 32-bit float. Parameter tying cuts it further: instead of six separate transformer blocks, ARTN uses one block and runs it six times. The halting gate then makes those six passes optional — each token position learns when it has "converged" and stops early. At inference, the model averages 3.57 passes instead of 6, cutting compute by 40% for a 0.08% NLL cost. Training runs all six passes for exact gradients; halting applies only at inference.

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

ARTN checkpoint `step_10000` — sign-aware repetition penalty 1.3, greedy decoding, seed 42. Each sample is from `broader_gen_eval.json` (config `C_penalty1.3_t1.0`, 20 prompts × 4 decoding configurations).

> **Prompt:** `once there was a little boy`
>
> **Output:** once there was a little boy who liked to play with his toys. One day, he saw something shiny in the box. He wanted to open it and see what it was. The little boy looked around and saw that it had many colors. It was so big that it could not touch it. But then the boy heard the noise and came out of the room. His mom said, "Please help me find it." She picked up the toy box and put it on the shelf. The little boy was very happy and thanked the boy for helping him. From that day on,

> **Prompt:** `a little bird flew high`
>
> **Output:** a little bird flew high in the sky. The bird was very happy to fly and it could fly high up high. But then one day, the bird saw something shiny on it. It had a long tail and it made a loud noise. Suddenly, the bird heard a voice. He looked around and saw a big bird. The bird said, "Hello, bird!" The bird smiled at the bird. The bird felt so happy. From that day on, the bird became friends and played together every day.

> **Prompt:** `a dragon lived in a cave`
>
> **Output:** a dragon lived in a cave and he was very happy. One day, he decided to go for a walk with his friends. He walked around the forest and saw a big tree. The bird looked so high that it could not find it. It had many friends like a long tail. They were happy and excited. But then, they heard a loud noise. A man came out of the window and saw the bird flying towards them. The bird said "Hello, bird," said the bird. "Do you want to help me get my friend?" The bird said, "Yes, I can help you." The

The model generates subject-specific vocabulary — the bird sample talks about flying and the sky; the dragon sample mentions a cave, a forest, and friends. Coherence degrades toward the end of each 100-word window, which is expected for a 2.9M-parameter model trained for 10k steps on TinyStories.

## Evaluation notes

Results are measured on TinyStories — a synthetic children's-story corpus — after 10,000 steps, and characterise that setting rather than general-domain text.

Generation quality is decoding-sensitive. Greedy decoding without a repetition penalty drifts into repetition; the sign-aware penalty of 1.3 used above removes it and produces the prose shown. Reproducing the example therefore means reproducing the decoding configuration, not just the checkpoint.

Prompt adherence for rare subjects remains the open problem, and is the focus of the ongoing work described below.

## Investigation

Every phase is documented. Every measurement is backed by an evidence file. Every failure is recorded — including the dead ends.

### The investigation

ARTN didn't appear out of nowhere. It took 390+ hours, 8 training attempts, 5 failures, and 62 phases of systematic investigation to get here — and every step is documented.

There's a full investigation record — **11,000+ lines** — that goes deeper than this README. It covers:

- A failure mode so subtle it masqueraded as success — and the forensics that finally exposed it
- Why a model can lose loss steadily and still produce gibberish
- How systematic elimination — not guesswork — narrowed months of dead ends to a single root cause
- Six diagnostic probes that mapped how knowledge is structured inside a 2.9M-parameter ternary model — and where it breaks
- A controlled four-arm experiment designed to settle a question — and the unexpected answer it returned
- Findings that defied every prediction and opened new questions the investigation is still chasing

Every measurement backed by an evidence file. Every dead end recorded.

This document is not in the public repo. It's available to anyone who wants to understand what it really takes to build a language model from scratch on a 4 GB laptop.

**If you want to see it, reach out.**

---

[`TRAINING_HISTORY.md`](TRAINING_HISTORY.md) — complete training history with configurations, loss trajectories, and results for every run — is in the repo.

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

## Research Programme

**The Mission: Advanced AI on Ordinary Hardware.**

Modern AI increasingly relies on massive computing clusters and proprietary cloud infrastructure. ARTN investigates an alternative foundation: architectures designed to compute efficiently, develop reusable capabilities, learn continuously, and expand without requiring ever-increasing hardware resources.

The objective is to make building, training, and specializing AI systems increasingly accessible on ordinary consumer hardware.

ARTN already provides an experimental foundation for this programme: a 2.9-million-parameter language model trained from scratch on a dual-core CPU with 4 GB of RAM, with documented architectural ablations, training investigations, inference benchmarks, and generation results. These experiments demonstrate what has been achieved while exposing the limitations that motivate the research directions below.

The following frontiers represent the next areas of investigation, not claims of completed capabilities.

| Frontier | Core Question | Architectural Direction |
| :--- | :--- | :--- |
| **1. Efficient Intelligence** | How can computation be allocated dynamically? | Ternary weights, recursive parameter tying, adaptive halting |
| **2. Reusable Intelligence** | How can learned representations transfer across tasks? | Compositional abstractions, shared computational structures |
| **3. Continual Intelligence** | How can models continue learning after initial training? | Offline memory consolidation, sleep/dreaming-inspired replay |
| **4. Expandable Intelligence** | How can compact models acquire additional capacity? | Modular expansion, selective activation, capability preservation |
| **5. Grounded Intelligence** | How can generation be separated from verified information and execution? | Local retrieval, deterministic tool execution, evidence verification |
| **6. Domain Specialization** | How can a shared architecture adapt to specialized fields? | Structural priors, domain-specific representations, specialized training |
| **7. Zero-Infrastructure AI** | How can we democratize training, not just inference? | CPU-first computation, low-RAM training, reproducible experimentation |

### 1. Efficient Intelligence — Rethinking Computation

Intelligence depends not just on how much compute is available, but on how selectively it is spent. Rather than relying exclusively on fixed-depth dense networks, ARTN investigates **adaptive computational depth**. Its current architecture combines ternary weights (`{-1, 0, 1}`), recursive parameter sharing, and learned adaptive halting to process representations through repeated computational passes.

> **Focus areas:** Dynamic compute allocation · Parameter reuse across depth · Adaptive early halting · CPU-efficient ternary operations · Compression-vs-throughput trade-offs

The long-term direction is to make computation more selective and efficient, rather than simply making models smaller.

### 2. Reusable Intelligence — Beyond Memorization

Memorizing surface patterns is not the same as developing transferable capabilities. ARTN's shared recursive block provides an architectural foundation for investigating whether repeated computation can develop useful internal operations that transfer across different contexts.

> **Focus areas:** Compositional abstractions · Shared computational structures · Modular recombination · Separating general vs. task-specific representations · Zero-rebuild transfer

The objective is to investigate whether compact models can develop reusable computational capabilities rather than relying primarily on memorizing individual patterns.

### 3. Continual Intelligence — Learning Beyond Training

Training should be the beginning of a model's development, not the end of it. This is one of my main goals. 
Most language models have a fixed training phase followed by inference, requiring costly retraining or external memory to incorporate new knowledge.

ARTN's longer-term direction investigates mechanisms for acquiring, retaining, and consolidating knowledge over time—drawing inspiration from **sleep, dreaming, and memory consolidation** (see Google Research's [Sleep Framework](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/) and [*Language Models Need Sleep*](https://arxiv.org/abs/2606.03979)). The central idea is separating active learning from offline consolidation: periodically revisiting accumulated experience, reinforcing useful knowledge, and reorganizing representations.

> **Focus areas:** Offline memory consolidation · Selective retention and pruning · Biological replay mechanisms · Synthetic experience generation · Mitigating catastrophic forgetting on low-resource hardware

This challenges the assumption that a model's development must stop when its initial training is complete.

### 4. Expandable Intelligence — A Foundation That Can Grow

Conventional architectures generally lock their capacity before training begins, meaning increased capability requires scaling up and repeating substantial portions of training. This direction investigates **progressive architectural growth**: starting with a compact computational foundation and introducing additional capacity as new requirements emerge.

> **Focus areas:** Modular capacity addition · Integrating independently trained components · Task-conditional activation · Combining modular growth with recursive execution · Capability preservation during expansion

The long-term ambition is to investigate architectures that grow incrementally rather than requiring the entire model to become larger whenever new capabilities are needed.

### 5. Grounded Intelligence — Knowledge, Tools, and Verification

Generating plausible tokens is not the same as establishing truth, and generating an instruction is not the same as successfully executing it. A compact model should not need to memorize an entire encyclopedia or independently reproduce every external capability. Instead, this direction combines learned representations with external knowledge and deterministic execution.

> **Focus areas:** Local knowledge retrieval · Deterministic tool execution · Evidence verification · Separating reasoning from validation · Execution-feedback loops

The broader objective is to investigate how compact models can serve as capable reasoning cores while keeping knowledge, tools, and execution accessible locally.

### 6. Domain Specialization — One Foundation, Multiple Fields

Language modelling is the current experimental focus of ARTN, but the architectural principles being investigated are not limited to natural language. Different technical domains have their own structural rules, representations, and constraints, requiring targeted adaptation rather than general-purpose text prediction alone.

> **Focus areas:** Domain-specific structural priors · Curated compact datasets · Recursive computation on structured data · Cross-domain ternary transfer · Low-infrastructure specialization across scientific, mathematical, logical, and engineering workflows

The objective is to investigate whether a shared architectural foundation can support multiple specialized models through targeted adaptation.

### 7. Intelligence Without Infrastructure

Open weights mean little if only organizations with data centers can meaningfully train, modify, and develop AI systems. Training ARTN from scratch on a dual-core CPU with 4 GB of RAM demonstrates that meaningful architectural experimentation is possible without GPU clusters—but true accessibility requires optimizing the entire development lifecycle, not just inference.

> **Focus areas:** Memory-conscious CPU training loops · Low-RAM optimization · CPU-native computational kernels · Local domain adaptation · Reproducible evaluation harnesses · Transparent failure forensics

The objective is to lower the barrier not just to *using* AI, but to building, training, understanding, and improving it.

### Active Work

My current ongoing work focuses on concrete limitations and extensions of the existing ARTN implementation:

- **Context and Prompt Adherence:** Reducing long-context repetition, improving subject fidelity, and evaluating generation quality across diverse prompts.
- **Halting and Training Efficiency:** Refining exit-gate calibration, investigating computation allocation, and improving CPU-native ternary operations.
- **Architectural Generalization:** Evaluating whether recursive representations and shared computational structures produce transferable capabilities beyond the original training distribution.

> *Advanced AI should not belong exclusively to those who can afford the infrastructure to train it.*


## License

**Code** is licensed under [GPL-3.0](LICENSE) — you may use, modify, and distribute, but derivatives must also be open-sourced under GPL.

**Documentation** (investigation reports, training history, and analysis files in `experiments/results/` and `outputs/`) is licensed under [CC-BY-NC 4.0](docs/LICENSE-CC-BY-NC) — you may read and learn, but may not use commercially without permission, and must provide attribution.

## Citation

```bibtex
@article{artn2026,
  title  = {Adaptive Recursive Ternary Networks: Training Transformer Language Models Under Extreme Hardware Constraints},
  author = {ARTN Project},
  year   = {2026},
  note   = {Active research project}
}
```

