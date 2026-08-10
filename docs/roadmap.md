# ARTN Project Roadmap

## Era 1: ARTN-Core (TinyStories)

### Milestone 0: Foundation ✅
- [x] Repository structure
- [x] MLOps tooling (pytest, ruff, black, pre-commit)
- [x] CI/CD pipeline (GitHub Actions)
- [ ] Initial git commit

### Milestone 1: Tokenization
- [ ] BPE tokenizer implementation
- [ ] Regex tokenizer (ARTN-Chem prep)
- [ ] Unit tests for tokenizers

### Milestone 2: Baseline Architecture
- [ ] Float32 GPT Decoder blocks
- [ ] Sinusoidal positional encoding
- [ ] Causal masked attention
- [ ] GELU feed-forward network

### Milestone 3: Training Pipeline
- [ ] Memory-mapped data loader
- [ ] Gradient accumulation
- [ ] Adafactor optimizer wrapper
- [ ] Metrics logging to JSON

### Milestone 4: Recursive Transformer
- [ ] Parameter tying implementation
- [ ] Loop-based forward pass
- [ ] Verify 80% parameter reduction

### Milestone 5: Adaptive Halting
- [ ] Halting gate linear layer
- [ ] Cumulative probability exit condition
- [ ] Average steps logging

### Milestone 6-7: Quantization
- [ ] INT8 dynamic quantization
- [ ] BitLinear 1.58-bit implementation
- [ ] Straight-Through Estimator (STE)
- [ ] Custom autograd.Function

### Milestone 8-9: Inference & Benchmarking
- [ ] Autoregressive generation
- [ ] Top-K sampling
- [ ] CPU profiling scripts
- [ ] Memory benchmarking

### Milestone 10-11: Documentation
- [ ] Research paper compilation
- [ ] Failure analysis documentation
- [ ] Final README polish

## Era 2: ARTN-Chem (SMILES)

*Planning phase - begins after Era 1 completion*

- [ ] SMILES data pipeline
- [ ] Chemical tokenizer
- [ ] Valency validation metrics
- [ ] Domain-specific benchmarks
