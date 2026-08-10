# ARTN - Adaptive Ternary Recurrent Network

## 1.58-bit Ternary Quantization for Language Models

Experimental implementation of ternary (1.58-bit) quantization for training small language models on limited hardware.

---

##Project Status

| Version | Status |
|---------|--------|
| v0.1-experimental | Development/Experimental |

---

## Hardware Target

| Spec | Value |
|------|-------|
| RAM | 4GB |
| CPU |2-core |
| Storage | ~20GB |

---

## ResultsSo Far

| Training | Status | Details |
|----------|--------|---------|
| Attempt 1 (5k steps) | SUCCESS | Verified pipeline works |
| Attempt 2 (50k steps) | FAILED | Beta1=None caused collapse |
| Attempt 3 (2k steps) | STOPPED | Wrong gradient accumulation |
| Attempt 4 (50k steps) | COMPLETED | Healthy entropy, gibberish output |

See [TRAINING_HISTORY.md](./TRAINING_HISTORY.md) for complete documentation.

---

## Key Lessons

1. Beta1=None causes collapse (entropy drops to 8%)
2. Beta1=0.9 too conservative (loss doesn't drop enough)
3.Entropy monitoring is CRITICAL
4. Need to find optimal beta1 (testing 0.5, 0.7)

---

## Project Structure

```
artn/
├── src/
│   ├── model/         # BitLinear GPT architecture
│   ├── tokenizer/     # BPE tokenizer
│   ├── training/      # Training loop, Adafactor
│   └── inference/     # Generation utilities
├── scripts/           # Training scripts
├── tests/             # Unit tests
├── docs/              # Documentation
└── config/            # Configuration files
```

---

## Quick Start

```bash
# Clone repo
git clone https://github.com/kishxrx/artn.git
cd artn

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run quick test
python test_quick.py
```

---

## Training

```bash
# Full training (50k steps)
python scripts/train_optimized.py --total-steps 50000
```

---

## Experimental

This is an experimental project to test if ternary quantization can work on extremely limited hardware. Not production-ready.

---

## License

MIT

---

## Author

kishxrx

======================================

**Last Updated:** August 2026
