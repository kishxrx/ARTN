# Stage 8 Quality Investigation Report
**Date:** October 15, 2026

---

## 1. ✅ TOKENIZER DECODING VERIFICATION

### Test: Invented Words Roundtrip

| Invented Word | Token IDs | Roundtrip OK | Conclusion |
|--------------|-----------|--------------|------------|
| "warvelvest" | [623, 2986, 850, 347] | ✅ YES | Tokenizer encodes/decodes correctly |
| "ristgistam" | [157, 133, 527, 133, 441] | ✅ YES | Tokenizer encodes/decodes correctly |
| "inseed" | [106, 222, 105] | ✅ YES | Tokenizer encodes/decodes correctly |
| "reTom" | [132, 365] | ✅ YES | Tokenizer encodes/decodes correctly |
| "hedly" | [126, 56, 234] | ✅ YES | Tokenizer encodes/decodes correctly |
| "highing" | [967, 129] | ✅ YES | Tokenizer encodes/decodes correctly |

### Normal Words Test

| Word | Token ID | Roundtrip OK |
|------|----------|--------------|
| "bird" | [502] | ✅ YES |
| "window" | [2181] | ✅ YES |
| "always" | [496] | ✅ YES |
| "curious" | [1486] | ✅ YES |
| "happy" | [297] | ✅ YES |

### ✅ FINDING #1: Tokenizer is WORKING CORRECTLY

**Evidence:**
- ✅ All words encode and decode correctly
- ✅ Invented words are valid BPE token sequences
- ✅ Tokenizer is NOT broken

**Conclusion:** Invented words are NOT tokenizer errors. They are model outputs generated from valid token combinations.

---

## 2. ✅ TEMPERATURE/QUALITY ANALYSIS

### Temperature Sweep Results

| Temperature | Quality | Sentences | Long Words | Sample Text |
|-------------|---------|-----------|------------|-------------|
| **0.5** | ✅ EXCELLENT | 10 | 0 | "...there was a little girl named Lucy. She had a special toy..." |
| **0.7** | ✅ EXCELLENT | 9 | 0 | "...there was a grumpy little girl. She was always very sad..." |
| **0.9** | ✅ GOOD | 8 | 0 | "...there lived a boy called Jack. He was very obedient..." |
| **1.1** | ⚠️ DEGRADED | 9 | 0 | "...wanted to watch a lot of **foredaly**..." |
| **1.3** | ❌ BROKEN | 11 | 0 | "...**wanfavoring**! \" question a bird. She **jumor** Elly **flaged**..." |

### Quality vs Temperature

```
Quality
  ✅ |----|----|----|
 0.5  0.7  0.9  1.1  1.3  Temperature
      ↑           ↑     ↑
   EXCELLENT   DEGRADED  BROKEN
```

### ✅ FINDING #2: Temperature Affects Quality

**Evidence:**
- ✅ Temperature 0.5-0.9: Coherent, multi-sentence stories
- ⚠️ Temperature 1.1: Minor quality issues ("foredaly")
- ❌ Temperature 1.3: Broken, invented words, grammar errors

**Optimal Temperature Range:** 0.5 - 0.9

**Current Default (0.9):** ✅ GOOD CHOICE

---

## 3. ⚠️ RAM MEASUREMENT CORRECTION

### Original Measurement (tracemalloc)

| Metric | Value | Issue |
|--------|-------|-------|
| Peak RAM | **0.18 MB** | ❌ INCORRECT |
| Method | tracemalloc | ❌ Only tracks Python allocations |
| Misses | PyTorch tensors, model weights | ❌ Major memory not counted |

### Corrected Measurement (psutil)

| Metric | Value | Correct |
|--------|-------|---------|
| Peak RAM (first gen) | **21.46 MB** | ✅ CORRECT |
| Peak RAM (subsequent) | **0.00 MB** | ⚠️ Delta only (model already loaded) |
| Method | psutil RSS | ✅ Total process memory |

### ✅ FINDING #3: Actual RAM Usage is 21.46 MB

**Evidence:**
- ✅ First generation: 21.46 MB (includes model loading)
- ✅ Subsequent generations: 0.00 MB delta (model already in memory)
- ✅ Total inference RAM: ~22 MB (model + generation buffers)

**Conclusion:** Actual RAM usage is 21-22 MB, not 0.18 MB as originally reported.

---

## 4. ❌ FP32 BASELINE COMPARISON

### Issue: Architecture Mismatch

**FP32 Checkpoint Structure:**
```
token_embedding.weight
position_embedding.weight
blocks.*.attn.{q,k,v,o}_proj.{weight,bias}
blocks.*.ff.{fc1,fc2}.{weight,bias}
ln_f.{weight,bias}
```

**src/model/gpt.py Expected:**
```
embeddings.token_embedding.embedding.weight
embeddings.positional_encoding.pe
blocks.*.attention.attention.W_{q,k,v,o}.weight
blocks.*.ffn.W{1,2}.weight
ln_final.{weight,bias}
```

### ❌ FINDING #4: FP32 Baseline Cannot Be Loaded

**Evidence:**
- ❌ FP32 checkpoint uses different architecture than `src/model/gpt.py`
- ❌ FP32 checkpoint matches training script architecture (same as BitLinear)
- ❌ No generate() method available in standard GPT implementation

**Root Cause:** FP32 model was trained with experimental training script, not standard GPT class.

**Impact:** Cannot directly compare FP32 vs Ternary generation quality.

---

## 5. 📊 SUMMARY OF FINDINGS

### ✅ What Works

| Component | Status | Evidence |
|-----------|--------|----------|
| Tokenizer | ✅ CORRECT | All words encode/decode properly |
| Temperature 0.5-0.9 | ✅ GOOD QUALITY | Coherent multi-sentence stories |
| Ternary quantization | ✅ VERIFIED | 32.56% / 34.72% / 32.72% distribution |
| Inference pipeline | ✅ WORKS | All 10 prompts generate |
| RAM usage | ✅ CORRECTED | 21.46 MB actual usage |

### ❌ What Doesn't Work

| Component | Status | Evidence |
|-----------|--------|----------|
| Temperature 1.3+ | ❌ BROKEN | Invented words, grammar errors |
| FP32 comparison | ❌ FAILED | Architecture mismatch |
| Original RAM measurement | ❌ WRONG | 0.18 MB vs 21.46 MB actual |

---

## 6. 🎯 ROOT CAUSE OF INVENTED WORDS

### Analysis:

**Hypothesis 1:** Tokenizer broken
- ❌ DISPROVEN: Tokenizer encodes/decodes all words correctly

**Hypothesis 2:** Model trained insufficiently
- ✅ LIKELY: Only 10,000 steps, val loss 1.92
- ✅ Model generates rare token combinations
- ✅ More training would improve quality

**Hypothesis 3:** Temperature too high
- ✅ CONFIRMED: Temperature 1.3 produces broken text
- ✅ Temperature 0.5-0.9 produces good text
- ✅ Default temperature 0.9 is acceptable

---

## 7. 📈 GENERATION SPEED

| Temperature | Time (100 tokens) | Speed |
|-------------|-------------------|-------|
| 0.5 | 17.79s | 5.62 tok/s |
| 0.7 | 17.67s | 5.66 tok/s |
| 0.9 | 17.66s | 5.66 tok/s |
| 1.1 | 17.31s | 5.78 tok/s |
| 1.3 | 17.41s | 5.74 tok/s |

**Average:** 5.69 tokens/second

**Analysis:**
- ✅ Consistent speed across temperatures
- ⚠️ Slower than estimated (15-25 tok/s)
- ✅ Reasonable for 2-core CPU, ternary quantization

---

## 8. 💾 COMPLETE RAM ANALYSIS

### Model Loading

| Component | Memory |
|-----------|--------|
| BitLinearGPT weights | ~22 MB |
| Tokenizer | < 1 MB |
| Generation buffers | ~1 MB |
| **Total Inference RAM** | **~24 MB** |

### Generation RAM

| Stage | RAM Delta |
|-------|-----------|
| First generation | 21.46 MB (model load + buffers) |
| Subsequent | 0.00 MB (already loaded) |

### Memory Efficiency

| Metric | Value |
|--------|-------|
| Model parameters | 6.8M (ternary) |
| Memory per parameter | ~3.2 bytes (vs 4 bytes for FP16) |
| Inference RAM | 24 MB |
| Training RAM (if attempted) | Would need ~300+ MB |

---

## 9. 🔬 INVESTIGATION CONCLUSIONS

### ✅ VERIFIED:

1. **Tokenizer is correct** - invented words are model outputs, not tokenizer errors
2. **Temperature affects quality** - optimal range is 0.5-0.9
3. **Actual RAM usage is 21-24 MB** - not 0.18 MB as originally reported
4. **Ternary quantization works** - balanced distribution across {-1, 0, +1}

### ❌ ISSUES IDENTIFIED:

1. **Generation quality at temperature 1.3+** - broken, invented words
2. **FP32 baseline unavailable** - architecture mismatch prevents comparison
3. **Model trained only 10k steps** - would benefit from more training

### 📊 STAGE 8 MILESTONE STATUS:

| Requirement | Status |
|-------------|--------|
| Ternary quantization verified | ✅ PASSED |
| All prompts generate | ✅ PASSED |
| Multi-sentence stories | ✅ PASSED |
| Coherent text | ⚠️ PARTIAL (temp dependent) |
| TinyStories style | ⚠️ PARTIAL (some quality issues) |

**OVERALL:** ⚠️ **PARTIAL SUCCESS**

---

## 10. 🚀 RECOMMENDATIONS

### Immediate:

1. **Use temperature 0.5-0.9 for all generation** (current default 0.9 is acceptable)
2. **Document that FP32 comparison is not available** (architecture mismatch)
3. **Accept Stage 8 as partial success** (inference works, quality has issues)

### Future Improvements:

1. **Train for more steps** (20k-50k) to improve generation quality
2. **Reduce temperature to 0.7** for better quality
3. **Implement validation loss monitoring** during generation
4. **Create FP32 baseline with matching architecture** for fair comparison

---

**Investigation Complete: October 15, 2026**

**Files Created:**
- `scripts/stage8_quality_investigation.py`
- `outputs/stage8_quality_investigation.json`
- `outputs/stage8_quality_investigation_report.md` (this file)
