# ARTN TODO List

## After Current 50k Training Completes

### PRIORITY 1: Build Checkpoint Evaluation System

User approved on Aug 14, 2026.

**Features to implement:**

1. **Fixed Prompts Evaluation**
   - Prompt 1: "Once upon a time"
   - Prompt 2: "The little boy went to"
   - Prompt 3: "In a small village"
   - Prompt 4: "The dog was"
   - Prompt 5: "One day, a young girl"

2. **Repetition Tracking**
   - 1-token repetition rate
   - 2-token repetition rate
   - Catch "a a a a" and "time time time" patterns

3. **Validation Loss**
   - Add validation set to training
   - Monitor train vs val loss gap
   - Detect overfitting

4. **Perplexity Tracking**
   - Calculate perplexity at each checkpoint

5. **Sample Quality Scoring**
   - Coherence (0-10)
   - Grammar (0-10)
   - Repetition (0-10)
   - Diversity (0-10)

6. **Automatic Checkpoint Comparison**
   - Save samples.txt for each checkpoint
   - Save metrics.json for each checkpoint
   - Auto-determine BEST checkpoint

7. **Output Format Example**
```
=== CHECKPOINT STEP XXXXX ===
Train Loss: X.XX
Validation Loss: X.XX
Perplexity: XXX

Repetition:
  1-token: X.X%
  2-token: X.X%

Sample Quality:
Coherence: XX%
Grammar: XX%
Repetition: XX%
Diversity: XX%
```

---

## Benefits

- Compare progression: 10k → 20k → 30k → 40k → 50k
- Detect overfitting early
- Find BEST checkpoint (not just final)
- Professional evaluation system
- Quantitative + qualitative assessment

---

## Status

- Created: Aug 14, 2026
- Approved by user: YES
- Priority: HIGH
- Timing: AFTER current 50k training completes
