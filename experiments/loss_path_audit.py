"""
LOSS PATH AUDIT - TRACING THE COMPLETE FLOW

Question: Why is 50k loss ~7 while Phase 22 loss ~0.05 if both use same copy task?
"""

# =============================================================================
# DATASET LEVEL (ALL SCRIPTS)
# =============================================================================
# train.py (Aug 1):           labels = input_ids.clone()  # NO SHIFT
# train_correct.py (Aug 6):   labels = input_ids.clone()  # NO SHIFT
# train_optimized.py (Aug 13): labels = input_ids.clone()  # NO SHIFT

# VERIFIED: ALL scripts use labels = input_ids.clone() with NO SHIFTING

# =============================================================================
# MODEL FORWARD PASS (BitLinearGPT)
# =============================================================================
# Line 480: logits = self.lm_head(x)  # Shape: (B, T, vocab_size)
# Line 485-489: loss = F.cross_entropy(
#     logits.reshape(-1, self.vocab_size),  # Shape: (B*T, vocab_size)
#     targets.reshape(-1),                   # Shape: (B*T,)
#     ignore_index=-100,
# )

# VERIFIED: NO INTERNAL SHIFTING IN MODEL

# =============================================================================
# KEY OBSERVATION: LOSS VALUES
# =============================================================================
# 50k training:   Loss ~7.1 after 50k steps
# Phase 22 FP32:  Loss ~0.05 after 5k steps
# Phase 22 Ternary: Loss ~0.11 after 5k steps

# =============================================================================
# QUESTION: Why such different losses?
# =============================================================================

# Hypothesis 1: Different data?
# - 50k used max_samples=50000
# - Phase 22 used max_samples=50000
# - Both use TinyStories
# UNLIKELY

# Hypothesis 2: Different training dynamics?
# - 50k trained longer (50k steps vs 5k)
# - But loss should be LOWER with more training on copy task!
# CONTRADICTS

# Hypothesis 3: Model architecture difference?
# - Both use BitLinearGPT
# - Same vocab_size=4000
# - Same d_model=256, n_layers=6
# UNLIKELY

# Hypothesis 4: ACTUALLY A REAL DIFFERENCE
# Let me check: is the causal mask correct?

print("""
ANALYSIS:

If labels = input_ids with causal mask:
- Position i predicts token at position i
- Position i can attend to positions 0..i
- This INCLUDES itself!

So the model CAN see the token it's predicting!

This should make the task TRIVIAL → loss should be ~0

But 50k has loss ~7!

This means EITHER:
A. 50k was trained on a DIFFERENT objective
B. 50k model is broken/not learning
C. Something else is wrong

Let me check if 50k training actually worked...
""")
