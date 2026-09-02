# Experiment 002: Recursive Transformer (Parameter Tying)

## Objective
Reduce layer RAM by instantiating a single transformer block and looping data through it N times.

## Method
- Implement ALBERT-style parameter tying
- Single `TransformerBlock` instance
- Forward pass: `for step in range(recursion_depth): x = block(x)`
- Compare parameter count and memory to baseline

## Expected Results
- ~80% reduction in parameter count
- Comparable loss to baseline
- Reduced checkpoint size

## Results
*To be populated after training*

## Observations
*To be populated during training*

## Next Decision
*To be determined based on results*
