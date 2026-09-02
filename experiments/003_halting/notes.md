# Experiment 003: Adaptive Halting Gate

## Objective
Reduce compute cycles by dynamically exiting the recursive loop early when the model is confident.

## Method
- Add `self.halting_gate = nn.Linear(d_model, 1)`
- At each recursion step, apply Sigmoid to get probability p_n
- When cumulative sum Σp_n ≥ 1.0, exit loop
- Log average compute steps per token

## Math
```
p_n = σ(halting_gate(x_n))  # Sigmoid probability
halt_when = (Σp_n >= 1.0)
```

## Expected Results
- Dynamic reduction in forward-pass compute
- Easy tokens use fewer steps
- Harder tokens use full recursion depth

## Results
*To be populated after training*

## Observations
*To be populated during training*

## Next Decision
*To be determined based on results*
