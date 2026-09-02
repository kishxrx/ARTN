# Experiment 005: Ternary BitLinear (1.58-bit)

## Objective
Replace all matrix multiplications with ternary addition/subtraction operations.

## Method
- Implement `BitLinear` class replacing `nn.Linear`
- Force weights to {-1, 0, 1}
- Use Straight-Through Estimator (STE) for backpropagation
- Custom `autograd.Function`: `grad_input = grad_output.clone()`

## Math
```
w_ternary = round(w / scale)  # Quantize to {-1, 0, 1}
y = x @ w_ternary  # Becomes addition/subtraction, not multiplication

# STE backward pass (bypasses round)
grad_input = grad_output.clone()
```

## Expected Results
- ~85% memory reduction vs Float32
- Matrix multiplication replaced by ternary addition
- Successful convergence (may need more steps)
- CPU-friendly integer arithmetic

## Results
*To be populated after training*

## Observations
*To be populated during training*

## Next Decision
*To be determined based on results*
