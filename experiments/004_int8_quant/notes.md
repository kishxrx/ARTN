# Experiment 004: INT8 Quantization (Safety Bridge)

## Objective
Validate that low-precision math doesn't break the architecture before pushing to ternary.

## Method
- Symmetric INT8 dynamic quantization during forward pass
- Find abs_max in tensor, calculate scaling factor
- Round float32 to integers between -127 and 127
- Investigate quantization impact on model quality

## Math
```
scale = 127 / max(|x|)
x_int8 = round(x / scale) * scale  # With STE for backward
```

## Expected Results
- Successful training with INT8 quantization
- Memory reduction compared to Float32
- Minimal accuracy degradation
- Confidence before ternary leap

## Results
*To be populated after training*

## Observations
*To be populated during training*

## Next Decision
*To be determined based on results*
