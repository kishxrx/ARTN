"""
Stage 8: Quantized Inference for BitLinear GPT

Minimal inference interface wrapping the existing BitLinearGPT.generate() method.
Provides model loading, generation utilities, and ternary weight verification.

Blueprint Reference: Stage 8 - Quantized Inference & Top-K Sampling
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Dict, Optional
import time
import tracemalloc
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import BitLinearGPT from experiments (matches checkpoint architecture)
from experiments.bitlinear_adamw_controlled import BitLinearGPT, BitLinear
from src.tokenizer.bpe_tokenizer import BPETokenizer


def load_model_from_checkpoint(
    checkpoint_path: str,
    tokenizer_path: str,
    device: str = "cpu",
) -> Tuple[BitLinearGPT, BPETokenizer]:
    """
    Load BitLinearGPT model and tokenizer from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint .pt file
        tokenizer_path: Path to tokenizer .json file
        device: Device to load model on ("cpu" or "cuda")
    
    Returns:
        Tuple of (model, tokenizer)
    
    Raises:
        FileNotFoundError: If checkpoint or tokenizer not found
        RuntimeError: If model architecture mismatch
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    config = checkpoint.get('config', {})
    
    # Create model with same architecture (using experiments implementation)
    model = BitLinearGPT(
        vocab_size=config.get('vocab_size', 4000),
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        max_seq_len=config.get('max_seq_len', 256),
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    return model, tokenizer


def verify_ternary_weights(model: BitLinearGPT) -> Dict:
    """
    Verify model uses ternary quantization and measure actual weight distribution.
    
    Args:
        model: BitLinearGPT model
    
    Returns:
        Dictionary with:
            - num_bitlinear_layers: Count of BitLinear layers
            - total_params: Total BitLinear parameters
            - ternary_distribution: Actual {-1, 0, +1} percentages
            - is_quantized: Whether weights are properly quantized
    """
    stats = {
        "num_bitlinear_layers": 0,
        "total_params": 0,
        "ternary_distribution": {"negative": 0.0, "zero": 0.0, "positive": 0.0},
        "is_quantized": False,
        "layer_stats": [],
    }
    
    total_neg = 0
    total_zero = 0
    total_pos = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            stats["num_bitlinear_layers"] += 1
            
            # Get ternary distribution
            distrib = module.get_ternary_distribution()
            
            layer_stats = {
                "name": name,
                "params": module.weight.numel(),
                "distribution": distrib,
            }
            stats["layer_stats"].append(layer_stats)
            
            # Accumulate totals
            total_neg += distrib["negative"] * module.weight.numel()
            total_zero += distrib["zero"] * module.weight.numel()
            total_pos += distrib["positive"] * module.weight.numel()
            total_params += module.weight.numel()
    
    stats["total_params"] = total_params
    
    if total_params > 0:
        stats["ternary_distribution"] = {
            "negative": total_neg / total_params,
            "zero": total_zero / total_params,
            "positive": total_pos / total_params,
        }
        
        # Check if quantized (all three values present)
        dist = stats["ternary_distribution"]
        stats["is_quantized"] = (
            dist["negative"] > 0.01 and 
            dist["zero"] > 0.01 and 
            dist["positive"] > 0.01
        )
    
    return stats


def generate_text(
    model: BitLinearGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.9,
    top_k: Optional[int] = 50,
    device: str = "cpu",
) -> Dict:
    """
    Generate text from prompt using existing BitLinearGPT.generate() method.
    
    Args:
        model: BitLinearGPT model
        tokenizer: BPETokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_k: Top-K filtering (None to disable)
        device: Device for inference
    
    Returns:
        Dictionary with:
            - prompt: Input prompt
            - generated_text: Full generated text (prompt + continuation)
            - continuation: Generated continuation only
            - num_tokens: Number of new tokens generated
            - generation_time: Time in seconds
            - tokens_per_second: Generation speed
            - peak_ram_mb: Peak RAM usage during generation
    """
    model.eval()
    
    # Start RAM tracking
    tracemalloc.start()
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    
    end_time = time.time()
    
    # Get peak RAM
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    # Extract continuation (generated part only)
    prompt_tokens = tokenizer.encode(prompt)
    continuation_tokens = output_ids[0].tolist()[len(prompt_tokens):]
    continuation = tokenizer.decode(continuation_tokens)
    
    # Calculate metrics
    generation_time = end_time - start_time
    num_tokens = len(continuation_tokens)
    tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0
    peak_ram_mb = peak / 1024 / 1024
    
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "continuation": continuation,
        "num_tokens": num_tokens,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second,
        "peak_ram_mb": peak_ram_mb,
    }


def generate_batch(
    model: BitLinearGPT,
    tokenizer: BPETokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 0.9,
    top_k: Optional[int] = 50,
    device: str = "cpu",
) -> List[Dict]:
    """
    Generate text for multiple prompts.
    
    Args:
        model: BitLinearGPT model
        tokenizer: BPETokenizer
        prompts: List of input prompts
        max_new_tokens: Maximum new tokens per prompt
        temperature: Sampling temperature
        top_k: Top-K filtering
        device: Device for inference
    
    Returns:
        List of generation result dictionaries
    """
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Generating {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        
        results.append(result)
    
    return results


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="BitLinear GPT Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K filtering")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_from_checkpoint(
        args.checkpoint,
        args.tokenizer,
        args.device,
    )
    
    # Verify ternary weights
    print("\nVerifying ternary weights...")
    stats = verify_ternary_weights(model)
    print(f"BitLinear layers: {stats['num_bitlinear_layers']}")
    print(f"Total BitLinear params: {stats['total_params']:,}")
    print(f"Ternary distribution: {stats['ternary_distribution']}")
    print(f"Is quantized: {stats['is_quantized']}")
    
    if args.prompt:
        # Single prompt generation
        print(f"\nGenerating from prompt: {args.prompt}")
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
        
        print(f"\nGenerated text:")
        print(result["generated_text"])
        print(f"\nSpeed: {result['tokens_per_second']:.2f} tokens/sec")
        print(f"Peak RAM: {result['peak_ram_mb']:.2f} MB")
