#!/usr/bin/env python3
"""
ARTN Text Generation Script.

Generate text from trained BitLinear model.

Usage:
    python scripts/generate.py --prompt "Once upon a time"
    python scripts/generate.py --prompt "The little girl" --max-length 100 --temperature 0.8
"""

import sys
import torch
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer


def top_k_sampling(logits, k=50):
    """Apply top-k sampling to logits."""
    # Get top k logits
    top_k = min(k, logits.size(-1))
    values, indices = torch.topk(logits, top_k)
    
    # Create distribution over top k
    probs = torch.softmax(values, dim=-1)
    
    # Sample from distribution
    sampled = torch.multinomial(probs, num_samples=1)
    
    # Map back to original indices
    return indices.gather(-1, sampled)


def generate(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=1.0,
    top_k=50,
):
    """Generate text from prompt."""
    
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    generated = input_ids.clone()
    
    print(f"\n[Generating from prompt: '{prompt}']")
    print(f"[Max length: {max_length}, Temperature: {temperature}, Top-k: {top_k}]")
    print("\n" + "="*60)
    print("GENERATED TEXT:")
    print("="*60)
    print(prompt, end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(generated)
            logits = outputs["logits"]
            
            # Get logits for last token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k sampling
            next_token = top_k_sampling(next_token_logits, k=top_k)
            
            # Append to generated
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Decode and print
            token_text = tokenizer.decode([next_token.item()])
            print(token_text, end="", flush=True)
            
            # Stop at end of sentence
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    print("\n" + "="*60)
    
    return tokenizer.decode(generated[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained ARTN model")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Starting prompt")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (default: final_model)")
    parser.add_argument("--model-dir", type=str, default="outputs/artn-core-v1", help="Model directory")
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer_path = Path(args.model_dir) / "tokenizer.json"
    print(f"\n[Loading tokenizer from {tokenizer_path}]")
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"[Vocab size: {len(tokenizer.vocab)}]")
    
    # Determine model path
    if args.checkpoint:
        model_path = Path(args.checkpoint)
        print(f"\n[Loading from checkpoint: {model_path}]")
        checkpoint = torch.load(model_path, map_location="cpu")
        model_config = checkpoint.get("config", {})
    else:
        model_path = Path(args.model_dir) / "final_model"
        print(f"\n[Loading from {model_path}]")
        checkpoint = None
        model_config = {}
    
    # Create model
    model = BitLinearGPT(
        vocab_size=len(tokenizer.vocab),
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
    )
    
    # Load weights
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Loaded checkpoint from step {checkpoint.get('step', 'unknown')}]")
    else:
        # Load final model
        model_file = model_path / "model.pt"
        if model_file.exists():
            model.load_state_dict(torch.load(model_file, map_location="cpu"))
            print(f"[Loaded final model]")
        else:
            print(f"[ERROR: No model found at {model_file}]")
            return 1
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model parameters: {num_params:,}]")
    
    # Generate
    generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
