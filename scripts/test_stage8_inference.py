#!/usr/bin/env python3
"""
Stage 8 Test Script: Quantized Inference Verification

Tests BitLinear + AdamW checkpoint with 10 fixed TinyStories prompts.
Measures generation speed, peak RAM, and output quality.
Verifies ternary quantization is used during inference.

Blueprint: Stage 8 - Quantized Inference & Top-K Sampling
Date: October 15, 2026
"""

import sys
from pathlib import Path
import torch
import json
import time

# Add ARTN to path
BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from src.inference.generate import (
    load_model_from_checkpoint,
    generate_text,
    verify_ternary_weights,
)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

CHECKPOINT_PATH = "outputs/bitlinear_adamw_20260910_143646/checkpoints/bitlinear_adamw/step_9500.pt"
TOKENIZER_PATH = "outputs/artn-optimized/tokenizer.json"
DEVICE = "cpu"

TEST_PROMPTS = [
    "Once upon a time",
    "A little girl named",
    "The boy went to the",
    "There was a big dog",
    "She looked at the sky and",
    "In the morning, the",
    "The cat sat on the",
    "He opened the door and saw",
    "Once there was a magical",
    "The children played in the",
]

GENERATION_CONFIG = {
    "max_new_tokens": 100,
    "temperature": 0.9,
}


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("=" * 80)
    print("STAGE 8: QUANTIZED INFERENCE TEST")
    print("=" * 80)
    print()
    
    # =========================================================================
    # STEP 1: Load Model and Tokenizer
    # =========================================================================
    print("STEP 1: Loading Model and Tokenizer")
    print("-" * 80)
    
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Device: {DEVICE}")
    print()
    
    try:
        model, tokenizer = load_model_from_checkpoint(
            CHECKPOINT_PATH,
            TOKENIZER_PATH,
            DEVICE,
        )
        print("✅ Model loaded successfully")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ FAILED to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # =========================================================================
    # STEP 2: Verify Ternary Weights
    # =========================================================================
    print("STEP 2: Verifying Ternary Quantization")
    print("-" * 80)
    
    try:
        stats = verify_ternary_weights(model)
        
        print(f"BitLinear layers: {stats['num_bitlinear_layers']}")
        print(f"Total BitLinear params: {stats['total_params']:,}")
        print()
        
        dist = stats['ternary_distribution']
        print(f"Actual Ternary Distribution:")
        print(f"   Negative (-1): {dist['negative']*100:.2f}%")
        print(f"   Zero (0):      {dist['zero']*100:.2f}%")
        print(f"   Positive (+1): {dist['positive']*100:.2f}%")
        print()
        
        if stats['is_quantized']:
            print("✅ Ternary quantization VERIFIED")
        else:
            print("⚠️  WARNING: Weights may not be properly quantized")
        
    except Exception as e:
        print(f"❌ FAILED to verify weights: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # =========================================================================
    # STEP 3: Test Generation with 10 Prompts
    # =========================================================================
    print("STEP 3: Testing Generation (10 Prompts)")
    print("-" * 80)
    
    results = []
    total_tokens = 0
    total_time = 0
    max_ram = 0
    
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{i+1}/10] Prompt: \"{prompt}\"")
        print("-" * 80)
        
        try:
            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
                device=DEVICE,
            )
            
            results.append(result)
            
            # Print generated text
            print(f"Generated: \"{result['generated_text'][:200]}...\"" if len(result['generated_text']) > 200 else f"Generated: \"{result['generated_text']}\"")
            print()
            print(f"   Tokens: {result['num_tokens']}")
            print(f"   Time: {result['generation_time']:.2f}s")
            print(f"   Speed: {result['tokens_per_second']:.2f} tokens/sec")
            print(f"   Peak RAM: {result['peak_ram_mb']:.2f} MB")
            
            # Track totals
            total_tokens += result['num_tokens']
            total_time += result['generation_time']
            if result['peak_ram_mb'] > max_ram:
                max_ram = result['peak_ram_mb']
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "prompt": prompt,
                "error": str(e),
            })
    
    print()
    print("=" * 80)
    
    # =========================================================================
    # STEP 4: Report Summary Statistics
    # =========================================================================
    print("STEP 4: Summary Statistics")
    print("=" * 80)
    
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    print(f"\nTotal prompts tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()
    
    if successful:
        avg_speed = total_tokens / total_time if total_time > 0 else 0
        
        print(f"Total tokens generated: {total_tokens}")
        print(f"Total generation time: {total_time:.2f}s")
        print(f"Average speed: {avg_speed:.2f} tokens/sec")
        print(f"Max peak RAM: {max_ram:.2f} MB")
        print()
        
        # Quality assessment
        print("Quality Assessment:")
        print("-" * 80)
        
        coherent_count = 0
        for result in successful:
            text = result['generated_text']
            # Simple coherence check: contains multiple sentences
            if text.count('.') >= 2 or text.count('!') >= 1 or text.count('?') >= 1:
                coherent_count += 1
        
        print(f"Multi-sentence generations: {coherent_count}/{len(successful)}")
        print(f"Coherence rate: {coherent_count/len(successful)*100:.1f}%")
    
    print()
    
    # =========================================================================
    # STEP 5: Save Results
    # =========================================================================
    print("STEP 5: Saving Results")
    print("-" * 80)
    
    output_file = BASE_DIR / "outputs" / "stage8_inference_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "test_date": "2026-10-15",
        "stage": "Stage 8 - Quantized Inference",
        "checkpoint": CHECKPOINT_PATH,
        "tokenizer": TOKENIZER_PATH,
        "device": DEVICE,
        "generation_config": GENERATION_CONFIG,
        "ternary_stats": stats,
        "results": results,
        "summary": {
            "total_prompts": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "total_tokens": total_tokens if successful else 0,
            "total_time": total_time if successful else 0,
            "avg_speed": avg_speed if successful else 0,
            "max_ram_mb": max_ram if successful else 0,
        },
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Results saved to: {output_file}")
    print()
    
    # =========================================================================
    # FINAL STATUS
    # =========================================================================
    print("=" * 80)
    print("STAGE 8 TEST COMPLETE")
    print("=" * 80)
    
    if stats['is_quantized'] and len(failed) == 0 and (coherent_count / len(successful) > 0.5 if successful else False):
        print("\n✅ STAGE 8 MILESTONE: PASSED")
        print("   - Ternary quantization verified")
        print("   - All prompts generated successfully")
        print("   - Coherent multi-sentence stories produced")
    else:
        print("\n⚠️  STAGE 8 STATUS: PARTIAL")
        if not stats['is_quantized']:
            print("   - Ternary quantization NOT verified")
        if failed:
            print(f"   - {len(failed)} prompt(s) failed to generate")
        if successful and coherent_count / len(successful) <= 0.5:
            print("   - Low coherence rate in generated text")


if __name__ == "__main__":
    main()
