#!/usr/bin/env python3
"""
GENERATION DIAGNOSIS FOR BITLINEAR + ADAMW
==========================================

Comprehensive analysis:
1. In-distribution vs out-of-distribution prompts
2. Decoding strategies (greedy, temperature, top-k, top-p)
3. Tokenizer artifact analysis
4. Checkpoint progression analysis
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
import json

BASE_DIR = Path('/root/agent-sandbox/ARTN')
sys.path.insert(0, str(BASE_DIR))

from src.tokenizer.bpe_tokenizer import BPETokenizer
from experiments.bitlinear_adamw_controlled import BitLinearGPT

# =============================================================================
# PROMPTS
# =============================================================================

# In-distribution prompts (TinyStories style - simple children's stories)
IN_DISTRIBUTION_PROMPTS = [
    "Once upon a time there was a little girl named",
    "There was a boy who loved to play with his",
    "One day, a happy dog went to the park and",
    "The little cat sat on the warm mat and",
    "A kind mother helped her son to",
]

# Out-of-distribution prompts (different style/content)
OUT_OF_DISTRIBUTION_PROMPTS = [
    "In a medieval village there was a princess",
    "The scientist discovered a new element that",
    "The quantum computer processed the data",
    "In the distant future, humans colonized",
    "The ancient prophecy foretold that a hero",
]

# =============================================================================
# LOAD MODEL AND TOKENIZER
# =============================================================================

def load_model(checkpoint_path, device='cpu'):
    """Load BitLinear model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = BitLinearGPT(
        vocab_size=4000,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=256,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def load_tokenizer():
    """Load tokenizer."""
    tokenizer_path = BASE_DIR / 'outputs' / 'artn-50k-best-config' / 'tokenizer.json'
    tokenizer = BPETokenizer.load(str(tokenizer_path))
    return tokenizer

# =============================================================================
# DECODING STRATEGIES
# =============================================================================

def generate_greedy(model, tokenizer, prompt, max_tokens=100):
    """Greedy decoding - always pick highest probability token."""
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)['logits']
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())

def generate_temperature(model, tokenizer, prompt, temperature=0.9, max_tokens=100):
    """Temperature sampling."""
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)['logits']
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())

def generate_top_k(model, tokenizer, prompt, k=50, temperature=1.0, max_tokens=100):
    """Top-k sampling."""
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)['logits']
            logits = logits[:, -1, :]
            
            # Top-k filtering
            top_k_logits, top_k_indices = torch.topk(logits, k)
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(1, top_k_indices, top_k_logits)
            
            logits_filtered = logits_filtered / temperature
            probs = F.softmax(logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())

def generate_top_p(model, tokenizer, prompt, p=0.9, temperature=1.0, max_tokens=100):
    """Nucleus (top-p) sampling."""
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)['logits']
            logits = logits[:, -1, :]
            
            # Sort and compute cumulative probabilities
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())

# =============================================================================
# TOKENIZER ANALYSIS
# =============================================================================

def analyze_tokenizer_artifacts(tokenizer, generated_text, prompt):
    """Check if invented words are tokenizer artifacts or model errors."""
    
    # Find words that look invented
    import re
    words = re.findall(r'\b\w+\b', generated_text.lower())
    
    # Check vocabulary
    vocab = set(tokenizer.vocab.keys()) if hasattr(tokenizer, 'vocab') else set()
    
    # Tokenize individual words
    results = []
    for word in words:
        if len(word) > 5:  # Only check longer words
            try:
                tokens = tokenizer.encode(word)
                decoded = tokenizer.decode(tokens)
                
                # Check if word is split into multiple tokens
                is_split = len(tokens) > 1
                
                # Check if decoded matches original
                matches = decoded.strip().lower() == word
                
                results.append({
                    'word': word,
                    'tokens': tokens,
                    'decoded': decoded,
                    'is_split': is_split,
                    'matches': matches,
                    'is_artifact': not matches
                })
            except:
                pass
    
    return results

# =============================================================================
# DIAGNOSTIC TESTS
# =============================================================================

def test_in_distribution_prompts(model, tokenizer):
    """Test in-distribution prompts with various decoding strategies."""
    
    print("\n" + "="*80)
    print("IN-DISTRIBUTION PROMPT TESTS (TinyStories style)")
    print("="*80)
    
    results = []
    
    for prompt in IN_DISTRIBUTION_PROMPTS:
        print(f"\n--- Prompt: {prompt} ---\n")
        
        # Greedy
        greedy_gen = generate_greedy(model, tokenizer, prompt, max_tokens=80)
        print(f"GREEDY: {greedy_gen}")
        
        # Temperature 0.7
        temp_07 = generate_temperature(model, tokenizer, prompt, temperature=0.7, max_tokens=80)
        print(f"TEMP 0.7: {temp_07}")
        
        # Temperature 1.0
        temp_10 = generate_temperature(model, tokenizer, prompt, temperature=1.0, max_tokens=80)
        print(f"TEMP 1.0: {temp_10}")
        
        # Top-k 50
        topk = generate_top_k(model, tokenizer, prompt, k=50, temperature=0.9, max_tokens=80)
        print(f"TOP-K 50: {topk}")
        
        # Top-p 0.9
        topp = generate_top_p(model, tokenizer, prompt, p=0.9, temperature=0.9, max_tokens=80)
        print(f"TOP-P 0.9: {topp}")
        
        results.append({
            'prompt': prompt,
            'greedy': greedy_gen,
            'temp_07': temp_07,
            'temp_10': temp_10,
            'topk': topk,
            'topp': topp,
        })
    
    return results

def test_out_of_distribution_prompts(model, tokenizer):
    """Test out-of-distribution prompts."""
    
    print("\n" + "="*80)
    print("OUT-OF-DISTRIBUTION PROMPT TESTS")
    print("="*80)
    
    results = []
    
    for prompt in OUT_OF_DISTRIBUTION_PROMPTS:
        print(f"\n--- Prompt: {prompt} ---\n")
        
        # Greedy
        greedy_gen = generate_greedy(model, tokenizer, prompt, max_tokens=80)
        print(f"GREEDY: {greedy_gen}")
        
        # Temperature 0.9
        temp = generate_temperature(model, tokenizer, prompt, temperature=0.9, max_tokens=80)
        print(f"TEMP 0.9: {temp}")
        
        results.append({
            'prompt': prompt,
            'greedy': greedy_gen,
            'temp': temp,
        })
    
    return results

def test_tokenizer_artifacts(model, tokenizer):
    """Analyze if invented words are tokenizer artifacts."""
    
    print("\n" + "="*80)
    print("TOKENIZER ARTIFACT ANALYSIS")
    print("="*80)
    
    # Generate a sample
    prompt = "Once upon a time there was a princess named"
    generated = generate_temperature(model, tokenizer, prompt, temperature=0.9, max_tokens=100)
    
    print(f"\nGenerated text: {generated}")
    print("\nAnalyzing potential artifact words...\n")
    
    # Analyze
    artifacts = analyze_tokenizer_artifacts(tokenizer, generated, prompt)
    
    # Show only potential artifacts
    artifact_words = [a for a in artifacts if a['is_artifact'] or not a['matches']]
    
    if artifact_words:
        print("Potential tokenizer artifacts found:")
        for a in artifact_words[:10]:  # Show first 10
            print(f"  Word: '{a['word']}'")
            print(f"    Tokens: {a['tokens']}")
            print(f"    Decoded: '{a['decoded']}'")
            print(f"    Is artifact: {a['is_artifact']}")
            print()
    else:
        print("No obvious tokenizer artifacts found.")
    
    return artifacts

def test_checkpoint_progression(tokenizer):
    """Test generation quality across checkpoints."""
    
    print("\n" + "="*80)
    print("CHECKPOINT PROGRESSION ANALYSIS")
    print("="*80)
    
    checkpoints_dir = BASE_DIR / 'outputs' / 'bitlinear_adamw_20260910_143646' / 'checkpoints' / 'bitlinear_adamw'
    checkpoints = sorted(checkpoints_dir.glob('step_*.pt'))
    
    # Test specific checkpoints
    test_steps = [1000, 2000, 5000, 8000, 10000]
    
    prompt = "Once upon a time there was a little girl who"
    
    results = []
    
    for step in test_steps:
        ckpt_path = checkpoints_dir / f'step_{step}.pt'
        if not ckpt_path.exists():
            print(f"Checkpoint step_{step} not found, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"CHECKPOINT STEP {step}")
        print(f"{'='*80}\n")
        
        # Load checkpoint
        model = load_model(ckpt_path)
        
        # Generate with temperature
        generated = generate_temperature(model, tokenizer, prompt, temperature=0.9, max_tokens=80)
        
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        
        results.append({
            'step': step,
            'generated': generated,
        })
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

def analyze_vocabulary_usage(model, tokenizer):
    """Analyze what vocabulary the model actually uses."""
    
    print("\n" + "="*80)
    print("VOCABULARY USAGE ANALYSIS")
    print("="*80)
    
    # Generate multiple samples and collect token usage
    all_tokens = []
    
    for prompt in IN_DISTRIBUTION_PROMPTS[:3]:
        generated = generate_temperature(model, tokenizer, prompt, temperature=0.9, max_tokens=80)
        tokens = tokenizer.encode(generated)
        all_tokens.extend(tokens)
    
    # Count token frequency
    token_counts = Counter(all_tokens)
    
    # Get most common tokens
    most_common = token_counts.most_common(20)
    
    print(f"\nMost frequently generated tokens:")
    for token_id, count in most_common:
        try:
            decoded = tokenizer.decode([token_id])
            print(f"  Token {token_id}: '{decoded}' (count: {count})")
        except:
            print(f"  Token {token_id}: (decode error) (count: {count})")
    
    # Check for rare/unusual tokens
    rare_tokens = [t for t, c in token_counts.items() if c == 1]
    print(f"\nRare tokens (used only once): {len(rare_tokens)}")
    
    if len(rare_tokens) > 0:
        print(f"Examples of rare tokens:")
        for token_id in rare_tokens[:5]:
            try:
                decoded = tokenizer.decode([token_id])
                print(f"  Token {token_id}: '{decoded}'")
            except:
                pass
    
    return token_counts

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("BITLINEAR + ADAMW GENERATION DIAGNOSIS")
    print("="*80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()
    print(f"Tokenizer vocab size: {len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else 'unknown'}")
    
    # Load final checkpoint
    print("\nLoading model...")
    checkpoint_path = BASE_DIR / 'outputs' / 'bitlinear_adamw_20260910_143646' / 'checkpoints' / 'bitlinear_adamw' / 'step_10000.pt'
    model = load_model(checkpoint_path)
    
    # Run tests
    print("\n" + "="*80)
    print("RUNNING DIAGNOSTIC TESTS")
    print("="*80)
    
    # Test 1: In-distribution prompts
    in_dist_results = test_in_distribution_prompts(model, tokenizer)
    
    # Test 2: Out-of-distribution prompts
    out_dist_results = test_out_of_distribution_prompts(model, tokenizer)
    
    # Test 3: Tokenizer artifacts
    artifact_results = test_tokenizer_artifacts(model, tokenizer)
    
    # Test 4: Vocabulary usage
    vocab_results = analyze_vocabulary_usage(model, tokenizer)
    
    # Test 5: Checkpoint progression (separate command - long running)
    print("\n" + "="*80)
    print("CHECKPOINT PROGRESSION")
    print("="*80)
    print("\nTo test checkpoint progression, run:")
    print("  python experiments/generation_diagnosis.py --checkpoints")
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    
    # Save results
    results_path = BASE_DIR / 'outputs' / 'generation_diagnosis_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'in_distribution': in_dist_results,
            'out_of_distribution': out_dist_results,
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    import sys
    
    if '--checkpoints' in sys.argv:
        # Run only checkpoint progression test
        tokenizer = load_tokenizer()
        test_checkpoint_progression(tokenizer)
    else:
        main()
