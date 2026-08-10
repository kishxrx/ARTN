#!/usr/bin/env python3
"""
COMPREHENSIVE GENERATION TESTING FOR ARTN MODEL
Tests multiple sampling strategies and reports exact results.
"""
import torch
import sys
sys.path.insert(0, '.')
from src.model.bitlinear import BitLinearGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer

def top_k_filter(logits, k):
    """Keep only top-k tokens, set rest to -inf"""
    if k <= 0:
        return logits
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits

def top_p_filter(logits, p):
    """Nucleus sampling - keep tokens until cumulative prob reaches p"""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits

def generate(model, tokenizer, prompt, max_tokens=50, temperature=1.0, top_k=0, top_p=1.0, seed=42):
    """Generate text with specified sampling parameters"""
    torch.manual_seed(seed)
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(generated)
            logits = outputs['logits'][0, -1, :].clone()
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k
            logits = top_k_filter(logits, top_k)
            
            # Apply top-p
            logits = top_p_filter(logits, top_p)
            
            # Sample
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.view(1, 1)], dim=1)
    
    return tokenizer.decode(generated[0].tolist())

def main():
    print('='*80)
    print('COMPREHENSIVE GENERATION TESTING - ARTN MODEL')
    print('='*80)
    
    # Load model
    print('\n[1] LOADING MODEL AND TOKENIZER')
    print('-'*80)
    tokenizer = BPETokenizer.load('outputs/artn-optimized/tokenizer.json')
    model = BitLinearGPT(vocab_size=4000, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_seq_len=256)
    ckpt = torch.load('outputs/artn-optimized/checkpoints/checkpoint_step_50000.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Model loaded: {ckpt["loss"]:.4f} loss')
    
    # Analyze probability distribution
    print('\n[2] PROBABILITY DISTRIBUTION ANALYSIS')
    print('-'*80)
    test_prompts = [
        'Once upon a time',
        'The dog ran',
        'A girl named Lily',
        'In the forest there was',
    ]
    
    for prompt in test_prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs['logits'][0, -1, :]
            probs = torch.softmax(logits, dim=0)
            
            top5 = torch.topk(probs, k=5)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            max_entropy = torch.log(torch.tensor(4000.0)).item()
            
            print(f'\nPrompt: "{prompt}"')
            print(f'  Entropy: {entropy:.4f} / {max_entropy:.4f} ({entropy/max_entropy*100:.1f}%)')
            print(f'  Top 5 predictions:')
            for i, (prob, idx) in enumerate(zip(top5.values, top5.indices)):
                token = tokenizer.decode([idx.item()])
                print(f'    {i+1}. "{token}" ({prob.item()*100:.2f}%)')
    
    # Test greedy decoding
    print('\n[3] GREEDY DECODING TEST (argmax)')
    print('-'*80)
    for prompt in test_prompts:
        torch.manual_seed(42)
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(30):
                outputs = model(generated)
                logits = outputs['logits'][0, -1, :]
                next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
        
        text = tokenizer.decode(generated[0].tolist())
        print(f'\n{prompt}: {text}')
    
    # Test temperature sampling (multiple temperatures)
    print('\n[4] TEMPERATURE SAMPLING TEST')
    print('-'*80)
    temperatures = [0.5, 0.7, 0.9, 1.0, 1.2]
    prompt = 'Once upon a time'
    
    for temp in temperatures:
        text = generate(model, tokenizer, prompt, max_tokens=50, temperature=temp, top_k=0, top_p=1.0, seed=42)
        print(f'\nTemp {temp}: {text}')
    
    # Test top-k sampling
    print('\n[5] TOP-K SAMPLING TEST')
    print('-'*80)
    k_values = [5, 10, 20, 50, 100]
    prompt = 'Once upon a time'
    
    for k in k_values:
        # Try multiple seeds to check consistency
        texts = []
        for seed in [42, 123, 456]:
            text = generate(model, tokenizer, prompt, max_tokens=50, temperature=0.9, top_k=k, top_p=1.0, seed=seed)
            texts.append(text)
        
        print(f'\nTop-k={k}:')
        for i, text in enumerate(texts):
            print(f'  Seed {[42, 123, 456][i]}: {text}')
    
    # Test top-p (nucleus) sampling
    print('\n[6] TOP-P (NUCLEUS) SAMPLING TEST')
    print('-'*80)
    p_values = [0.5, 0.8, 0.9, 0.95]
    prompt = 'Once upon a time'
    
    for p in p_values:
        texts = []
        for seed in [42, 123, 456]:
            text = generate(model, tokenizer, prompt, max_tokens=50, temperature=0.9, top_k=0, top_p=p, seed=seed)
            texts.append(text)
        
        print(f'\nTop-p={p}:')
        for i, text in enumerate(texts):
            print(f'  Seed {[42, 123, 456][i]}: {text}')
    
    # Test combined top-k + top-p
    print('\n[7] COMBINED TOP-K + TOP-P SAMPLING TEST')
    print('-'*80)
    combinations = [
        (0.7, 50, 0.95),
        (0.8, 50, 0.9),
        (0.9, 40, 0.95),
    ]
    prompt = 'Once upon a time'
    
    for temp, k, p in combinations:
        texts = []
        for seed in [42, 123, 456]:
            text = generate(model, tokenizer, prompt, max_tokens=50, temperature=temp, top_k=k, top_p=p, seed=seed)
            texts.append(text)
        
        print(f'\nTemp={temp}, Top-k={k}, Top-p={p}:')
        for i, text in enumerate(texts):
            print(f'  Seed {[42, 123, 456][i]}: {text}')
    
    # Test different prompts with best settings
    print('\n[8] DIFFERENT PROMPTS TEST (Best Settings: Temp=0.9, Top-k=50, Top-p=0.95)')
    print('-'*80)
    
    for prompt in test_prompts:
        text = generate(model, tokenizer, prompt, max_tokens=60, temperature=0.9, top_k=50, top_p=0.95, seed=42)
        print(f'\n{prompt}: {text}')
    
    # Test longer generation
    print('\n[9] LONGER GENERATION TEST (100 tokens)')
    print('-'*80)
    prompt = 'Once upon a time there was a little girl named Lily who lived in a small house near the forest'
    text = generate(model, tokenizer, prompt, max_tokens=100, temperature=0.9, top_k=50, top_p=0.95, seed=42)
    print(f'\nPrompt: "{prompt}"')
    print(f'Generated: {text}')
    
    # Summary
    print('\n' + '='*80)
    print('TESTING COMPLETE')
    print('='*80)
    print('\nRecommendations based on results:')
    print('1. Check which sampling method produces most readable text')
    print('2. Check if entropy indicates overconfidence')
    print('3. Check if different prompts produce different outputs')
    print('4. Determine if model needs retraining or sampling adjustment')

if __name__ == '__main__':
    main()
