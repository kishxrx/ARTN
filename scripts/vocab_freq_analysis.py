
import torch, sys, json, re
from collections import Counter, defaultdict

sys.path.insert(0, '/root/agent-sandbox/ARTN')
from src.tokenizer.bpe_tokenizer import BPETokenizer

# Load tokenized data
train_tokens = torch.load('/root/agent-sandbox/ARTN/outputs/tokenized_cache/train_tokenized.pt', weights_only=False)
val_tokens = torch.load('/root/agent-sandbox/ARTN/outputs/tokenized_cache/val_tokenized.pt', weights_only=False)

# Load tokenizer
tok = BPETokenizer.load('/root/agent-sandbox/ARTN/outputs/artn-50k-best-config/tokenizer.json')

# Decode all train stories to text
train_texts = []
for i, tokens in enumerate(train_tokens):
    text = tok.decode(tokens)
    train_texts.append(text)

# Decode all val stories
val_texts = []
for tokens in val_tokens:
    text = tok.decode(tokens)
    val_texts.append(text)

# Target words to analyze
# Uncommon (struggling) subjects
uncommon_words = ['king', 'kings', 'queen', 'wizard', 'knight', 'teacher', 'peacock']
# Common (succeeding) subjects
common_words = ['dog', 'bird', 'fish', 'girl', 'boy', 'tree']

# Also include variations
all_words = uncommon_words + common_words + ['royal', 'dragon', 'robot', 'cat', 'robot', 'children', 'students']

# 1. WORD FREQUENCY
print("=== 1. WORD FREQUENCY (total occurrences) ===\n")
word_freq_train = {}
word_freq_val = {}
doc_freq_train = {}
doc_freq_val = {}

for word in all_words:
    # Train
    total_train = 0
    docs_train = 0
    for text in train_texts:
        count = len(re.findall(r'\b' + re.escape(word) + r'\b', text.lower()))
        total_train += count
        if count > 0:
            docs_train += 1
    word_freq_train[word] = total_train
    doc_freq_train[word] = docs_train
    
    # Val
    total_val = 0
    docs_val = 0
    for text in val_texts:
        count = len(re.findall(r'\b' + re.escape(word) + r'\b', text.lower()))
        total_val += count
        if count > 0:
            docs_val += 1
    word_freq_val[word] = total_val
    doc_freq_val[word] = docs_val

# Print frequency table
print(f"{'Word':<15} {'Train Count':<15} {'Train Docs':<15} {'Train Doc%':<12} {'Val Count':<12} {'Val Docs':<12} {'Val Doc%':<10}")
print("-" * 91)
for word in all_words:
    tc = word_freq_train[word]
    td = doc_freq_train[word]
    td_pct = td / len(train_texts) * 100
    vc = word_freq_val[word]
    vd = doc_freq_val[word]
    vd_pct = vd / len(val_texts) * 100
    print(f"{word:<15} {tc:<15} {td:<15} {td_pct:<12.2f} {vc:<12} {vd:<12} {vd_pct:<10.2f}")

print(f"\nTotal train stories: {len(train_texts)}")
print(f"Total val stories: {len(val_texts)}")

# 2. BPE TOKENIZATION
print("\n=== 2. BPE TOKENIZATION ===\n")
for word in all_words:
    # Encode the word
    tokens = tok.encode(word)
    decoded_tokens = [tok.decode([t]) for t in tokens]
    token_strs = [f"{t}='{tok.decode([t])}'" for t in tokens]
    
    # Get token frequencies in training data
    token_freqs = []
    for t in tokens:
        freq = sum(1 for sample in train_tokens if t in sample)
        token_freqs.append(f"{t}({freq}docs)")
    
    print(f"  {word:<12} → tokens={tokens} → decoded={decoded_tokens} → freq={token_freqs}")

# 3. CONTEXT ANALYSIS
print("\n=== 3. CONTEXT ANALYSIS ===\n")
# Find sentences containing each uncommon word
for word in ['king', 'teacher', 'peacock', 'dog', 'bird']:
    print(f"\n--- '{word}' context samples (first 5 from training) ---")
    found = 0
    for i, text in enumerate(train_texts):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            if re.search(r'\b' + re.escape(word) + r'\b', sent.lower()):
                print(f"  [{i}] {sent.strip()[:150]}")
                found += 1
                break
        if found >= 5:
            break
    if found == 0:
        print(f"  No occurrences found")

# 4. CO-OCCURRENCE ANALYSIS
print("\n=== 4. CO-OCCURRENCE ANALYSIS ===\n")
# For each target word, find what other words co-occur in the same story
for word in ['king', 'teacher', 'peacock', 'dog', 'bird']:
    co_occur = Counter()
    docs_with = 0
    for text in train_texts:
        if re.search(r'\b' + re.escape(word) + r'\b', text.lower()):
            docs_with += 1
            words_in_text = set(re.findall(r'\b[a-z]+\b', text.lower()))
            for w in words_in_text:
                if w != word and len(w) > 2:
                    co_occur[w] += 1
    
    print(f"  '{word}' (appears in {docs_with} train stories):")
    if co_occur:
        for w, c in co_occur.most_common(10):
            print(f"    {w}: {c} stories")
    else:
        print(f"    No occurrences")
    print()

# Save results to JSON
results = {
    "experiment": "vocabulary_frequency_analysis",
    "date": "2026-11-02",
    "train_samples": len(train_texts),
    "val_samples": len(val_texts),
    "word_freq_train": word_freq_train,
    "word_freq_val": word_freq_val,
    "doc_freq_train": doc_freq_train,
    "doc_freq_val": doc_freq_val,
    "doc_freq_train_pct": {k: v/len(train_texts)*100 for k,v in doc_freq_train.items()},
    "doc_freq_val_pct": {k: v/len(val_texts)*100 for k,v in doc_freq_val.items()},
}

with open('/root/agent-sandbox/ARTN/outputs/artn_recursive_20261025_002725/vocab_freq_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== Results saved to vocab_freq_analysis.json ===")
