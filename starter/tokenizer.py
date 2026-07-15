"""
A Byte-Pair Encoding (BPE) tokenizer that replaces the naive byte-level fallback.
This heavily compresses Hindi (Devanagari) characters by merging frequent byte pairs.
"""
import json
import os

class BPETokenizer:
    def __init__(self, merges=None):
        # We store merges internally as (p0, p1) -> new_id
        self.merges = {}
        if merges:
            for k, v in merges.items():
                p0, p1 = map(int, k.split("_"))
                self.merges[(p0, p1)] = v
                
        self.vocab_size = 256 + len(self.merges)
        
        # Build inverse vocabulary for lossless, O(1) decoding lookups
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        if not self.merges:
            return tokens
            
        while len(tokens) >= 2:
            lowest_pair = None
            lowest_rank = float('inf')
            
            # Find the most eligible pair to merge using a linear sweep
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merges and self.merges[pair] < lowest_rank:
                    lowest_pair = pair
                    lowest_rank = self.merges[pair]
            
            if lowest_pair is None:
                break 
                
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == lowest_pair:
                    new_tokens.append(self.merges[lowest_pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return tokens

    def decode(self, ids):
        # Direct byte mapping ensures 100% lossless text reconstruction
        bytes_list = [self.vocab[idx] for idx in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")

    def save(self, path):
        # JSON keys must be strings
        saved_merges = {f"{p0}_{p1}": idx for (p0, p1), idx in self.merges.items()}
        with open(path, "w") as f:
            json.dump({"merges": saved_merges}, f)


def load(path=None):
    """Return the tokenizer used by evaluate.py. Required interface."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "bpe_merges.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return BPETokenizer(data.get("merges", {}))
    return BPETokenizer()


# --- STANDALONE TRAINING SCRIPT ---
if __name__ == "__main__":
    # Train the tokenizer dynamically on the allowed corpus
    corpus_path = os.path.join(os.path.dirname(__file__), "../data/train_corpus.txt")
    out_path = os.path.join(os.path.dirname(__file__), "bpe_merges.json")
    
    print("Training BPE Tokenizer...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        # Sample the first 500,000 characters to keep execution time under 15 seconds.
        # The Hindi byte structures are repetitive enough that this captures them perfectly.
        text = f.read(500000)
    
    tokens = list(text.encode("utf-8"))
    vocab_size = 256
    num_merges = 256  # Adds 256 new tokens, bringing total vocab to 512
    
    merges = {}
    for i in range(num_merges):
        # 1. Count adjacent pairs
        counts = {}
        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j+1])
            counts[pair] = counts.get(pair, 0) + 1
            
        if not counts:
            break
            
        # 2. Greedily select the highest-frequency pair
        best_pair = max(counts, key=counts.get)
        new_id = vocab_size + i
        merges[best_pair] = new_id
        
        # 3. Replace all occurrences in the token array
        new_tokens = []
        j = 0
        while j < len(tokens):
            if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best_pair:
                new_tokens.append(new_id)
                j += 2
            else:
                new_tokens.append(tokens[j])
                j += 1
        tokens = new_tokens
        
        if (i + 1) % 64 == 0:
            print(f"Merged {i+1}/{num_merges} pairs...")

    tok = BPETokenizer(merges)
    tok.save(out_path)
    print(f"Saved {len(merges)} merges to {out_path}. Vocab size is now {tok.vocab_size}.")
