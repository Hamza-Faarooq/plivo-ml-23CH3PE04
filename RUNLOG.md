"""A heavily optimized GPT variant utilizing Pre-LN, RMSNorm, SwiGLU, 
and Rotary Positional Embeddings to minimize Bits Per Byte (bpb).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    vocab_size = 256
    block_size = 128
    n_layer = 5        # Adjusted to accommodate SwiGLU params under the 2M cap
    n_head = 4
    n_embd = 160       # Optimized multiple-of-32/64 block allocation
    dropout = 0.0
    tie_weights = True # Parameter allocation savings

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=128):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # Cast for broadcasting shapes
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim, cfg.block_size)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(v, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

class SwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = int(2 * (4 * n_embd) / 3) # Parameter optimized dimension
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd) 
        self.attn = SelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
            
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            # Scale standard normal distributions based on network depth
            std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Note: Absolute pos_emb is completely eliminated to let RoPE manage sequence tracking.
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())








"""Baseline trainer optimized for maximum data compression efficiency
within a restricted 2,000 step budget.
"""
import argparse
import time
import math

import torch
from torch.nn.utils import clip_grad_norm_

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step, max_steps=2000, warmup_steps=150, base_lr=6e-4, min_lr=6e-5):
    """Computes a Cosine Learning Rate Schedule with a Linear Warmup phase."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)  # Bumped from 8 to 16 for better gradient accuracy
    ap.add_argument("--lr", type=float, default=6e-4) # Higher base learning rate paired with schedule
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    # Upgraded Optimization Recipe: AdamW with specific decoupled decay weights
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    model.train()
    t0 = time.time()
    losses = []
   
    for step in range(1, args.steps + 1):
        # 1. Update learning rate based on current step progression
        lr = get_lr(step, max_steps=args.steps, base_lr=args.lr)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
           
        # 2. Extract batch tensors and calculate loss
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
       
        # 3. Clean gradients efficiently
        opt.zero_grad(set_to_none=True)
        loss.backward()
       
        # 4. Clip gradients to safeguard residual tracking stability
        clip_grad_norm_(model.parameters(), max_norm=1.0)
       
        opt.step()
        losses.append(loss.item())
       
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)  lr {lr:.2e}")

    # Serialize weights alongside metadata for evaluation extraction
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()

corpus: 7,318,592 bytes -> 7,318,592 tokens (vocab 256)
model: 1,577,120 params
step     1  loss 5.5324  (329 ms/step)  lr 4.00e-06
step   100  loss 4.3104  (230 ms/step)  lr 4.00e-04
step   200  loss 2.3540  (242 ms/step)  lr 5.99e-04
step   300  loss 2.0198  (262 ms/step)  lr 5.91e-04
step   400  loss 1.8915  (271 ms/step)  lr 5.76e-04
step   500  loss 1.8036  (276 ms/step)  lr 5.54e-04
step   600  loss 1.7452  (276 ms/step)  lr 5.25e-04
step   700  loss 1.7159  (281 ms/step)  lr 4.91e-04
step   800  loss 1.6753  (286 ms/step)  lr 4.52e-04
step   900  loss 1.6199  (289 ms/step)  lr 4.09e-04
step  1000  loss 1.5534  (292 ms/step)  lr 3.64e-04
step  1100  loss 1.5475  (293 ms/step)  lr 3.19e-04
step  1200  loss 1.5461  (294 ms/step)  lr 2.73e-04
step  1300  loss 1.5150  (297 ms/step)  lr 2.29e-04
step  1400  loss 1.4869  (299 ms/step)  lr 1.88e-04
step  1500  loss 1.4661  (303 ms/step)  lr 1.52e-04
step  1600  loss 1.4461  (319 ms/step)  lr 1.20e-04
step  1700  loss 1.4081  (320 ms/step)  lr 9.43e-05
step  1800  loss 1.4088  (323 ms/step)  lr 7.54e-05
step  1900  loss 1.4090  (330 ms/step)  lr 6.39e-05
step  2000  loss 1.4071  (331 ms/step)  lr 6.00e-05
saved ckpt.pt  (663s total)

{"bpb": 2.0467, "n_params": 1577120, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}


EXP 2:
n layer = 4
    n_embd = 192      
results:
corpus: 7,318,592 bytes -> 7,318,592 tokens (vocab 256)
model: 1,820,352 params
step     1  loss 5.5479  (355 ms/step)  lr 4.00e-06
step   100  loss 4.1226  (301 ms/step)  lr 4.00e-04
step   200  loss 2.2294  (315 ms/step)  lr 5.99e-04
step   300  loss 1.9664  (306 ms/step)  lr 5.91e-04
step   400  loss 1.8382  (301 ms/step)  lr 5.76e-04
step   500  loss 1.7743  (297 ms/step)  lr 5.54e-04
step   600  loss 1.6914  (301 ms/step)  lr 5.25e-04
step   700  loss 1.6339  (301 ms/step)  lr 4.91e-04
step   800  loss 1.6149  (299 ms/step)  lr 4.52e-04
step   900  loss 1.5785  (299 ms/step)  lr 4.09e-04
step  1000  loss 1.5463  (302 ms/step)  lr 3.64e-04
step  1100  loss 1.5177  (301 ms/step)  lr 3.19e-04
step  1200  loss 1.4667  (299 ms/step)  lr 2.73e-04
step  1300  loss 1.4544  (299 ms/step)  lr 2.29e-04
step  1400  loss 1.4417  (300 ms/step)  lr 1.88e-04
step  1500  loss 1.4238  (301 ms/step)  lr 1.52e-04
step  1600  loss 1.4034  (301 ms/step)  lr 1.20e-04
step  1700  loss 1.4009  (299 ms/step)  lr 9.43e-05
step  1800  loss 1.3737  (299 ms/step)  lr 7.54e-05
step  1900  loss 1.3618  (302 ms/step)  lr 6.39e-05
step  2000  loss 1.3779  (303 ms/step)  lr 6.00e-05
saved ckpt.pt  (606s total)
((env) ) hamza@Hamza:~/speedrun/llm_handout$ python starter/evaluate.py --checkpoint ckpt.pt --text_file data/dev_eval.txt
{"bpb": 2.0196, "n_params": 1820352, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}



EXPERIMENT 3:
BLOCK SIZE = 256 increased from 128
results:
corpus: 7,318,592 bytes -> 7,318,592 tokens (vocab 256)
model: 1,820,352 params
step     1  loss 5.5441  (680 ms/step)  lr 4.00e-06
step   100  loss 4.1058  (569 ms/step)  lr 4.00e-04
step   200  loss 2.2004  (618 ms/step)  lr 5.99e-04
step   300  loss 1.9094  (626 ms/step)  lr 5.91e-04
step   400  loss 1.7505  (634 ms/step)  lr 5.76e-04
step   500  loss 1.7098  (639 ms/step)  lr 5.54e-04
step   600  loss 1.6043  (639 ms/step)  lr 5.25e-04
step   700  loss 1.5426  (640 ms/step)  lr 4.91e-04
step   800  loss 1.5189  (641 ms/step)  lr 4.52e-04
step   900  loss 1.4518  (643 ms/step)  lr 4.09e-04
step  1000  loss 1.4372  (646 ms/step)  lr 3.64e-04
step  1100  loss 1.4094  (660 ms/step)  lr 3.19e-04
step  1200  loss 1.3768  (664 ms/step)  lr 2.73e-04
step  1300  loss 1.3644  (676 ms/step)  lr 2.29e-04
step  1400  loss 1.3359  (677 ms/step)  lr 1.88e-04
step  1500  loss 1.3085  (673 ms/step)  lr 1.52e-04
step  1600  loss 1.2982  (670 ms/step)  lr 1.20e-04
step  1700  loss 1.2797  (665 ms/step)  lr 9.43e-05
step  1800  loss 1.2891  (663 ms/step)  lr 7.54e-05
step  1900  loss 1.2701  (666 ms/step)  lr 6.39e-05
step  2000  loss 1.2594  (666 ms/step)  lr 6.00e-05
saved ckpt.pt  (1333s total)
((env) ) hamza@Hamza:~/speedrun/llm_handout$ python starter/evaluate.py --checkpoint ckpt.pt --text_file data/dev_eval.txt
{"bpb": 1.934, "n_params": 1820352, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}



EXPERIMENT 4:
File to Edit: train.py
The Change: In the main() function, change the default learning rate argument to 1e-3 (ap.add_argument("--lr", type=float, default=1e-3)).
Results: 
model: 1,820,352 params
step     1  loss 5.5441  (612 ms/step)  lr 6.67e-06
step   100  loss 3.7663  (682 ms/step)  lr 6.67e-04
step   200  loss 2.0858  (680 ms/step)  lr 9.98e-04
step   300  loss 1.8631  (671 ms/step)  lr 9.85e-04
step   400  loss 1.7121  (682 ms/step)  lr 9.58e-04
step   500  loss 1.6705  (670 ms/step)  lr 9.19e-04
step   600  loss 1.5666  (660 ms/step)  lr 8.69e-04
step   700  loss 1.5092  (653 ms/step)  lr 8.09e-04
step   800  loss 1.4831  (646 ms/step)  lr 7.42e-04
step   900  loss 1.4239  (641 ms/step)  lr 6.68e-04
step  1000  loss 1.4104  (647 ms/step)  lr 5.90e-04
step  1100  loss 1.3812  (644 ms/step)  lr 5.10e-04
step  1200  loss 1.3491  (642 ms/step)  lr 4.31e-04
step  1300  loss 1.3400  (643 ms/step)  lr 3.55e-04
step  1400  loss 1.3101  (645 ms/step)  lr 2.84e-04
step  1500  loss 1.2822  (648 ms/step)  lr 2.19e-04
step  1600  loss 1.2720  (651 ms/step)  lr 1.64e-04
step  1700  loss 1.2525  (650 ms/step)  lr 1.20e-04
step  1800  loss 1.2613  (653 ms/step)  lr 8.68e-05
step  1900  loss 1.2437  (654 ms/step)  lr 6.68e-05
step  2000  loss 1.2345  (652 ms/step)  lr 6.00e-05
saved ckpt.pt  (1305s total)
((env) ) hamza@Hamza:~/speedrun/llm_handout$ python starter/evaluate.py --checkpoint ckpt.pt --text_file data/dev_eval.txt
{"bpb": 1.9094, "n_params": 1820352, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}






EXPERIMENT 5:
File to Edit: model.py
The Change: In your Config class, change dropout = 0.0 to dropout = 0.05

RESULTS:





EXPERIMENT 6:
Because Hindi text in UTF-8 requires 3 bytes per character, a 128-token context window is effectively looking at a microscopic fragment of Hindi text compared to English. You need to implement a Byte-Pair Encoding (BPE) algorithm to compress these sequences.

Rather than relying on heavy external libraries (which are banned), we can write a highly efficient greedy merging loop right inside tokenizer.py. Taking an algorithmic approach to counting and sweeping array structures will allow us to train the tokenizer natively in just a few seconds
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


    Results:
    



