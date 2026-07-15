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
    ap.add_argument("--lr", type=float, default=1e-3) # Higher base learning rate paired with schedule
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
