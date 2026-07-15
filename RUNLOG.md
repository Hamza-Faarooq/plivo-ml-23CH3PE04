# RUNLOG

## Run 1: Architectural Baseline Overhaul
* **Hypothesis:** The default script uses standard LayerNorm, a basic Adam optimizer, and no learning rate schedule. Moving to a Pre-LN RMSNorm architecture with weight tying, AdamW, a Cosine LR schedule (6e-4), and gradient clipping will drastically improve optimization stability within the 2,000-step limit.
* **What Changed:** Swapped to `RMSNorm` and Pre-LN. Enabled `tie_weights = True`. Updated optimizer to `AdamW`. Added Cosine Annealing with a 150-step warmup. Added `clip_grad_norm_`. Adjusted to 5 layers, 160 dimensions.
* **Result:** Dev `bpb` established at **2.0467** (1,577,120 params).
* **Conclusion:** This configuration provides a highly stable, fast-converging baseline that stays safely underneath the 2,000,000 parameter ceiling.

## Run 2: The "Wider vs. Deeper" Trade-off
* **Hypothesis:** A wider hidden dimension with fewer layers will provide larger representational capacity per layer, which may capture the complex mixed English/Hindi vocabulary better than a deeper, narrower network.
* **What Changed:** Adjusted `Config.n_layer` from 5 to 4. Adjusted `Config.n_embd` from 160 to 192.
* **Result:** Dev `bpb` dropped from 2.0467 to **2.0196** (1,820,352 params).
* **Conclusion:** Success. The wider residual stream improved the model's ability to compress the mixed text distribution effectively without exceeding the parameter cap.

## Run 3: Context Window Expansion
* **Hypothesis:** Devanagari script requires 3 bytes per character in UTF-8. A block size of 128 effectively cripples the model's temporal context on Hindi text. Expanding the block size to 256 will allow the attention mechanism to track longer-range linguistic dependencies.
* **What Changed:** Adjusted `Config.block_size` from 128 to 256.
* **Result:** Dev `bpb` dropped from 2.0196 to **1.9340**.
* **Conclusion:** Massive success. Giving the model twice the context window significantly improved predictive accuracy, especially for the multi-byte Hindi sequences.

## Run 4: Aggressive Optimizer Scaling
* **Hypothesis:** Given the strict 2,000-step budget, increasing the maximum learning rate in the cosine schedule will allow for larger gradient updates during the warmup phase, potentially reaching a deeper loss minimum before the decay sequence takes over.
* **What Changed:** Increased `lr` argument in `train.py` from 6e-4 to 1e-3.
* **Result:** Dev `bpb` dropped from 1.9340 to **1.9094**.
* **Conclusion:** Success. The higher learning rate capitalized on the short 2,000-step sprint, forcing faster convergence. This is the best configuration so far.

## Run 5: Regularization Injection
* **Hypothesis:** With only 7MB of training data, the model might be overfitting near the end of the 2,000 steps. Adding a slight dropout rate will introduce necessary regularization to force more robust feature representations.
* **What Changed:** Adjusted `Config.dropout` from 0.0 to 0.05.
* **Result:** Dev `bpb` worsened from 1.9094 to **1.9104**.
* **Conclusion:** Failed. Given the small dataset and the very short training duration, adding regularization hindered the network's ability to memorize and compress the text optimally. Dropout should remain at 0.0.

## Run 6: Custom Byte-Pair Encoding (BPE)
* **Hypothesis:** Replacing the naive byte-level tokenizer with a custom BPE tokenizer (256 merges, 512 vocab) will heavily compress the 3-byte Devanagari characters, vastly improving the effective sequence length and compression ratio.
* **What Changed:** Replaced `tokenizer.py` with a custom greedy BPE implementation trained dynamically on the corpus.
* **Result:** Dev `bpb` worsened to **1.9504** (Training Loss: 1.291).
* **Conclusion:** Failed. While BPE successfully compressed the text sequences logically, expanding the vocabulary from 256 to 512 added new, randomly initialized embedding weights. Within a strict 2,000-step limit, the optimizer did not have enough time or iterations to properly train these new embedding parameters, causing a regression in compression efficiency compared to the raw byte fallback.
