# Plivo ML Assignment: 2,000-Step LLM Speedrun

## Project Overview
This repository contains the optimized implementation for the Plivo AI/ML track assignment. The objective was to train a GPT-style language model from scratch on a provided mixed English/Hindi corpus (~7MB) under strict constraints:
- **Maximum 2,000 optimizer steps.**
- **Maximum 2,000,000 total parameters.**
- **CPU-only training.**

The goal was to minimize the Bits Per Byte (bpb) metric on a held-out evaluation set while ensuring architectural stability[cite: 6].

## Architectural Improvements
The baseline model was deliberately mediocre[cite: 6]. The following optimizations were implemented to improve convergence speed and predictive accuracy:

*   **Pre-LN Architecture:** Migrated from Post-LN to Pre-LN to ensure stable gradient flow from the start of training.
*   **RMSNorm:** Replaced standard LayerNorm with RMSNorm for computational efficiency.
*   **Rotary Positional Embeddings (RoPE):** Eliminated absolute positional embeddings in favor of RoPE for superior context tracking.
*   **SwiGLU Activation:** Replaced GELU with SwiGLU layers to enhance feature extraction capabilities.
*   **Weight Tying:** Linked input embeddings and output projection weights, reclaiming ~30% of the parameter budget, which allowed for a wider (160 dim) hidden state.
*   **AdamW & Cosine Schedule:** Upgraded the optimizer to AdamW with weight decay and a linear-warmup/cosine-decay learning rate schedule for optimal loss descent[cite: 6].

## Tokenizer Optimization Analysis
A critical part of this project was addressing the Hindi (Devanagari) script issue. Because Hindi text in UTF-8 requires 3 bytes per character, the baseline byte-level tokenizer (vocab 256) inflated sequence lengths, effectively cutting the model's context window in the Hindi portions of the corpus.

**Experiment 6** involved implementing a custom Byte-Pair Encoding (BPE) tokenizer to compress these sequences. While this technically compressed the data representation, it expanded the model's vocabulary size to 512. Within the strict 2,000-step training limit, the optimizer lacked sufficient iterations to adequately train the newly initialized embedding weights for these merged tokens, resulting in a regression in validation `bpb` (1.9095 vs 1.9094). This proved that within short-step speedruns, architecture efficiency often beats complex vocabulary engineering[cite: 6].

## Experimental Results
Each experiment isolated a single variable to measure its impact on the `bpb` metric[cite: 6].

| Exp | Modification | Params | BPB (Dev) |
| :--- | :--- | :--- | :--- |
| Baseline | Standard setup | 1.57M | 2.0467 |
| Exp 2 | Wider (n_embd=192, 4 layers) | 1.82M | 2.0196 |
| Exp 3 | Block Size = 256 | 1.82M | 1.9340 |
| **Exp 4** | **LR 1e-3 (Winner)** | **1.82M** | **1.9094** |
| Exp 5 | Dropout 0.05 | 1.82M | 1.9104 |
| Exp 6 | BPE Tokenizer (512 vocab) | 1.82M | 1.9095 |

## Setup Instructions

### 1. Environment Requirements
Ensure you are operating within a Linux/WSL2 environment as specified by the assignment instructions[cite: 1, 6].

```bash
# Navigate to project root
cd ~/speedrun/llm_handout

# Activate the virtual environment
source env/bin/activate
