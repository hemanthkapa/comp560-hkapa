# Latent Chain-of-Thought in NanoGPT

## Project Overview

This project implements latent Chain-of-Thought (CoT) reasoning in NanoGPT using looped/recurrent transformer architectures. The goal is to explore whether internal latent reasoning (via iterative computation) can match or exceed the reasoning capability of explicit token-based CoT.

## Background

**Latent CoT** refers to reasoning that happens within the model's hidden states rather than being explicitly generated as tokens. Key approaches include:

1. **Looped Transformers**: Apply the same weight-tied transformer block multiple times, allowing the hidden state to iteratively refine
2. **Continuous Thought Tokens (COCONUT)**: Insert latent thought vectors between input/output
3. **Hidden State Pondering**: Cache and loop back final hidden states

Reference papers:
- "Reasoning with Latent Thoughts" (Saunshi et al., 2025) - ICLR 2025
- "Scaling Latent Reasoning via Looped Language Models" (Ouro) - 2025
- "Beyond Chains of Thought" - Benchmarking latent reasoning

## Directory Structure

```
latent-cot-nanogpt/
├── README.md                  # This file
├── model.py                   # Modified NanoGPT with LoopedGPT
├── data/
│   └── arithmetic/
│       ├── train.bin          # Generated training data
│       ├── val.bin            # Validation data
│       └── meta.pkl           # Metadata (vocab, etc)
├── scripts/
│   ├── generate_data.py       # Arithmetic data generator
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation harness
└── experiments/               # Results and checkpoints
    └── baseline/              # Standard model runs
    └── looped/                # Looped model runs
```

## Implementation Plan

### Phase 1: Baseline Model (Week 1)

**Objective**: Establish baseline performance with standard NanoGPT

- [ ] Generate arithmetic dataset (2-4 digit addition)
- [ ] Train standard 4-layer NanoGPT
- [ ] Verify MPS backend works
- [ ] Evaluate on held-out test set

**Dataset Specs**:
- Task: Multi-digit addition (e.g., "12345+67890=")
- Vocab: ~20 tokens (0-9, +, =, <pad>, <eos>)
- Train: 100K examples
- Val/Test: 10K each

### Phase 2: Looped Model (Week 2-3)

**Objective**: Implement and train LoopedGPT

- [ ] Implement `LoopedGPT` class with weight-tied blocks
- [ ] 2-layer block looped 4× (8 effective layers)
- [ ] Compare against 8-layer non-looped baseline
- [ ] Test inference with varying loop counts

**Model Config**:
| Config | Block Layers | Loop Count | Effective Depth | Parameters |
|--------|-------------|------------|-----------------|------------|
| A | 2 | 4× | 8 | ~2M |
| B | 2 | 6× | 12 | ~2M |
| C | 4 | 2× | 8 | ~4M |
| Baseline | 8 | 1× | 8 | ~8M |

### Phase 3: Adaptive Depth (Week 4)

**Objective**: Learn when to stop reasoning

- [ ] Add learned exit head
- [ ] Train with variable loop count
- [ ] Analyze when model decides to exit early

### Phase 4: Analysis (Week 5)

**Objective**: Understand latent reasoning patterns

- [ ] Visualize hidden state evolution across loops
- [ ] Compare with explicit CoT (if time permits)
- [ ] Test generalization to harder tasks (4+ digit addition)

## Model Configuration

### For M3 Pro (MPS Backend)

```python
# Minimal config for fast iteration
n_embd = 128       # Embedding dimension
n_head = 4         # Number of heads
n_layer = 2        # Layers in one block (will loop)
vocab_size = 16   # 0-9, +, =, <pad>
block_size = 32   # Max sequence length
batch_size = 64   # Adjust based on memory
```

### Training Hyperparameters

```python
learning_rate = 3e-4
max_iters = 10000
warmup_iters = 500
eval_interval = 500
eval_iters = 200
```

## Expected Outcomes

1. **Looped models should match non-looped** at same effective depth
2. **More loops = better reasoning** on complex problems
3. **Learned exit** should save computation on easy problems
4. **Latent reasoning** should be more efficient than explicit CoT (no token overhead)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| MPS backend issues | Fall back to CPU for debugging |
| Training instability | Use gradient clipping, warmup |
| Poor looped performance | Check weight initialization |
| Memory issues | Reduce batch size, embedding dim |

## References

- Saunshi et al., "Reasoning with Latent Thoughts", ICLR 2025
- "Scaling Latent Reasoning via Looped Language Models" (Ouro)
- Karpathy's NanoGPT: https://github.com/karpathy/nanoGPT
