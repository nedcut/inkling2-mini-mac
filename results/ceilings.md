# Ceilings for Inkling 2 Mini on an M4 Pro (24 GB)

Computed the way the [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill)
describes, from byte counts read out of the checkpoint (`ceiling-inputs.json`) and throughput measured on this Mac
(218 GB/s streaming bandwidth, 5.86 TFLOP/s dense BF16 matmul).

Bytes read per generated token: dense weights 1.045 GB (the middle 12 layers counted three times, because
the model loops them), LM head 0.261 GB, and 8 of 128 experts in each of the 48 layer visits,
1.84 GB of the 29.43 GB of looped expert weights. About
3.2 GB per token, so decode is bandwidth-bound.

## Decode ceiling, aggregate tokens/s

| Users | 512 ctx | 4K | 32K | 128K | 4K at the 273 GB/s spec |
|---|---|---|---|---|---|
| 1 | 69 | 66 | 52 | 29 | 83 |
| 2 | 89 | 85 | 62 | 32 | 106 |
| 4 | 108 | 102 | 71 | 35 | 128 |

Several users share the dense weights and LM head but each routes to its own experts, so the gain from batching is
less than linear. (The skill's `ceiling.py` multiplies the aggregate by the number of users twice; the table uses
aggregate = bandwidth × users / bytes per step.)

Speculative decoding does not lift the single-user ceiling much for this model: checking k draft tokens routes each
of them to its own 8 experts, so the bytes per step grow nearly as fast as the tokens gained. Even a draft that is
accepted for 3 tokens per step gives about 66 tokens/s; the bundled draft model was accepted for 1.3-1.4.

## Prefill ceiling (75% of measured BF16 matmul throughput)

| Prompt | 4K | 32K | 128K |
|---|---|---|---|
| tokens/s | 399 | 354 | 255 |

## Where this setup is

- Single-user decode: 44.5 tokens/s at short context, **about 64% of the 69 tokens/s ceiling** (31 tokens/s, 45%,
  with the unpatched build). Most of the remaining time is in MLX's quantized expert matmul kernels.
- Prefill: about 400-410 tokens/s at 4K, **at the practical ceiling**.
