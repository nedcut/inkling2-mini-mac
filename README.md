# Inkling 2 Mini on a Mac

Run [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26.4B MoE, 3.4B active)
on an Apple-silicon Mac as a local OpenAI-compatible server, at about **44 tokens/s** on a 24 GB M4 Pro,
with output identical to the reference implementation.

It uses the MLX 4-bit build (`mlx-community/Inkling-2-Mini-4bit`), mlx-lm from the open pull request that adds
the model ([ml-explore/mlx-lm#2086](https://github.com/ml-explore/mlx-lm/pull/2086)), and a patch on top of
it that makes generation about 30% faster and prompt reading about 7% faster without changing a single
output token. A small server calls the
library directly and returns the model's thinking, answer and tool calls in separate fields.

## Quick start

```zsh
git clone https://github.com/nedcut/inkling2-mini-mac && cd inkling2-mini-mac
./install.sh                              # mlx-lm at the PR commit + patch, pinned packages, 17.3 GB of weights
sudo sysctl iogpu.wired_limit_mb=21504    # 24 GB Macs only, see below; resets at reboot
./serve.sh                                # http://127.0.0.1:8790/v1, model "inkling-2-mini"
```

```zsh
curl -s localhost:8790/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 17*23?"}]}'
```

The reply has `content` (the answer) and `reasoning_content` (the thinking). Streaming, tools (`tools` /
`tool_calls`) and `"reasoning_effort"` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) work like the
OpenAI API. `POST /v1/token/encode` returns the model's token ids for a text.

Requirements: macOS on Apple silicon, **24 GB of unified memory or more**, about 20 GB of free disk, git, and
Python 3.11-3.13 (or [uv](https://docs.astral.sh/uv/)). No full Xcode, no Homebrew packages.

### 24 GB Macs: raise the GPU memory cap

macOS lets the GPU use about 70% of RAM by default, which is 17.8 GB on a 24 GB Mac. The weights alone are
17.3 GB, so with the default cap the server runs out of memory on any prompt longer than a few hundred tokens.
`sudo sysctl iogpu.wired_limit_mb=21504` raises the cap to 21 GB and leaves about 3 GB for macOS. It resets at
reboot. Close other large apps while serving; the server warns when the cap is at the default. Macs with
32 GB or more don't need this.

## Results (Apple M4 Pro, 24 GB, 20-core GPU)

| | Generation, 512-token prompt | Generation, 4k prompt | Prompt reading, 4k | Peak memory, 4k |
|---|---|---|---|---|
| **This server (`./serve.sh`)** | **44.4-44.5 tokens/s** | **42.5-42.7 tokens/s** | 404 tokens/s | text-only |
| mlx-lm library, this patch | 44.1-44.6 | 42.0-42.7 | 410 | 18.77 GB |
| mlx-lm library, generation-only patch | 43.8-44.2 | 42.2-42.4 | 384 | 19.31 GB |
| mlx-lm library, unpatched PR build | 31.1-31.5 | 30.1-30.2 | ~350 | 19.31 GB |
| oMLX 0.8.2 (oQ4e build) | 34.6-34.9 | 33.0-33.2 | ~300 | |

The unpatched and oMLX rows were measured earlier the same day, when this Mac ran about 6% slower overall
(the generation-only patch measured 40.2-40.8 / 38.7-39.1 then, against 31.1-31.5 / 30.1-30.2 unpatched).

- At 16k tokens of context: prompt reading 384-391 tokens/s, generation 38-39 tokens/s, peak 19.17 GB.
- Multi-turn chat with a 5.6k-token history: the wait for the first token of turns 2-5 drops from about 15 s
  to 0.6-1.8 s with the prompt cache, and every reply (thinking, answer, tool calls) is identical to the
  uncached run (`tools/cache_equivalence.py`, `results/cache-conversation-*.json`).
- [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) gates with a 64k context window:
  **all six pass** (load, chat, reasoning, tools, context, speed). The speed gate measured 40.6 tokens/s
  (temperature 0.8, counted with the server's tokenizer); the context gate retrieved a code planted near the end
  of a 45,470-token prompt in 146 s at a peak of 18.16 GiB (`results/registry-gates-ctx65536.json`).
- A full 64k prompt through the server (`tools/needle.py`): a code planted 90% of the way into 63,666 tokens came
  back exactly; prompt reading 288 tokens/s, generation 24.7 tokens/s, peak 20.14 GB
  (`results/needle-64k-server.jsonl`).
- How close this is to the hardware limit: `results/ceilings.md`. Single-user generation is at about 64% of the
  bandwidth ceiling (about 69 tokens/s at short context); prompt reading is at its practical ceiling.

Every number above was measured on this machine with nothing else loaded on the GPU and greedy decoding; ranges
span 2-4 runs (`tools/bench.py`, `tools/http_bench.py`). `results/` has the raw rows.

## What the patch changes

`patches/mlx-lm-inkling2-mini-speed.patch` applies to mlx-lm at `0c4c991` (PR #2086).

For generating tokens (single-token decode) it:

- compiles each decoder layer's work before and after attention into two fused MLX graphs (fewer, larger GPU
  dispatches and less Python overhead per token);
- builds the relative-position attention mask with one small Metal kernel from the cache's latest slot;
- fuses the four-tap depthwise convolution, its state update, rounding and residual into one Metal kernel;
- fuses the weighted sum of the eight expert outputs with the following convolution and residual add.

For reading the prompt (prefill, in 512-token chunks) it:

- fuses the expert outputs' un-sort and weighted reduction into one Metal kernel, so neither the gathered nor the
  weighted copy of the expert outputs is materialized;
- keeps only the last three rows of each convolution's input as state;
- compiles the full 512-token chunk stages;
- submits each group of layers to the GPU as soon as it reaches a full-attention layer, so its intermediates are
  freed earlier (about 0.55 GB lower peak memory at every prompt length).

Batched and padded inputs, training and tensor parallelism keep the original code paths. The patch skips no
work: all 48 decoder visits per token (the middle 12 layers run three times), all eight routed experts, and the
original quantized matmul kernels. Every change was profiled and A/B-tested on its own.

The server also caps MLX's free-buffer cache (64 MiB) and sets a memory guideline 2.35 GB below the GPU working
set. Without these, prompts of 32k tokens and more made MLX keep gigabytes of freed buffers and pushed macOS into
swap.

## Correctness

`./check.sh` greedy-decodes four fixed prompts (one of about 2.5k tokens) and compares all 800 generated tokens,
and the top-20 first-step log-probabilities, with `reference/`, which was made with the **unpatched** mlx-lm build.
On an M4 Pro with MLX 0.32.4 the patched build matches token for token. Another chip generation or MLX version can
differ in the last bits of some kernels; the check then reports where the outputs first diverge.

`tools/cache_equivalence.py` runs the same five-turn greedy conversation (a long document, two tool calls) with
the prompt cache off and on and requires identical replies.

`tools/functional_test.py` exercises the server: a plain answer, streaming, a tool call and the reply that uses
its result, and `reasoning_effort`. `tools/run_registry_gates.py` runs the six recipe gates of
[local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) (load, chat, reasoning, tools,
context, speed) with the registry's own `lab.py`.

## Limits

- **One request at a time.** Concurrent requests queue; this is a single-user local server.
- **Text only.** The vision tower is not loaded (saves 0.64 GB), so image inputs are refused.
- **Context.** `serve.sh` refuses prompts over `INKLING_MAX_CONTEXT` tokens (default 65536) instead of running out of GPU
  memory; see the results for what fits.
- **One cached prompt.** The server keeps the state of the previous prompt (up to 16k tokens by default,
  `INKLING_PROMPT_CACHE_TOKENS`) and reuses it when the next prompt starts with the same tokens, which is what
  a chat client sends. Longer prompts, edited history or a different conversation are processed from scratch
  (about 400 tokens/s for short prompts, under 300 tokens/s at 64k). `INKLING_PROMPT_CACHE=0` turns it off.
- Depends on an unmerged mlx-lm pull request, pinned by commit. When it merges, the patch will need a rebase.

## Things that measured slower here

- The bundled DSpark draft model (speculative decoding, via oMLX): 30-32 tokens/s against 37 without it. Drafts
  were accepted about half the time, and verifying a draft costs more than it saves on this GPU.
- `mlx_lm.server`: about 20% slower than calling `stream_generate`, and it returns raw control tokens.
- oMLX 0.8.2 with the oQ4e build: 34.6-34.9 tokens/s at 512 tokens, measured the same afternoon as the unpatched
  build's 31.1-31.5.
- Custom NVFP4 and 4-bit matrix-vector kernels, packed attention projections, grouped loop gates and larger Metal
  command buffers: no gain or under 1%.

## Layout

| Path | What |
|---|---|
| `install.sh`, `serve.sh`, `check.sh` | install, serve, verify |
| `inkling_server.py` | the server (standard library plus mlx-lm) |
| `patches/` | the speed patch against mlx-lm `0c4c991` |
| `requirements.lock` | exact package versions |
| `reference/` | reference greedy outputs |
| `tools/` | benchmarks, the server test, the registry gate runner |
| `results/` | measured numbers |

## Credits and license

- [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by Thinking Machines (Apache-2.0); MLX
  conversion by [mlx-community](https://huggingface.co/mlx-community/Inkling-2-Mini-4bit).
- [mlx-lm](https://github.com/ml-explore/mlx-lm) and its Inkling 2 support (PR #2086), MIT, Copyright Apple Inc.
  The patch in `patches/` modifies files from it and keeps their notices.
- Measured and published following the
  [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill); the gates are
  [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry)'s.

Everything in this repository is MIT licensed (`LICENSE`).
