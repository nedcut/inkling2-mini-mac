# Inkling 2 Mini on a Mac

Run [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26.4B MoE, 3.4B active)
on an Apple-silicon Mac as a local OpenAI-compatible server, at about **46.5 tokens/s** on a 24 GB M4 Pro,
with output identical to the reference implementation. Prompts up to **131,072 tokens** fit with an 8-bit
cache profile, and an opt-in batching mode serves up to four requests at once.

It uses the MLX 4-bit build (`mlx-community/Inkling-2-Mini-4bit`), mlx-lm from the open pull request that adds
the model ([ml-explore/mlx-lm#2086](https://github.com/ml-explore/mlx-lm/pull/2086)), and a patch on top of
it that makes generation about 40% faster than the unpatched build without changing a single output token. A
small server calls the library directly and returns the model's thinking, answer and tool calls in separate
fields.

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

To update an existing checkout: `git pull && ./install.sh` (it replaces the previously applied patch).

### 24 GB Macs: raise the GPU memory cap

macOS lets the GPU use about 70% of RAM by default, which is 17.8 GB on a 24 GB Mac. The weights alone are
17.3 GB, so with the default cap the server runs out of memory on any prompt longer than a few hundred tokens.
`sudo sysctl iogpu.wired_limit_mb=21504` raises the cap to 21 GB and leaves about 3 GB for macOS. It resets at
reboot. Close other large apps while serving; the server warns when the cap is at the default. Macs with
32 GB or more don't need this.

### Long prompts: up to 131,072 tokens

```zsh
INKLING_KV_BITS=8 INKLING_MAX_CONTEXT=131072 ./serve.sh
```

The default profile keeps every attention cache in BF16 and stops at 65,536 tokens. This profile keeps BF16 for
the first 65,536 tokens, then packs the eight full-attention caches to 8-bit (the sliding-window and convolution
caches stay as they are). Up to that point the output is identical to the default profile; after it, packing can
change some generated tokens (see Correctness). `INKLING_KV_START` moves the switch point.

On a 24 GB Mac a 128k prompt peaks at about 18.9 GiB of GPU memory, close to the 21 GB cap, and macOS swaps
other apps out while it runs. Quit browsers and other large apps first: in one of two runs with a browser and
an editor open, the server was killed partway through a 128k prompt, most likely by macOS for memory. A 128k
prompt takes about 10 minutes to read the first time. Batching and this profile cannot be combined.

### Several users at once

```zsh
INKLING_DECODE_BATCH_SIZE=4 ./serve.sh     # also accepts 2 or 3
```

Requests that arrive together are read one at a time and then generate together, which raises total
throughput (about 61 tokens/s for four 512-token requests, against about 43 for one). Each request waits longer
for its first token, and with long prompts most of the time goes to reading them, so the gain shrinks. Batched
greedy output can differ from serial output by rounding (see Correctness). A single request still takes the
serial path, at the same speed and with the same prompt cache.

## Results (Apple M4 Pro, 24 GB, 20-core GPU)

Generation speed through the server (`./serve.sh`, `tools/http_bench.py`), greedy, 256 tokens, medians of 6 runs
per arm, the previous release and this one alternating in one session (`results/server-speed-release2.jsonl`):

| | Generation, 512-token prompt | Generation, 4k prompt |
|---|---|---|
| **This release** | **46.5-46.6 tokens/s** | **44.8-45.0 tokens/s** |
| Previous release (`7fb83b4`) | 43.8-44.0 | 42.2 |
| Unpatched PR build (measured on day one) | 31.1-31.5 | 30.1-30.2 |

Every run of this release was faster than every run of the previous one. An earlier session the same day, with
Spotlight indexing in the background, measured +5.8% / +6.4% (42.8 / 41.3 against 40.5 / 38.8 tokens/s).

Earlier library measurements (day one, `tools/bench.py`): prompt reading about 400-410 tokens/s at 4k and 384-391
at 16k; the unpatched build read prompts at about 350 tokens/s. oMLX 0.8.2 with its oQ4e build generated
34.6-34.9 tokens/s at 512 tokens.

- **131,072-token profile** (`INKLING_KV_BITS=8`): a code planted 90% of the way into a 128,291-token prompt came
  back exactly; prompt reading 219 tokens/s, generation 16.6 tokens/s, peak 20.26 GB (18.87 GiB)
  (`results/needle-128k-server.jsonl`, memory trace in `results/needle-128k-memory.log`).
- [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) gates at a **131,072-token** context
  window: **all six pass** (load, chat, reasoning, tools, context, speed). The speed gate measured 45.0 tokens/s
  (temperature 0.8, counted with the server's tokenizer); the context gate retrieved a code from a 91,890-token
  prompt in 341 s at a peak of 18.82 GiB (`results/registry-gates-ctx131072.json`). The default profile passes at
  32k and 64k (`results/registry-gates-ctx*.json`).
- A full 64k prompt with the default profile: a code planted 90% of the way into 63,666 tokens came back exactly;
  prompt reading 288 tokens/s, generation 24.7 tokens/s, peak 20.14 GB (`results/needle-64k-server.jsonl`).
- Multi-turn chat with a 5.7k-token history: the wait for the first token of turns 2-5 drops from about 16 s to
  0.5-0.7 s with the prompt cache, and every reply (thinking, answer, tool calls) is identical to the uncached run
  (`results/cache-conversation-*.json`).
- Batching, four 512-token requests at once: 60.2-60.9 tokens/s combined while generating, against 43.2-43.5 for
  a single request on the same server in the same session (`results/batch-c1-c4-512.jsonl`). With 4k prompts
  (measured before this release's decode kernels): 58.0 tokens/s combined against 39.8-42.4 serial, but the full
  wave finished only 1% sooner and the median wait for a first token rose from about 31 s to 44 s.
- How close this is to the hardware limit: `results/ceilings.md`. Single-user generation is at about 67% of the
  bandwidth ceiling (about 69 tokens/s at short context); prompt reading is at its practical ceiling.

Every number above was measured on this machine with nothing else loaded on the GPU. `results/` has the raw rows.

## What the patch changes

`patches/mlx-lm-inkling2-mini-speed.patch` applies to mlx-lm at `0c4c991` (PR #2086).

For generating tokens (single-token decode) it:

- compiles each decoder layer's work before and after attention into two fused MLX graphs (fewer, larger GPU
  dispatches and less Python overhead per token), capturing the layer's parameters once;
- builds the relative-position attention mask with one small Metal kernel from the cache's latest slot, and
  computes the 16-dimensional relative-position projection inside that kernel instead of materializing a
  24 × 1,024 bias for each of the 48 decoder visits;
- runs the message gate (two projections, tanh and the blend) as one Metal kernel. Its two weight matrices are
  kept in BF16 instead of being widened to FP32 at load: every value in the checkpoint is an exact BF16 value
  (checked at load), so nothing changes except the bytes read;
- replaces the depth-attention kernel with a single-pass version that keeps its inputs in registers, with the
  same reduction order;
- fuses the four-tap depthwise convolution, its state update, rounding and residual into one Metal kernel, and
  keeps only the last three rows as state;
- fuses the weighted sum of the eight expert outputs with the following convolution and residual add;
- skips the log-scaled attention temperature while it is exactly 1 (the first 8,192 positions).

The new kernels reproduce MLX's accumulation order exactly. Versions that rounded differently by one step were
faster in isolation but flipped expert choices and changed output tokens, and were dropped.

For reading the prompt (prefill, in 512-token chunks) it:

- fuses the expert outputs' un-sort and weighted reduction into one Metal kernel, so neither the gathered nor the
  weighted copy of the expert outputs is materialized;
- keeps only the last three rows of each convolution's input as state;
- compiles the full 512-token chunk stages;
- submits each group of layers to the GPU as soon as it reaches a full-attention layer, so its intermediates are
  freed earlier (about 0.55 GB lower peak memory at every prompt length).

For the 8-bit profile it adds a packed cache for the eight full-attention visits, an attention kernel that reads
packed keys and values during prefill, and a decode path built on MLX's quantized matmuls that includes the
learned attention sink. For batching it lets the compiled decode stages run at any batch width and gives each
row its own fused attention call.

The patch retains all 48 decoder visits per token (the middle 12 layers run three times), all eight routed
experts, and the original quantized matmul kernels for the experts. Every change was profiled and A/B-tested
on its own.

The server also caps MLX's free-buffer cache (64 MiB) and sets a memory guideline 2.35 GB below the GPU working
set. Without these, prompts of 32k tokens and more made MLX keep gigabytes of freed buffers and pushed macOS into
swap.

## Correctness

`./check.sh` greedy-decodes four fixed prompts (one of about 2.5k tokens) and compares all 800 generated tokens,
and the top-20 first-step log-probabilities, with `reference/`, which was made with the **unpatched** mlx-lm build.
On an M4 Pro with MLX 0.32.4 the patched build matches token for token. Another chip generation or MLX version can
differ in the last bits of some kernels; the check then reports where the outputs first diverge.

`tools/cache_equivalence.py` runs the same five-turn greedy conversation (a long document, two tool calls) with
the prompt cache off and on and requires identical replies. It also passes with the 8-bit profile and the switch
moved to 2,048 tokens, so that the conversation crosses it (`results/cache-conversation-kv8-switch2048-*.json`).

8-bit cache: with the default switch point, a 60k-token prompt gives output identical to BF16. With packing
forced from 8,192 tokens on the same prompts, the next-token choice agreed with BF16 at 98.4% of positions and
the mean KL divergence was 0.0036 nats; greedy replies first differed after 38 and 168 tokens.

Batching: a request's tokens are bit-identical whichever requests run beside it, but they differ from serial
decoding by rounding (median KL divergence 0.0005 nats per token against the serial run, maximum 0.15), so long
greedy replies can end up worded differently.

`tools/functional_test.py` exercises the server: a plain answer, streaming, a tool call and the reply that uses
its result, and `reasoning_effort`. It passes with and without batching. `tools/run_registry_gates.py` runs the
six recipe gates of [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) (load, chat,
reasoning, tools, context, speed) with the registry's own `lab.py`.

## Limits

- **Serial by default.** Concurrent requests queue unless batching is on (see above).
- **Text only.** The vision tower is not loaded (saves 0.64 GB), so image inputs are refused.
- **Context.** `serve.sh` refuses prompts over `INKLING_MAX_CONTEXT` tokens (default 65536) instead of running out
  of GPU memory. Use the 8-bit profile for up to 131,072. The limit counts the prompt only; a long reply adds to
  the cache.
- **One cached prompt.** The server keeps the state of the previous prompt (up to 16k tokens by default,
  `INKLING_PROMPT_CACHE_TOKENS`) and reuses it when the next prompt starts with the same tokens, which is what
  a chat client sends. Longer prompts, edited history or a different conversation are processed from scratch.
  `INKLING_PROMPT_CACHE=0` turns it off.
- Depends on an unmerged mlx-lm pull request, pinned by commit (the installer fetches the commit by hash, since
  the pull request has been force-pushed since). When it merges, the patch will need a rebase.

## Things that measured slower here

- The bundled DSpark draft model (speculative decoding, via oMLX): 30-32 tokens/s against 37 without it. Drafts
  were accepted about half the time, and verifying a draft costs more than it saves on this GPU.
- `mlx_lm.server`: about 20% slower than calling `stream_generate`, and it returns raw control tokens.
- oMLX 0.8.2 with the oQ4e build: 34.6-34.9 tokens/s at 512 tokens, measured the same afternoon as the unpatched
  build's 31.1-31.5.
- MLX built from its main branch (November 2026): within 1% of MLX 0.32.4.
- Custom NVFP4 and 4-bit matrix-vector kernels, packed attention projections, grouped loop gates and larger Metal
  command buffers: no gain or under 1%.
- 8-bit KV from the first token: fits 131k but generates about 10% slower at short prompts, hence the switch point.

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
