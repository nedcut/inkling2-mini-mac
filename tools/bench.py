"""Inkling 2 Mini MLX benchmark: prompt and generation speed, peak memory.

usage: bench.py MODEL [--contexts 512,4096] [--gen 256] [--no-kernels]
"""

import argparse
import json
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--contexts", default="512,4096")
ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--no-kernels", action="store_true")
ap.add_argument("--label", default="")
ap.add_argument("--prefill-step", type=int, default=2048)
ap.add_argument("--kv-bits", type=int, default=None)
a = ap.parse_args()

if a.no_kernels:
    from mlx_lm.models.inkling2 import layers

    layers._use_kernels = lambda: False

t0 = time.time()
model, tok = load(a.model)
load_s = time.time() - t0

text = (
    "Write a detailed, multi-section technical essay about the history of "
    "computing hardware, from mechanical calculators to modern GPUs. "
) * 2000
ids = tok.encode(text)
sampler = make_sampler(temp=0.0)


def run(prompt, n):
    r = None
    for r in stream_generate(model, tok, prompt, max_tokens=n, sampler=sampler,
        prefill_step_size=a.prefill_step, kv_bits=a.kv_bits):
        pass
    return r


run(ids[:64], 8)  # warm-up / compile
rows = []
for ctx in [int(c) for c in a.contexts.split(",")]:
    for i in range(a.runs):
        mx.reset_peak_memory()
        r = run(ids[:ctx], a.gen)
        row = dict(
            label=a.label,
            prefill_step=a.prefill_step,
            kv_bits=a.kv_bits,
            ctx=ctx,
            run=i,
            prompt_tps=round(r.prompt_tps, 1),
            gen_tps=round(r.generation_tps, 2),
            gen_tokens=r.generation_tokens,
            peak_gb=round(r.peak_memory, 2),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
print(json.dumps(dict(label=a.label, load_s=round(load_s, 1))))
