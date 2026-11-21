"""Streamed OpenAI-compatible chat benchmark (stdlib only).

Reports time to first token (prefill), decode tokens/s, and server usage counts.
usage: http_bench.py URL MODEL [--contexts 512,4096] [--gen 256] [--runs 2] [--label X]
"""

import argparse
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("model")
ap.add_argument("--contexts", default="512,4096")
ap.add_argument("--gen", type=int, default=256)
ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--label", default="")
ap.add_argument("--task", default="essay", choices=["essay", "code"])
a = ap.parse_args()

FILL = (
    "The history of computing hardware runs from mechanical calculators "
    "through vacuum tubes and transistors to modern GPUs. "
)
TASKS = {
    "essay": "Write a detailed multi-section essay on that history.",
    "code": "Now write a complete Python implementation of a red-black tree "
    "with insert, delete and search, with docstrings.",
}


def prompt_for(ctx_words):
    # ~1.3 tokens per word for this filler; the server reports the real count.
    n = max(1, int(ctx_words / 1.3 / len(FILL.split())))
    return FILL * n + "\n\n" + TASKS[a.task]


def run(ctx, gen):
    body = dict(
        model=a.model,
        messages=[{"role": "user", "content": prompt_for(ctx)}],
        max_tokens=gen,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    req = urllib.request.Request(
        a.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    first = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    chunks += 1
                    if first is None:
                        first = time.time()
    end = time.time()
    ptoks = usage.get("prompt_tokens")
    ctoks = usage.get("completion_tokens") or chunks
    ttft = (first or end) - t0
    decode_s = end - (first or end)
    return dict(
        label=a.label,
        ctx=ptoks,
        ttft_s=round(ttft, 2),
        prompt_tps=round(ptoks / ttft, 1) if ptoks else None,
        gen_tokens=ctoks,
        gen_tps=round((ctoks - 1) / decode_s, 2) if decode_s > 0 else None,
    )


run(64, 8)  # warm-up
for ctx in [int(c) for c in a.contexts.split(",")]:
    for i in range(a.runs):
        r = run(ctx, a.gen)
        r["run"] = i
        print(json.dumps(r), flush=True)
