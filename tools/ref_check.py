"""Check that this install reproduces the reference greedy outputs.

Greedy-decodes four fixed prompts (one of ~2.5k tokens, so prefill crosses several
512-token chunks) and compares the 200 generated token ids per prompt, and the top-20
first-step log-probabilities, with reference/greedy-m4-pro-mlx-0.32.4.json. The
reference was made on an M4 Pro with the unpatched mlx-lm build; the decode patch is
meant to reproduce it exactly.

  .venv/bin/python tools/ref_check.py [MODEL_DIR] [--write OUT.json]

Exit 0 and "REF_CHECK PASS" when every token matches. Another chip generation or
MLX version can legitimately differ in the last bits of some kernels; then the report
shows where the first divergence is and how large the step-0 log-prob difference is.
"""

import argparse
import json
import pathlib

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = ROOT / "reference" / "greedy-m4-pro-mlx-0.32.4.json"
PROMPTS = [
    "Explain, step by step, how a CPU executes a single ADD instruction.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "List five differences between TCP and UDP.",
    ("The history of computing hardware runs from mechanical calculators through "
     "vacuum tubes and transistors to modern GPUs. " * 120)
    + "\n\nSummarize the passage above in one sentence.",
]
N = 200

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default=str(ROOT / "models" / "Inkling-2-Mini-4bit"))
ap.add_argument("--prefill-step", type=int, default=512)
ap.add_argument("--write", help="write a new reference here instead of checking")
a = ap.parse_args()

mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
model, tok = load(a.model, model_config={"vision_config": None})
out = []
for p in PROMPTS:
    ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
    toks, lp0 = [], None
    for i, (t, logprobs) in enumerate(
        generate_step(mx.array(ids), model, max_tokens=N, prefill_step_size=a.prefill_step)
    ):
        if i == 0:
            lp0 = logprobs.astype(mx.float32)
        toks.append(int(t))
    top = mx.argsort(-lp0)[:20]
    out.append({"prompt_tokens": len(ids), "tokens": toks, "top20_ids": top.tolist(),
                "top20_logprobs": [round(float(v), 5) for v in lp0[top].tolist()]})

if a.write:
    pathlib.Path(a.write).write_text(json.dumps({"meta": {}, "prompts": out}))
    print("wrote", a.write)
    raise SystemExit(0)

ref = json.loads(REF.read_text())["prompts"]
ok = True
for i, (r, o) in enumerate(zip(ref, out)):
    n = min(len(r["tokens"]), len(o["tokens"]))
    div = next((k for k in range(n) if r["tokens"][k] != o["tokens"][k]), None)
    match = div is None and len(r["tokens"]) == len(o["tokens"])
    # Compare as id -> log-prob maps: tied values can come back in a different order.
    ref_lp = dict(zip(r["top20_ids"], r["top20_logprobs"]))
    our_lp = dict(zip(o["top20_ids"], o["top20_logprobs"]))
    common = ref_lp.keys() & our_lp.keys()
    same_top = len(common) == len(ref_lp)
    d0 = max(abs(ref_lp[k] - our_lp[k]) for k in common) if common else None
    ok &= match
    print(json.dumps({"prompt": i, "prompt_tokens": o["prompt_tokens"], "match": match,
                      "first_divergence": div, "top20_same": same_top,
                      "top20_logprob_maxabs": None if d0 is None else round(d0, 5)}))
print("REF_CHECK", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
