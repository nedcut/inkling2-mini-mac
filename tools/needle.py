"""Long-context recall through a running server: plant a code near the end of a prompt of
about N tokens and ask for it back. Reports prompt tokens, seconds, peak memory and the answer.

  python3 -I tools/needle.py --tokens 65000 [--endpoint http://127.0.0.1:8790]
"""

import argparse
import json
import random
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="http://127.0.0.1:8790")
ap.add_argument("--tokens", type=int, required=True)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
url = a.endpoint.rstrip("/")


def post(path, body, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


rng = random.Random(a.seed)
code = f"{rng.randrange(10**7, 10**8)}"
line = lambda i: (f"Entry {i}: crate {rng.randrange(1000, 9999)} moved from rack {rng.randrange(1, 60)} "
                  f"to rack {rng.randrange(1, 60)} at {rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}.")
lines, n = [], 0
per_line = post("/v1/token/encode", {"text": "\n".join(line(i) for i in range(50))})["length"] / 50
target_lines = int(a.tokens / per_line)
lines = [line(i) for i in range(target_lines)]
lines.insert(int(len(lines) * 0.9), f"Note: the vault code is {code}.")
prompt = "\n".join(lines) + "\n\nWhat is the vault code? Reply with only the code."
t0 = time.time()
out = post("/v1/chat/completions", {"model": "inkling-2-mini", "temperature": 0, "max_tokens": 4000,
                                    "reasoning_effort": "low",
                                    "messages": [{"role": "user", "content": prompt}]})
secs = time.time() - t0
msg = out["choices"][0]["message"]
u = out.get("usage", {})
ok = code in (msg.get("content") or "")
print(json.dumps({"recall": "PASS" if ok else "FAIL", "code": code, "content": msg.get("content"),
                  "finish": out["choices"][0].get("finish_reason"),
                  "reasoning_tail": (msg.get("reasoning_content") or "")[-300:],
                  "completion_tokens": u.get("completion_tokens"),
                  "prompt_tokens": u.get("prompt_tokens"), "seconds": round(secs, 1),
                  "prompt_tps": u.get("prompt_tps"), "generation_tps": u.get("generation_tps"),
                  "peak_memory_gb": u.get("peak_memory_gb")}))
raise SystemExit(0 if ok else 1)
