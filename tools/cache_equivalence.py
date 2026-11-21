"""Multi-turn conversation against a running server, greedy; records every reply and
time to first token. Run once with the prompt cache on and once with it off, then
compare: replies must be identical, and turns 2+ should start much sooner with it on.

  python3 -I tools/cache_equivalence.py run URL OUT.json [--doc-words 3000]
  python3 -I tools/cache_equivalence.py compare OFF.json ON.json
"""

import json
import sys
import time
import urllib.request

TOOLS = [{"type": "function", "function": {
    "name": "lookup_part", "description": "Look up a part number in the inventory database",
    "parameters": {"type": "object", "properties": {"part": {"type": "string"}},
                   "required": ["part"]}}}]


def document(words):
    rows = []
    i = 0
    while sum(len(r.split()) for r in rows) < words:
        rows.append(f"Section {i}: part P-{1000 + i * 7 % 900} ships from bay {i % 17} "
                    f"with tolerance {i % 5}.{i % 9} mm and inspector code K{i * 13 % 97}.")
        i += 1
    return "\n".join(rows)


def chat(url, messages, stream=True):
    body = {"model": "inkling-2-mini", "messages": messages, "tools": TOOLS,
            "temperature": 0, "max_tokens": 600, "stream": stream,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    reasoning = content = ""
    calls, finish, usage = [], None, {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if (piece or d.get("tool_calls")) and first is None:
                    first = time.time()
                reasoning += d.get("reasoning_content") or ""
                content += d.get("content") or ""
                calls += d.get("tool_calls") or []
                finish = ch.get("finish_reason") or finish
    return {"ttft_s": round((first or time.time()) - t0, 3), "reasoning": reasoning,
            "content": content, "finish": finish, "prompt_tokens": usage.get("prompt_tokens"),
            "tool_calls": [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                           for c in calls], "raw_calls": calls}


def run(url, out, doc_words):
    doc = document(doc_words)
    msgs = [{"role": "user", "content": doc + "\n\nWhich bay ships part P-1007? One sentence."}]
    turns = []
    follow_ups = [
        "And what is that section's inspector code?",
        "Check part P-1014 in the inventory database with the tool.",
        None,  # tool result turn
        "Thanks. Summarize everything we found in two sentences.",
    ]
    for step, follow in enumerate([None] + follow_ups):
        if step > 0:
            if follow is None:
                prev = turns[-1]
                if not prev["raw_calls"]:
                    break
                call = prev["raw_calls"][0]
                msgs.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps({"part": "P-1014", "in_stock": 42, "bin": "B7"})})
            else:
                msgs.append({"role": "user", "content": follow})
        r = chat(url, msgs)
        turns.append(r)
        msg = {"role": "assistant", "content": r["content"]}
        if r["raw_calls"]:
            msg["tool_calls"] = [{k: c[k] for k in ("id", "type", "function")} for c in r["raw_calls"]]
        msgs.append(msg)
        print(json.dumps({"turn": len(turns), "prompt_tokens": r["prompt_tokens"],
                          "ttft_s": r["ttft_s"], "finish": r["finish"],
                          "content": r["content"][:80], "tools": r["tool_calls"]}), flush=True)
    for t in turns:
        t.pop("raw_calls")
    json.dump(turns, open(out, "w"), indent=1)


def compare(off_path, on_path):
    off, on = json.load(open(off_path)), json.load(open(on_path))
    ok = len(off) == len(on)
    for i, (a, b) in enumerate(zip(off, on), 1):
        same = all(a[k] == b[k] for k in ("reasoning", "content", "finish", "tool_calls", "prompt_tokens"))
        ok &= same
        print(json.dumps({"turn": i, "prompt_tokens": a["prompt_tokens"], "identical": same,
                          "ttft_off_s": a["ttft_s"], "ttft_on_s": b["ttft_s"]}))
    print("CACHE_EQUIVALENCE", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "run":
        words = int(sys.argv[sys.argv.index("--doc-words") + 1]) if "--doc-words" in sys.argv else 3000
        run(sys.argv[2], sys.argv[3], words)
    else:
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
