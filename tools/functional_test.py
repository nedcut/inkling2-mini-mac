"""Functional checks for inkling_server.py (stdlib client). usage: server_functional.py URL"""

import json
import sys
import urllib.request

URL = sys.argv[1].rstrip("/") + "/v1/chat/completions"


def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def stream(body):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    reasoning = content = ""
    finish = usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                reasoning += d.get("reasoning_content") or ""
                content += d.get("content") or ""
                finish = ch.get("finish_reason") or finish
    return reasoning, content, finish, usage


ok = True

def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ": " + detail, flush=True)


r = post({"messages": [{"role": "user", "content": "What is 17*23? Answer with just the number."}],
          "temperature": 0, "max_tokens": 400})
m = r["choices"][0]["message"]
check("plain answer", m["content"].strip() == "391" and "<|" not in m["content"]
      and m.get("reasoning_content"), json.dumps(m)[:300])

reasoning, content, finish, usage = stream(
    {"messages": [{"role": "user", "content": "Name the capital of France in one word."}],
     "temperature": 0, "max_tokens": 400})
check("streaming", "paris" in content.lower() and "<|" not in content + reasoning
      and finish == "stop" and usage, f"content={content!r} finish={finish} usage={usage}")

tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
r = post({"messages": msgs, "tools": tools, "temperature": 0, "max_tokens": 600})
c = r["choices"][0]
calls = c["message"].get("tool_calls") or []
check("tool call", c["finish_reason"] == "tool_calls" and calls
      and calls[0]["function"]["name"] == "get_weather"
      and "paris" in calls[0]["function"]["arguments"].lower(), json.dumps(c)[:400])
if calls:
    msgs += [c["message"], {"role": "tool", "tool_call_id": calls[0]["id"],
                            "content": json.dumps({"temp_c": 18, "sky": "cloudy"})}]
    r = post({"messages": msgs, "tools": tools, "temperature": 0, "max_tokens": 600})
    c = r["choices"][0]
    check("tool result used", "18" in (c["message"]["content"] or ""), json.dumps(c)[:400])

r = post({"messages": [{"role": "user", "content": "What is 2+2?"}],
          "reasoning_effort": "low", "temperature": 0, "max_tokens": 300})
m = r["choices"][0]["message"]
check("reasoning_effort low", "4" in m["content"], json.dumps(m)[:300])

print("FUNCTIONAL", "PASS" if ok else "FAIL")
