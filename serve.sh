#!/bin/zsh
# Serve Inkling 2 Mini (OpenAI-compatible) on http://127.0.0.1:8790/v1, model "inkling-2-mini".
#
#   ./serve.sh
#   INKLING_PORT=9000 INKLING_MAX_CONTEXT=16384 ./serve.sh
#   INKLING_HOST=0.0.0.0 ./serve.sh   # listen on the network too (no authentication!)
#   INKLING_PROMPT_CACHE=0 ./serve.sh # turn off reuse of the previous prompt's prefix
#
# One request at a time. Replies put the answer in `content`, the thinking in
# `reasoning_content` and tool calls in `tool_calls`. Pass "reasoning_effort"
# (none/minimal/low/medium/high/xhigh/max) to change how long it thinks.
set -euo pipefail
ROOT=${0:A:h}
[[ -x $ROOT/.venv/bin/python && -f $ROOT/models/Inkling-2-Mini-4bit/config.json ]] \
  || { echo "run ./install.sh first" >&2; exit 1; }

mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
if (( mem_gb < 32 )) && (( $(sysctl -n iogpu.wired_limit_mb) < 21504 )); then
  echo "warning: the GPU memory cap is the macOS default; prompts over ~1k tokens will run out of memory." >&2
  echo "         sudo sysctl iogpu.wired_limit_mb=21504   (resets at reboot)" >&2
fi

cache_flags=(--prompt-cache-max-tokens "${INKLING_PROMPT_CACHE_TOKENS:-16384}")
[[ ${INKLING_PROMPT_CACHE:-1} == 0 ]] && cache_flags=(--no-prompt-cache)

exec "$ROOT/.venv/bin/python" "$ROOT/inkling_server.py" \
  --model "$ROOT/models/Inkling-2-Mini-4bit" \
  --host "${INKLING_HOST:-127.0.0.1}" --port "${INKLING_PORT:-8790}" \
  --prefill-step 512 \
  --max-context "${INKLING_MAX_CONTEXT:-65536}" \
  "${cache_flags[@]}"
