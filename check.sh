#!/bin/zsh
# Correctness check: this install must reproduce the reference greedy outputs (~3 min).
# Stop any running ./serve.sh first: the Mac fits one copy of the model at a time.
set -euo pipefail
ROOT=${0:A:h}
exec "$ROOT/.venv/bin/python" "$ROOT/tools/ref_check.py" "$@"
