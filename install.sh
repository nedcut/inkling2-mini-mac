#!/bin/zsh
# Install Inkling 2 Mini (mlx-community 4-bit) with the faster decode patch, on Apple silicon.
# Safe to re-run: every step checks what is already there. Nothing outside this folder is
# changed except the Hugging Face cache (where the weights are downloaded).
#
#   ./install.sh
set -euo pipefail
ROOT=${0:A:h}
cd "$ROOT"

MLX_LM_REPO=https://github.com/ml-explore/mlx-lm
MLX_LM_PIN=0c4c99192230f1943dd468d297f0b8a64ae7b7c2   # ml-explore/mlx-lm PR #2086 (Inkling 2 Mini support)
PATCH=patches/mlx-lm-inkling2-mini-speed.patch
MODEL_REPO=mlx-community/Inkling-2-Mini-4bit
MODEL_REV=9856dc53edead71405f8348930937c3638ff21a2
MODEL_DIR=models/Inkling-2-Mini-4bit

say() { print -P "%B==>%b $*"; }
die() { print -P "%F{red}error:%f $*" >&2; exit 1; }

# --- machine checks ---------------------------------------------------------------
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || die "needs macOS on Apple silicon"
mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
(( mem_gb >= 24 )) || die "needs at least 24 GB of unified memory (this Mac has ${mem_gb} GB); the weights alone are 17.3 GB"
free_gb=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
have_weights=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-2-Mini-4bit/snapshots/$MODEL_REV 2>/dev/null || true)
if [[ -z $have_weights ]] && (( free_gb < 22 )); then
  die "needs about 20 GB free disk for the weights plus room to spare (${free_gb} GB free)"
fi
command -v git >/dev/null || die "needs git (xcode-select --install)"

# --- python environment -----------------------------------------------------------
if command -v uv >/dev/null; then
  UV=$(command -v uv)
elif [[ -x ~/.local/bin/uv ]]; then
  UV=~/.local/bin/uv
else
  UV=""
fi

# --- mlx-lm at the pinned PR commit, plus the decode patch --------------------------
if [[ ! -d vendor/mlx-lm/.git ]]; then
  say "cloning mlx-lm"
  mkdir -p vendor
  git clone -q "$MLX_LM_REPO" vendor/mlx-lm
fi
if [[ $(git -C vendor/mlx-lm rev-parse HEAD) != "$MLX_LM_PIN" ]]; then
  git -C vendor/mlx-lm diff --quiet || die "vendor/mlx-lm has local changes and is not at the pin; remove vendor/ and re-run"
  say "checking out mlx-lm ${MLX_LM_PIN:0:12} (PR #2086)"
  # Fetch the pinned commit by hash: the pull request branch can be force-pushed past it.
  git -C vendor/mlx-lm fetch -q origin "$MLX_LM_PIN" || git -C vendor/mlx-lm fetch -q origin "pull/2086/head"
  git -C vendor/mlx-lm checkout -q --detach "$MLX_LM_PIN"
fi
patch_sha=$(shasum -a 256 "$PATCH" | cut -d' ' -f1)
marker=vendor/.applied-patch-sha256
if [[ -f $marker && $(<$marker) == "$patch_sha" ]] && ! git -C vendor/mlx-lm diff --quiet; then
  say "speed patch already applied"
else
  if [[ -f $marker ]]; then
    # Only ever undo a patch this script applied itself (recorded in the marker),
    # including any files it added.
    say "replacing the previously applied patch"
    git -C vendor/mlx-lm checkout -q -- .
    git -C vendor/mlx-lm clean -fdq
  elif ! git -C vendor/mlx-lm diff --quiet; then
    die "vendor/mlx-lm has local changes this installer did not make; remove vendor/ and re-run"
  fi
  say "applying the speed patch"
  git -C vendor/mlx-lm apply "$ROOT/$PATCH"
  print -r -- "$patch_sha" > "$marker"
fi

if [[ ! -x .venv/bin/python ]]; then
  say "creating the Python environment"
  if [[ -n $UV ]]; then
    "$UV" venv -q --python 3.12 .venv
  else
    py=$(command -v python3.12 || command -v python3.13 || command -v python3.11 || true)
    [[ -n $py ]] || die "needs Python 3.11-3.13 (or uv: https://docs.astral.sh/uv/)"
    "$py" -m venv .venv
  fi
fi
say "installing pinned packages"
if [[ -n $UV ]]; then
  "$UV" pip install -q --python .venv/bin/python -r requirements.lock -e vendor/mlx-lm
else
  .venv/bin/python -m pip install -q -r requirements.lock -e vendor/mlx-lm
fi

# --- weights ------------------------------------------------------------------------
say "downloading ${MODEL_REPO}@${MODEL_REV:0:8} (17.3 GB; resumes if interrupted)"
snapshot=$(.venv/bin/hf download "$MODEL_REPO" --revision "$MODEL_REV" --quiet 2>/dev/null | tail -1)
[[ -d $snapshot ]] || snapshot=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--Inkling-2-Mini-4bit/snapshots/$MODEL_REV)
[[ -f $snapshot/config.json ]] || die "download did not produce a model folder ($snapshot)"

# A model folder that links the downloaded files and fixes one config value:
# transformers 5.24 rejects this checkpoint unless vision_config.text_hidden_size
# matches the text model's hidden size (2304); its loader only sets it when the
# checkpoint has no vision config at all.
say "preparing $MODEL_DIR"
mkdir -p "$MODEL_DIR"
for f in "$snapshot"/*; do
  name=${f:t}
  [[ $name == config.json ]] && continue
  ln -sfn "${f:A}" "$MODEL_DIR/$name"
done
.venv/bin/python -I - "$snapshot/config.json" "$MODEL_DIR/config.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
c = json.load(open(src))
c["vision_config"]["text_hidden_size"] = c["text_config"]["hidden_size"]
json.dump(c, open(dst, "w"), indent=2)
PY

# --- GPU memory cap -------------------------------------------------------------------
say "installed"
if (( mem_gb < 32 )); then
  limit=$(sysctl -n iogpu.wired_limit_mb)
  if (( limit < 21504 )); then
    print ""
    print "On a ${mem_gb} GB Mac the default GPU memory cap (~70% of RAM) is below the 17.3 GB of weights"
    print "plus working memory. Raise it before serving (resets at reboot; leaves ~3 GB for macOS):"
    print ""
    print "  sudo sysctl iogpu.wired_limit_mb=21504"
  fi
fi
print ""
print "Next:  ./serve.sh            # OpenAI-compatible server on http://127.0.0.1:8790/v1"
print "Check: ./check.sh            # correctness against the reference outputs (~3 min)"
