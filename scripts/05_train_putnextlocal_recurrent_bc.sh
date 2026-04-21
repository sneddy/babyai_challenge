#!/usr/bin/env bash
set -euo pipefail

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi

  local candidate
  for candidate in python3 python /Users/sneddy/anaconda3/envs/babyai/bin/python; do
    if [[ "$candidate" = /* ]]; then
      [[ -x "$candidate" ]] || continue
    else
      command -v "$candidate" >/dev/null 2>&1 || continue
    fi
    if "$candidate" - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
import torch  # noqa: F401
PY
    then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "Could not find a Python interpreter with numpy and torch. Set PYTHON_BIN explicitly." >&2
  return 1
}

PYTHON_BIN="$(resolve_python)"

"$PYTHON_BIN" -m sneddy_baby_ai.cli.train_bc \
  --demos sneddy_baby_ai/artifacts/demos/easy/putnextlocal.npz \
  --checkpoint sneddy_baby_ai/artifacts/exports/putnextlocal_recurrent_bc.pt \
  --vocab sneddy_baby_ai/artifacts/global_vocab.json \
  --config bc_easy \
  --model-preset advance \
  --seed 42 \
  --eval-envs BabyAI-PutNextLocal-v0 \
  --warm-start sneddy_baby_ai/artifacts/exports/advance_bc_easy_v2_best.pt \
  --recurrent
