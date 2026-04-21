#!/usr/bin/env bash
set -euo pipefail

python3 -m sneddy_baby_ai.cli.train_bc \
  --demos sneddy_baby_ai/artifacts/demos/easy/putnextlocal.npz \
  --checkpoint sneddy_baby_ai/artifacts/exports/putnextlocal_large.pt \
  --vocab sneddy_baby_ai/artifacts/global_vocab.json \
  --config bc_easy \
  --model-preset advance_large \
  --seed 42 \
  --eval-envs BabyAI-PutNextLocal-v0 \
  --aux-preset aux_v1 \
  --recurrent
