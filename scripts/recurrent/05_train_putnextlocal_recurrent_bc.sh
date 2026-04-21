#!/usr/bin/env bash
set -euo pipefail

python3 -m sneddy_baby_ai.cli.train_bc \
  --demos sneddy_baby_ai/artifacts/demos_aux/aux_v1/easy/putnextlocal.npz \
  --checkpoint sneddy_baby_ai/artifacts/exports/putnextlocal_recurrent_bc_aux_v1.pt \
  --vocab sneddy_baby_ai/artifacts/global_vocab.json \
  --config bc_easy \
  --model-preset advance \
  --seed 42 \
  --eval-envs BabyAI-PutNextLocal-v0 \
  --warm-start sneddy_baby_ai/artifacts/exports/advance_bc_easy_recurrent_aux_v1_best.pt \
  --aux-preset aux_v1 \
  --recurrent
