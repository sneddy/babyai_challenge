#!/usr/bin/env bash
set -euo pipefail

python3 -m sneddy_baby_ai.cli.train_bc \
  --demos \
    sneddy_baby_ai/artifacts/demos/easy/gotoobj.npz \
    sneddy_baby_ai/artifacts/demos/easy/gotolocal.npz \
    sneddy_baby_ai/artifacts/demos/easy/gotoredballgrey.npz \
    sneddy_baby_ai/artifacts/demos/easy/pickuploc.npz \
    sneddy_baby_ai/artifacts/demos/easy/putnextlocal.npz \
  --checkpoint sneddy_baby_ai/artifacts/exports/advance_bc_easy_v2.pt \
  --vocab sneddy_baby_ai/artifacts/global_vocab.json \
  --config bc_easy_finetune \
  --model-preset advance \
  --seed 42 \
  --eval-envs BabyAI-GoToObj-v0,BabyAI-GoToLocal-v0,BabyAI-GoToRedBallGrey-v0,BabyAI-PickupLoc-v0,BabyAI-PutNextLocal-v0 \
  --warm-start sneddy_baby_ai/artifacts/exports/advance_bc_easy_best.pt
