#!/usr/bin/env bash
set -euo pipefail

python3 -m sneddy_baby_ai.cli.train_bc \
  --demos \
    sneddy_baby_ai/artifacts/demos/easy/gotoobj.npz \
    sneddy_baby_ai/artifacts/demos/easy/gotolocal.npz \
    sneddy_baby_ai/artifacts/demos/easy/gotoredballgrey.npz \
    sneddy_baby_ai/artifacts/demos/easy/pickuploc.npz \
    sneddy_baby_ai/artifacts/demos/easy/putnextlocal.npz \
    sneddy_baby_ai/artifacts/demos/moderate/gotoredball.npz \
    sneddy_baby_ai/artifacts/demos/moderate/gotoobjmaze.npz \
    sneddy_baby_ai/artifacts/demos/moderate/goto.npz \
    sneddy_baby_ai/artifacts/demos/moderate/pickup.npz \
    sneddy_baby_ai/artifacts/demos/moderate/open.npz \
  --checkpoint sneddy_baby_ai/artifacts/exports/advance_bc_easy_moderate_recurrent_aux_v1.pt \
  --vocab sneddy_baby_ai/artifacts/global_vocab.json \
  --config bc_moderate \
  --model-preset advance_largest \
  --seed 42 \
  --eval-envs BabyAI-GoToObj-v0,BabyAI-GoToLocal-v0,BabyAI-GoToRedBallGrey-v0,BabyAI-PickupLoc-v0,BabyAI-PutNextLocal-v0,BabyAI-GoToRedBall-v0,BabyAI-GoToObjMaze-v0,BabyAI-GoTo-v0,BabyAI-Pickup-v0,BabyAI-Open-v0 \
  --warm-start sneddy_baby_ai/artifacts/exports/advance_bc_easy_recurrent_aux_v1_best.pt \
  --aux-preset aux_v1 \
  --recurrent
