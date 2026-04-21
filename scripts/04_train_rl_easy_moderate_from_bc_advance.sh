#!/usr/bin/env bash
set -euo pipefail

python3 -m sneddy_baby_ai.cli.train_rl \
  --envs easy,moderate \
  --run-name advance_rl_easy_moderate_v1 \
  --config bc_warmstart_push \
  --model-preset advance \
  --seed 42 \
  --timesteps 10000000 \
  --warm-start sneddy_baby_ai/artifacts/exports/advance_bc_all_tiers_v1_best.pt
