#!/usr/bin/env bash
set -euo pipefail

python3 -m sneddy_baby_ai.cli.train_rl \
  --envs easy,moderate \
  --run-name rl_auc_rec \
  --config bc_warmstart_push_recurrent \
  --model-preset advance_largest \
  --seed 42 \
  --timesteps 10000000 \
  --recurrent \
  --warm-start sneddy_baby_ai/artifacts/exports/advance_bc_easy_moderate_recurrent_aux_v1_best.pt
