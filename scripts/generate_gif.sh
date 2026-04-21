/Users/sneddy/anaconda3/envs/babyai/bin/python -m sneddy_baby_ai.cli.view_rollout \
  --env BabyAI-PutNextLocal-v0 \
  --seed 42 \
  --checkpoint sneddy_baby_ai/artifacts/exports/advance_bc_all_tiers_v1_best.pt \
  --mode compare \
  --output-dir sneddy_baby_ai/artifacts/rollouts/putnextlocal_compare \
  --count 10 \
  --max-steps 128 \
  --format mp4
