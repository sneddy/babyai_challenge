# sneddy_baby_ai

## Difference vs baseline

Compared with [baseline/README.md](/Users/sneddy/research/baby_ai/babyai-ml8103-leaderboard-2026/baseline/README.md), this pipeline changes the main weak points of the starter:

- baseline ignores mission text; this pipeline uses **mission language explicitly**
- baseline uses a simple CNN over the symbolic grid; this pipeline uses **symbol embeddings + GRU mission encoder + FiLM residual blocks**
- baseline is effectively **single-task**; this pipeline supports **one shared multi-task policy**
- multitask sampling is **adaptive**: harder envs get sampled more often based on per-env validation success
- baseline has no real curriculum/replay path; this pipeline supports **stage training with replay-weighted sampling**
- baseline does not include BC warm-start in the RL path; this pipeline supports **`--bc-checkpoint` and `--bc-demos`**
- baseline is a plain PPO starter; this pipeline supports **feedforward PPO with BC warm-start and adaptive multitask sampling**
- baseline does not select checkpoints by cross-env validation; this pipeline keeps **`latest` and `best` by validation success rate**

Short version:

- baseline = simple PPO starter
- `sneddy_baby_ai` = language-grounded symbolic BabyAI policy with multitask, BC warm-start, and export-safe submission packaging

## Difference vs BabyAI 1.1

This codebase is BabyAI-1.1-inspired, not a direct port.

What is similar:

- symbolic bag-of-words tile embeddings
- GRU mission encoding
- FiLM conditioning
- residual visual blocks

What is different:

- built for the **leaderboard submission contract**, not the original BabyAI stack
- uses a **global vocab across leaderboard envs**
- supports **one shared multi-task policy**
- uses **cross-env validation** to choose `best`
- integrates **BC warm-start into the RL runner**
- exports directly into `demo_submission/`

Short version:

- BabyAI 1.1 = architecture ideas in the original research stack
- `sneddy_baby_ai` = BabyAI-style architecture adapted to one final leaderboard submission

## Run

Useful config presets:

- `bootstrap`: from-scratch multitask bootstrap
- `stable_multitask`: main long-run shared-policy preset
- `hard_focus`: stronger pressure on still-unsolved tasks
- `conservative_finetune`: gentle continuation of a good checkpoint
- `bc_recovery`: BC-heavy recovery preset for stuck manipulation tasks
- `debug`: short smoke test

Examples:

```bash
python3 -m sneddy_baby_ai.cli.train_rl --env BabyAI-GoToObj-v0 --config bootstrap --seed 42 --timesteps 500000
python3 -m sneddy_baby_ai.cli.train_rl --all-envs --run-name leaderboard_multitask --config stable_multitask --seed 42 --timesteps 1500000
```

Single env:

```bash
python3 -m sneddy_baby_ai.cli.train_rl \
  --env BabyAI-GoToObj-v0 \
  --config default \
  --seed 42 \
  --timesteps 1000000
```

Multi-task:

```bash
python3 -m sneddy_baby_ai.cli.train_rl \
  --envs BabyAI-GoToObj-v0,BabyAI-GoToLocal-v0,BabyAI-GoToRedBallGrey-v0 \
  --run-name easy_nav_multitask \
  --config default \
  --seed 42 \
  --timesteps 1000000
```

Tier aliases also work:

```bash
python3 -m sneddy_baby_ai.cli.train_rl --envs easy,moderate --run-name easy_moderate --config stable_multitask --seed 42 --timesteps 1000000
python3 -m sneddy_baby_ai.cli.train_rl --easy --run-name full_easy --config stable_multitask --seed 42 --timesteps 1000000
```

All leaderboard envs:

```bash
python3 -m sneddy_baby_ai.cli.train_rl \
  --all-envs \
  --run-name leaderboard_multitask \
  --config default \
  --seed 42 \
  --timesteps 1000000
```

BC warm-start from demos:

```bash
python3 -m sneddy_baby_ai.cli.train_rl \
  --env BabyAI-GoToObj-v0 \
  --config default \
  --seed 42 \
  --timesteps 1000000 \
  --bc-demos sneddy_baby_ai/artifacts/demos/gotoobj_demos.npz
```

Artifacts:

- feedforward latest: `sneddy_baby_ai/artifacts/checkpoints/<RUN>_latest.zip`
- latest export: `sneddy_baby_ai/artifacts/exports/<RUN>_policy.pt`
- best export: `sneddy_baby_ai/artifacts/exports/<RUN>_best.pt`

## Continue

Resume the same run:

```bash
python3 -m sneddy_baby_ai.cli.train_rl \
  --all-envs \
  --run-name leaderboard_multitask \
  --config default \
  --seed 42 \
  --timesteps 1000000 \
  --resume sneddy_baby_ai/artifacts/checkpoints/leaderboard_multitask_latest.zip
```

Warm-start a new run from an old checkpoint:

```bash
python3 -m sneddy_baby_ai.cli.train_rl \
  --env BabyAI-GoToLocal-v0 \
  --config default \
  --seed 42 \
  --timesteps 1000000 \
  --warm-start sneddy_baby_ai/artifacts/checkpoints/GoToObj_latest.zip
```

`Ctrl+C` keeps resumable artifacts and the latest exported policy.

## Validate

Export:

```bash
python3 -m sneddy_baby_ai.cli.export_submission \
  --checkpoint sneddy_baby_ai/artifacts/exports/advance_bc_easy_moderate_recurrent_aux_v1_best.pt \
  --zip-output submissions/largest_aux_rec.zip
```

Validate ZIP:

```bash
python3 babyai-ml8103-leaderboard-2026/evaluation/validate_submission.py demo_submission.zip
```

Official local evaluation:

```bash
python3 babyai-ml8103-leaderboard-2026/evaluation/evaluate_submission.py \
  --submission demo_submission.zip \
  --seed-dir babyai-ml8103-leaderboard-2026/eval_seeds
```
