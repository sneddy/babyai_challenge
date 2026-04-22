# sneddy_baby_ai Runbook

This README is intentionally operational. It documents:

- which CLI entrypoints exist
- which config files matter
- how to launch the main training workflows
- where checkpoints, demos, and submissions are written

## Main Entrypoints

The package exposes four CLIs that cover the normal workflow:

| Task | Command |
| --- | --- |
| generate demonstrations | `python3 -m sneddy_baby_ai.cli.generate_demos` |
| train PPO / RL | `python3 -m sneddy_baby_ai.cli.train_rl` |
| train behavior cloning | `python3 -m sneddy_baby_ai.cli.train_bc` |
| export leaderboard submission | `python3 -m sneddy_baby_ai.cli.export_submission` |

For the exact experiment launchers used in this repository, see:

- `scripts/generate_easy.sh`
- `scripts/generate_moderate.sh`
- `scripts/generate_hard.sh`
- `scripts/feedforward/*.sh`
- `scripts/recurrent/*.sh`

## Important Config Files

### Shared config

| File | Purpose |
| --- | --- |
| `configs/base.yaml` | global defaults for tokenizer, PPO, BC, artifact paths, and training defaults |
| `configs/envs/tiers.yaml` | tier aliases for `easy`, `moderate`, `hard` |

### Training presets

These are the presets you can pass through `--config`.

| Preset | File | Typical use |
| --- | --- | --- |
| `default` | `configs/presets/default.yaml` | plain default PPO / BC values |
| `stable_multitask` | `configs/presets/stable_multitask.yaml` | main shared-policy RL training |
| `bc_easy` | `configs/presets/bc_easy.yaml` | BC on easy-tier demos |
| `bc_moderate` | `configs/presets/bc_moderate.yaml` | BC on easy + moderate demos |
| `bc_hard` | `configs/presets/bc_hard.yaml` | BC on all tiers |
| `bc_warmstart_push` | `configs/presets/bc_warmstart_push.yaml` | feedforward RL fine-tuning after BC |
| `bc_warmstart_push_recurrent` | `configs/presets/bc_warmstart_push_recurrent.yaml` | recurrent RL fine-tuning after BC |

### Model presets

These are the presets you can pass through `--model-preset`.

| Preset | File | Typical use |
| --- | --- | --- |
| `base` | `configs/models/base.yaml` | generic default model |
| `minimalistic` | `configs/models/minimalistic.yaml` | small grounded RL baseline |
| `advance` | `configs/models/advance.yaml` | main feedforward BC model |
| `advance_large` | `configs/models/advance_large.yaml` | larger recurrent probe |
| `advance_largest` | `configs/models/advance_largest.yaml` | main recurrent + auxiliary model |

### Auxiliary supervision

| File | Purpose |
| --- | --- |
| `sneddy_baby_ai/auxiliary/specs.py` | defines auxiliary head presets |

Current auxiliary preset:

- `aux_v1`

## Environment Selection

All CLIs that accept env groups support tier aliases from `configs/envs/tiers.yaml`.

Examples:

- `--env BabyAI-GoToObj-v0`
- `--envs BabyAI-GoToObj-v0,BabyAI-GoToLocal-v0`
- `--envs easy`
- `--envs easy,moderate`
- `--all-envs`
- `--easy`
- `--moderate`
- `--hard`

## Common Workflows

### 1. Generate demonstrations

Generate easy-tier demos:

```bash
python3 -m sneddy_baby_ai.cli.generate_demos \
  --envs easy \
  --episodes 5000 \
  --output-dir sneddy_baby_ai/artifacts/demos/easy \
  --aux-preset aux_v1
```

Generate moderate-tier demos:

```bash
python3 -m sneddy_baby_ai.cli.generate_demos \
  --envs moderate \
  --episodes 5000 \
  --output-dir sneddy_baby_ai/artifacts/demos/moderate \
  --aux-preset aux_v1
```

Generate hard-tier demos:

```bash
python3 -m sneddy_baby_ai.cli.generate_demos \
  --envs hard \
  --episodes 1000 \
  --output-dir sneddy_baby_ai/artifacts/demos/hard \
  --aux-preset aux_v1
```

Notes:

- `--aux-preset aux_v1` is recommended if you will later train recurrent BC with auxiliary heads.
- Use the `scripts/generate_*.sh` wrappers if you want the repository defaults directly.

### 2. Train RL / PPO

Single environment:

```bash
python3 -m sneddy_baby_ai.cli.train_rl \
  --env BabyAI-GoToObj-v0 \
  --run-name gotoobj_rl \
  --config default \
  --model-preset base \
  --seed 42 \
  --timesteps 1000000
```

Multitask RL over tiers:

```bash
python3 -m sneddy_baby_ai.cli.train_rl \
  --envs easy,moderate \
  --run-name rl_easy_moderate \
  --config stable_multitask \
  --model-preset minimalistic \
  --seed 42 \
  --timesteps 1500000
```

Important flags:

- `--run-name`: controls checkpoint/export naming
- `--config`: training preset from `configs/presets/`
- `--model-preset`: architecture preset from `configs/models/`
- `--warm-start`: initialize PPO from an older `.zip` or exported `.pt`
- `--bc-checkpoint`: initialize PPO from a BC checkpoint
- `--bc-demos`: run BC first from demos, then warm-start PPO from it
- `--recurrent`: train recurrent PPO instead of feedforward PPO
- `--resume`: continue an existing PPO run from an SB3 `.zip`

### 3. Train feedforward BC

Easy-tier BC:

```bash
python3 -m sneddy_baby_ai.cli.train_bc \
  --demos \
    sneddy_baby_ai/artifacts/demos/easy/gotoobj.npz \
    sneddy_baby_ai/artifacts/demos/easy/gotolocal.npz \
    sneddy_baby_ai/artifacts/demos/easy/gotoredballgrey.npz \
    sneddy_baby_ai/artifacts/demos/easy/pickuploc.npz \
    sneddy_baby_ai/artifacts/demos/easy/putnextlocal.npz \
  --checkpoint sneddy_baby_ai/artifacts/exports/advance_bc_easy.pt \
  --vocab sneddy_baby_ai/artifacts/global_vocab.json \
  --config bc_easy \
  --model-preset advance \
  --seed 42 \
  --eval-envs BabyAI-GoToObj-v0,BabyAI-GoToLocal-v0,BabyAI-GoToRedBallGrey-v0,BabyAI-PickupLoc-v0,BabyAI-PutNextLocal-v0
```

Easy + moderate BC:

```bash
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
  --checkpoint sneddy_baby_ai/artifacts/exports/advance_bc_easy_moderate.pt \
  --vocab sneddy_baby_ai/artifacts/global_vocab.json \
  --config bc_moderate \
  --model-preset advance \
  --seed 42 \
  --eval-envs BabyAI-GoToObj-v0,BabyAI-GoToLocal-v0,BabyAI-GoToRedBallGrey-v0,BabyAI-PickupLoc-v0,BabyAI-PutNextLocal-v0,BabyAI-GoToRedBall-v0,BabyAI-GoToObjMaze-v0,BabyAI-GoTo-v0,BabyAI-Pickup-v0,BabyAI-Open-v0
```

Important BC flags:

- `--checkpoint`: output `.pt` path
- `--vocab`: mission vocabulary JSON, usually `sneddy_baby_ai/artifacts/global_vocab.json`
- `--warm-start`: initialize BC model from an older `.zip` or exported `.pt`
- `--eval-envs`: strongly recommended for holdout evaluation and `best` checkpoint selection

Behavior notes:

- if `--eval-envs` is given and you pass multiple demo files, BC uses holdout environment evaluation
- with multiple tasks, adaptive BC resampling is driven by per-env `val_success_rate`
- if `--eval-envs` is omitted, BC falls back to a `90/10` demo split

### 4. Train recurrent BC with auxiliary heads

Main recurrent + auxiliary pattern:

```bash
python3 -m sneddy_baby_ai.cli.train_bc \
  --demos \
    sneddy_baby_ai/artifacts/demos/easy/gotoobj.npz \
    sneddy_baby_ai/artifacts/demos/easy/gotolocal.npz \
    sneddy_baby_ai/artifacts/demos/easy/gotoredballgrey.npz \
    sneddy_baby_ai/artifacts/demos/easy/pickuploc.npz \
    sneddy_baby_ai/artifacts/demos/easy/putnextlocal.npz \
  --checkpoint sneddy_baby_ai/artifacts/exports/advance_bc_easy_recurrent_aux.pt \
  --vocab sneddy_baby_ai/artifacts/global_vocab.json \
  --config bc_easy \
  --model-preset advance_largest \
  --seed 42 \
  --recurrent \
  --aux-preset aux_v1 \
  --eval-envs BabyAI-GoToObj-v0,BabyAI-GoToLocal-v0,BabyAI-GoToRedBallGrey-v0,BabyAI-PickupLoc-v0,BabyAI-PutNextLocal-v0
```

Use this pattern when you want:

- recurrent BC instead of feedforward BC
- auxiliary state heads from `aux_v1`
- the larger recurrent architecture

### 5. Fine-tune RL from BC

Feedforward RL after BC:

```bash
python3 -m sneddy_baby_ai.cli.train_rl \
  --envs easy,moderate \
  --run-name rl_from_bc \
  --config bc_warmstart_push \
  --model-preset advance \
  --seed 42 \
  --timesteps 1000000 \
  --warm-start sneddy_baby_ai/artifacts/exports/advance_bc_all_tiers_v1_best.pt
```

Recurrent RL after recurrent BC:

```bash
python3 -m sneddy_baby_ai.cli.train_rl \
  --envs easy,moderate \
  --run-name rl_from_recurrent_bc \
  --config bc_warmstart_push_recurrent \
  --model-preset advance_largest \
  --seed 42 \
  --timesteps 1000000 \
  --recurrent \
  --warm-start sneddy_baby_ai/artifacts/exports/advance_bc_easy_moderate_recurrent_aux_v1_best.pt
```

### 6. Resume and warm-start

Resume an RL run from an SB3 checkpoint:

```bash
python3 -m sneddy_baby_ai.cli.train_rl \
  --envs easy,moderate \
  --run-name rl_easy_moderate \
  --config stable_multitask \
  --seed 42 \
  --resume sneddy_baby_ai/artifacts/checkpoints/rl_easy_moderate_latest.zip
```

Warm-start a new RL run from an older PPO or exported policy:

```bash
python3 -m sneddy_baby_ai.cli.train_rl \
  --env BabyAI-GoToLocal-v0 \
  --run-name gotolocal_warm \
  --config default \
  --seed 42 \
  --warm-start sneddy_baby_ai/artifacts/exports/goto_easy_minimalistic_best.pt
```

Warm-start BC from an older checkpoint:

```bash
python3 -m sneddy_baby_ai.cli.train_bc \
  --demos sneddy_baby_ai/artifacts/demos/easy/putnextlocal.npz \
  --checkpoint sneddy_baby_ai/artifacts/exports/putnextlocal_recurrent_bc.pt \
  --vocab sneddy_baby_ai/artifacts/global_vocab.json \
  --config bc_easy \
  --model-preset advance_large \
  --seed 42 \
  --recurrent \
  --aux-preset aux_v1 \
  --warm-start sneddy_baby_ai/artifacts/exports/advance_bc_easy_recurrent_aux_v1_best.pt \
  --eval-envs BabyAI-PutNextLocal-v0
```

## Export and Evaluation

Export a trained policy into a leaderboard submission zip:

```bash
python3 -m sneddy_baby_ai.cli.export_submission \
  --checkpoint sneddy_baby_ai/artifacts/exports/advance_bc_all_recurrent_aux_v1_best.pt \
  --zip-output submissions/advance_bc_all_recurrent_aux_v1_best.zip
```

Validate the zip:

```bash
python3 babyai-ml8103-leaderboard-2026/evaluation/validate_submission.py \
  submissions/advance_bc_all_recurrent_aux_v1_best.zip
```

Run official local evaluation:

```bash
python3 babyai-ml8103-leaderboard-2026/evaluation/evaluate_submission.py \
  --submission submissions/advance_bc_all_recurrent_aux_v1_best.zip \
  --seed-dir babyai-ml8103-leaderboard-2026/eval_seeds
```

## Artifacts

Default artifact roots are defined in `configs/base.yaml`.

Main outputs:

| Path | Contents |
| --- | --- |
| `sneddy_baby_ai/artifacts/demos/` | generated expert demos |
| `sneddy_baby_ai/artifacts/checkpoints/` | resumable RL `.zip` checkpoints |
| `sneddy_baby_ai/artifacts/exports/` | exported `.pt` policies and `*_best.pt` checkpoints |
| `sneddy_baby_ai/artifacts/logs/` | JSONL training metrics |
| `submissions/` | zipped submission packages |

Typical naming:

- RL latest checkpoint: `sneddy_baby_ai/artifacts/checkpoints/<RUN>_latest.zip`
- RL exported policy: `sneddy_baby_ai/artifacts/exports/<RUN>_policy.pt`
- BC latest checkpoint: path passed to `--checkpoint`
- best BC / best RL export: usually `*_best.pt`

## Fastest Path

If you just want the shortest route from data to leaderboard zip:

1. generate demos with `scripts/generate_easy.sh`, `scripts/generate_moderate.sh`, `scripts/generate_hard.sh`
2. train BC or recurrent BC using the matching scripts under `scripts/feedforward/` or `scripts/recurrent/`
3. export the best `.pt` checkpoint with `python3 -m sneddy_baby_ai.cli.export_submission`
4. validate and evaluate the resulting zip with the official scripts under `babyai-ml8103-leaderboard-2026/evaluation/`
