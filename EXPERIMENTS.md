# BabyAI Experiment History and Reproducibility Notes

This file is the technical companion to the main report. It records how the checkpoint ladder was produced in the current repository, which presets and scripts correspond to each stage, and where the relevant artifacts live. Repo-specific paths are intentional here: the goal is reproducibility, not narrative elegance.

## Public Score Summary

All overall scores below are the official weighted leaderboard score with tier weights `0.5 / 0.35 / 0.15` for `easy / moderate / hard`.

| ID | System | Easy | Moderate | Hard | Overall | `PutNextLocal` | Result file |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `B0` | provided baseline | `0.26` | `0.05` | `0.00` | `0.1475` | `0.00` | `babyai-ml8103-leaderboard-2026/evaluation/results/unknown_public_20260422_180119.json` |
| `C1` | `goto_easy_minimalistic_best` | `0.58` | `0.22` | `0.10` | `0.3820` | `0.00` | `evaluation/results/goto_easy_minimalistic_best.json` |
| `C2` | `advance_bc_easy_v2_best` | `0.78` | `0.30` | `0.08` | `0.5070` | `0.20` | `evaluation/results/advance_bc_easy_v2_best.json` |
| `C3` | `advance_bc_easy_moderate_v1_best` | `0.75` | `0.80` | `0.39` | `0.7135` | `0.10` | `evaluation/results/advance_bc_easy_moderate_v1_best.json` |
| `C4` | `advance_bc_all_tiers_v1_best` | `0.76` | `0.77` | `0.39` | `0.7080` | `0.15` | `evaluation/results/advance_bc_all_tiers_v1_best.json` |
| `C5` | `advance_bc_easy_recurrent_aux_v1_best` | `0.81` | `0.27` | `0.16` | `0.5235` | `0.50` | `evaluation/results/advance_bc_easy_recurrent_aux_v1_best.json` |
| `C6` | `advance_bc_easy_moderate_recurrent_aux_v1_best` | `0.83` | `0.97` | `0.57` | `0.8400` | `0.45` | `evaluation/results/advance_bc_easy_moderate_recurrent_aux_v1_best.json` |
| `C7` | `advance_bc_all_recurrent_aux_v1_best` | `0.86` | `0.93` | `0.61` | `0.8470` | `0.50` | `evaluation/results/advance_bc_all_recurrent_aux_v1_best.json` |
| `C8` | `rl_auc_rec_best` | `pending` | `pending` | `pending` | `pending` | `pending` | pending |

## Shared Reproducibility Anchors

### Artifact locations

- exported checkpoints: `sneddy_baby_ai/artifacts/exports/`
- public evaluation JSON files: `evaluation/results/`
- packaged submissions: `submissions/`
- demo datasets: `sneddy_baby_ai/artifacts/demos/`

### Tier aliases and demo generation

The environment groups are defined in [configs/envs/tiers.yaml](configs/envs/tiers.yaml).

| Tier | Env IDs | Demo script | Episodes per task |
| --- | --- | --- | ---: |
| easy | `BabyAI-GoToObj-v0`, `BabyAI-GoToLocal-v0`, `BabyAI-GoToRedBallGrey-v0`, `BabyAI-PickupLoc-v0`, `BabyAI-PutNextLocal-v0` | [scripts/generate_easy.sh](scripts/generate_easy.sh) | `5000` |
| moderate | `BabyAI-GoToRedBall-v0`, `BabyAI-GoToObjMaze-v0`, `BabyAI-GoTo-v0`, `BabyAI-Pickup-v0`, `BabyAI-Open-v0` | [scripts/generate_moderate.sh](scripts/generate_moderate.sh) | `5000` |
| hard | `BabyAI-Unlock-v0`, `BabyAI-UnblockPickup-v0`, `BabyAI-PutNextS7N4-v0`, `BabyAI-Synth-v0`, `BabyAI-SynthLoc-v0` | [scripts/generate_hard.sh](scripts/generate_hard.sh) | `1000` |

All three demo-generation scripts attach `--aux-preset aux_v1`, so the saved datasets include the auxiliary labels used by the later recurrent BC runs.

### Model presets

The main model presets are stored in [configs/models](configs/models/).

| Preset | File | Scale | Primary use |
| --- | --- | --- | --- |
| `minimalistic` | [configs/models/minimalistic.yaml](configs/models/minimalistic.yaml) | `96 / 32 / 64 / 128` | first grounded PPO baseline |
| `advance` | [configs/models/advance.yaml](configs/models/advance.yaml) | `160 / 80 / 160 / 320` | feedforward BC ladder |
| `advance_large` | [configs/models/advance_large.yaml](configs/models/advance_large.yaml) | `192 / 96 / 192 / 448` | isolated recurrent `PutNextLocal` probe |
| `advance_largest` | [configs/models/advance_largest.yaml](configs/models/advance_largest.yaml) | `256 / 128 / 256 / 512` | recurrent + auxiliary multitask ladder |

### Training presets

All presets inherit from [configs/base.yaml](configs/base.yaml).

| Preset | File | Regime | Key settings |
| --- | --- | --- | --- |
| `stable_multitask` | [configs/presets/stable_multitask.yaml](configs/presets/stable_multitask.yaml) | RL | `1.5M` timesteps, `lr=7.5e-5`, `ent_coef=0.01`, `sampling_uniform_alpha=0.2` |
| `bc_easy` | [configs/presets/bc_easy.yaml](configs/presets/bc_easy.yaml) | BC | `epochs=8`, `batch_size=256`, `lr=1e-3`, `aux_weight=0.9`, `action_reweight_on_holding_target=2.0` |
| `bc_moderate` | [configs/presets/bc_moderate.yaml](configs/presets/bc_moderate.yaml) | BC | `epochs=5`, `batch_size=256`, `lr=3e-4`, `aux_weight=2.0`, `sampling_episode_budget_per_task=1000` |
| `bc_hard` | [configs/presets/bc_hard.yaml](configs/presets/bc_hard.yaml) | BC | `epochs=5`, `batch_size=256`, `lr=2e-4`, `aux_weight=1.0`, `sampling_episode_budget_per_task=1000` |
| `bc_warmstart_push` | [configs/presets/bc_warmstart_push.yaml](configs/presets/bc_warmstart_push.yaml) | RL after BC | `1.0M` timesteps, `lr=2e-4 -> 1e-6`, `sampling_uniform_alpha=0.05` |
| `bc_warmstart_push_recurrent` | [configs/presets/bc_warmstart_push_recurrent.yaml](configs/presets/bc_warmstart_push_recurrent.yaml) | recurrent RL after BC | `1.0M` timesteps, `recurrence=16`, `batch_size=2048`, `lr=1e-4 -> 1e-6` |

### Auxiliary preset

The auxiliary head definition used by the recurrent BC ladder is [sneddy_baby_ai/auxiliary/specs.py](sneddy_baby_ai/auxiliary/specs.py).

`aux_v1` contains:

- `in_front_of_what`
- `obj_in_instr_visible`
- `holding_target_object`
- `adjacent_to_target_object`
- `valid_drop_position`
- `valid_pickup_action`
- `fixed_target_visible`

The carry-phase emphasis used in the report comes from `action_reweight_on_holding_target: 2.0` in the BC preset files.

## Main Checkpoint Ledger

| ID | Export checkpoint | Train tiers | Model | Preset | Launcher | Warm start | Overall |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `B0` | external provided baseline | all benchmark tiers | official starter | external | `babyai-ml8103-leaderboard-2026/baseline` | none | `0.1475` |
| `C1` | `goto_easy_minimalistic_best.pt` | easy | `minimalistic` | recovered `minimalistic_finetune` | not retained in current `scripts/` | none | `0.3820` |
| `C2` | `advance_bc_easy_v2_best.pt` | easy | `advance` | `bc_easy` | [scripts/feedforward/01_train_bc_easy_finetune_advance.sh](scripts/feedforward/01_train_bc_easy_finetune_advance.sh) | `advance_bc_easy_best.pt` | `0.5070` |
| `C3` | `advance_bc_easy_moderate_v1_best.pt` | easy + moderate | `advance` | `bc_moderate` | [scripts/feedforward/02_train_bc_easy_moderate_advance.sh](scripts/feedforward/02_train_bc_easy_moderate_advance.sh) | `advance_bc_easy_v2_best.pt` | `0.7135` |
| `C4` | `advance_bc_all_tiers_v1_best.pt` | easy + moderate + hard | `advance` | `bc_hard` | [scripts/feedforward/03_train_bc_all_tiers_advance.sh](scripts/feedforward/03_train_bc_all_tiers_advance.sh) | `advance_bc_easy_moderate_v1.pt` | `0.7080` |
| `C5` | `advance_bc_easy_recurrent_aux_v1_best.pt` | easy | `advance_largest` | `bc_easy` + `aux_v1` | [scripts/recurrent/01_train_bc_easy_advance.sh](scripts/recurrent/01_train_bc_easy_advance.sh) | none in current script | `0.5235` |
| `C6` | `advance_bc_easy_moderate_recurrent_aux_v1_best.pt` | easy + moderate | `advance_largest` | `bc_moderate` + `aux_v1` | [scripts/recurrent/02_train_bc_easy_moderate_advance.sh](scripts/recurrent/02_train_bc_easy_moderate_advance.sh) | `advance_bc_easy_recurrent_aux_v1_best_trimmed.pt` | `0.8400` |
| `C7` | `advance_bc_all_recurrent_aux_v1_best.pt` | easy + moderate + hard | `advance_largest` | `bc_hard` + `aux_v1` | [scripts/recurrent/03_train_bc_all_tiers_advance.sh](scripts/recurrent/03_train_bc_all_tiers_advance.sh) | `advance_bc_easy_moderate_recurrent_aux_v1_best.pt` | `0.8470` |
| `C8` | `rl_auc_rec_best.pt` | easy + moderate RL fine-tuning | `advance_largest` | `bc_warmstart_push_recurrent` | [scripts/recurrent/04_train_rl.sh](scripts/recurrent/04_train_rl.sh) | `advance_bc_easy_moderate_recurrent_aux_v1_best.pt` | `pending` |

## Stage Notes

### `B0`: external provided baseline

The true external zero point was evaluated in:

- `babyai-ml8103-leaderboard-2026/evaluation/results/unknown_public_20260422_180119.json`

Its official scores were:

- easy: `0.26`
- moderate: `0.05`
- hard: `0.00`
- overall: `0.1475`

This baseline is important historically, but it is not a strong internal research baseline. It solves `GoToObj` while remaining near zero on most of the rest of the benchmark.

### `C1`: grounded minimalistic RL baseline

`C1` is the first measured internal baseline, not the official starter itself.

- export artifact: `sneddy_baby_ai/artifacts/exports/goto_easy_minimalistic_best.pt`
- public evaluation: `evaluation/results/goto_easy_minimalistic_best.json`
- policy metadata also exists as `goto_easy_minimalistic_policy.json`
- model shape matches [configs/models/minimalistic.yaml](configs/models/minimalistic.yaml)

The exact launcher script for `minimalistic_finetune` is not retained in the current `scripts/` directory, but the preserved artifact and recovered metadata show that this run already used symbolic observations, mission grounding, easy-tier multitask training, and adaptive sampling.

### Pre-`C2` easy BC bootstrap

There is an earlier easy-only BC artifact that is not part of the main public comparison table but is required to understand the warm-start chain:

- export artifact: `sneddy_baby_ai/artifacts/exports/advance_bc_easy_best.pt`
- public evaluation: `evaluation/results/advance_bc_easy_best.json`

The dedicated launcher for this pretrain is not retained in `scripts/`, but its existence is confirmed by [scripts/feedforward/01_train_bc_easy_finetune_advance.sh](scripts/feedforward/01_train_bc_easy_finetune_advance.sh), which warm-starts directly from `advance_bc_easy_best.pt`.

### `C2` to `C4`: feedforward BC ladder

These are the main feedforward multitask BC checkpoints.

| ID | Focus | Launcher | Notes |
| --- | --- | --- | --- |
| `C2` | easy-only BC continuation | [scripts/feedforward/01_train_bc_easy_finetune_advance.sh](scripts/feedforward/01_train_bc_easy_finetune_advance.sh) | easy demos only, `advance` model, warm-start from `advance_bc_easy_best.pt` |
| `C3` | extend to moderate tier | [scripts/feedforward/02_train_bc_easy_moderate_advance.sh](scripts/feedforward/02_train_bc_easy_moderate_advance.sh) | adds `GoToRedBall`, `GoToObjMaze`, `GoTo`, `Pickup`, `Open` |
| `C4` | extend to all tiers | [scripts/feedforward/03_train_bc_all_tiers_advance.sh](scripts/feedforward/03_train_bc_all_tiers_advance.sh) | adds `Unlock`, `UnblockPickup`, `PutNextS7N4`, `Synth`, `SynthLoc` |

Important detail: `C4` warm-starts from `advance_bc_easy_moderate_v1.pt`, not the `*_best.pt` export. That is reflected in the launcher and is worth preserving if the run is reproduced exactly.

### `C5` to `C7`: recurrent + auxiliary ladder

These checkpoints form the main architectural contribution in the report.

Shared properties across `C5-C7`:

- `--recurrent`
- `--aux-preset aux_v1`
- `advance_largest` model preset
- BC presets with `action_reweight_on_holding_target=2.0`

The warm-start chain is:

- `C5`: no warm start in the current script
- `C6`: warm-start from `advance_bc_easy_recurrent_aux_v1_best_trimmed.pt`
- `C7`: warm-start from `advance_bc_easy_moderate_recurrent_aux_v1_best.pt`

This chain is the cleanest reproducibility explanation for the final result:

1. train recurrent + auxiliary BC on easy tasks
2. extend that policy to easy + moderate
3. extend again to all tiers

### Diagnostic runs outside the main series

The repository also contains supporting probes that explain how the final recipe was chosen.

| Run | Launcher | Main artifact | Result | Purpose |
| --- | --- | --- | --- | --- |
| isolated `PutNextLocal` large recurrent probe | [scripts/recurrent/00_train_putnextlocal.sh](scripts/recurrent/00_train_putnextlocal.sh) | `putnextlocal_large_best.pt` | `evaluation/results/putnextlocal_large_best.json` | test whether a larger recurrent model helps on the diagnostic task |
| isolated recurrent + aux `PutNextLocal` BC | [scripts/recurrent/05_train_putnextlocal_recurrent_bc.sh](scripts/recurrent/05_train_putnextlocal_recurrent_bc.sh) | `putnextlocal_recurrent_bc_best.pt` | `evaluation/results/putnextlocal_recurrent_bc_best.json` | test recurrent BC with `aux_v1` on a single diagnostic task |
| public recurrent + aux submission probe | export-time run | `submissions/largest_aux_rec_v2.zip` | `evaluation/results/largest_aux_public_20260422_074755.json` | public evaluation showing `PutNextLocal = 0.35` before the final `C7` checkpoint |

These runs are useful because they separate three hypotheses:

- capacity alone
- recurrence alone
- recurrence plus auxiliary task-state shaping

### RL follow-up branches

Two RL-after-BC branches are present in the repository, but neither is part of the main report claim.

| Branch | Launcher | Warm start | Artifact status |
| --- | --- | --- | --- |
| feedforward RL after BC | [scripts/feedforward/04_train_rl_easy_moderate_from_bc_advance.sh](scripts/feedforward/04_train_rl_easy_moderate_from_bc_advance.sh) | `advance_bc_all_tiers_v1_best.pt` | export present as `advance_rl_easy_moderate_v1_best.pt` |
| recurrent RL after BC | [scripts/recurrent/04_train_rl.sh](scripts/recurrent/04_train_rl.sh) | `advance_bc_easy_moderate_recurrent_aux_v1_best.pt` | export present as `rl_auc_rec_best.pt`, public result still pending |

The report therefore treats RL-after as follow-up work, not as the mechanism behind the strongest published score.

## Practical Reproduction Order

If the goal is to reproduce the main story rather than every side branch, the shortest path is:

1. generate demos with [scripts/generate_easy.sh](scripts/generate_easy.sh), [scripts/generate_moderate.sh](scripts/generate_moderate.sh), and [scripts/generate_hard.sh](scripts/generate_hard.sh)
2. run the feedforward BC ladder: `C2 -> C3 -> C4`
3. run the recurrent + auxiliary ladder: `C5 -> C6 -> C7`
4. export and evaluate the resulting `*_best.pt` checkpoints into `submissions/` and `evaluation/results/`

If the goal is to reproduce the scientific narrative more faithfully, keep `B0` as the true external zero point and `C1` as the first meaningful internal baseline.

## Caveats

- The exact launcher for `C1` is not retained in the current `scripts/` directory; reproduction of that stage relies on the preserved artifact and recovered metadata.
- The easy-only pretrain `advance_bc_easy_best.pt` is preserved as an artifact and as a warm-start dependency, but its original dedicated launcher is also not retained.
- The result tables in the report use public evaluation JSONs as the canonical score source.
- `C8` is real as an exported checkpoint, but it should remain outside the main claims until a final public evaluation file is added.
