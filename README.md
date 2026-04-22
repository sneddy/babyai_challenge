# BabyAI Training History

## Summary Table

| ID | Checkpoint | Category | Public overall score | Result file |
| --- | --- | ---: | --- |
| `C1` | `goto_easy_minimalistic_best` | Minimalistic RL baseline | `0.3820` | `evaluation/results/goto_easy_minimalistic_best.json` |
| `C2` | `advance_bc_easy_best` | Feedforward BC, easy only | `0.5070` | `evaluation/results/advance_bc_easy_best.json` |
| `C3` | `advance_bc_easy_v2_best` | Feedforward BC, easy finetune | `0.5070` | `evaluation/results/advance_bc_easy_v2_best.json` |
| `C4` | `advance_bc_easy_moderate_v1_best` | Feedforward BC, easy + moderate | `0.7135` | `evaluation/results/advance_bc_easy_moderate_v1_best.json` |
| `C5` | `advance_bc_easy_moderate_v1` | Feedforward BC, easy + moderate latest | `0.7220` | `evaluation/results/advance_bc_easy_moderate_v1.json` |
| `C6` | `advance_bc_all_tiers_v1_best` | Feedforward BC, all tiers | `0.7080` | `evaluation/results/advance_bc_all_tiers_v1_best.json` |
| `C7` | `advance_bc_all_recurrent_aux_v1_best` | Recurrent multitask BC + auxiliary heads | `0.8470` | `evaluation/results/advance_bc_all_recurrent_aux_v1_best.json` |
| `C8` | `largest_aux_rec_v2` | Larger recurrent multitask + auxiliary run | see per-env result | `evaluation/results/largest_aux_public_20260422_074755.json` |

| ID | Easy | Moderate | Hard | Overall |
| --- | ---: | ---: | ---: | ---: |
| `C1` | `0.7800` | `0.3200` | `0.0000` | `0.3820` |
| `C2` | `0.7800` | `0.3000` | `0.0800` | `0.5070` |
| `C3` | `0.7800` | `0.3000` | `0.0800` | `0.5070` |
| `C4` | `0.9200` | `0.6100` | `0.2800` | `0.7135` |
| `C5` | `0.9200` | `0.6500` | `0.2500` | `0.7220` |
| `C6` | `0.9000` | `0.6100` | `0.3900` | `0.7080` |
| `C7` | `0.9400` | `0.8400` | `0.7100` | `0.8470` |

| ID | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 | T12 | T13 | T14 | T15 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| `C1` | `0.90` | `0.50` | `0.80` | `0.70` | `0.00` | `0.70` | `0.10` | `0.05` | `0.25` | `0.00` | `0.00` | `0.15` | `0.00` | `0.10` | `0.25` |
| `C2` | `1.00` | `0.95` | `0.95` | `0.80` | `0.20` | `1.00` | `0.10` | `0.10` | `0.30` | `0.00` | `0.00` | `0.15` | `0.00` | `0.10` | `0.15` |
| `C3` | `1.00` | `0.95` | `0.95` | `0.80` | `0.20` | `1.00` | `0.10` | `0.10` | `0.30` | `0.00` | `0.00` | `0.15` | `0.00` | `0.10` | `0.15` |
| `C4` | `1.00` | `1.00` | `0.90` | `0.75` | `0.10` | `1.00` | `0.65` | `0.85` | `0.65` | `0.85` | `0.15` | `0.60` | `0.10` | `0.45` | `0.65` |
| `C5` | `1.00` | `1.00` | `0.95` | `0.75` | `0.15` | `1.00` | `0.60` | `0.75` | `0.80` | `0.85` | `0.15` | `0.55` | `0.05` | `0.45` | `0.70` |
| `C6` | `1.00` | `1.00` | `0.90` | `0.75` | `0.15` | `1.00` | `0.55` | `0.75` | `0.80` | `0.75` | `0.20` | `0.50` | `0.10` | `0.45` | `0.70` |
| `C7` | `1.00` | `0.95` | `1.00` | `0.85` | `0.50` | `0.95` | `0.95` | `0.95` | `0.90` | `0.90` | `0.15` | `0.95` | `0.55` | `0.75` | `0.65` |

Checkpoint IDs:
- `C1`: `goto_easy_minimalistic_best`
- `C2`: `advance_bc_easy_best`
- `C3`: `advance_bc_easy_v2_best`
- `C4`: `advance_bc_easy_moderate_v1_best`
- `C5`: `advance_bc_easy_moderate_v1`
- `C6`: `advance_bc_all_tiers_v1_best`
- `C7`: `advance_bc_all_recurrent_aux_v1_best`
- `C8`: `largest_aux_rec_v2`

Task IDs:
- `T1`: `GoToObj`
- `T2`: `GoToLocal`
- `T3`: `GoToRedBallGrey`
- `T4`: `PickupLoc`
- `T5`: `PutNextLocal`
- `T6`: `GoToRedBall`
- `T7`: `GoToObjMaze`
- `T8`: `GoTo`
- `T9`: `Pickup`
- `T10`: `Open`
- `T11`: `Unlock`
- `T12`: `UnblockPickup`
- `T13`: `PutNextS7N4`
- `T14`: `Synth`
- `T15`: `SynthLoc`

## 0. Sampling strategy

From the start, the project did not use a staged curriculum-learning schedule.
Instead, the main training regimes were kept as fixed environment sets:
- `easy`,
- `easy + moderate`,
- `easy + moderate + hard`.

For RL, task selection inside each fixed set used performance-based sampling mixed with a
uniform component, rather than a hand-written curriculum progression.

The RL sampler used:

```text
s_i = val_success_rate(i)
N = number of active tasks
u = uniform_alpha

adaptive_i = 1 - s_i
mixed_i = (1 - u) * adaptive_i + u * (1 / N)
w_i = mixed_i / sum_j mixed_j
```

where `w_i` is the normalized sampling probability for task `i`.
In the implementation this was expressed in percent form before the final normalization,
but mathematically it is the same formula.

For BC, sampling later became per-task demo resampling, so weaker tasks could be
upweighted directly at the dataset level instead of relying on a curriculum schedule.

The BC sampler used a per-task keep-rate instead of normalized task probabilities:

```text
s_i = val_success_rate(i)
p_min = min_sampling_proba
B_i = min(total_episodes_i, sampling_episode_budget_per_task)

keep_i = clip((1 - s_i) * (1 - p_min) + p_min, p_min, 1)
count_i = ceil(B_i * keep_i)
```

where `count_i` is the number of demo episodes retained for task `i` in the next epoch.
So RL redistributed rollout probability mass across tasks, while BC directly changed the
number of training episodes kept per task.

## 1. Minimalistic RL baseline

Initial approach: a minimalistic feedforward PPO policy.

Even this first `minimalistic` version already included several meaningful upgrades over
`babyai-ml8103-leaderboard-2026/baseline`:
- it used the BabyAI mission text instead of ignoring language completely,
- it trained on symbolic `image` triplets (`object/color/state`) with learned tile embeddings instead of a plain RGB CNN pipeline,
- it conditioned vision on the mission through a BabyAI-style FiLM encoder instead of a standalone image extractor,
- it was set up for multi-env easy-tier training with adaptive environment sampling instead of only a fixed single-env baseline run.

This was not just a tuned version of the shipped baseline. The leaderboard baseline
treated BabyAI as a single-env, image-only PPO problem, while `minimalistic`
reframed it as a language-grounded symbolic multi-task PPO run. The gain came from
better state representation and training setup, not only from hyperparameter tuning:
mission conditioning, tile embeddings, FiLM fusion, multi-env training, adaptive
sampling, and a much longer training budget.

| Axis | `babyai-ml8103-leaderboard-2026/baseline` | `minimalistic` |
| --- | --- | --- |
| Observation | image only via `ImgObsWrapper`; mission ignored | symbolic `image` plus tokenized mission |
| Visual representation | 3-layer CNN over the 7x7x3 grid | learned embeddings for `object/color/state` channels |
| Language | not used in the shipped checkpoint | BiGRU mission encoder |
| Vision-language fusion | none | FiLM-conditioned residual visual encoder |
| Policy | SB3 `CnnPolicy` | feedforward actor-critic over a BabyAI-style encoder |
| Training scope | one env: `BabyAI-GoToObj-v0` | five easy envs: `GoToObj`, `GoToLocal`, `GoToRedBallGrey`, `PickupLoc`, `PutNextLocal` |
| Sampling | fixed single-env training | adaptive per-env sampling with a uniform mix |
| PPO setup | `lr=2.5e-4`, `batch=256`, `gae_lambda=0.95`, `ent_coef=0.01` | `lr=5e-5`, `batch=512`, `gae_lambda=0.99`, `ent_coef=0.005`, `sampling_uniform_alpha=0.1` |
| Training budget | `1.0M` timesteps | `5.14M` actual timesteps before interrupt |
| Checkpoint selection | periodic saves | fixed-seed cross-env evaluation with `best` export |

Training path:
- first trained RL on the first env,
- then expanded to all easy envs with sampling,
- resulting artifact: `sneddy_baby_ai/artifacts/exports/goto_easy_minimalistic_best.pt`

Known outcome:
- this solved the simpler easy tasks reasonably well,
- but did not solve `PutNextLocal`,
- the key limitation was not data quality, but insufficient temporal memory in the policy.

Checkpoint:
- `sneddy_baby_ai/artifacts/exports/goto_easy_minimalistic_best.pt`
Results:
- `evaluation/results/goto_easy_minimalistic_best.json`

Metadata recovered from export:
- run name: `goto_easy_minimalistic`
- algorithm: `ppo`
- config: `minimalistic_finetune`
- seed: `42`
- train/eval envs:
  - `BabyAI-GoToObj-v0`
  - `BabyAI-GoToLocal-v0`
  - `BabyAI-GoToRedBallGrey-v0`
  - `BabyAI-PickupLoc-v0`
  - `BabyAI-PutNextLocal-v0`

Model config:
```json
{
  "action_dim": 7,
  "image_embedding_dim": 96,
  "mission_embedding_dim": 32,
  "mission_hidden_dim": 64,
  "attention": false,
  "features_dim": 128,
  "recurrent_hidden_dim": 128,
  "tile_vocab_sizes": {
    "object": 16,
    "color": 16,
    "state": 8
  }
}
```

Optim config:
```json
{
  "algorithm": "ppo",
  "config_name": "minimalistic_finetune",
  "learning_rate": 5e-05,
  "n_steps": 256,
  "batch_size": 512,
  "n_epochs": 4,
  "gamma": 0.99,
  "gae_lambda": 0.99,
  "clip_range": 0.2,
  "ent_coef": 0.005,
  "vf_coef": 0.5,
  "max_grad_norm": 0.5,
  "sampling_uniform_alpha": 0.1
}
```

## 2. Switch to supervised pretraining with a larger model

After seeing that the minimalistic RL path did not fix `PutNextLocal`, the next approach was supervised pretraining on expert demonstrations.

Reason for the switch:
- BC converged much more reliably than direct RL,
- once that worked, the model capacity was increased from `minimalistic` to `advance`.

### 2.1 Easy-only BC pretrain

Goal:
- train a larger feedforward policy on easy-tier demos only.

Main checkpoint:
- `sneddy_baby_ai/artifacts/exports/advance_bc_easy_best.pt`
Results:
- `evaluation/results/advance_bc_easy_best.json`

Observed result:
- strong improvement on easy tasks,
- `PutNextLocal` still 0%.

Model config:
```json
{
  "action_dim": 7,
  "image_embedding_dim": 160,
  "mission_embedding_dim": 80,
  "mission_hidden_dim": 160,
  "attention": true,
  "features_dim": 320,
  "recurrent_hidden_dim": 320,
  "tile_vocab_sizes": {
    "object": 16,
    "color": 16,
    "state": 8
  }
}
```

Optim config:
```json
{
  "algorithm": "behavior_cloning",
  "config_name": "bc_easy",
  "epochs": 16,
  "batch_size": 1024,
  "learning_rate": 1e-04,
  "weight_decay": 1e-06,
  "clip_grad_norm": 0.5,
  "min_sampling_proba": 0.1,
  "n_eval_episodes": 12
}
```

### 2.2 Easy BC finetune

This was the follow-up easy finetune pass that continued from the previous easy BC checkpoint.

Script:
- `scripts/feedforward/01_train_bc_easy_finetune_advance.sh`

Warm start:
- `sneddy_baby_ai/artifacts/exports/advance_bc_easy_best.pt`

Produced checkpoint:
- `sneddy_baby_ai/artifacts/exports/advance_bc_easy_v2_best.pt`
Results:
- `evaluation/results/advance_bc_easy_v2_best.json`

Train demos:
- easy tier only:
  - `gotoobj`
  - `gotolocal`
  - `gotoredballgrey`
  - `pickuploc`
  - `putnextlocal`

Model config:
```json
{
  "action_dim": 7,
  "image_embedding_dim": 160,
  "mission_embedding_dim": 80,
  "mission_hidden_dim": 160,
  "attention": true,
  "features_dim": 320,
  "recurrent_hidden_dim": 320,
  "tile_vocab_sizes": {
    "object": 16,
    "color": 16,
    "state": 8
  }
}
```

Optim config:
```json
{
  "algorithm": "behavior_cloning",
  "config_name": "bc_easy_finetune",
  "epochs": 5,
  "batch_size": 1024,
  "learning_rate": 1e-05,
  "weight_decay": 1e-06,
  "clip_grad_norm": 0.5,
  "min_sampling_proba": 0.1,
  "n_eval_episodes": 10,
  "warm_start": "sneddy_baby_ai/artifacts/exports/advance_bc_easy_best.pt"
}
```

## 3. Larger supervised curriculum over more envs

### 3.1 Easy + moderate BC

Goal:
- extend the stronger BC initialization to the moderate tier.

Checkpoint:
- `sneddy_baby_ai/artifacts/exports/advance_bc_easy_moderate_v1_best.pt`
Results:
- `evaluation/results/advance_bc_easy_moderate_v1_best.json`

Script:
- `scripts/feedforward/02_train_bc_easy_moderate_advance.sh`

Warm start:
- `sneddy_baby_ai/artifacts/exports/advance_bc_easy_v2_best.pt`

Train demos:
- all easy demos,
- plus moderate demos:
  - `gotoredball`
  - `gotoobjmaze`
  - `goto`
  - `pickup`
  - `open`

Model config:
```json
{
  "action_dim": 7,
  "image_embedding_dim": 160,
  "mission_embedding_dim": 80,
  "mission_hidden_dim": 160,
  "attention": true,
  "features_dim": 320,
  "recurrent_hidden_dim": 320,
  "tile_vocab_sizes": {
    "object": 16,
    "color": 16,
    "state": 8
  }
}
```

Optim config:
```json
{
  "algorithm": "behavior_cloning",
  "config_name": "bc_moderate",
  "epochs": 5,
  "batch_size": 512,
  "learning_rate": 2e-05,
  "weight_decay": 0.0,
  "clip_grad_norm": 0.5,
  "min_sampling_proba": 0.1,
  "n_eval_episodes": 20,
  "warm_start": "sneddy_baby_ai/artifacts/exports/advance_bc_easy_v2_best.pt"
}
```

### 3.2 All tiers BC

Goal:
- build a single larger BC checkpoint over easy + moderate + hard tiers.

Checkpoint:
- `sneddy_baby_ai/artifacts/exports/advance_bc_all_tiers_v1_best.pt`

Script:
- `scripts/feedforward/03_train_bc_all_tiers_advance.sh`

Warm start:
- `sneddy_baby_ai/artifacts/exports/advance_bc_easy_moderate_v1.pt`

Train demos:
- all easy demos,
- all moderate demos,
- hard demos:
  - `unlock`
  - `unblockpickup`
  - `putnexts7n4`
  - `synth`
  - `synthloc`

Model config:
```json
{
  "action_dim": 7,
  "image_embedding_dim": 160,
  "mission_embedding_dim": 80,
  "mission_hidden_dim": 160,
  "attention": true,
  "features_dim": 320,
  "recurrent_hidden_dim": 320,
  "tile_vocab_sizes": {
    "object": 16,
    "color": 16,
    "state": 8
  }
}
```

Optim config:
```json
{
  "algorithm": "behavior_cloning",
  "config_name": "bc_hard",
  "epochs": 5,
  "batch_size": 512,
  "learning_rate": 1e-05,
  "weight_decay": 0.0,
  "clip_grad_norm": 0.5,
  "min_sampling_proba": 0.1,
  "sampling_episode_budget_per_task": 1000,
  "n_eval_episodes": 20,
  "warm_start": "sneddy_baby_ai/artifacts/exports/advance_bc_easy_moderate_v1.pt"
}
```

## 4. Why `PutNextLocal` still failed

Key conclusion from the code-level debug:
- demos were valid,
- there was no action-label shift,
- mission tokenization was not broken,
- success criterion was correct,
- the actual issue was architectural.

Root cause:
- `PutNextLocal` was not failing because of broken data,
- it was failing because the policy had no memory,
- all checkpoints above were pure feedforward per-step policies.

Why this matters specifically for `PutNextLocal`:
- `PickupLoc` can often be solved reactively from the current observation,
- `PutNextLocal` has a two-stage dependency:
  - find and pick object `X`,
  - then place it next to object `Y`,
- after pickup, the agent must still preserve task-relevant context across later timesteps.

In short:
- feedforward policy: action is computed only from the current encoded observation,
- recurrent policy: action is computed from the current observation plus a learned hidden state carried across timesteps.

## 5. Next step: add memory / recurrence

Current direction:
- move from feedforward BC to recurrent BC for `PutNextLocal`.

What changed in the model:
- previously:
  - `encoder(obs_t) -> actor/critic`,
  - no persistent hidden state,
  - each step treated independently.
- now:
  - `encoder(obs_t) -> LSTMCell(hidden_{t-1}, cell_{t-1}) -> actor/critic`,
  - hidden state is passed from one step to the next,
  - training is done on full episodes rather than shuffled independent transitions.

This is a model-level difference, not a demo-format change.

Important note:
- demos do not need to be regenerated for this fix,
- the existing expert trajectories were already valid,
- the missing component was temporal state inside the policy.

Current recurrent entrypoint:
- `scripts/recurrent/05_train_putnextlocal_recurrent_bc.sh`

Current recurrent warm start:
- `sneddy_baby_ai/artifacts/exports/advance_bc_easy_v2_best.pt`

Seeded recurrent checkpoint:
- `sneddy_baby_ai/artifacts/exports/putnextlocal_recurrent_bc_best.pt`

Model config:
```json
{
  "action_dim": 7,
  "image_embedding_dim": 160,
  "mission_embedding_dim": 80,
  "mission_hidden_dim": 160,
  "attention": true,
  "features_dim": 320,
  "recurrent_hidden_dim": 320,
  "tile_vocab_sizes": {
    "object": 16,
    "color": 16,
    "state": 8
  }
}
```

Optim config:
```json
{
  "algorithm": "behavior_cloning",
  "config_name": "bc_easy",
  "epochs": 16,
  "batch_size": 1024,
  "effective_episode_batch_size": "derived from mean episode length",
  "learning_rate": 1e-04,
  "weight_decay": 1e-06,
  "clip_grad_norm": 0.5,
  "min_sampling_proba": 0.1,
  "n_eval_episodes": 12,
  "recurrent": true,
  "warm_start": "sneddy_baby_ai/artifacts/exports/advance_bc_easy_v2_best.pt"
}
```

## 6. Practical report summary

Short narrative for a report:
- minimalistic RL worked as a useful baseline but did not solve `PutNextLocal`,
- supervised pretraining with a larger `advance` model improved most tasks significantly,
- scaling BC from easy to moderate to all tiers still left `PutNextLocal` unresolved,
- code-level debugging showed the failure was not caused by demos or label alignment,
- the real bottleneck was the lack of temporal memory in the policy,
- the next correct step is recurrent BC, where the same visual/language encoder is augmented with an LSTM memory state and trained on whole trajectories.

## 7. Recurrent BC follow-up and auxiliary head experiments

### 7.1 Recurrent BC by itself still did not reliably solve `PutNextLocal`

After moving from feedforward BC to recurrent BC, the expectation was that the added LSTM memory would close the gap on the two-stage manipulation structure of `PutNextLocal`.

What was observed:
- recurrent BC improved optimization stability,
- but `PutNextLocal` still did not converge reliably,
- rollout success remained inconsistent even when per-step imitation accuracy looked reasonable.

Conclusion:
- adding memory alone was not sufficient,
- the policy still appeared to struggle with the manipulation-specific internal state, especially around carrying the correct object through the pickup-to-drop transition.

### 7.2 First auxiliary head set

To make the recurrent encoder represent more structured task state, auxiliary supervision was added on top of BC.

First head set:
- `in_front_of_what`
- `obj_in_instr_visible`
- `holding_target_object`
- `adjacent_to_target_object`
- `valid_drop_position`

Intent:
- keep the action BC loss as the main objective,
- add lightweight supervised heads for state predicates that should help the recurrent encoder organize the manipulation phases better.

Observed pattern:
- the heads trained cleanly on demo states,
- but rollout success on `PutNextLocal` still lagged,
- the clearest weak head during policy rollouts was `holding_target_object`.

### 7.3 Expanded auxiliary head set

Because the first head set still seemed too focused on local geometry and late placement, the auxiliary set was expanded:
- `valid_pickup_action`
- `fixed_target_visible`

Reason for the expansion:
- explicitly supervise whether the policy is in the pickup phase or drop phase,
- distinguish visibility of the fixed target from the generic instruction-object visibility signal,

Practical outcome:
- this gave better instrumentation and made it easier to see that the policy often recognized local pickup/drop conditions,
- but still failed to consistently enter and maintain the correct carry phase on its own rollouts.
- across these auxiliary runs, the most persistent weak signal was `holding_target_object`,
- this became the clearest diagnosis of the remaining `PutNextLocal` failure mode:
  the policy often understood local pickup/drop geometry, but still did not reliably pick up and keep carrying the correct object.

### 7.4 Larger recurrent model

Since the recurrent `advance` model plus auxiliary heads still did not make `PutNextLocal` converge reliably, model capacity was increased further for isolated `PutNextLocal` runs.

Larger recurrent preset introduced:
- `advance_largest`

Compared with `advance`, this increased:
- image embedding width,
- mission embedding width,
- mission hidden size,
- shared feature size,
- recurrent hidden size.

Model config:
```json
{
  "action_dim": 7,
  "image_embedding_dim": 256,
  "mission_embedding_dim": 128,
  "mission_hidden_dim": 256,
  "attention": true,
  "features_dim": 512,
  "recurrent_hidden_dim": 512,
  "tile_vocab_sizes": {
    "object": 16,
    "color": 16,
    "state": 8
  }
}
```

Current interpretation:
- recurrent BC was directionally correct but not sufficient on its own,
- auxiliary heads improved observability and representation shaping,
- the most persistent bottleneck remained the policy's ability to actually hold and carry the correct target object on its own rollouts,
- larger recurrent capacity was introduced as the next isolated experiment rather than as proof that architecture size alone solves the task.

### 7.5 Larger recurrent + auxiliary multitask run finally made `PutNextLocal` move

After combining all three ingredients
- recurrent BC,
- auxiliary losses,
- and the larger `advance_largest` recurrent model,

the previously stuck `PutNextLocal` task finally started to show non-trivial public-eval success instead of staying effectively collapsed.

Reference result:
- `evaluation/results/largest_aux_public_20260422_074755.json`
- submission: `submissions/largest_aux_rec_v2.zip`

Observed public outcome:
- `BabyAI-PutNextLocal-v0` reached `35%` success (`7/20`) on the public evaluation,
- which was the first clear sign that this task was beginning to converge under the recurrent + auxiliary + larger-capacity setup.

Interpretation:
- the effect was more noticeable in the broader multitask setting than in isolated single-task experiments,
- auxiliary heads appeared to learn more meaningful internal structure when trained across a larger set of tasks,
- this supports the view that the auxiliary objectives were not just acting as local regularizers, but were helping the recurrent encoder organize more reusable task state across tasks,
- in practice, this was the first setup where `PutNextLocal` stopped looking completely stalled and began to benefit from the added memory, representation shaping, and model capacity together.
