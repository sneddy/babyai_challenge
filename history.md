# BabyAI Training History

## 1. Minimalistic RL baseline

Initial approach: a minimalistic feedforward PPO policy.

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

Observed result:
- strong improvement on easy tasks,
- `PutNextLocal` still remained weak.

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

Because the first head set still seemed too focused on local geometry and late placement, the auxiliary set was expanded to cover earlier pickup and phase-tracking signals.

Additional heads added later:
- `valid_pickup_action`
- `fixed_target_visible`
- `need_pickup_phase`
- `need_drop_phase`

Reason for the expansion:
- explicitly supervise whether the policy is in the pickup phase or drop phase,
- distinguish visibility of the fixed target from the generic instruction-object visibility signal,
- add a direct pickup-side analogue of `valid_drop_position`.

Practical outcome:
- this gave better instrumentation and made it easier to see that the policy often recognized local pickup/drop conditions,
- but still failed to consistently enter and maintain the correct carry phase on its own rollouts.
- across these auxiliary runs, the most persistent weak signal was `holding_target_object`,
- this became the clearest diagnosis of the remaining `PutNextLocal` failure mode:
  the policy often understood local pickup/drop geometry, but still did not reliably pick up and keep carrying the correct object.

### 7.4 Larger recurrent model

Since the recurrent `advance` model plus auxiliary heads still did not make `PutNextLocal` converge reliably, model capacity was increased further for isolated `PutNextLocal` runs.

Larger recurrent preset introduced:
- `advance_large`

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
  "image_embedding_dim": 192,
  "mission_embedding_dim": 96,
  "mission_hidden_dim": 192,
  "attention": true,
  "features_dim": 448,
  "recurrent_hidden_dim": 448,
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
