# BabyAI Under a Limited Demonstration Budget

## Abstract

This report studies how far a BabyAI agent can be pushed when the supervision budget is deliberately kept small and fixed. The true external zero point is the provided leaderboard baseline, whose public evaluation reached **0.1475** overall weighted score. Starting from that weak reference system, the project rebuilt the agent around a different principle: diagnose the concrete failure mode of the current policy, then add only the mechanism required to fix that failure. Under a budget of **5000 demonstrations per easy task**, **5000 demonstrations per moderate task**, and **1000 demonstrations per hard task**, the strongest checkpoint reached **0.8470** overall, with **0.860 / 0.930 / 0.610** on easy, moderate, and hard tasks. The main scientific claim is that this gain came less from raw scaling than from identifying and repairing a specific hidden-state bottleneck exposed by `PutNextLocal`.

## Main Contributions

- The provided leaderboard baseline at **0.1475** overall was turned into a language-grounded symbolic multitask RL baseline at **0.3820** before behavior cloning or recurrence entered the picture.
- Strong multitask performance was obtained under a fixed and limited demonstration budget, so the final result should be read as a **sample-efficient training result**, not a data-scaling result.
- `PutNextLocal` was used as the main diagnostic task to show that the remaining bottleneck was missing temporal task-state, especially the hidden carry phase after pickup.
- Recurrent memory, auxiliary state prediction, and targeted carry-phase loss reweighting were combined into a coherent fix that raised the best score from the strongest feedforward BC checkpoint (**0.7080**) to the final recurrent checkpoint (**0.8470**).

## Benchmark Setting and Supervision Budget

The benchmark was organized into fixed task groups rather than a hand-written curriculum schedule.

| Tier | Tasks | Demonstrations per task |
| --- | --- | ---: |
| easy | `GoToObj`, `GoToLocal`, `GoToRedBallGrey`, `PickupLoc`, `PutNextLocal` | `5000` |
| moderate | `GoToRedBall`, `GoToObjMaze`, `GoTo`, `Pickup`, `Open` | `5000` |
| hard | `Unlock`, `UnblockPickup`, `PutNextS7N4`, `Synth`, `SynthLoc` | `1000` |

Training then progressed through three fixed multitask regimes:

- `easy`
- `easy + moderate`
- `easy + moderate + hard`

Across stages, the supervision budget stayed small and largely fixed; the major changes came from representation, memory, and training structure.

## Chronological Evolution

### 1. Building the Minimalistic RL Baseline

The project did not begin from a strong architecture. The provided leaderboard baseline, evaluated in `babyai-ml8103-leaderboard-2026/evaluation/results/unknown_public_20260422_180119.json`, reached only **0.1475** overall weighted score. Its behavior was highly uneven: it solved `GoToObj` almost perfectly, but remained near zero on most of the benchmark, including `PutNextLocal`, `PickupLoc`, and essentially all hard tasks. In practical terms, it was too weak to serve as the main scientific baseline.

The first real stage of the work was therefore to build a minimalistic grounded RL system that still stayed simple, but changed the problem formulation in the right direction. Instead of treating BabyAI as an image-only control problem, this model used symbolic observations, explicit mission language, grounded vision-language fusion, and multitask training across the easy tier. That step alone raised the score to **0.3820** overall. The minimalistic stage matters because it already showed that BabyAI responds strongly to the right symbolic and language-grounded inductive bias even before behavior cloning, recurrence, or auxiliary supervision are introduced.

### 2. BC PPO: Breaking the Zero-Success Regime

After the minimalistic RL stage, the next question was not whether the policy needed more capacity, but whether pure RL was failing to bootstrap on tasks that were still producing almost no correct behavior. The practical motivation was simple: if a task stays at or near **0%** success, then the policy has almost no successful history to learn from, and convergence can stall even if the representation is already better than the provided baseline. That was especially relevant for `PutNextLocal`.

This motivated the switch to behavior cloning on the same general policy family. The goal was to inject correct trajectories directly into training and move difficult tasks out of the zero-success regime. This worked immediately. The easy-tier BC checkpoint `C2` reached **0.5070** overall, the `easy + moderate` checkpoint `C3` reached **0.7135**, and the all-tier feedforward checkpoint `C4` stabilized at **0.7080**. BC was therefore not just a convenience trick. It was the first genuinely reliable way to obtain a strong multitask policy under the limited demonstration budget.

At the same time, the BC ladder exposed an important limit. Aggregate performance improved strongly, but `PutNextLocal` remained weak: **0.20 -> 0.10 -> 0.15** across `C2-C4`. So better supervision and better optimization alone were not enough to fix the core manipulation bottleneck.

### 3. Recurrent Policy, Auxiliary Heads, and the Largest Model

At this point `PutNextLocal` became the main diagnostic task. The failure pattern was informative: the problem was not broken demonstrations, not misaligned labels, and not tokenization. The task itself has a hidden multi-stage structure. The agent must pick up the correct object, remember that it is already carrying it, and only then execute the placement phase correctly. That is a temporal state-tracking problem.

This diagnosis motivated the recurrent stage. The first recurrent step was meant to fix memory, but recurrence alone was not enough. The model still needed a more structured internal state, especially around whether it was holding the correct object. That led to auxiliary state heads and to explicit emphasis on `holding_target_object` through `action_reweight_on_holding_target_object`.

The final recipe combined three ingredients:

- recurrence for temporal memory
- auxiliary state supervision for interpretable task phase
- the `advance_largest` preset to give the recurrent model enough capacity once the right inductive bias had been identified

This sequence produced the recurrent checkpoint ladder:

- `C5 = 0.5235`
- `C6 = 0.8400`
- `C7 = 0.8470`

The crucial point is that the recurrent stage was not introduced to make the model generically bigger. It was introduced to repair memory and, in particular, to make the hidden variable `holding_target_object` usable enough for `PutNextLocal` and related multistep tasks.

## Method Summary

The final method is a language-grounded symbolic policy. Mission text and symbolic grid observations are encoded jointly, spatial features are conditioned on the instruction, and the policy acts on the resulting grounded representation. The early feedforward models already showed that this representation change matters, but the decisive later improvement came from acknowledging that some BabyAI tasks require hidden task-state across time.

### Adaptive sampling

The project used adaptive sampling in both RL and BC, but in different ways.

For RL, if `s_i` is the validation success rate of task `i`, `N` is the number of active tasks, and `u` is the uniform-mixture coefficient, then the unnormalized task weight is

$$
\tilde{w}_i = (1 - u)(1 - s_i) + \frac{u}{N}
$$

and the final sampling probability is

$$
w_i = \frac{\tilde{w}_i}{\sum_j \tilde{w}_j}.
$$

For BC, adaptive sampling happens at the dataset level. Let `B_i` be the number of available episodes for task `i`, let `B_i^{\max}` be the optional per-task cap from `sampling_episode_budget_per_task`, and define the effective episode pool

$$
\tilde{B}_i = \min(B_i,\; B_i^{\max})
$$

when the cap is used. If `s_i` is the validation success rate and `p_{\min}` is the minimum keep probability, then the keep ratio is

$$
r_i = (1 - s_i)(1 - p_{\min}) + p_{\min} = 1 - s_i(1 - p_{\min}).
$$

In the implementation this ratio is also clamped to `[p_{\min}, 1]` as a defensive measure. The retained episode count is then

$$
\mathrm{count}_i = \max\left(1,\; \min\left(\tilde{B}_i,\; \left\lceil \tilde{B}_i \, r_i \right\rceil\right)\right).
$$

So RL redistributes rollout probability mass across tasks, while BC increases supervision density directly on weaker tasks.

### Model scale by stage

| Stage | Checkpoints | Model scale | Interpretation |
| --- | --- | --- | --- |
| provided baseline | `B0` | external starter | image-only PPO reference system |
| lightweight grounded model | `C1` | `96 / 32 / 64 / 128` | image embedding / mission embedding / mission hidden / shared features |
| `advance` preset | `C2-C4` | `160 / 80 / 160 / 320` | stronger feedforward BC model |
| larger recurrent preset | `C5-C7` | `256 / 128 / 256 / 512` | higher-capacity recurrent + auxiliary model |

Capacity matters, but the history of the results shows that capacity alone does not explain the improvement. The decisive change is better explained by the right inductive bias: memory for temporal tasks, plus supervision that shapes the hidden state into something useful.

### Final auxiliary supervision

The final recurrent model used the following auxiliary state targets:

- `in_front_of_what`
- `obj_in_instr_visible`
- `holding_target_object`
- `adjacent_to_target_object`
- `valid_drop_position`
- `valid_pickup_action`
- `fixed_target_visible`

These heads were paired with `action_reweight_on_holding_target_object`, which increases the action loss on states where the target object is already being carried. This became important because mistakes in the carry phase are especially damaging for `PutNextLocal`.

## Main Results

All overall scores below are the official weighted leaderboard score with tier weights `0.5 / 0.35 / 0.15` for `easy / moderate / hard`.

The main public-score progression was:

- provided baseline: **0.1475**
- minimalistic grounded RL baseline: **0.3820**
- stronger feedforward BC checkpoints: **0.5070 -> 0.7135 -> 0.7080**
- recurrent multitask BC with auxiliary structure: **0.5235 -> 0.8400 -> 0.8470**

The strongest checkpoint, `advance_bc_all_recurrent_aux_v1_best`, reached:

- **0.860** on easy tasks
- **0.930** on moderate tasks
- **0.610** on hard tasks
- **0.8470** overall

### Compact result table

| ID | Checkpoint | Stage | Easy | Moderate | Hard | Overall | `PutNextLocal` | Main takeaway |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `B0` | provided baseline | official provided baseline | `0.26` | `0.05` | `0.00` | `0.1475` | `0.00` | The external starter solves only the simplest navigation case and collapses on most of the benchmark. |
| `C1` | `goto_easy_minimalistic_best` | minimalistic RL baseline | `0.58` | `0.22` | `0.10` | `0.3820` | `0.00` | A grounded PPO agent is viable, but still far too weak for the full leaderboard. |
| `C2` | `advance_bc_easy_v2_best` | BC | `0.78` | `0.30` | `0.08` | `0.5070` | `0.20` | BC immediately stabilizes learning and gives a much stronger starting point. |
| `C3` | `advance_bc_easy_moderate_v1_best` | BC | `0.75` | `0.80` | `0.39` | `0.7135` | `0.10` | Extending BC to more tasks yields the first major jump in leaderboard quality. |
| `C4` | `advance_bc_all_tiers_v1_best` | BC | `0.76` | `0.77` | `0.39` | `0.7080` | `0.15` | More tasks alone do not solve the core manipulation bottleneck. |
| `C5` | `advance_bc_easy_recurrent_aux_v1_best` | BC + recurrence + aux | `0.81` | `0.27` | `0.16` | `0.5235` | `0.50` | Memory and task-state supervision sharply improve the diagnostic task before broader generalization appears. |
| `C6` | `advance_bc_easy_moderate_recurrent_aux_v1_best` | BC + recurrence + aux | `0.83` | `0.97` | `0.57` | `0.8400` | `0.45` | Once trained on a broader multitask set, the gain becomes global. |
| `C7` | `advance_bc_all_recurrent_aux_v1_best` | BC + recurrence + aux | `0.86` | `0.93` | `0.61` | `0.8470` | `0.50` | Best overall checkpoint: strong multitask performance plus a real fix for the temporal bottleneck. |
| `C8` | `rl_auc_rec_best` | RL after BC | `pending` | `pending` | `pending` | `pending` | `pending` | RL fine-tuning after strong recurrent BC initialization was explored, but is not part of the main claims yet. |

`C8` is intentionally excluded from the main claims because its final exported public evaluation is still pending.

### What improved most from `C4` to `C7`

The most revealing change is not only the aggregate score increase from **0.7080** to **0.8470**, but which tasks improved most.

| Task | `C4` | `C7` | Delta |
| --- | ---: | ---: | ---: |
| `PutNextS7N4` | `0.10` | `0.55` | `+0.45` |
| `UnblockPickup` | `0.50` | `0.95` | `+0.45` |
| `GoToObjMaze` | `0.55` | `0.95` | `+0.40` |
| `PutNextLocal` | `0.15` | `0.50` | `+0.35` |
| `Synth` | `0.45` | `0.75` | `+0.30` |

This pattern strongly supports the core claim of the report: recurrent memory and auxiliary state shaping did not merely inflate the average score. They helped most on tasks that require a richer internal state, multistep structure, or more reliable phase tracking.

## Full Per-Task Public Score Table

| Task ID | Task | `B0` | `C1` | `C2` | `C3` | `C4` | `C5` | `C6` | `C7` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `T1` | `GoToObj` | `1.00` | `0.90` | `1.00` | `1.00` | `1.00` | `1.00` | `1.00` | `1.00` |
| `T2` | `GoToLocal` | `0.10` | `0.50` | `0.95` | `1.00` | `1.00` | `0.80` | `0.90` | `0.95` |
| `T3` | `GoToRedBallGrey` | `0.20` | `0.80` | `0.95` | `0.90` | `0.90` | `1.00` | `1.00` | `1.00` |
| `T4` | `PickupLoc` | `0.00` | `0.70` | `0.80` | `0.75` | `0.75` | `0.75` | `0.80` | `0.85` |
| `T5` | `PutNextLocal` | `0.00` | `0.00` | `0.20` | `0.10` | `0.15` | `0.50` | `0.45` | `0.50` |
| `T6` | `GoToRedBall` | `0.10` | `0.70` | `1.00` | `1.00` | `1.00` | `0.85` | `0.95` | `0.95` |
| `T7` | `GoToObjMaze` | `0.10` | `0.10` | `0.10` | `0.65` | `0.55` | `0.10` | `1.00` | `0.95` |
| `T8` | `GoTo` | `0.05` | `0.05` | `0.10` | `0.85` | `0.75` | `0.10` | `1.00` | `0.95` |
| `T9` | `Pickup` | `0.00` | `0.25` | `0.30` | `0.65` | `0.80` | `0.30` | `0.95` | `0.90` |
| `T10` | `Open` | `0.00` | `0.00` | `0.00` | `0.85` | `0.75` | `0.00` | `0.95` | `0.90` |
| `T11` | `Unlock` | `0.00` | `0.00` | `0.00` | `0.15` | `0.20` | `0.00` | `0.30` | `0.15` |
| `T12` | `UnblockPickup` | `0.00` | `0.15` | `0.15` | `0.60` | `0.50` | `0.15` | `0.80` | `0.95` |
| `T13` | `PutNextS7N4` | `0.00` | `0.00` | `0.00` | `0.10` | `0.10` | `0.25` | `0.50` | `0.55` |
| `T14` | `Synth` | `0.00` | `0.10` | `0.10` | `0.45` | `0.45` | `0.15` | `0.65` | `0.75` |
| `T15` | `SynthLoc` | `0.00` | `0.25` | `0.15` | `0.65` | `0.70` | `0.25` | `0.60` | `0.65` |

## Interpretation

The final result did not come from a single lucky trick. It came from a constrained chain of interventions, each motivated by the failure mode of the previous stage:

1. start from the weak provided baseline
2. build a grounded symbolic multitask RL baseline
3. move to BC to escape the zero-success regime on difficult tasks
4. use `PutNextLocal` to diagnose that feedforward policies still miss temporal task-state
5. add recurrence because the task is temporal
6. add auxiliary supervision and carry-phase reweighting because memory alone is too weak without structure

This is why the final result is scientifically stronger than a generic "bigger model, better score" story. The gain from **0.1475** to **0.8470** is tied to a causal sequence of justified changes rather than one isolated trick. The secondary claim is equally important: this was achieved without increasing the demonstration budget aggressively. The final system is therefore best understood as a **sample-efficient multitask method under limited supervision**.
