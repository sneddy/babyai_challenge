"""Training callbacks and evaluation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

from ..auxiliary.labels import build_aux_targets
from ..auxiliary.specs import AuxHeadSpec
from ..common.constants import EVAL_TRAIN_SEED_BASE, EVAL_VAL_SEED_BASE
from ..envs.runtime import ensure_babyai_envs_registered, suppress_noisy_env_output
from ..envs.wrappers import TokenizedMissionWrapper


@dataclass
class EvalSnapshot:
    timesteps: int
    train_success_rate: float
    val_success_rate: float
    train_mean_length: float
    val_mean_length: float
    best_val_success_rate: float
    env_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


def create_progress_bar(total: int, desc: str, unit: str = "ts"):
    return tqdm(total=total, desc=desc, unit=unit)


def _evaluate_on_seeds(model, env_name: str, vocab, seeds: list[int], max_steps: int = 1000) -> tuple[float, float, dict[str, float]]:
    ensure_babyai_envs_registered()
    with suppress_noisy_env_output():
        env = gym.make(env_name)
    env = TokenizedMissionWrapper(env, vocab)
    successes: list[int] = []
    lengths: list[int] = []

    try:
        for seed in seeds:
            with suppress_noisy_env_output():
                obs, _ = env.reset(seed=seed)
            done = False
            reward = 0.0
            steps = 0
            while not done and steps < max_steps:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
            successes.append(int(reward > 0))
            lengths.append(steps)
    finally:
        env.close()

    success_rate = float(np.mean(successes) if successes else 0.0)
    mean_length = float(np.mean(lengths) if lengths else 0.0)
    return success_rate, mean_length, {}


def evaluate_torch_policy_on_seeds(policy, env_name: str, vocab, seeds: list[int], device="cpu", max_steps: int = 1000) -> tuple[float, float, dict[str, float]]:
    ensure_babyai_envs_registered()
    with suppress_noisy_env_output():
        env = gym.make(env_name)
    env = TokenizedMissionWrapper(env, vocab)
    successes: list[int] = []
    lengths: list[int] = []
    device = torch.device(device)

    aux_specs: tuple[AuxHeadSpec, ...] = tuple(getattr(policy, "aux_specs", ()))
    aux_totals: dict[str, float] = {}
    for spec in aux_specs:
        aux_totals[f"{spec.name}_loss_sum"] = 0.0
        aux_totals[f"{spec.name}_acc_sum"] = 0.0
        aux_totals[f"{spec.name}_active"] = 0.0

    try:
        for seed in seeds:
            with suppress_noisy_env_output():
                obs, _ = env.reset(seed=seed)
            done = False
            reward = 0.0
            steps = 0
            state = policy.initial_state(batch_size=1, device=device) if hasattr(policy, "initial_state") else None
            while not done and steps < max_steps:
                tensor_obs = {
                    "image": torch.as_tensor(obs["image"], device=device).unsqueeze(0),
                    "mission_tokens": torch.as_tensor(obs["mission_tokens"], device=device).unsqueeze(0),
                    "mission_mask": torch.as_tensor(obs["mission_mask"], device=device).unsqueeze(0),
                }
                if aux_specs and hasattr(policy, "forward_with_aux"):
                    labels, masks = build_aux_targets(env.unwrapped, aux_specs)
                    if state is None:
                        logits, _value, aux_predictions = policy.forward_with_aux(tensor_obs)
                        action = logits.argmax(dim=-1)
                    else:
                        logits, _value, state, aux_predictions = policy.forward_with_aux(tensor_obs, state=state)
                        action = logits.argmax(dim=-1)
                    _update_aux_eval_totals(aux_totals, aux_predictions, labels, masks, aux_specs)
                else:
                    if state is None:
                        action, _value = policy.act(tensor_obs, deterministic=True)
                    else:
                        action, _value, state = policy.act(tensor_obs, state=state, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated
                steps += 1
            successes.append(int(reward > 0))
            lengths.append(steps)
    finally:
        env.close()

    success_rate = float(np.mean(successes) if successes else 0.0)
    mean_length = float(np.mean(lengths) if lengths else 0.0)
    metrics = _finalize_aux_eval_totals(aux_totals, aux_specs)
    return success_rate, mean_length, metrics


def _update_aux_eval_totals(
    aux_totals: dict[str, float],
    aux_predictions: dict[str, torch.Tensor],
    labels: dict[str, int],
    masks: dict[str, int],
    aux_specs: tuple[AuxHeadSpec, ...],
) -> None:
    for spec in aux_specs:
        if not masks.get(spec.name, 0):
            continue
        prediction = aux_predictions[spec.name]
        target_value = labels[spec.name]
        if spec.head_type == "binary":
            target = torch.tensor([float(target_value)], device=prediction.device)
            loss = F.binary_cross_entropy_with_logits(prediction.reshape(-1), target, reduction="mean")
            acc = float(((prediction.reshape(-1) > 0).float() == target).float().mean().item())
        else:
            target = torch.tensor([int(target_value)], device=prediction.device, dtype=torch.long)
            loss = F.cross_entropy(prediction, target, reduction="mean")
            acc = float((prediction.argmax(dim=-1) == target).float().mean().item())
        aux_totals[f"{spec.name}_loss_sum"] += float(loss.item())
        aux_totals[f"{spec.name}_acc_sum"] += acc
        aux_totals[f"{spec.name}_active"] += 1.0


def _finalize_aux_eval_totals(aux_totals: dict[str, float], aux_specs: tuple[AuxHeadSpec, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for spec in aux_specs:
        active = aux_totals.get(f"{spec.name}_active", 0.0)
        if active > 0:
            metrics[f"{spec.name}_loss"] = aux_totals[f"{spec.name}_loss_sum"] / active
            metrics[f"{spec.name}_acc"] = aux_totals[f"{spec.name}_acc_sum"] / active
        else:
            metrics[f"{spec.name}_loss"] = 0.0
            metrics[f"{spec.name}_acc"] = 0.0
        metrics[f"{spec.name}_active"] = active
    return metrics


def compute_adaptive_env_sampling_weights(
    env_metrics: dict[str, dict[str, float]],
    *,
    uniform_alpha: float = 0.2,
) -> dict[str, float]:
    env_names = list(env_metrics)
    n_envs = len(env_names)
    if n_envs == 0:
        return {}
    uniform_alpha = float(np.clip(uniform_alpha, 0.0, 1.0))
    uniform_percent = 100.0 / n_envs
    raw_weights = {}
    for env_name, metrics in env_metrics.items():
        success_percent = 100.0 * float(metrics.get("val_success_rate", 0.0))
        adaptive_percent = 100.0 - success_percent
        raw_weights[env_name] = max(
            0.0,
            adaptive_percent * (1.0 - uniform_alpha) + uniform_percent * uniform_alpha,
        )
    total = sum(raw_weights.values())
    if total <= 0:
        uniform = 1.0 / n_envs
        return {env_name: uniform for env_name in env_names}
    return {env_name: weight / total for env_name, weight in raw_weights.items()}


def format_sampling_summary(sampling_weights: dict[str, float] | None) -> str:
    if not sampling_weights:
        return ""
    return " ".join(
        f"{env_name.replace('BabyAI-', '').replace('-v0', '')}={weight:.3f}"
        for env_name, weight in sampling_weights.items()
    )


def evaluate_env_suite(
    *,
    predictor,
    env_names: list[str],
    vocab,
    seed: int,
    n_eval_episodes: int,
    max_steps: int = 1000,
    on_env_complete=None,
) -> tuple[dict[str, dict[str, float]], float, float, float, float]:
    env_metrics: dict[str, dict[str, float]] = {}
    train_success_rates: list[float] = []
    val_success_rates: list[float] = []
    train_lengths: list[float] = []
    val_lengths: list[float] = []

    for env_index, env_name in enumerate(env_names):
        train_seed_base = EVAL_TRAIN_SEED_BASE + seed
        val_seed_base = EVAL_VAL_SEED_BASE + seed
        train_seeds = [train_seed_base + env_index * 1_000 + i for i in range(n_eval_episodes)]
        val_seeds = [val_seed_base + env_index * 1_000 + i for i in range(n_eval_episodes)]
        train_success_rate, train_mean_length, train_aux_metrics = predictor(env_name, train_seeds, max_steps)
        val_success_rate, val_mean_length, val_aux_metrics = predictor(env_name, val_seeds, max_steps)
        env_metrics[env_name] = {
            "train_success_rate": train_success_rate,
            "val_success_rate": val_success_rate,
            "train_mean_length": train_mean_length,
            "val_mean_length": val_mean_length,
        }
        for key, value in train_aux_metrics.items():
            env_metrics[env_name][f"train_{key}"] = value
        for key, value in val_aux_metrics.items():
            env_metrics[env_name][f"val_{key}"] = value
        if on_env_complete is not None:
            on_env_complete(env_name, env_metrics[env_name], env_index, len(env_names))
        train_success_rates.append(train_success_rate)
        val_success_rates.append(val_success_rate)
        train_lengths.append(train_mean_length)
        val_lengths.append(val_mean_length)

    return (
        env_metrics,
        float(np.mean(train_success_rates) if train_success_rates else 0.0),
        float(np.mean(val_success_rates) if val_success_rates else 0.0),
        float(np.mean(train_lengths) if train_lengths else 0.0),
        float(np.mean(val_lengths) if val_lengths else 0.0),
    )


class ProgressEvalCallback(BaseCallback):  # type: ignore[misc]
    """Live progress bar + fixed-seed train/validation evaluation."""

    def __init__(
        self,
        *,
        run_name: str,
        env_names: list[str],
        vocab,
        seed: int,
        total_timesteps: int,
        eval_freq: int,
        n_eval_episodes: int,
        sampling_uniform_alpha: float,
        log_dir: str,
        save_on_eval=None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.run_name = run_name
        self.env_names = list(env_names)
        self.vocab = vocab
        self.seed = seed
        self.total_timesteps = total_timesteps
        self.eval_freq = max(int(eval_freq), 1)
        self.n_eval_episodes = max(int(n_eval_episodes), 1)
        self.sampling_uniform_alpha = float(sampling_uniform_alpha)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_log_path = self.log_dir / f"{run_name}_eval_log.jsonl"
        self.best_val_success_rate = float("-inf")
        self._progress = None
        self._last_postfix: dict[str, str] = {}
        self.save_on_eval = save_on_eval
        self._last_best_val_success_rate = float("-inf")

    def _training_postfix(self) -> dict[str, str]:
        values = getattr(self.logger, "name_to_value", {})
        postfix: dict[str, str] = {}

        ep_rew = values.get("rollout/ep_rew_mean")
        ep_len = values.get("rollout/ep_len_mean")

        if ep_rew is None or ep_len is None:
            ep_info_buffer = getattr(self.model, "ep_info_buffer", None)
            if ep_info_buffer:
                rewards = [float(item["r"]) for item in ep_info_buffer if "r" in item]
                lengths = [float(item["l"]) for item in ep_info_buffer if "l" in item]
                if ep_rew is None and rewards:
                    ep_rew = float(np.mean(rewards))
                if ep_len is None and lengths:
                    ep_len = float(np.mean(lengths))

        if ep_rew is not None:
            postfix["rew"] = f"{float(ep_rew):.3f}"
        if ep_len is not None:
            postfix["len"] = f"{float(ep_len):.1f}"

        postfix.update(self._last_postfix)
        return postfix

    def _on_training_start(self) -> None:
        self._progress = create_progress_bar(total=self.total_timesteps, desc=f"train {self.run_name}", unit="ts")

    def _on_step(self) -> bool:
        if self._progress is not None:
            self._progress.n = min(self.num_timesteps, self.total_timesteps)
            postfix = self._training_postfix()
            if postfix:
                self._progress.set_postfix(**postfix)
            self._progress.refresh()

        if self.num_timesteps % self.eval_freq != 0:
            return True

        def _print_eval_env_metric(env_name: str, metrics: dict[str, float], env_index: int, total_envs: int) -> None:
            if not self.verbose:
                return
            short_name = env_name.replace("BabyAI-", "").replace("-v0", "")
            print(
                f"[rl eval env @ {self.num_timesteps}] {env_index + 1}/{total_envs} {short_name}: "
                f"train_sr={metrics['train_success_rate']:.2%} "
                f"val_sr={metrics['val_success_rate']:.2%}"
            )

        env_metrics, train_success_rate, val_success_rate, train_mean_length, val_mean_length = evaluate_env_suite(
            predictor=lambda env_name, seeds, max_steps: _evaluate_on_seeds(
                self.model,
                env_name,
                self.vocab,
                seeds,
                max_steps=max_steps,
            ),
            env_names=self.env_names,
            vocab=self.vocab,
            seed=self.seed,
            n_eval_episodes=self.n_eval_episodes,
            on_env_complete=_print_eval_env_metric,
        )

        previous_best = self.best_val_success_rate
        self.best_val_success_rate = max(self.best_val_success_rate, val_success_rate)
        snapshot = EvalSnapshot(
            timesteps=self.num_timesteps,
            train_success_rate=train_success_rate,
            val_success_rate=val_success_rate,
            train_mean_length=train_mean_length,
            val_mean_length=val_mean_length,
            best_val_success_rate=self.best_val_success_rate,
            env_metrics=env_metrics,
        )
        with self.eval_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(snapshot), sort_keys=True) + "\n")

        self._last_postfix = {
            "train_sr": f"{train_success_rate:.2%}",
            "val_sr": f"{val_success_rate:.2%}",
            "best_val": f"{self.best_val_success_rate:.2%}",
        }
        if self._progress is not None:
            self._progress.set_postfix(**self._training_postfix())

        if self.verbose:
            print(
                f"\n[eval @ {self.num_timesteps}] "
                f"train_sr={train_success_rate:.2%} "
                f"val_sr={val_success_rate:.2%} "
                f"best_val={self.best_val_success_rate:.2%}"
            )
            env_summary = " ".join(
                f"{env_name.replace('BabyAI-', '').replace('-v0', '')}={metrics['val_success_rate']:.0%}"
                for env_name, metrics in env_metrics.items()
            )
            if env_summary:
                print(env_summary)
            if len(self.env_names) > 1:
                sampling_summary = format_sampling_summary(
                    compute_adaptive_env_sampling_weights(
                        env_metrics,
                        uniform_alpha=self.sampling_uniform_alpha,
                    )
                )
                if sampling_summary:
                    print(f"[sampling] {sampling_summary}")
        if self.save_on_eval is not None:
            self.save_on_eval(
                timesteps=self.num_timesteps,
                train_success_rate=train_success_rate,
                val_success_rate=val_success_rate,
                best_val_success_rate=self.best_val_success_rate,
                env_metrics=env_metrics,
                is_new_best=self.best_val_success_rate > previous_best,
            )

        return True

    def _on_training_end(self) -> None:
        if self._progress is not None:
            self._progress.n = self.total_timesteps
            self._progress.refresh()
            self._progress.close()
