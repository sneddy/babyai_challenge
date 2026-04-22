"""RL training service for single-task and multi-task BabyAI runs."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
import torch

from ..bc.service import train_bc
from ..evaluation import (
    EvalSnapshot,
    ProgressEvalCallback,
    create_progress_bar,
    compute_adaptive_env_sampling_weights,
    evaluate_torch_policy_on_seeds,
    evaluate_env_suite,
    format_sampling_summary,
)
from ...config.loader import get_config
from ...data.vocabulary import MissionVocabulary, TokenizationConfig, build_global_vocab
from ...envs.catalog import ALL_LEADERBOARD_ENVS, LEADERBOARD_TIERS
from ...envs.wrappers import make_env, make_vec_env
from ...models.core import BabyAIRecurrentActorCritic, ModelConfig, load_exported_checkpoint, save_exported_checkpoint
from ...models.sb3 import BabyAIFeaturesExtractor, export_feedforward_policy_from_sb3
from ...models.transfer import (
    checkpoint_looks_like_torch_export,
    copy_matching_state_dict,
    initialize_feedforward_sb3_from_exported_checkpoint,
    initialize_torch_policy_from_checkpoint,
)

@dataclass(frozen=True)
class RunSpec:
    run_name: str
    train_env_names: list[str]
    eval_env_names: list[str]
    sampling_weights: dict[str, float] | None
    all_envs: bool = False


@dataclass(frozen=True)
class ArtifactPaths:
    latest_state_path: Path
    best_state_path: Path
    interrupted_state_path: Path
    latest_export_path: Path
    best_export_path: Path
    metadata_path: Path


def _env_stem(env_name: str) -> str:
    return env_name.replace("BabyAI-", "").replace("-v0", "")


def _parse_envs(raw_envs: str | None) -> list[str]:
    if not raw_envs:
        return []
    alias_map = {
        "easy": LEADERBOARD_TIERS["easy"],
        "medium": LEADERBOARD_TIERS["moderate"],
        "moderate": LEADERBOARD_TIERS["moderate"],
        "hard": LEADERBOARD_TIERS["hard"],
        "all": ALL_LEADERBOARD_ENVS,
    }
    env_names: list[str] = []
    for item in raw_envs.split(","):
        token = item.strip()
        if not token:
            continue
        expanded = alias_map.get(token.lower())
        if expanded is not None:
            env_names.extend(expanded)
        else:
            env_names.append(token)
    return list(dict.fromkeys(env_names))
def resolve_run_spec(args, config: dict) -> RunSpec:
    tier_flags = [args.easy, args.moderate, args.hard]
    selector_count = sum(
        bool(value)
        for value in [args.env, args.envs, args.all_envs, any(tier_flags)]
    )
    if selector_count > 1:
        raise RuntimeError("Use only one selector source: --env, --envs, --all-envs, or tier flags.")

    if args.env:
        env_names = [args.env]
        run_name = args.run_name or _env_stem(args.env)
        return RunSpec(
            run_name=run_name,
            train_env_names=env_names,
            eval_env_names=env_names,
            sampling_weights=None,
        )

    if any(tier_flags):
        env_names: list[str] = []
        if args.easy:
            env_names.extend(LEADERBOARD_TIERS["easy"])
        if args.moderate:
            env_names.extend(LEADERBOARD_TIERS["moderate"])
        if args.hard:
            env_names.extend(LEADERBOARD_TIERS["hard"])
        env_names = list(dict.fromkeys(env_names))
        if not env_names:
            raise RuntimeError("Tier flags were provided but no environments were selected.")
        if args.run_name:
            run_name = args.run_name
        else:
            selected_tiers = [
                name
                for name, enabled in [
                    ("easy", args.easy),
                    ("moderate", args.moderate),
                    ("hard", args.hard),
                ]
                if enabled
            ]
            run_name = "_".join(selected_tiers)
        return RunSpec(
            run_name=run_name,
            train_env_names=env_names,
            eval_env_names=list(env_names),
            sampling_weights=None,
        )

    if args.envs:
        env_names = _parse_envs(args.envs)
        if not env_names:
            raise RuntimeError("--envs was provided but no environment names were parsed.")
        run_name = args.run_name or ("multitask" if len(env_names) > 1 else _env_stem(env_names[0]))
        return RunSpec(
            run_name=run_name,
            train_env_names=env_names,
            eval_env_names=list(env_names),
            sampling_weights=None,
        )

    if args.all_envs:
        return RunSpec(
            run_name=args.run_name or "leaderboard_multitask",
            train_env_names=list(ALL_LEADERBOARD_ENVS),
            eval_env_names=list(ALL_LEADERBOARD_ENVS),
            sampling_weights=None,
            all_envs=True,
        )

    default_env = LEADERBOARD_TIERS["easy"][0]
    return RunSpec(
        run_name=args.run_name or _env_stem(default_env),
        train_env_names=[default_env],
        eval_env_names=[default_env],
        sampling_weights=None,
    )


def _tokenization_config_from_config(config: dict) -> TokenizationConfig:
    return TokenizationConfig(**config["tokenizer"])


def _load_or_build_global_vocab(config: dict, seed: int, rebuild: bool = False) -> MissionVocabulary:
    vocab_path = config["artifacts"]["vocab_path"]
    tokenization = _tokenization_config_from_config(config)
    if not rebuild and os.path.exists(vocab_path):
        vocab = MissionVocabulary.load(vocab_path)
        if vocab.tokenization == tokenization:
            return vocab
    return build_global_vocab(
        output_path=vocab_path,
        episodes_per_env=64,
        seed=seed,
        tokenization=tokenization,
    )


def _ensure_bc_checkpoint(
    *,
    run_spec: RunSpec,
    config_name: str,
    model_preset: str | None,
    config: dict,
    seed: int,
    bc_checkpoint_path: str | None,
    bc_demo_paths: list[str] | None,
    warm_start_path: str | None,
    eval_env_names: list[str] | None,
    bc_min_sampling_proba: float | None,
    recurrent: bool,
) -> str | None:
    if bc_checkpoint_path and bc_demo_paths:
        raise RuntimeError("Use either --bc-checkpoint or --bc-demos, not both.")
    if bc_checkpoint_path:
        return bc_checkpoint_path
    if not bc_demo_paths:
        return None

    _load_or_build_global_vocab(config, seed, rebuild=False)
    export_dir = Path(config["artifacts"]["exports"])
    export_dir.mkdir(parents=True, exist_ok=True)
    target_stem = f"{run_spec.run_name}_bc_recurrent" if recurrent else f"{run_spec.run_name}_bc"
    target_path = export_dir / f"{target_stem}.pt"
    train_bc(
        demo_path=bc_demo_paths,
        checkpoint_path=str(target_path),
        vocab_path=config["artifacts"]["vocab_path"],
        config_name=config_name,
        model_preset=model_preset,
        seed=seed,
        warm_start_path=warm_start_path,
        eval_env_names=eval_env_names,
        min_sampling_proba=bc_min_sampling_proba,
        recurrent=recurrent,
    )
    return str(target_path)


def _artifact_paths(run_name: str, algorithm: str, artifacts: dict) -> ArtifactPaths:
    checkpoint_ext = ".pt" if algorithm == "ppo_recurrent" else ".zip"
    latest_state_path = Path(artifacts["checkpoints"]) / f"{run_name}_latest{checkpoint_ext}"
    best_state_path = Path(artifacts["checkpoints"]) / f"{run_name}_best{checkpoint_ext}"
    interrupted_state_path = Path(artifacts["checkpoints"]) / f"{run_name}_interrupted{checkpoint_ext}"
    latest_export_path = Path(artifacts["exports"]) / f"{run_name}_policy.pt"
    best_export_path = Path(artifacts["exports"]) / f"{run_name}_best.pt"
    metadata_path = Path(artifacts["exports"]) / f"{run_name}_policy.json"
    return ArtifactPaths(
        latest_state_path=latest_state_path,
        best_state_path=best_state_path,
        interrupted_state_path=interrupted_state_path,
        latest_export_path=latest_export_path,
        best_export_path=best_export_path,
        metadata_path=metadata_path,
    )


def _build_learning_rate_schedule(config: dict) -> float | Callable[[float], float]:
    learning_rate = config["ppo"]["learning_rate"]
    end_learning_rate = config["ppo"].get("end_learning_rate")
    learning_rate_period = config["ppo"].get("learning_rate_period")
    if end_learning_rate is None and learning_rate_period is None:
        return float(learning_rate)

    start_lr = float(learning_rate)
    end_lr = float(0.0 if end_learning_rate is None else end_learning_rate)
    total_timesteps = max(int(config["training"]["total_timesteps"]), 1)
    period_timesteps = total_timesteps if learning_rate_period is None else max(int(learning_rate_period), 1)

    def learning_rate_fn(progress_remaining: float) -> float:
        elapsed_timesteps = (1.0 - float(progress_remaining)) * total_timesteps
        if learning_rate_period is None:
            cosine_progress = min(max(elapsed_timesteps / period_timesteps, 0.0), 1.0)
        else:
            cosine_progress = (elapsed_timesteps % period_timesteps) / period_timesteps
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        return end_lr + (start_lr - end_lr) * cosine_value

    return learning_rate_fn


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _create_ppo_model(config: dict, train_env, vocab, model_config: ModelConfig, seed: int):
    policy_kwargs = {
        "features_extractor_class": BabyAIFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": model_config.features_dim,
            "vocab_size": vocab.size,
            "model_config": model_config.to_dict(),
        },
        "net_arch": config["ppo"]["net_arch"],
    }
    learning_rate = _build_learning_rate_schedule(config)
    return PPO(
        config["ppo"]["policy"],
        train_env,
        learning_rate=learning_rate,
        n_steps=config["ppo"]["n_steps"],
        batch_size=config["ppo"]["batch_size"],
        n_epochs=config["ppo"]["n_epochs"],
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_range=config["ppo"]["clip_range"],
        ent_coef=config["ppo"]["ent_coef"],
        vf_coef=config["ppo"]["vf_coef"],
        max_grad_norm=config["ppo"]["max_grad_norm"],
        verbose=config["ppo"]["verbose"],
        seed=seed,
        policy_kwargs=policy_kwargs,
        device=config["training"]["device"],
    )


def _resolve_feedforward_model_config(
    *,
    config: dict,
    warm_start_path: str | None,
    train_env,
) -> ModelConfig:
    model_config = ModelConfig.from_dict(config["model"])
    if not warm_start_path:
        return model_config

    if checkpoint_looks_like_torch_export(warm_start_path):
        try:
            payload = load_exported_checkpoint(warm_start_path, map_location="cpu")
            checkpoint_model_config = payload.get("model_config")
            if checkpoint_model_config:
                return ModelConfig.from_dict(checkpoint_model_config)
        except Exception:
            return model_config
        return model_config

    try:
        source_model = PPO.load(warm_start_path, env=train_env, device=config["training"]["device"])
        source_config = getattr(source_model.policy.features_extractor.core, "config", None)
        if source_config is not None:
            return ModelConfig.from_dict(source_config.to_dict())
    except Exception:
        return model_config
    return model_config


def _initialize_feedforward_policy(model, checkpoint_path: str, train_env, device: str) -> None:
    if checkpoint_looks_like_torch_export(checkpoint_path):
        initialize_feedforward_sb3_from_exported_checkpoint(model, checkpoint_path, map_location=device)
        return

    source_model = PPO.load(checkpoint_path, env=train_env, device=device)
    model.policy.load_state_dict(source_model.policy.state_dict(), strict=True)


def _resolve_recurrent_model_config(
    *,
    config: dict,
    warm_start_path: str | None,
    train_env,
) -> ModelConfig:
    model_config = ModelConfig.from_dict(config["model"])
    if not warm_start_path:
        return model_config

    if checkpoint_looks_like_torch_export(warm_start_path):
        try:
            payload = load_exported_checkpoint(warm_start_path, map_location="cpu")
            checkpoint_model_config = payload.get("model_config")
            if checkpoint_model_config:
                return ModelConfig.from_dict(checkpoint_model_config)
        except Exception:
            return model_config
        return model_config

    try:
        source_model = PPO.load(warm_start_path, env=train_env, device=config["training"]["device"])
        source_config = getattr(source_model.policy.features_extractor.core, "config", None)
        if source_config is not None:
            return ModelConfig.from_dict(source_config.to_dict())
    except Exception:
        return model_config
    return model_config


def _initialize_recurrent_policy(model, checkpoint_path: str, train_env, device: str) -> None:
    if checkpoint_looks_like_torch_export(checkpoint_path):
        initialize_torch_policy_from_checkpoint(model, checkpoint_path, map_location=device)
        return

    source_model = PPO.load(checkpoint_path, env=train_env, device=device)
    copy_matching_state_dict(model.encoder, source_model.policy.features_extractor.core.state_dict())
    copy_matching_state_dict(model.actor, source_model.policy.action_net.state_dict())
    copy_matching_state_dict(model.critic, source_model.policy.value_net.state_dict())


def _tensorize_vec_obs(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "image": torch.as_tensor(obs["image"], device=device),
        "mission_tokens": torch.as_tensor(obs["mission_tokens"], device=device),
        "mission_mask": torch.as_tensor(obs["mission_mask"], device=device),
    }


class RecurrentPPOTrainer:
    """Minimal recurrent PPO trainer on top of VecEnv rollouts."""

    def __init__(
        self,
        *,
        env,
        model: BabyAIRecurrentActorCritic,
        device: torch.device,
        num_steps: int,
        recurrence: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        beta1: float,
        beta2: float,
        adam_eps: float,
        discount: float,
        gae_lambda: float,
        clip_eps: float,
        entropy_coef: float,
        value_loss_coef: float,
        max_grad_norm: float,
    ) -> None:
        self.env = env
        self.model = model
        self.device = device
        self.num_steps = int(num_steps)
        self.recurrence = int(recurrence)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.discount = float(discount)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.value_loss_coef = float(value_loss_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.num_envs = int(env.num_envs)
        self.num_frames = self.num_steps * self.num_envs
        if self.num_steps % self.recurrence != 0:
            raise ValueError("ppo.n_steps must be divisible by ppo.recurrence for recurrent PPO")
        if self.batch_size % self.recurrence != 0:
            raise ValueError("ppo.batch_size must be divisible by ppo.recurrence for recurrent PPO")

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(learning_rate),
            betas=(float(beta1), float(beta2)),
            eps=float(adam_eps),
        )
        self.obs = self.env.reset()
        self.state = self.model.initial_state(batch_size=self.num_envs, device=self.device)
        self.mask = torch.ones(self.num_envs, 1, device=self.device)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def set_learning_rate(self, learning_rate: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = float(learning_rate)

    def get_state(self) -> dict:
        return {
            "policy_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }

    def load_state(self, payload: dict) -> None:
        self.model.load_state_dict(payload["policy_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])

    @staticmethod
    def _flatten_rollout_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(0, 1).reshape(-1, *tensor.shape[2:])

    def _allocate_obs_storage(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: torch.zeros((self.num_steps, self.num_envs, *value.shape[1:]), dtype=value.dtype, device=self.device)
            for key, value in obs.items()
        }

    def collect_rollouts(self) -> tuple[dict[str, torch.Tensor | dict[str, torch.Tensor]], dict[str, float | int | list[float] | list[int]]]:
        obs_storage: dict[str, torch.Tensor] | None = None
        memory_storage = torch.zeros((self.num_steps, self.num_envs, self.model.memory_size), dtype=torch.float32, device=self.device)
        mask_storage = torch.zeros((self.num_steps, self.num_envs, 1), dtype=torch.float32, device=self.device)
        action_storage = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long, device=self.device)
        value_storage = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        reward_storage = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        log_prob_storage = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []

        self.model.train()
        for step in range(self.num_steps):
            obs_tensors = _tensorize_vec_obs(self.obs, self.device)
            if obs_storage is None:
                obs_storage = self._allocate_obs_storage(obs_tensors)
            for key, value in obs_tensors.items():
                obs_storage[key][step] = value

            current_memory = torch.cat(self.state, dim=1)
            memory_storage[step] = current_memory
            mask_storage[step] = self.mask

            with torch.no_grad():
                outputs = self.model(obs_tensors, current_memory * self.mask)
                dist = outputs["dist"]
                value = outputs["value"]
                next_memory = outputs["memory"]

            action = dist.sample()
            next_obs, reward, done, _infos = self.env.step(action.cpu().numpy())
            reward_array = np.asarray(reward, dtype=np.float32)
            done_array = np.asarray(done, dtype=bool)

            action_storage[step] = action
            value_storage[step] = value
            reward_storage[step] = torch.as_tensor(reward_array, device=self.device)
            log_prob_storage[step] = dist.log_prob(action)

            self.episode_returns += reward_array
            self.episode_lengths += 1
            for env_index, done_flag in enumerate(done_array.tolist()):
                if done_flag:
                    completed_returns.append(float(self.episode_returns[env_index]))
                    completed_lengths.append(int(self.episode_lengths[env_index]))
                    self.episode_returns[env_index] = 0.0
                    self.episode_lengths[env_index] = 0

            self.obs = next_obs
            self.state = self.model._split_memory(next_memory, device=self.device, batch_size=self.num_envs)
            self.mask = (1.0 - torch.as_tensor(done_array, device=self.device, dtype=torch.float32)).unsqueeze(1)

        assert obs_storage is not None
        next_obs_tensors = _tensorize_vec_obs(self.obs, self.device)
        with torch.no_grad():
            next_outputs = self.model(next_obs_tensors, torch.cat(self.state, dim=1) * self.mask)
            next_value = next_outputs["value"]

        advantage_storage = torch.zeros_like(reward_storage)
        running_advantage = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_mask = self.mask.squeeze(1)
                next_step_value = next_value
            else:
                next_mask = mask_storage[step + 1].squeeze(1)
                next_step_value = value_storage[step + 1]
            delta = reward_storage[step] + self.discount * next_step_value * next_mask - value_storage[step]
            running_advantage = delta + self.discount * self.gae_lambda * running_advantage * next_mask
            advantage_storage[step] = running_advantage

        rollout = {
            "obs": {key: self._flatten_rollout_tensor(value) for key, value in obs_storage.items()},
            "memory": self._flatten_rollout_tensor(memory_storage),
            "mask": self._flatten_rollout_tensor(mask_storage),
            "action": self._flatten_rollout_tensor(action_storage.unsqueeze(-1)).squeeze(-1),
            "value": self._flatten_rollout_tensor(value_storage.unsqueeze(-1)).squeeze(-1),
            "reward": self._flatten_rollout_tensor(reward_storage.unsqueeze(-1)).squeeze(-1),
            "advantage": self._flatten_rollout_tensor(advantage_storage.unsqueeze(-1)).squeeze(-1),
            "log_prob": self._flatten_rollout_tensor(log_prob_storage.unsqueeze(-1)).squeeze(-1),
        }
        rollout["returnn"] = rollout["value"] + rollout["advantage"]
        logs = {
            "return_per_episode": completed_returns,
            "num_frames_per_episode": completed_lengths,
            "episodes_done": len(completed_returns),
            "num_frames": self.num_frames,
        }
        return rollout, logs

    def _get_batches_starting_indexes(self) -> list[np.ndarray]:
        indexes = np.arange(0, self.num_frames, self.recurrence)
        indexes = np.random.permutation(indexes)
        num_indexes = self.batch_size // self.recurrence
        return [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

    def update_parameters(self) -> dict[str, float | int | list[float] | list[int]]:
        rollout, logs = self.collect_rollouts()
        memories = rollout["memory"].clone()

        log_entropies: list[float] = []
        log_values: list[float] = []
        log_policy_losses: list[float] = []
        log_value_losses: list[float] = []
        log_grad_norms: list[float] = []
        log_losses: list[float] = []

        for _ in range(self.epochs):
            for batch_starts in self._get_batches_starting_indexes():
                batch_entropy = 0.0
                batch_value = 0.0
                batch_policy_loss = 0.0
                batch_value_loss = 0.0
                batch_loss = 0.0

                memory = memories[batch_starts]
                for step_offset in range(self.recurrence):
                    indexes = batch_starts + step_offset
                    obs_batch = {key: value[indexes] for key, value in rollout["obs"].items()}
                    mask_batch = rollout["mask"][indexes]
                    outputs = self.model(obs_batch, memory * mask_batch)
                    dist = outputs["dist"]
                    value = outputs["value"]
                    memory = outputs["memory"]

                    entropy = dist.entropy().mean()
                    ratio = torch.exp(dist.log_prob(rollout["action"][indexes]) - rollout["log_prob"][indexes])
                    surr1 = ratio * rollout["advantage"][indexes]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * rollout["advantage"][indexes]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = rollout["value"][indexes] + torch.clamp(
                        value - rollout["value"][indexes],
                        -self.clip_eps,
                        self.clip_eps,
                    )
                    value_loss = torch.max(
                        (value - rollout["returnn"][indexes]).pow(2),
                        (value_clipped - rollout["returnn"][indexes]).pow(2),
                    ).mean()
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    batch_entropy += float(entropy.item())
                    batch_value += float(value.mean().item())
                    batch_policy_loss += float(policy_loss.item())
                    batch_value_loss += float(value_loss.item())
                    batch_loss = batch_loss + loss

                    if step_offset < self.recurrence - 1:
                        memories[indexes + 1] = memory.detach()

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss = batch_loss / self.recurrence

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(
                    p.grad.data.norm(2) ** 2
                    for p in self.model.parameters()
                    if p.grad is not None
                ) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(float(grad_norm.item()))
                log_losses.append(float(batch_loss.item()))

        logs["entropy"] = float(np.mean(log_entropies) if log_entropies else 0.0)
        logs["value"] = float(np.mean(log_values) if log_values else 0.0)
        logs["policy_loss"] = float(np.mean(log_policy_losses) if log_policy_losses else 0.0)
        logs["value_loss"] = float(np.mean(log_value_losses) if log_value_losses else 0.0)
        logs["grad_norm"] = float(np.mean(log_grad_norms) if log_grad_norms else 0.0)
        logs["loss"] = float(np.mean(log_losses) if log_losses else 0.0)
        return logs


def _run_initial_warm_start_eval(
    *,
    model,
    run_spec: RunSpec,
    vocab,
    config: dict,
    seed: int,
):
    if len(run_spec.train_env_names) <= 1:
        return None

    def _print_env_metric(env_name: str, metrics: dict[str, float], env_index: int, total_envs: int) -> None:
        short_name = env_name.replace("BabyAI-", "").replace("-v0", "")
        print(
            f"[rl warm-start eval env] {env_index + 1}/{total_envs} {short_name}: "
            f"train_sr={metrics['train_success_rate']:.2%} "
            f"val_sr={metrics['val_success_rate']:.2%}"
        )

    env_metrics, train_success_rate, val_success_rate, _train_len, _val_len = evaluate_env_suite(
        predictor=lambda env_name, seeds, max_steps: model_predictor(
            model,
            env_name,
            vocab,
            seeds,
            max_steps,
        ),
        env_names=run_spec.eval_env_names,
        vocab=vocab,
        seed=seed,
        n_eval_episodes=config["training"]["n_eval_episodes"],
        on_env_complete=_print_env_metric,
    )
    updated_weights = compute_adaptive_env_sampling_weights(
        env_metrics,
        uniform_alpha=float(config["curriculum"]["sampling_uniform_alpha"]),
    )
    print(
        f"[warm-start eval] "
        f"train_sr={train_success_rate:.2%} "
        f"val_sr={val_success_rate:.2%}"
    )
    env_summary = " ".join(
        f"{env_name.replace('BabyAI-', '').replace('-v0', '')}={metrics['val_success_rate']:.0%}"
        for env_name, metrics in env_metrics.items()
    )
    if env_summary:
        print(env_summary)
    sampling_summary = format_sampling_summary(updated_weights)
    if sampling_summary:
        print(f"[warm-start sampling] {sampling_summary}")
    return updated_weights


def _run_initial_recurrent_warm_start_eval(
    *,
    model: BabyAIRecurrentActorCritic,
    run_spec: RunSpec,
    vocab,
    config: dict,
    seed: int,
    device: torch.device,
):
    if len(run_spec.train_env_names) <= 1:
        return None, None

    def _print_env_metric(env_name: str, metrics: dict[str, float], env_index: int, total_envs: int) -> None:
        short_name = env_name.replace("BabyAI-", "").replace("-v0", "")
        print(
            f"[rl warm-start eval env] {env_index + 1}/{total_envs} {short_name}: "
            f"train_sr={metrics['train_success_rate']:.2%} "
            f"val_sr={metrics['val_success_rate']:.2%}"
        )

    env_metrics, train_success_rate, val_success_rate, _train_len, _val_len = evaluate_env_suite(
        predictor=lambda env_name, seeds, max_steps: evaluate_torch_policy_on_seeds(
            model,
            env_name,
            vocab,
            seeds,
            device=str(device),
            max_steps=max_steps,
        ),
        env_names=run_spec.eval_env_names,
        vocab=vocab,
        seed=seed,
        n_eval_episodes=config["training"]["n_eval_episodes"],
        max_steps=int(config["training"].get("eval_max_steps", 1000)),
        on_env_complete=_print_env_metric,
    )
    updated_weights = compute_adaptive_env_sampling_weights(
        env_metrics,
        uniform_alpha=float(config["curriculum"]["sampling_uniform_alpha"]),
    )
    print(
        f"[warm-start eval] "
        f"train_sr={train_success_rate:.2%} "
        f"val_sr={val_success_rate:.2%}"
    )
    env_summary = " ".join(
        f"{env_name.replace('BabyAI-', '').replace('-v0', '')}={metrics['val_success_rate']:.0%}"
        for env_name, metrics in env_metrics.items()
    )
    if env_summary:
        print(env_summary)
    sampling_summary = format_sampling_summary(updated_weights)
    if sampling_summary:
        print(f"[warm-start sampling] {sampling_summary}")
    return updated_weights, {
        "env_metrics": env_metrics,
        "train_success_rate": train_success_rate,
        "val_success_rate": val_success_rate,
    }


def _evaluate_recurrent_policy(
    *,
    model: BabyAIRecurrentActorCritic,
    vocab,
    env_names: list[str],
    seed: int,
    n_eval_episodes: int,
    max_steps: int,
    device: torch.device,
    log_prefix: str | None = None,
) -> tuple[dict[str, dict[str, float]], float, float, float, float]:
    def _print_env_metric(env_name: str, metrics: dict[str, float], env_index: int, total_envs: int) -> None:
        if not log_prefix:
            return
        short_name = env_name.replace("BabyAI-", "").replace("-v0", "")
        print(
            f"{log_prefix} {env_index + 1}/{total_envs} {short_name}: "
            f"train_sr={metrics['train_success_rate']:.2%} "
            f"val_sr={metrics['val_success_rate']:.2%}"
        )

    return evaluate_env_suite(
        predictor=lambda env_name, seeds, eval_max_steps: evaluate_torch_policy_on_seeds(
            model,
            env_name,
            vocab,
            seeds,
            device=str(device),
            max_steps=eval_max_steps,
        ),
        env_names=env_names,
        vocab=vocab,
        seed=seed,
        n_eval_episodes=n_eval_episodes,
        max_steps=max_steps,
        on_env_complete=_print_env_metric,
    )


def model_predictor(model, env_name: str, vocab, seeds: list[int], max_steps: int):
    from ..evaluation import _evaluate_on_seeds

    return _evaluate_on_seeds(model, env_name, vocab, seeds, max_steps=max_steps)


def _save_feedforward_latest(
    *,
    model,
    vocab,
    model_config: ModelConfig,
    artifact_paths: ArtifactPaths,
    run_spec: RunSpec,
    config_name: str,
    model_preset: str | None,
    seed: int,
    total_timesteps_requested: int,
    save_tag: str,
    vocab_path: str,
) -> None:
    model.save(str(artifact_paths.latest_state_path))
    export_policy = export_feedforward_policy_from_sb3(
        model,
        vocab_payload=vocab.to_dict(),
        model_config=model_config.to_dict(),
    )
    save_exported_checkpoint(
        str(artifact_paths.latest_export_path),
        policy_state=export_policy.state_dict(),
        model_config=model_config,
        vocab_payload=vocab.to_dict(),
        recurrent=False,
    )
    metadata = {
        "algorithm": "ppo",
        "run_name": run_spec.run_name,
        "train_envs": run_spec.train_env_names,
        "eval_envs": run_spec.eval_env_names,
        "config_name": config_name,
        "model_preset": model_preset,
        "seed": seed,
        "num_timesteps": model.num_timesteps,
        "total_timesteps_requested": total_timesteps_requested,
        "latest_state_path": str(artifact_paths.latest_state_path),
        "latest_export_path": str(artifact_paths.latest_export_path),
        "vocab_path": vocab_path,
        "save_tag": save_tag,
    }
    artifact_paths.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _save_feedforward_best(
    *,
    model,
    vocab,
    model_config: ModelConfig,
    artifact_paths: ArtifactPaths,
) -> None:
    model.save(str(artifact_paths.best_state_path))
    export_policy = export_feedforward_policy_from_sb3(
        model,
        vocab_payload=vocab.to_dict(),
        model_config=model_config.to_dict(),
    )
    save_exported_checkpoint(
        str(artifact_paths.best_export_path),
        policy_state=export_policy.state_dict(),
        model_config=model_config,
        vocab_payload=vocab.to_dict(),
        recurrent=False,
    )


def _save_recurrent_training_checkpoint(
    path: Path,
    *,
    trainer: RecurrentPPOTrainer,
    model_config: ModelConfig,
    vocab: MissionVocabulary,
    run_spec: RunSpec,
    config_name: str,
    model_preset: str | None,
    seed: int,
    total_timesteps_requested: int,
    num_timesteps: int,
    update_index: int,
    best_val_success_rate: float,
    save_tag: str,
) -> None:
    payload = {
        **trainer.get_state(),
        "algorithm": "ppo_recurrent",
        "recurrent": True,
        "model_config": model_config.to_dict(),
        "vocab": vocab.to_dict(),
        "run_name": run_spec.run_name,
        "train_envs": run_spec.train_env_names,
        "eval_envs": run_spec.eval_env_names,
        "config_name": config_name,
        "model_preset": model_preset,
        "seed": seed,
        "num_timesteps": int(num_timesteps),
        "update_index": int(update_index),
        "total_timesteps_requested": int(total_timesteps_requested),
        "best_val_success_rate": float(best_val_success_rate),
        "save_tag": save_tag,
    }
    torch.save(payload, path)


def _load_recurrent_training_checkpoint(path: str, map_location: torch.device | str = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "optimizer_state" not in payload:
        raise RuntimeError("--resume for recurrent PPO expects a recurrent training checkpoint, not an exported policy.")
    return payload


def _save_recurrent_export(
    path: Path,
    *,
    model: BabyAIRecurrentActorCritic,
    vocab: MissionVocabulary,
    model_config: ModelConfig,
) -> None:
    save_exported_checkpoint(
        str(path),
        policy_state=model.export_state_dict(),
        model_config=model_config,
        vocab_payload=vocab.to_dict(),
        recurrent=True,
    )


def _write_recurrent_metadata(
    path: Path,
    *,
    run_spec: RunSpec,
    config_name: str,
    model_preset: str | None,
    seed: int,
    num_timesteps: int,
    total_timesteps_requested: int,
    latest_state_path: Path,
    latest_export_path: Path,
    vocab_path: str,
    save_tag: str,
) -> None:
    payload = {
        "algorithm": "ppo_recurrent",
        "run_name": run_spec.run_name,
        "train_envs": run_spec.train_env_names,
        "eval_envs": run_spec.eval_env_names,
        "config_name": config_name,
        "model_preset": model_preset,
        "seed": seed,
        "num_timesteps": int(num_timesteps),
        "total_timesteps_requested": int(total_timesteps_requested),
        "latest_state_path": str(latest_state_path),
        "latest_export_path": str(latest_export_path),
        "vocab_path": vocab_path,
        "save_tag": save_tag,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _save_recurrent_latest(
    *,
    trainer: RecurrentPPOTrainer,
    model: BabyAIRecurrentActorCritic,
    vocab: MissionVocabulary,
    model_config: ModelConfig,
    artifact_paths: ArtifactPaths,
    run_spec: RunSpec,
    config_name: str,
    model_preset: str | None,
    seed: int,
    total_timesteps_requested: int,
    num_timesteps: int,
    update_index: int,
    best_val_success_rate: float,
    save_tag: str,
    vocab_path: str,
) -> None:
    _save_recurrent_training_checkpoint(
        artifact_paths.latest_state_path,
        trainer=trainer,
        model_config=model_config,
        vocab=vocab,
        run_spec=run_spec,
        config_name=config_name,
        model_preset=model_preset,
        seed=seed,
        total_timesteps_requested=total_timesteps_requested,
        num_timesteps=num_timesteps,
        update_index=update_index,
        best_val_success_rate=best_val_success_rate,
        save_tag=save_tag,
    )
    _save_recurrent_export(
        artifact_paths.latest_export_path,
        model=model,
        vocab=vocab,
        model_config=model_config,
    )
    _write_recurrent_metadata(
        artifact_paths.metadata_path,
        run_spec=run_spec,
        config_name=config_name,
        model_preset=model_preset,
        seed=seed,
        num_timesteps=num_timesteps,
        total_timesteps_requested=total_timesteps_requested,
        latest_state_path=artifact_paths.latest_state_path,
        latest_export_path=artifact_paths.latest_export_path,
        vocab_path=vocab_path,
        save_tag=save_tag,
    )


def _save_recurrent_best(
    *,
    trainer: RecurrentPPOTrainer,
    model: BabyAIRecurrentActorCritic,
    vocab: MissionVocabulary,
    model_config: ModelConfig,
    artifact_paths: ArtifactPaths,
    run_spec: RunSpec,
    config_name: str,
    model_preset: str | None,
    seed: int,
    total_timesteps_requested: int,
    num_timesteps: int,
    update_index: int,
    best_val_success_rate: float,
) -> None:
    _save_recurrent_training_checkpoint(
        artifact_paths.best_state_path,
        trainer=trainer,
        model_config=model_config,
        vocab=vocab,
        run_spec=run_spec,
        config_name=config_name,
        model_preset=model_preset,
        seed=seed,
        total_timesteps_requested=total_timesteps_requested,
        num_timesteps=num_timesteps,
        update_index=update_index,
        best_val_success_rate=best_val_success_rate,
        save_tag="best",
    )
    _save_recurrent_export(
        artifact_paths.best_export_path,
        model=model,
        vocab=vocab,
        model_config=model_config,
    )


def train_recurrent(
    *,
    run_spec: RunSpec,
    config_name: str,
    model_preset: str | None,
    seed: int,
    total_timesteps: int | None,
    resume_path: str | None,
    warm_start_path: str | None,
    bc_checkpoint_path: str | None,
    rebuild_vocab: bool = False,
) -> Path:
    config = get_config(config_name, model_preset=model_preset)
    if total_timesteps is not None:
        config["training"]["total_timesteps"] = total_timesteps
    config["training"]["seed"] = seed

    artifacts = config["artifacts"]
    os.makedirs(artifacts["checkpoints"], exist_ok=True)
    os.makedirs(artifacts["logs"], exist_ok=True)
    os.makedirs(artifacts["exports"], exist_ok=True)

    vocab = _load_or_build_global_vocab(config, seed, rebuild=rebuild_vocab)
    artifact_paths = _artifact_paths(run_spec.run_name, "ppo_recurrent", artifacts)
    train_env = make_vec_env(
        env_name=run_spec.train_env_names if len(run_spec.train_env_names) > 1 else run_spec.train_env_names[0],
        vocab=vocab,
        n_envs=config["training"]["n_envs"],
        seed=seed,
        monitor_dir=os.path.join(artifacts["checkpoints"], "monitors", run_spec.run_name),
        sampling_weights=run_spec.sampling_weights,
    )
    device = _resolve_device(config["training"]["device"])
    initial_checkpoint_path = bc_checkpoint_path or warm_start_path
    model_config = _resolve_recurrent_model_config(
        config=config,
        warm_start_path=initial_checkpoint_path,
        train_env=train_env,
    )
    model = BabyAIRecurrentActorCritic(
        vocab_size=vocab.size,
        config=model_config,
        pad_id=vocab.pad_id,
    ).to(device)
    learning_rate_schedule = _build_learning_rate_schedule(config)
    initial_learning_rate = (
        learning_rate_schedule(1.0)
        if callable(learning_rate_schedule)
        else float(learning_rate_schedule)
    )
    trainer = RecurrentPPOTrainer(
        env=train_env,
        model=model,
        device=device,
        num_steps=int(config["ppo"]["n_steps"]),
        recurrence=int(config["ppo"]["recurrence"]),
        batch_size=int(config["ppo"]["batch_size"]),
        epochs=int(config["ppo"]["n_epochs"]),
        learning_rate=float(initial_learning_rate),
        beta1=float(config["ppo"].get("beta1", 0.9)),
        beta2=float(config["ppo"].get("beta2", 0.999)),
        adam_eps=float(config["ppo"].get("adam_eps", 1e-5)),
        discount=float(config["ppo"]["gamma"]),
        gae_lambda=float(config["ppo"]["gae_lambda"]),
        clip_eps=float(config["ppo"]["clip_range"]),
        entropy_coef=float(config["ppo"]["ent_coef"]),
        value_loss_coef=float(config["ppo"]["vf_coef"]),
        max_grad_norm=float(config["ppo"]["max_grad_norm"]),
    )

    num_timesteps = 0
    update_index = 0
    best_val_success_rate = float("-inf")
    if resume_path:
        resume_payload = _load_recurrent_training_checkpoint(resume_path, map_location=device)
        trainer.load_state(resume_payload)
        num_timesteps = int(resume_payload.get("num_timesteps", 0))
        update_index = int(resume_payload.get("update_index", 0))
        best_val_success_rate = float(resume_payload.get("best_val_success_rate", float("-inf")))
    else:
        if warm_start_path:
            _initialize_recurrent_policy(model, warm_start_path, train_env, str(device))
        if bc_checkpoint_path:
            _initialize_recurrent_policy(model, bc_checkpoint_path, train_env, str(device))
        if initial_checkpoint_path and len(run_spec.train_env_names) > 1:
            initial_weights, initial_eval = _run_initial_recurrent_warm_start_eval(
                model=model,
                run_spec=run_spec,
                vocab=vocab,
                config=config,
                seed=seed,
                device=device,
            )
            if initial_weights is not None:
                train_env.env_method("set_sampling_weights", initial_weights)
            assert initial_eval is not None
            env_metrics = initial_eval["env_metrics"]
            train_success_rate = float(initial_eval["train_success_rate"])
            val_success_rate = float(initial_eval["val_success_rate"])
            best_val_success_rate = float(val_success_rate)
            _save_recurrent_best(
                trainer=trainer,
                model=model,
                vocab=vocab,
                model_config=model_config,
                artifact_paths=artifact_paths,
                run_spec=run_spec,
                config_name=config_name,
                model_preset=model_preset,
                seed=seed,
                total_timesteps_requested=int(config["training"]["total_timesteps"]),
                num_timesteps=num_timesteps,
                update_index=update_index,
                best_val_success_rate=best_val_success_rate,
            )
            print(f"[warm-start seeded best] train_sr={train_success_rate:.2%} val_sr={val_success_rate:.2%}")
            env_summary = " ".join(
                f"{env_name.replace('BabyAI-', '').replace('-v0', '')}={metrics['val_success_rate']:.0%}"
                for env_name, metrics in env_metrics.items()
            )
            if env_summary:
                print(env_summary)

    total_timesteps_requested = int(config["training"]["total_timesteps"])
    eval_freq = max(int(config["training"]["eval_freq"]), 1)
    save_freq = max(int(config["training"]["save_freq"]), 1)
    eval_max_steps = int(config["training"].get("eval_max_steps", 1000))
    next_eval_timestep = ((num_timesteps // eval_freq) + 1) * eval_freq
    next_save_timestep = ((num_timesteps // save_freq) + 1) * save_freq
    eval_log_path = Path(artifacts["logs"]) / f"{run_spec.run_name}_eval_log.jsonl"
    progress = create_progress_bar(total=total_timesteps_requested, desc=f"train {run_spec.run_name}", unit="ts")
    last_postfix: dict[str, str] = {}
    interrupted = False

    print(f"[rl] recurrent=True")
    print(f"[rl] eval_max_steps={eval_max_steps}")
    if len(run_spec.train_env_names) > 1:
        print(f"[rl] adaptive sampling envs: {', '.join(run_spec.eval_env_names)}")

    try:
        while num_timesteps < total_timesteps_requested:
            progress_remaining = max(0.0, 1.0 - (num_timesteps / max(total_timesteps_requested, 1)))
            current_learning_rate = (
                learning_rate_schedule(progress_remaining)
                if callable(learning_rate_schedule)
                else float(learning_rate_schedule)
            )
            trainer.set_learning_rate(float(current_learning_rate))
            logs = trainer.update_parameters()
            num_timesteps += int(logs["num_frames"])
            update_index += 1

            postfix = {
                "loss": f"{float(logs['loss']):.4f}",
                "policy": f"{float(logs['policy_loss']):.4f}",
                "value": f"{float(logs['value_loss']):.4f}",
                "lr": f"{float(current_learning_rate):.2e}",
            }
            if logs["return_per_episode"]:
                postfix["rew"] = f"{float(np.mean(logs['return_per_episode'])):.3f}"
            if logs["num_frames_per_episode"]:
                postfix["len"] = f"{float(np.mean(logs['num_frames_per_episode'])):.1f}"
            postfix.update(last_postfix)
            progress.n = min(num_timesteps, total_timesteps_requested)
            progress.set_postfix(**postfix)
            progress.refresh()

            if num_timesteps < next_eval_timestep:
                continue
            while next_eval_timestep <= num_timesteps:
                next_eval_timestep += eval_freq

            model.eval()
            env_metrics, train_success_rate, val_success_rate, train_mean_length, val_mean_length = _evaluate_recurrent_policy(
                model=model,
                vocab=vocab,
                env_names=run_spec.eval_env_names,
                seed=seed,
                n_eval_episodes=config["training"]["n_eval_episodes"],
                max_steps=eval_max_steps,
                device=device,
                log_prefix=f"[rl eval env @ {num_timesteps}]",
            )
            previous_best = best_val_success_rate
            best_val_success_rate = max(best_val_success_rate, float(val_success_rate))
            snapshot = EvalSnapshot(
                timesteps=num_timesteps,
                train_success_rate=train_success_rate,
                val_success_rate=val_success_rate,
                train_mean_length=train_mean_length,
                val_mean_length=val_mean_length,
                best_val_success_rate=best_val_success_rate,
                env_metrics=env_metrics,
            )
            with eval_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(snapshot), sort_keys=True) + "\n")

            last_postfix = {
                "train_sr": f"{train_success_rate:.2%}",
                "val_sr": f"{val_success_rate:.2%}",
                "best_val": f"{best_val_success_rate:.2%}",
            }
            progress.set_postfix(**(postfix | last_postfix))

            print(
                f"\n[eval @ {num_timesteps}] "
                f"train_sr={train_success_rate:.2%} "
                f"val_sr={val_success_rate:.2%} "
                f"best_val={best_val_success_rate:.2%}"
            )
            env_summary = " ".join(
                f"{env_name.replace('BabyAI-', '').replace('-v0', '')}={metrics['val_success_rate']:.0%}"
                for env_name, metrics in env_metrics.items()
            )
            if env_summary:
                print(env_summary)
            if len(run_spec.train_env_names) > 1:
                updated_weights = compute_adaptive_env_sampling_weights(
                    env_metrics,
                    uniform_alpha=float(config["curriculum"]["sampling_uniform_alpha"]),
                )
                train_env.env_method("set_sampling_weights", updated_weights)
                sampling_summary = format_sampling_summary(updated_weights)
                if sampling_summary:
                    print(f"[sampling] {sampling_summary}")

            if num_timesteps >= next_save_timestep:
                _save_recurrent_latest(
                    trainer=trainer,
                    model=model,
                    vocab=vocab,
                    model_config=model_config,
                    artifact_paths=artifact_paths,
                    run_spec=run_spec,
                    config_name=config_name,
                    model_preset=model_preset,
                    seed=seed,
                    total_timesteps_requested=total_timesteps_requested,
                    num_timesteps=num_timesteps,
                    update_index=update_index,
                    best_val_success_rate=best_val_success_rate,
                    save_tag=f"latest_eval_{num_timesteps}",
                    vocab_path=artifacts["vocab_path"],
                )
                while next_save_timestep <= num_timesteps:
                    next_save_timestep += save_freq

            if best_val_success_rate > previous_best:
                _save_recurrent_best(
                    trainer=trainer,
                    model=model,
                    vocab=vocab,
                    model_config=model_config,
                    artifact_paths=artifact_paths,
                    run_spec=run_spec,
                    config_name=config_name,
                    model_preset=model_preset,
                    seed=seed,
                    total_timesteps_requested=total_timesteps_requested,
                    num_timesteps=num_timesteps,
                    update_index=update_index,
                    best_val_success_rate=best_val_success_rate,
                )
                print(f"[milestone] new best recurrent RL holdout val success rate: {best_val_success_rate:.2%}")
            model.train()
    except KeyboardInterrupt:
        interrupted = True
        _save_recurrent_training_checkpoint(
            artifact_paths.interrupted_state_path,
            trainer=trainer,
            model_config=model_config,
            vocab=vocab,
            run_spec=run_spec,
            config_name=config_name,
            model_preset=model_preset,
            seed=seed,
            total_timesteps_requested=total_timesteps_requested,
            num_timesteps=num_timesteps,
            update_index=update_index,
            best_val_success_rate=best_val_success_rate,
            save_tag="interrupt",
        )
        _save_recurrent_latest(
            trainer=trainer,
            model=model,
            vocab=vocab,
            model_config=model_config,
            artifact_paths=artifact_paths,
            run_spec=run_spec,
            config_name=config_name,
            model_preset=model_preset,
            seed=seed,
            total_timesteps_requested=total_timesteps_requested,
            num_timesteps=num_timesteps,
            update_index=update_index,
            best_val_success_rate=best_val_success_rate,
            save_tag="interrupt",
            vocab_path=artifacts["vocab_path"],
        )
    finally:
        progress.n = min(num_timesteps, total_timesteps_requested)
        progress.refresh()
        progress.close()
        train_env.close()

    if not interrupted:
        _save_recurrent_latest(
            trainer=trainer,
            model=model,
            vocab=vocab,
            model_config=model_config,
            artifact_paths=artifact_paths,
            run_spec=run_spec,
            config_name=config_name,
            model_preset=model_preset,
            seed=seed,
            total_timesteps_requested=total_timesteps_requested,
            num_timesteps=num_timesteps,
            update_index=update_index,
            best_val_success_rate=best_val_success_rate,
            save_tag="final",
            vocab_path=artifacts["vocab_path"],
        )

    return artifact_paths.latest_export_path


def train_feedforward(
    *,
    run_spec: RunSpec,
    config_name: str,
    model_preset: str | None,
    seed: int,
    total_timesteps: int | None,
    resume_path: str | None,
    warm_start_path: str | None,
    bc_checkpoint_path: str | None,
    rebuild_vocab: bool = False,
) -> Path:

    config = get_config(config_name, model_preset=model_preset)
    if total_timesteps is not None:
        config["training"]["total_timesteps"] = total_timesteps
    config["training"]["seed"] = seed

    artifacts = config["artifacts"]
    os.makedirs(artifacts["checkpoints"], exist_ok=True)
    os.makedirs(artifacts["logs"], exist_ok=True)
    os.makedirs(artifacts["exports"], exist_ok=True)

    vocab = _load_or_build_global_vocab(config, seed, rebuild=rebuild_vocab)
    artifact_paths = _artifact_paths(run_spec.run_name, "ppo", artifacts)
    train_env = make_vec_env(
        env_name=run_spec.train_env_names if len(run_spec.train_env_names) > 1 else run_spec.train_env_names[0],
        vocab=vocab,
        n_envs=config["training"]["n_envs"],
        seed=seed,
        monitor_dir=os.path.join(artifacts["checkpoints"], "monitors", run_spec.run_name),
        sampling_weights=run_spec.sampling_weights,
    )
    model_config = _resolve_feedforward_model_config(
        config=config,
        warm_start_path=warm_start_path,
        train_env=train_env,
    )

    if resume_path:
        if checkpoint_looks_like_torch_export(resume_path):
            raise RuntimeError("--resume for PPO expects an SB3 .zip checkpoint, not a torch export.")
        model = PPO.load(resume_path, env=train_env, device=config["training"]["device"])
        model.set_env(train_env)
    else:
        model = _create_ppo_model(config, train_env, vocab, model_config, seed)
        if warm_start_path:
            _initialize_feedforward_policy(model, warm_start_path, train_env, config["training"]["device"])
        if bc_checkpoint_path:
            _initialize_feedforward_policy(model, bc_checkpoint_path, train_env, config["training"]["device"])
        if warm_start_path and len(run_spec.train_env_names) > 1:
            initial_weights = _run_initial_warm_start_eval(
                model=model,
                run_spec=run_spec,
                vocab=vocab,
                config=config,
                seed=seed,
            )
            if initial_weights is not None:
                train_env.env_method("set_sampling_weights", initial_weights)

    def save_on_eval(**eval_metrics):
        if len(run_spec.train_env_names) > 1 and eval_metrics.get("env_metrics"):
            updated_weights = compute_adaptive_env_sampling_weights(
                eval_metrics["env_metrics"],
                uniform_alpha=float(config["curriculum"]["sampling_uniform_alpha"]),
            )
            train_env.env_method("set_sampling_weights", updated_weights)
        if eval_metrics["timesteps"] % config["training"]["save_freq"] == 0:
            _save_feedforward_latest(
                model=model,
                vocab=vocab,
                model_config=model_config,
                artifact_paths=artifact_paths,
                run_spec=run_spec,
                config_name=config_name,
                model_preset=model_preset,
                seed=seed,
                total_timesteps_requested=config["training"]["total_timesteps"],
                save_tag=f"latest_eval_{eval_metrics['timesteps']}",
                vocab_path=artifacts["vocab_path"],
            )
        if eval_metrics.get("is_new_best"):
            _save_feedforward_best(
                model=model,
                vocab=vocab,
                model_config=model_config,
                artifact_paths=artifact_paths,
            )

    progress_callback = ProgressEvalCallback(
        run_name=run_spec.run_name,
        env_names=run_spec.eval_env_names,
        vocab=vocab,
        seed=seed,
        total_timesteps=config["training"]["total_timesteps"],
        eval_freq=config["training"]["eval_freq"],
        n_eval_episodes=config["training"]["n_eval_episodes"],
        sampling_uniform_alpha=float(config["curriculum"]["sampling_uniform_alpha"]),
        log_dir=artifacts["logs"],
        save_on_eval=save_on_eval,
        verbose=1,
    )

    interrupted = False
    try:
        learn_kwargs = {
            "total_timesteps": config["training"]["total_timesteps"],
            "callback": CallbackList([progress_callback]),
        }
        if resume_path:
            remaining_timesteps = max(config["training"]["total_timesteps"] - model.num_timesteps, 0)
            learn_kwargs["total_timesteps"] = remaining_timesteps
            learn_kwargs["reset_num_timesteps"] = False
        model.learn(**learn_kwargs)
    except KeyboardInterrupt:
        interrupted = True
        model.save(str(artifact_paths.interrupted_state_path))
        _save_feedforward_latest(
            model=model,
            vocab=vocab,
            model_config=model_config,
            artifact_paths=artifact_paths,
            run_spec=run_spec,
            config_name=config_name,
            model_preset=model_preset,
            seed=seed,
            total_timesteps_requested=config["training"]["total_timesteps"],
            save_tag="interrupt",
            vocab_path=artifacts["vocab_path"],
        )

    if not interrupted:
        _save_feedforward_latest(
            model=model,
            vocab=vocab,
            model_config=model_config,
            artifact_paths=artifact_paths,
            run_spec=run_spec,
            config_name=config_name,
            model_preset=model_preset,
            seed=seed,
            total_timesteps_requested=config["training"]["total_timesteps"],
            save_tag="final",
            vocab_path=artifacts["vocab_path"],
        )

    train_env.close()
    return artifact_paths.latest_export_path


def _selector_args(run_spec: RunSpec) -> str:
    if run_spec.all_envs:
        selector = "--all-envs"
        if run_spec.run_name != "leaderboard_multitask":
            selector += f" --run-name {run_spec.run_name}"
        return selector
    if len(run_spec.train_env_names) == 1:
        selector = f"--env {run_spec.train_env_names[0]}"
        if run_spec.run_name != _env_stem(run_spec.train_env_names[0]):
            selector += f" --run-name {run_spec.run_name}"
        return selector
    return f"--envs {','.join(run_spec.train_env_names)} --run-name {run_spec.run_name}"


def _print_next_steps(
    run_spec: RunSpec,
    config_name: str,
    seed: int,
    total_timesteps: int,
    artifacts: ArtifactPaths,
    *,
    recurrent: bool,
) -> None:
    print(f"\n[artifacts] latest checkpoint: {artifacts.latest_state_path}")
    print(f"[artifacts] exported submission policy: {artifacts.latest_export_path}")
    print(f"[artifacts] best checkpoint: {artifacts.best_state_path}")
    print(f"[artifacts] best exported policy: {artifacts.best_export_path}")
    print("[next] export submission:")
    print(f"python3 -m sneddy_baby_ai.cli.export_submission --checkpoint {artifacts.latest_export_path}")
    print("[next] export best submission:")
    print(f"python3 -m sneddy_baby_ai.cli.export_submission --checkpoint {artifacts.best_export_path}")
    print("[next] validate submission:")
    print("python3 babyai-ml8103-leaderboard-2026/evaluation/validate_submission.py demo_submission.zip")
    print("[next] local leaderboard evaluation:")
    print(
        "python3 babyai-ml8103-leaderboard-2026/evaluation/evaluate_submission.py "
        "--submission demo_submission.zip --seed-dir babyai-ml8103-leaderboard-2026/eval_seeds"
    )
    print("[next] resume training:")
    recurrent_flag = " --recurrent" if recurrent else ""
    print(
        f"python3 -m sneddy_baby_ai.cli.train_rl {_selector_args(run_spec)} --config {config_name} --seed {seed} "
        f"--timesteps {total_timesteps}{recurrent_flag} --resume {artifacts.latest_state_path}"
    )


def run_train_rl(
    *,
    env: str | None = None,
    envs: str | None = None,
    all_envs: bool = False,
    easy: bool = False,
    moderate: bool = False,
    hard: bool = False,
    run_name: str | None = None,
    config_name: str = "default",
    model_preset: str | None = None,
    seed: int = 42,
    total_timesteps: int | None = None,
    resume_path: str | None = None,
    warm_start_path: str | None = None,
    bc_checkpoint_path: str | None = None,
    bc_demo_paths: list[str] | None = None,
    bc_min_sampling_proba: float | None = None,
    recurrent: bool = False,
    rebuild_vocab: bool = False,
) -> Path:
    if resume_path and warm_start_path:
        raise RuntimeError("Use either --resume or --warm-start, not both.")

    config = get_config(config_name, model_preset=model_preset)
    selector_args = SimpleNamespace(
        env=env,
        envs=envs,
        all_envs=all_envs,
        easy=easy,
        moderate=moderate,
        hard=hard,
        run_name=run_name,
    )
    run_spec = resolve_run_spec(selector_args, config)

    resolved_bc_checkpoint_path = _ensure_bc_checkpoint(
        run_spec=run_spec,
        config_name=config_name,
        model_preset=model_preset,
        config=config,
        seed=seed,
        bc_checkpoint_path=bc_checkpoint_path,
        bc_demo_paths=bc_demo_paths,
        warm_start_path=warm_start_path,
        eval_env_names=run_spec.eval_env_names,
        bc_min_sampling_proba=bc_min_sampling_proba,
        recurrent=recurrent,
    )
    effective_warm_start_path = None if resolved_bc_checkpoint_path is not None else warm_start_path

    export_path = (
        train_recurrent(
            run_spec=run_spec,
            config_name=config_name,
            model_preset=model_preset,
            seed=seed,
            total_timesteps=total_timesteps,
            resume_path=resume_path,
            warm_start_path=effective_warm_start_path,
            bc_checkpoint_path=resolved_bc_checkpoint_path,
            rebuild_vocab=rebuild_vocab,
        )
        if recurrent
        else train_feedforward(
            run_spec=run_spec,
            config_name=config_name,
            model_preset=model_preset,
            seed=seed,
            total_timesteps=total_timesteps,
            resume_path=resume_path,
            warm_start_path=effective_warm_start_path,
            bc_checkpoint_path=resolved_bc_checkpoint_path,
            rebuild_vocab=rebuild_vocab,
        )
    )

    resolved_total_timesteps = int(total_timesteps or config["training"]["total_timesteps"])
    artifact_paths = _artifact_paths(run_spec.run_name, "ppo_recurrent" if recurrent else "ppo", config["artifacts"])
    _print_next_steps(run_spec, config_name, seed, resolved_total_timesteps, artifact_paths, recurrent=recurrent)
    return export_path
