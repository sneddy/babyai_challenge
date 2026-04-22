"""Behavior-cloning training service."""

from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
from tqdm.auto import tqdm

from ...auxiliary.losses import compute_auxiliary_loss
from ...auxiliary.specs import AuxHeadSpec, apply_aux_weight, get_aux_specs
from ..evaluation import (
    evaluate_env_suite,
    evaluate_torch_policy_on_seeds,
)
from ...config.loader import get_config
from ...data.demos import DemoBatch, DemoDataset, load_demo_file, load_demo_files
from ...data.vocabulary import MissionVocabulary
from ...envs.catalog import ALL_LEADERBOARD_ENVS
from ...models.core import (
    BabyAIFeedForwardPolicy,
    BabyAIRecurrentPolicy,
    ModelConfig,
    load_exported_checkpoint,
    save_exported_checkpoint,
)
from ...models.transfer import (
    checkpoint_looks_like_torch_export,
    copy_matching_state_dict,
    initialize_torch_policy_from_checkpoint,
)


def _save_bc_checkpoint(
    *,
    path: str | Path,
    model,
    model_config: ModelConfig,
    vocab: MissionVocabulary,
    recurrent: bool,
    aux_specs: tuple[AuxHeadSpec, ...] = (),
    aux_preset: str | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    export_state = model.export_state_dict() if hasattr(model, "export_state_dict") else model.state_dict()
    aux_state = model.auxiliary_state_dict() if hasattr(model, "auxiliary_state_dict") else {}
    save_exported_checkpoint(
        str(checkpoint_path),
        policy_state=export_state,
        model_config=model_config,
        vocab_payload=vocab.to_dict(),
        recurrent=recurrent,
        aux_state=aux_state,
        aux_config={
            "preset": aux_preset,
            "heads": [spec.to_dict() for spec in aux_specs],
        } if aux_specs else None,
    )


def _best_bc_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_best{checkpoint_path.suffix}")


def _env_stem(env_name: str) -> str:
    return env_name.replace("BabyAI-", "").replace("-v0", "").lower()


def parse_eval_envs(raw_envs: str | None) -> list[str]:
    if not raw_envs:
        return []
    env_names = []
    for item in raw_envs.split(","):
        token = item.strip()
        if token:
            env_names.append(token)
    return list(dict.fromkeys(env_names))


def _demo_paths(demo_path: str | list[str]) -> list[Path]:
    if isinstance(demo_path, (list, tuple)):
        return [Path(path) for path in demo_path]
    return [Path(demo_path)]


def _infer_env_name_from_demo_path(path: str | Path) -> str | None:
    stem_to_env = {_env_stem(env_name): env_name for env_name in ALL_LEADERBOARD_ENVS}
    return stem_to_env.get(Path(path).stem.lower())


def _infer_eval_env_names(demo_path: str | list[str]) -> list[str]:
    inferred = []
    for path in _demo_paths(demo_path):
        env_name = _infer_env_name_from_demo_path(path)
        if env_name is None:
            return []
        inferred.append(env_name)
    return list(dict.fromkeys(inferred))


def _load_demo_batches_by_env(demo_path: str | list[str]) -> dict[str, DemoBatch]:
    batches_by_env: dict[str, DemoBatch] = {}
    for path in _demo_paths(demo_path):
        env_name = _infer_env_name_from_demo_path(path)
        if env_name is None:
            continue
        batches_by_env[env_name] = load_demo_file(path)
    return batches_by_env


def _count_demo_episodes(batch: DemoBatch, indices: list[int] | None = None) -> int:
    done = batch.done if indices is None else batch.done[np.asarray(indices, dtype=np.int64)]
    return int(np.asarray(done, dtype=np.int64).sum())


def _episode_ranges(batch: DemoBatch) -> list[tuple[int, int]]:
    done = np.asarray(batch.done, dtype=np.int64)
    end_indices = np.flatnonzero(done == 1)
    if end_indices.size == 0:
        return [(0, int(done.shape[0]))] if done.shape[0] else []

    ranges: list[tuple[int, int]] = []
    start = 0
    for end_index in end_indices.tolist():
        ranges.append((start, end_index + 1))
        start = end_index + 1
    if start < int(done.shape[0]):
        ranges.append((start, int(done.shape[0])))
    return ranges


class EpisodeDataset:
    """Episode-wise BC dataset for recurrent imitation learning."""

    def __init__(self, batch: DemoBatch, episode_ids: list[int] | None = None) -> None:
        self.batch = batch
        self.ranges = _episode_ranges(batch)
        self.episode_ids = list(range(len(self.ranges))) if episode_ids is None else list(episode_ids)

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        start, end = self.ranges[self.episode_ids[index]]
        sample = {
            "image": self.batch.image[start:end],
            "mission_tokens": self.batch.mission_tokens[start:end],
            "mission_mask": self.batch.mission_mask[start:end],
            "action": self.batch.action[start:end],
            "done": self.batch.done[start:end],
        }
        if self.batch.aux_targets:
            sample["aux_targets"] = {name: values[start:end] for name, values in self.batch.aux_targets.items()}
        if self.batch.aux_masks:
            sample["aux_masks"] = {name: values[start:end] for name, values in self.batch.aux_masks.items()}
        return sample


def _collate_episode_batch(samples):
    if not samples:
        raise ValueError("samples must be non-empty")

    batch_size = len(samples)
    lengths = torch.as_tensor([sample["action"].shape[0] for sample in samples], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch: dict[str, torch.Tensor] = {"lengths": lengths}

    for key in ["image", "mission_tokens", "mission_mask", "action", "done"]:
        first = torch.as_tensor(samples[0][key])
        padded = torch.zeros((batch_size, max_len, *first.shape[1:]), dtype=first.dtype)
        for sample_index, sample in enumerate(samples):
            value = torch.as_tensor(sample[key])
            padded[sample_index, : value.shape[0]] = value
        batch[key] = padded

    aux_target_names = sorted(samples[0].get("aux_targets", {}))
    if aux_target_names:
        batch["aux_targets"] = {}
        batch["aux_masks"] = {}
        for name in aux_target_names:
            first_target = torch.as_tensor(samples[0]["aux_targets"][name])
            padded_targets = torch.zeros((batch_size, max_len, *first_target.shape[1:]), dtype=first_target.dtype)
            first_mask = torch.as_tensor(samples[0]["aux_masks"][name])
            padded_masks = torch.zeros((batch_size, max_len, *first_mask.shape[1:]), dtype=first_mask.dtype)
            for sample_index, sample in enumerate(samples):
                target_value = torch.as_tensor(sample["aux_targets"][name])
                mask_value = torch.as_tensor(sample["aux_masks"][name])
                padded_targets[sample_index, : target_value.shape[0]] = target_value
                padded_masks[sample_index, : mask_value.shape[0]] = mask_value
            batch["aux_targets"][name] = padded_targets
            batch["aux_masks"][name] = padded_masks

    return batch


def _mean_episode_length(batch: DemoBatch) -> float:
    episode_count = _count_demo_episodes(batch)
    if episode_count <= 0:
        return float(max(int(batch.action.shape[0]), 1))
    return float(batch.action.shape[0]) / float(episode_count)


def _resolve_recurrent_episode_batch_size(
    *,
    target_step_batch_size: int,
    demo_batches: dict[str, DemoBatch] | None = None,
    combined_batch: DemoBatch | None = None,
) -> int:
    if target_step_batch_size <= 0:
        return 1

    if demo_batches:
        total_steps = sum(int(batch.action.shape[0]) for batch in demo_batches.values())
        total_episodes = sum(_count_demo_episodes(batch) for batch in demo_batches.values())
        mean_length = float(total_steps) / float(max(total_episodes, 1))
    elif combined_batch is not None:
        mean_length = _mean_episode_length(combined_batch)
    else:
        mean_length = 16.0

    return max(1, int(target_step_batch_size / max(mean_length, 1.0)))


def _build_weighted_train_loader(
    *,
    demo_batches_by_env: dict[str, DemoBatch],
    datasets_by_env: dict[str, DemoDataset],
    env_metrics: dict[str, dict[str, float]] | None,
    batch_size: int,
    min_sampling_proba: float,
    sampling_episode_budget_per_task: int | None,
    generator,
):
    env_names = list(datasets_by_env)
    if not env_names:
        raise ValueError("datasets_by_env must be non-empty")

    selected_datasets = []
    selected_episode_counts: dict[str, int] = {}
    sampling_ratios: dict[str, float] = {}

    for env_name in env_names:
        dataset = datasets_by_env[env_name]
        batch = demo_batches_by_env[env_name]
        dataset_len = len(dataset)
        if dataset_len <= 0:
            continue
        episode_ranges = _episode_ranges(batch)
        total_episodes = len(episode_ranges)
        if total_episodes <= 0:
            continue
        effective_total_episodes = total_episodes
        if sampling_episode_budget_per_task is not None:
            effective_total_episodes = min(effective_total_episodes, int(sampling_episode_budget_per_task))
        success_rate = float((env_metrics or {}).get(env_name, {}).get("val_success_rate", 0.0))
        sampling_ratio = (1.0 - success_rate) * (1.0 - min_sampling_proba) + min_sampling_proba
        sampling_ratio = float(np.clip(sampling_ratio, min_sampling_proba, 1.0))
        sample_episode_count = max(
            1,
            min(effective_total_episodes, int(math.ceil(effective_total_episodes * sampling_ratio))),
        )
        permutation = torch.randperm(total_episodes, generator=generator).tolist()
        selected_episode_ids = permutation[:sample_episode_count]
        selected_indices: list[int] = []
        for episode_id in selected_episode_ids:
            start, end = episode_ranges[episode_id]
            selected_indices.extend(range(start, end))
        selected_datasets.append(Subset(dataset, selected_indices))
        selected_episode_counts[env_name] = sample_episode_count
        sampling_ratios[env_name] = sampling_ratio

    if not selected_datasets:
        raise ValueError("No BC samples were selected for this epoch")

    concat_dataset = ConcatDataset(selected_datasets)
    # Shuffle at the combined dataset level so batches mix tasks instead of
    # iterating through one task subset after another.
    loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    loader.selected_episode_counts = selected_episode_counts  # type: ignore[attr-defined]
    loader.sampling_ratios = sampling_ratios  # type: ignore[attr-defined]
    loader.selected_total_episodes = sum(selected_episode_counts.values())  # type: ignore[attr-defined]
    return loader


def _build_weighted_episode_train_loader(
    *,
    demo_batches_by_env: dict[str, DemoBatch],
    datasets_by_env: dict[str, EpisodeDataset],
    env_metrics: dict[str, dict[str, float]] | None,
    batch_size: int,
    min_sampling_proba: float,
    sampling_episode_budget_per_task: int | None,
    generator,
):
    env_names = list(datasets_by_env)
    if not env_names:
        raise ValueError("datasets_by_env must be non-empty")

    selected_datasets = []
    selected_episode_counts: dict[str, int] = {}
    sampling_ratios: dict[str, float] = {}

    for env_name in env_names:
        dataset = datasets_by_env[env_name]
        total_episodes = len(dataset)
        if total_episodes <= 0:
            continue
        effective_total_episodes = total_episodes
        if sampling_episode_budget_per_task is not None:
            effective_total_episodes = min(effective_total_episodes, int(sampling_episode_budget_per_task))
        success_rate = float((env_metrics or {}).get(env_name, {}).get("val_success_rate", 0.0))
        sampling_ratio = (1.0 - success_rate) * (1.0 - min_sampling_proba) + min_sampling_proba
        sampling_ratio = float(np.clip(sampling_ratio, min_sampling_proba, 1.0))
        sample_episode_count = max(
            1,
            min(effective_total_episodes, int(math.ceil(effective_total_episodes * sampling_ratio))),
        )
        permutation = torch.randperm(total_episodes, generator=generator).tolist()
        selected_episode_ids = permutation[:sample_episode_count]
        selected_datasets.append(Subset(dataset, selected_episode_ids))
        selected_episode_counts[env_name] = sample_episode_count
        sampling_ratios[env_name] = sampling_ratio

    if not selected_datasets:
        raise ValueError("No BC episodes were selected for this epoch")

    concat_dataset = ConcatDataset(selected_datasets)
    loader = DataLoader(
        concat_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=_collate_episode_batch,
    )
    loader.selected_episode_counts = selected_episode_counts  # type: ignore[attr-defined]
    loader.sampling_ratios = sampling_ratios  # type: ignore[attr-defined]
    loader.selected_total_episodes = sum(selected_episode_counts.values())  # type: ignore[attr-defined]
    return loader


def _validate_aux_supervision(batch: DemoBatch, aux_specs: tuple[AuxHeadSpec, ...], *, source: str) -> None:
    if not aux_specs:
        return
    required_names = {spec.name for spec in aux_specs}
    missing_targets = sorted(required_names.difference(batch.aux_targets))
    missing_masks = sorted(required_names.difference(batch.aux_masks))
    if missing_targets or missing_masks:
        raise RuntimeError(
            f"Missing auxiliary supervision in {source}: "
            f"targets={missing_targets or 'ok'} masks={missing_masks or 'ok'}"
        )


def _init_aux_metric_totals(aux_specs: tuple[AuxHeadSpec, ...]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for spec in aux_specs:
        totals[f"{spec.name}_weighted_loss_sum"] = 0.0
        totals[f"{spec.name}_weighted_acc_sum"] = 0.0
        totals[f"{spec.name}_active"] = 0.0
    return totals


def _accumulate_aux_metric_totals(
    totals: dict[str, float],
    batch_metrics: dict[str, float],
    aux_specs: tuple[AuxHeadSpec, ...],
) -> None:
    for spec in aux_specs:
        active = float(batch_metrics.get(f"{spec.name}_active", 0.0))
        totals[f"{spec.name}_active"] += active
        totals[f"{spec.name}_weighted_loss_sum"] += float(batch_metrics.get(f"{spec.name}_loss", 0.0)) * active
        totals[f"{spec.name}_weighted_acc_sum"] += float(batch_metrics.get(f"{spec.name}_acc", 0.0)) * active


def _finalize_aux_metric_totals(
    totals: dict[str, float],
    aux_specs: tuple[AuxHeadSpec, ...],
    denom_count: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    denom = max(float(denom_count), 1.0)
    for spec in aux_specs:
        active = totals.get(f"{spec.name}_active", 0.0)
        if active > 0:
            metrics[f"{spec.name}_loss"] = totals[f"{spec.name}_weighted_loss_sum"] / active
            metrics[f"{spec.name}_acc"] = totals[f"{spec.name}_weighted_acc_sum"] / active
        else:
            metrics[f"{spec.name}_loss"] = 0.0
            metrics[f"{spec.name}_acc"] = 0.0
        metrics[f"{spec.name}_active"] = active
        metrics[f"{spec.name}_active_fraction"] = active / denom
    return metrics


def _print_aux_head_metrics(prefix: str, metrics: dict[str, float], aux_specs: tuple[AuxHeadSpec, ...]) -> None:
    if not aux_specs:
        return
    for spec in aux_specs:
        print(
            f"{prefix} {spec.name}: "
            f"loss={metrics.get(f'{spec.name}_loss', 0.0):.4f} "
            f"acc={metrics.get(f'{spec.name}_acc', 0.0):.2%}"
        )


def _print_aux_head_train_val_metrics(
    prefix: str,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    aux_specs: tuple[AuxHeadSpec, ...],
) -> None:
    if not aux_specs:
        return
    for spec in aux_specs:
        print(
            f"{prefix} {spec.name}: "
            f"train_acc={train_metrics.get(f'{spec.name}_acc', 0.0):.2%} "
            f"train_loss={train_metrics.get(f'{spec.name}_loss', 0.0):.4f} "
            f"val_acc={val_metrics.get(f'{spec.name}_acc', 0.0):.2%} "
            f"val_loss={val_metrics.get(f'{spec.name}_loss', 0.0):.4f}"
        )


def _aggregate_env_aux_metrics(
    env_metrics: dict[str, dict[str, float]],
    aux_specs: tuple[AuxHeadSpec, ...],
    split_prefix: str,
) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for spec in aux_specs:
        active_key = f"{split_prefix}_{spec.name}_active"
        loss_key = f"{split_prefix}_{spec.name}_loss"
        acc_key = f"{split_prefix}_{spec.name}_acc"
        total_active = sum(float(metrics.get(active_key, 0.0)) for metrics in env_metrics.values())
        if total_active > 0:
            aggregated[f"{spec.name}_loss"] = sum(
                float(metrics.get(loss_key, 0.0)) * float(metrics.get(active_key, 0.0))
                for metrics in env_metrics.values()
            ) / total_active
            aggregated[f"{spec.name}_acc"] = sum(
                float(metrics.get(acc_key, 0.0)) * float(metrics.get(active_key, 0.0))
                for metrics in env_metrics.values()
            ) / total_active
        else:
            aggregated[f"{spec.name}_loss"] = 0.0
            aggregated[f"{spec.name}_acc"] = 0.0
        aggregated[f"{spec.name}_active"] = total_active
    return aggregated


def _action_weights_from_aux_targets(
    *,
    aux_targets: dict[str, torch.Tensor] | None,
    aux_masks: dict[str, torch.Tensor] | None,
    action_reweight_on_holding_target: float,
    reference_tensor: torch.Tensor,
) -> torch.Tensor:
    weights = torch.ones_like(reference_tensor, dtype=torch.float)
    if action_reweight_on_holding_target <= 1.0:
        return weights
    if aux_targets is None or aux_masks is None:
        return weights
    if "holding_target_object" not in aux_targets or "holding_target_object" not in aux_masks:
        return weights
    holding = aux_targets["holding_target_object"].reshape_as(reference_tensor).float()
    active = aux_masks["holding_target_object"].reshape_as(reference_tensor).float()
    weights = weights + (float(action_reweight_on_holding_target) - 1.0) * holding * active
    return weights


def _compute_feedforward_bc_batch_metrics(
    model,
    batch,
    aux_specs: tuple[AuxHeadSpec, ...] = (),
    action_reweight_on_holding_target: float = 1.0,
):
    obs = {
        "image": batch["image"].long(),
        "mission_tokens": batch["mission_tokens"].long(),
        "mission_mask": batch["mission_mask"].long(),
    }
    if aux_specs:
        logits, _value, aux_predictions = model.forward_with_aux(obs)
    else:
        logits, _value = model(obs)
        aux_predictions = {}

    targets = batch["action"].long()
    action_weights = _action_weights_from_aux_targets(
        aux_targets=batch.get("aux_targets"),
        aux_masks=batch.get("aux_masks"),
        action_reweight_on_holding_target=action_reweight_on_holding_target,
        reference_tensor=targets,
    )
    per_item_action_loss = F.cross_entropy(logits, targets, reduction="none")
    action_loss = (per_item_action_loss * action_weights).sum() / action_weights.sum().clamp_min(1.0)
    aux_loss, aux_metrics = compute_auxiliary_loss(
        aux_predictions=aux_predictions if aux_specs else None,
        aux_targets=batch.get("aux_targets"),
        aux_masks=batch.get("aux_masks"),
        aux_specs=aux_specs,
        device=logits.device,
    )
    total_loss = action_loss + aux_loss
    predictions = logits.argmax(dim=-1)
    batch_correct = int((predictions == targets).sum().item())
    batch_count = int(targets.numel())
    metrics = {
        "action_loss": float(action_loss.item()),
        **aux_metrics,
    }
    return total_loss, batch_correct, batch_count, metrics


def _compute_recurrent_bc_batch_metrics(
    model,
    batch,
    aux_specs: tuple[AuxHeadSpec, ...] = (),
    action_reweight_on_holding_target: float = 1.0,
):
    lengths = batch["lengths"].long()
    if lengths.numel() == 0:
        zero = torch.zeros((), device=batch["image"].device)
        return zero, 0, 0, {"action_loss": 0.0, "aux_loss": 0.0, "aux_active": 0.0}

    batch_size = int(lengths.shape[0])
    max_len = int(lengths.max().item())
    state = model.initial_state(batch_size=batch_size, device=batch["image"].device)
    total_action_loss = torch.zeros((), device=batch["image"].device)
    total_action_weight = torch.zeros((), device=batch["image"].device)
    total_aux_loss = torch.zeros((), device=batch["image"].device)
    total_correct = 0
    total_count = 0
    total_aux_active = 0.0
    aux_metric_totals = _init_aux_metric_totals(aux_specs)

    for step_index in range(max_len):
        active = lengths > step_index
        if not torch.any(active):
            break

        obs = {
            "image": batch["image"][:, step_index].long(),
            "mission_tokens": batch["mission_tokens"][:, step_index].long(),
            "mission_mask": batch["mission_mask"][:, step_index].long(),
        }
        if aux_specs:
            logits, _value, next_state, aux_predictions = model.forward_with_aux(obs, state=state)
        else:
            logits, _value, next_state = model(obs, state=state)
            aux_predictions = {}
        targets = batch["action"][active, step_index].long()
        action_weights = _action_weights_from_aux_targets(
            aux_targets={
                name: batch["aux_targets"][name][active, step_index]
                for name in batch.get("aux_targets", {})
            } if batch.get("aux_targets") else None,
            aux_masks={
                name: batch["aux_masks"][name][active, step_index]
                for name in batch.get("aux_masks", {})
            } if batch.get("aux_masks") else None,
            action_reweight_on_holding_target=action_reweight_on_holding_target,
            reference_tensor=targets,
        )
        step_action_loss = F.cross_entropy(logits[active], targets, reduction="none")
        total_action_loss = total_action_loss + (step_action_loss * action_weights).sum()
        total_action_weight = total_action_weight + action_weights.sum()
        predictions = logits[active].argmax(dim=-1)
        total_correct += int((predictions == targets).sum().item())
        total_count += int(targets.numel())
        if aux_specs:
            step_aux_loss, step_aux_metrics = compute_auxiliary_loss(
                aux_predictions={name: prediction[active] for name, prediction in aux_predictions.items()},
                aux_targets={
                    name: batch["aux_targets"][name][active, step_index]
                    for name in batch.get("aux_targets", {})
                },
                aux_masks={
                    name: batch["aux_masks"][name][active, step_index]
                    for name in batch.get("aux_masks", {})
                },
                aux_specs=aux_specs,
                device=logits.device,
            )
            total_aux_loss = total_aux_loss + step_aux_loss * max(int(targets.numel()), 1)
            total_aux_active += step_aux_metrics.get("aux_active", 0.0)
            _accumulate_aux_metric_totals(aux_metric_totals, step_aux_metrics, aux_specs)

        active_mask = active.unsqueeze(-1).to(next_state[0].dtype)
        state = (
            next_state[0] * active_mask + state[0] * (1.0 - active_mask),
            next_state[1] * active_mask + state[1] * (1.0 - active_mask),
        )

    action_loss = total_action_loss / total_action_weight.clamp_min(1.0)
    aux_loss = total_aux_loss / max(total_count, 1)
    total_loss = action_loss + aux_loss
    metrics = {
        "action_loss": float(action_loss.item()),
        "aux_loss": float(aux_loss.item()),
        "aux_active": float(total_aux_active),
    }
    metrics.update(_finalize_aux_metric_totals(aux_metric_totals, aux_specs, total_count))
    return total_loss, total_correct, total_count, metrics


def _evaluate_bc_policy(
    *,
    model,
    vocab,
    env_names: list[str],
    seed: int,
    n_eval_episodes: int,
    max_steps: int,
    log_prefix: str | None = None,
):
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
        predictor=lambda env_name, seeds, max_steps: evaluate_torch_policy_on_seeds(
            model,
            env_name,
            vocab,
            seeds,
            device="cpu",
            max_steps=max_steps,
        ),
        env_names=env_names,
        vocab=vocab,
        seed=seed,
        n_eval_episodes=n_eval_episodes,
        max_steps=max_steps,
        on_env_complete=_print_env_metric,
    )


def _initialize_bc_model(model, checkpoint_path: str, device: str = "cpu") -> None:
    if checkpoint_looks_like_torch_export(checkpoint_path):
        initialize_torch_policy_from_checkpoint(model, checkpoint_path, map_location=device)
        payload = load_exported_checkpoint(checkpoint_path, map_location=device)
        if hasattr(model, "load_auxiliary_state"):
            model.load_auxiliary_state(payload.get("aux_state"))
        return

    source_model = PPO.load(checkpoint_path, device=device)
    copy_matching_state_dict(model.encoder, source_model.policy.features_extractor.core.state_dict())
    copy_matching_state_dict(model.actor, source_model.policy.action_net.state_dict())
    copy_matching_state_dict(model.critic, source_model.policy.value_net.state_dict())


def train_bc(
    demo_path: str | list[str],
    checkpoint_path: str,
    vocab_path: str,
    config_name: str = "default",
    model_preset: str | None = None,
    seed: int = 42,
    warm_start_path: str | None = None,
    eval_env_names: list[str] | None = None,
    min_sampling_proba: float | None = None,
    recurrent: bool = False,
    aux_preset: str | None = None,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = get_config(config_name, model_preset=model_preset)
    aux_specs = apply_aux_weight(get_aux_specs(aux_preset), config["bc"].get("aux_weight"))
    action_reweight_on_holding_target = float(config["bc"].get("action_reweight_on_holding_target", 1.0))
    if min_sampling_proba is not None:
        config["bc"]["min_sampling_proba"] = float(min_sampling_proba)
    vocab = MissionVocabulary.load(vocab_path)
    combined_demos = load_demo_files([str(path) for path in _demo_paths(demo_path)])
    _validate_aux_supervision(combined_demos, aux_specs, source="combined demos")
    resolved_eval_env_names = list(eval_env_names) if eval_env_names is not None else _infer_eval_env_names(demo_path)
    use_holdout_eval = bool(resolved_eval_env_names)
    demo_batches_by_env = _load_demo_batches_by_env(demo_path)
    for env_name, batch in demo_batches_by_env.items():
        _validate_aux_supervision(batch, aux_specs, source=env_name)
    adaptive_demo_sampling = use_holdout_eval and len(demo_batches_by_env) > 1
    weighted_sampler_generator = torch.Generator().manual_seed(seed)
    recurrent_episode_batch_size = _resolve_recurrent_episode_batch_size(
        target_step_batch_size=int(config["bc"]["batch_size"]),
        demo_batches=demo_batches_by_env if demo_batches_by_env else None,
        combined_batch=combined_demos,
    ) if recurrent else None

    if use_holdout_eval:
        if adaptive_demo_sampling:
            datasets_by_env = (
                {env_name: EpisodeDataset(batch) for env_name, batch in demo_batches_by_env.items()}
                if recurrent
                else {env_name: DemoDataset(batch) for env_name, batch in demo_batches_by_env.items()}
            )
            train_loader = None
        else:
            dataset = EpisodeDataset(combined_demos) if recurrent else DemoDataset(combined_demos)
            train_loader = DataLoader(
                dataset,
                batch_size=recurrent_episode_batch_size if recurrent else config["bc"]["batch_size"],
                shuffle=True,
                collate_fn=_collate_episode_batch if recurrent else None,
            )
        val_loader = None
    else:
        dataset = EpisodeDataset(combined_demos) if recurrent else DemoDataset(combined_demos)
        val_size = max(1, int(0.1 * len(dataset)))
        train_size = max(1, len(dataset) - val_size)
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=recurrent_episode_batch_size if recurrent else config["bc"]["batch_size"],
            shuffle=True,
            collate_fn=_collate_episode_batch if recurrent else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=recurrent_episode_batch_size if recurrent else config["bc"]["batch_size"],
            shuffle=False,
            collate_fn=_collate_episode_batch if recurrent else None,
        )

    model_config = ModelConfig.from_dict(config["model"])
    model = (
        BabyAIRecurrentPolicy(vocab_size=vocab.size, config=model_config, pad_id=vocab.pad_id, aux_specs=aux_specs)
        if recurrent
        else BabyAIFeedForwardPolicy(vocab_size=vocab.size, config=model_config, pad_id=vocab.pad_id, aux_specs=aux_specs)
    )
    if warm_start_path:
        _initialize_bc_model(model, warm_start_path, device="cpu")
        print(f"[bc] warm-started from {warm_start_path}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["bc"]["learning_rate"],
        weight_decay=config["bc"]["weight_decay"],
    )

    logs_dir = Path(config["artifacts"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / f"{Path(checkpoint_path).stem}_bc_metrics.jsonl"
    latest_checkpoint_path = Path(checkpoint_path)
    best_checkpoint_path = _best_bc_checkpoint_path(checkpoint_path)

    best_score = float("-inf")
    print(f"[bc] latest checkpoint path: {latest_checkpoint_path}")
    print(f"[bc] best checkpoint path: {best_checkpoint_path}")
    print(f"[bc] recurrent={recurrent}")
    if aux_specs:
        print(
            f"[bc] auxiliary preset={aux_preset} aux_weight={config['bc'].get('aux_weight')} "
            f"heads={', '.join(spec.name for spec in aux_specs)}"
        )
        if action_reweight_on_holding_target > 1.0:
            print(f"[bc] action_reweight_on_holding_target={action_reweight_on_holding_target:.2f}")
    if recurrent_episode_batch_size is not None:
        print(
            f"[bc] recurrent episode batch size: {recurrent_episode_batch_size} "
            f"(target step budget {config['bc']['batch_size']})"
        )
    if use_holdout_eval:
        print(f"[bc] holdout eval envs: {', '.join(resolved_eval_env_names)}")
        print(f"[bc] eval_max_steps={int(config['training'].get('eval_max_steps', 1000))}")
        if adaptive_demo_sampling:
            print(f"[bc] adaptive demo sampling envs: {', '.join(demo_batches_by_env)}")
    else:
        print("[bc] holdout eval envs: unavailable, using 90/10 demo split")

    latest_env_metrics: dict[str, dict[str, float]] | None = None
    if use_holdout_eval and warm_start_path:
        model.eval()
        env_metrics, train_success_rate, val_success_rate, train_mean_length, val_mean_length = _evaluate_bc_policy(
            model=model,
            vocab=vocab,
            env_names=resolved_eval_env_names,
            seed=seed,
            n_eval_episodes=config["training"]["n_eval_episodes"],
            max_steps=int(config["training"].get("eval_max_steps", 1000)),
            log_prefix="[bc warm-start eval env]",
        )
        print(
            f"[bc warm-start eval] train_sr={train_success_rate:.2%} "
            f"val_sr={val_success_rate:.2%}"
        )
        if aux_specs:
            _print_aux_head_train_val_metrics(
                "[bc warm-start aux]",
                _aggregate_env_aux_metrics(env_metrics, aux_specs, "train"),
                _aggregate_env_aux_metrics(env_metrics, aux_specs, "val"),
                aux_specs,
            )
        env_summary = " ".join(
            f"{env_name.replace('BabyAI-', '').replace('-v0', '')}={env_metrics[env_name]['val_success_rate']:.0%}"
            for env_name in resolved_eval_env_names
        )
        if env_summary:
            print(env_summary)
        if adaptive_demo_sampling:
            latest_env_metrics = env_metrics
        best_score = float(val_success_rate)
        _save_bc_checkpoint(
            path=best_checkpoint_path,
            model=model,
            model_config=model_config,
            vocab=vocab,
            recurrent=recurrent,
            aux_specs=aux_specs,
            aux_preset=aux_preset,
        )
        print(f"[bc] warm-start seeded best checkpoint at {best_checkpoint_path}")

    for epoch in range(1, config["bc"]["epochs"] + 1):
        if adaptive_demo_sampling:
            train_loader = (
                _build_weighted_episode_train_loader(
                    demo_batches_by_env=demo_batches_by_env,
                    datasets_by_env=datasets_by_env,
                    env_metrics=latest_env_metrics,
                    batch_size=recurrent_episode_batch_size,
                    min_sampling_proba=float(config["bc"]["min_sampling_proba"]),
                    sampling_episode_budget_per_task=config["bc"].get("sampling_episode_budget_per_task"),
                    generator=weighted_sampler_generator,
                )
                if recurrent
                else _build_weighted_train_loader(
                    demo_batches_by_env=demo_batches_by_env,
                    datasets_by_env=datasets_by_env,
                    env_metrics=latest_env_metrics,
                    batch_size=config["bc"]["batch_size"],
                    min_sampling_proba=float(config["bc"]["min_sampling_proba"]),
                    sampling_episode_budget_per_task=config["bc"].get("sampling_episode_budget_per_task"),
                    generator=weighted_sampler_generator,
                )
            )
        model.train()
        train_losses = []
        train_action_losses = []
        train_aux_losses = []
        train_aux_metric_totals = _init_aux_metric_totals(aux_specs)
        train_correct = 0
        train_count = 0
        progress = tqdm(train_loader, desc=f"bc epoch {epoch}", leave=True)
        ema_loss: float | None = None
        ema_acc: float | None = None
        ema_aux_loss: float | None = None
        for batch in progress:
            optimizer.zero_grad()
            if recurrent:
                loss, batch_correct, batch_count, batch_metrics = _compute_recurrent_bc_batch_metrics(
                    model,
                    batch,
                    aux_specs=aux_specs,
                    action_reweight_on_holding_target=action_reweight_on_holding_target,
                )
            else:
                loss, batch_correct, batch_count, batch_metrics = _compute_feedforward_bc_batch_metrics(
                    model,
                    batch,
                    aux_specs=aux_specs,
                    action_reweight_on_holding_target=action_reweight_on_holding_target,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["bc"]["clip_grad_norm"])
            optimizer.step()
            batch_loss = float(loss.item())
            train_losses.append(batch_loss)
            train_action_losses.append(batch_metrics["action_loss"])
            train_aux_losses.append(batch_metrics["aux_loss"])
            _accumulate_aux_metric_totals(train_aux_metric_totals, batch_metrics, aux_specs)
            train_correct += batch_correct
            train_count += batch_count

            batch_acc = float(batch_correct / batch_count if batch_count else 0.0)
            ema_loss = batch_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * batch_loss
            ema_acc = batch_acc if ema_acc is None else 0.9 * ema_acc + 0.1 * batch_acc
            ema_aux_loss = (
                batch_metrics["aux_loss"]
                if ema_aux_loss is None
                else 0.9 * ema_aux_loss + 0.1 * batch_metrics["aux_loss"]
            )
            postfix = {
                "ema_loss": f"{ema_loss:.4f}",
                "ema_acc": f"{ema_acc:.2%}",
                "avg_loss": f"{np.mean(train_losses):.4f}",
                "avg_acc": f"{(train_correct / train_count) if train_count else 0.0:.2%}",
            }
            if aux_specs:
                postfix["ema_aux"] = f"{ema_aux_loss:.4f}"
                postfix["avg_aux"] = f"{np.mean(train_aux_losses):.4f}"
            selected_total_episodes = getattr(train_loader, "selected_total_episodes", None)
            if selected_total_episodes is not None:
                postfix["epoch_episodes"] = str(selected_total_episodes)
            progress.set_postfix(**postfix)

        train_loss = float(np.mean(train_losses) if train_losses else 0.0)
        train_action_loss = float(np.mean(train_action_losses) if train_action_losses else 0.0)
        train_aux_loss = float(np.mean(train_aux_losses) if train_aux_losses else 0.0)
        train_acc = float(train_correct / train_count if train_count else 0.0)
        train_aux_head_metrics = _finalize_aux_metric_totals(train_aux_metric_totals, aux_specs, train_count)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_action_loss": train_action_loss,
            "train_aux_loss": train_aux_loss,
            "train_accuracy": train_acc,
        }
        metrics.update(train_aux_head_metrics)

        if use_holdout_eval:
            model.eval()
            env_metrics, train_success_rate, val_success_rate, train_mean_length, val_mean_length = _evaluate_bc_policy(
                model=model,
                vocab=vocab,
                env_names=resolved_eval_env_names,
                seed=seed,
                n_eval_episodes=config["training"]["n_eval_episodes"],
                max_steps=int(config["training"].get("eval_max_steps", 1000)),
                log_prefix=f"[bc eval env epoch={epoch}]",
            )
            score = float(val_success_rate)
            metrics.update(
                {
                    "env_metrics": env_metrics,
                    "train_success_rate": train_success_rate,
                    "val_success_rate": val_success_rate,
                    "train_mean_length": train_mean_length,
                    "val_mean_length": val_mean_length,
                    "best_val_success_rate": max(best_score, score),
                }
            )
            if aux_specs:
                train_env_aux_metrics = _aggregate_env_aux_metrics(env_metrics, aux_specs, "train")
                val_env_aux_metrics = _aggregate_env_aux_metrics(env_metrics, aux_specs, "val")
                metrics.update({f"eval_train_{key}": value for key, value in train_env_aux_metrics.items()})
                metrics.update({f"eval_val_{key}": value for key, value in val_env_aux_metrics.items()})
                _print_aux_head_train_val_metrics(f"[bc aux eval epoch={epoch}]", train_env_aux_metrics, val_env_aux_metrics, aux_specs)
            if adaptive_demo_sampling:
                latest_env_metrics = env_metrics
                metrics["sampling_ratios"] = dict(getattr(train_loader, "sampling_ratios", {}))
                metrics["selected_episode_counts"] = dict(getattr(train_loader, "selected_episode_counts", {}))
        else:
            model.eval()
            val_losses = []
            val_action_losses = []
            val_aux_losses = []
            val_aux_metric_totals = _init_aux_metric_totals(aux_specs)
            val_correct = 0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    if recurrent:
                        loss, batch_correct, batch_count, batch_metrics = _compute_recurrent_bc_batch_metrics(
                            model,
                            batch,
                            aux_specs=aux_specs,
                            action_reweight_on_holding_target=action_reweight_on_holding_target,
                        )
                    else:
                        loss, batch_correct, batch_count, batch_metrics = _compute_feedforward_bc_batch_metrics(
                            model,
                            batch,
                            aux_specs=aux_specs,
                            action_reweight_on_holding_target=action_reweight_on_holding_target,
                        )
                    val_losses.append(float(loss.item()))
                    val_action_losses.append(batch_metrics["action_loss"])
                    val_aux_losses.append(batch_metrics["aux_loss"])
                    _accumulate_aux_metric_totals(val_aux_metric_totals, batch_metrics, aux_specs)
                    val_correct += batch_correct
                    val_count += batch_count
            val_loss = float(np.mean(val_losses) if val_losses else 0.0)
            val_action_loss = float(np.mean(val_action_losses) if val_action_losses else 0.0)
            val_aux_loss = float(np.mean(val_aux_losses) if val_aux_losses else 0.0)
            val_acc = float(val_correct / val_count if val_count else 0.0)
            val_aux_head_metrics = _finalize_aux_metric_totals(val_aux_metric_totals, aux_specs, val_count)
            score = -val_loss
            metrics.update(
                {
                    "val_loss": val_loss,
                    "val_action_loss": val_action_loss,
                    "val_aux_loss": val_aux_loss,
                    "val_accuracy": val_acc,
                    "best_val_loss": -max(best_score, score),
                }
            )
            metrics.update({f"val_{key}": value for key, value in val_aux_head_metrics.items()})
            _print_aux_head_train_val_metrics(f"[bc aux epoch={epoch}]", train_aux_head_metrics, val_aux_head_metrics, aux_specs)

        previous_best = best_score
        best_score = max(best_score, score)

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")

        _save_bc_checkpoint(
            path=latest_checkpoint_path,
            model=model,
            model_config=model_config,
            vocab=vocab,
            recurrent=recurrent,
            aux_specs=aux_specs,
            aux_preset=aux_preset,
        )
        if use_holdout_eval:
            env_summary = " ".join(
                f"{env_name.replace('BabyAI-', '').replace('-v0', '')}={env_metrics[env_name]['val_success_rate']:.0%}"
                for env_name in resolved_eval_env_names
            )
            print(
                f"[bc] epoch={epoch} train_loss={train_loss:.4f} train_aux={train_aux_loss:.4f} train_acc={train_acc:.2%} "
                f"train_sr={train_success_rate:.2%} val_sr={val_success_rate:.2%}"
            )
            if env_summary:
                print(env_summary)
            if adaptive_demo_sampling:
                ratio_summary = " ".join(
                    f"{env_name.replace('BabyAI-', '').replace('-v0', '')}={getattr(train_loader, 'sampling_ratios', {}).get(env_name, 0.0):.3f}"
                    for env_name in demo_batches_by_env
                )
                count_summary = " ".join(
                    f"{env_name.replace('BabyAI-', '').replace('-v0', '')}={getattr(train_loader, 'selected_episode_counts', {}).get(env_name, 0)}"
                    for env_name in demo_batches_by_env
                )
                if ratio_summary:
                    print(f"[bc sampling ratios] {ratio_summary}")
                if count_summary:
                    print(f"[bc sampling counts] {count_summary}")
        else:
            print(
                f"[bc] epoch={epoch} train_loss={train_loss:.4f} train_aux={train_aux_loss:.4f} "
                f"val_loss={val_loss:.4f} val_aux={val_aux_loss:.4f} "
                f"train_acc={train_acc:.2%} val_acc={val_acc:.2%}"
            )
        if best_score > previous_best:
            _save_bc_checkpoint(
                path=best_checkpoint_path,
                model=model,
                model_config=model_config,
                vocab=vocab,
                recurrent=recurrent,
                aux_specs=aux_specs,
                aux_preset=aux_preset,
            )
            if use_holdout_eval:
                print(f"[milestone] new best BC holdout val success rate: {best_score:.2%}")
            else:
                print(f"[milestone] new best BC validation loss: {-best_score:.4f}")
