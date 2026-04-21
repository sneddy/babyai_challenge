"""Checkpoint loading and weight-transfer helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def _normalize_map_location(map_location: str | object = "cpu"):
    if map_location == "auto":
        return "cpu"
    return map_location


def _load_torch_payload(path: str, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=_normalize_map_location(map_location))


def load_policy_state_for_transfer(path: str, map_location: str = "cpu") -> tuple[dict, dict]:
    payload = _load_torch_payload(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported checkpoint payload type at {path}")
    if "policy_state" not in payload:
        raise RuntimeError(f"Checkpoint {path} does not contain a policy_state")
    return payload["policy_state"], payload


def copy_matching_state_dict(target_module, source_state_dict: dict) -> dict[str, int]:
    target_state = target_module.state_dict()
    matched = {}
    skipped = 0
    for key, value in source_state_dict.items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value
            matched[key] = 1
        else:
            skipped += 1
    target_module.load_state_dict(target_state, strict=False)
    return {
        "matched_tensors": len(matched),
        "skipped_tensors": skipped,
    }


def initialize_torch_policy_from_checkpoint(policy, checkpoint_path: str, map_location: str = "cpu") -> dict[str, int]:
    source_state, _payload = load_policy_state_for_transfer(checkpoint_path, map_location=map_location)
    return copy_matching_state_dict(policy, source_state)


def initialize_feedforward_sb3_from_exported_checkpoint(sb3_model, checkpoint_path: str, map_location: str = "cpu") -> dict[str, int]:
    source_state, _payload = load_policy_state_for_transfer(checkpoint_path, map_location=map_location)
    target_policy = sb3_model.policy.features_extractor.core
    stats = copy_matching_state_dict(target_policy, {key.removeprefix("encoder."): value for key, value in source_state.items() if key.startswith("encoder.")})

    actor_state = {key.removeprefix("actor."): value for key, value in source_state.items() if key.startswith("actor.")}
    critic_state = {key.removeprefix("critic."): value for key, value in source_state.items() if key.startswith("critic.")}
    actor_stats = copy_matching_state_dict(sb3_model.policy.action_net, actor_state)
    critic_stats = copy_matching_state_dict(sb3_model.policy.value_net, critic_state)
    return {
        "matched_tensors": stats["matched_tensors"] + actor_stats["matched_tensors"] + critic_stats["matched_tensors"],
        "skipped_tensors": stats["skipped_tensors"] + actor_stats["skipped_tensors"] + critic_stats["skipped_tensors"],
    }


def checkpoint_looks_like_torch_export(path: str) -> bool:
    return Path(path).suffix.lower() in {".pt", ".pth"}
