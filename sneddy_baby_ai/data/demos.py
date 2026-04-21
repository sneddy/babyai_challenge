"""Behavior-cloning dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


_AUX_TARGET_PREFIX = "aux_target__"
_AUX_MASK_PREFIX = "aux_mask__"


@dataclass
class DemoBatch:
    image: np.ndarray
    mission_tokens: np.ndarray
    mission_mask: np.ndarray
    action: np.ndarray
    done: np.ndarray
    aux_targets: dict[str, np.ndarray] = field(default_factory=dict)
    aux_masks: dict[str, np.ndarray] = field(default_factory=dict)


def load_demo_file(path: str | Path) -> DemoBatch:
    payload = np.load(path, allow_pickle=False)
    aux_targets = {
        key.removeprefix(_AUX_TARGET_PREFIX): payload[key]
        for key in payload.files
        if key.startswith(_AUX_TARGET_PREFIX)
    }
    aux_masks = {
        key.removeprefix(_AUX_MASK_PREFIX): payload[key]
        for key in payload.files
        if key.startswith(_AUX_MASK_PREFIX)
    }
    return DemoBatch(
        image=payload["image"],
        mission_tokens=payload["mission_tokens"],
        mission_mask=payload["mission_mask"],
        action=payload["action"],
        done=payload["done"],
        aux_targets=aux_targets,
        aux_masks=aux_masks,
    )


def save_demo_file(path: str | Path, batch: DemoBatch) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": batch.image,
        "mission_tokens": batch.mission_tokens,
        "mission_mask": batch.mission_mask,
        "action": batch.action,
        "done": batch.done,
    }
    for name, values in batch.aux_targets.items():
        payload[f"{_AUX_TARGET_PREFIX}{name}"] = values
    for name, values in batch.aux_masks.items():
        payload[f"{_AUX_MASK_PREFIX}{name}"] = values
    np.savez_compressed(
        path,
        **payload,
    )


class DemoDataset:
    """Simple numpy-backed BC dataset."""

    def __init__(self, batch: DemoBatch) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return int(self.batch.action.shape[0])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        sample = {
            "image": self.batch.image[index],
            "mission_tokens": self.batch.mission_tokens[index],
            "mission_mask": self.batch.mission_mask[index],
            "action": self.batch.action[index],
            "done": self.batch.done[index],
        }
        if self.batch.aux_targets:
            sample["aux_targets"] = {name: values[index] for name, values in self.batch.aux_targets.items()}
        if self.batch.aux_masks:
            sample["aux_masks"] = {name: values[index] for name, values in self.batch.aux_masks.items()}
        return sample


def _validate_aux_alignment(batches: list[DemoBatch]) -> tuple[list[str], list[str]]:
    if not batches:
        return [], []

    target_keys = sorted(batches[0].aux_targets)
    mask_keys = sorted(batches[0].aux_masks)
    for batch in batches[1:]:
        if sorted(batch.aux_targets) != target_keys:
            raise ValueError("Demo batches do not share the same auxiliary target keys")
        if sorted(batch.aux_masks) != mask_keys:
            raise ValueError("Demo batches do not share the same auxiliary mask keys")
    return target_keys, mask_keys


def concatenate_demo_batches(batches: list[DemoBatch]) -> DemoBatch:
    if not batches:
        raise ValueError("batches must be non-empty")
    if len(batches) == 1:
        return batches[0]
    aux_target_keys, aux_mask_keys = _validate_aux_alignment(batches)
    return DemoBatch(
        image=np.concatenate([batch.image for batch in batches], axis=0),
        mission_tokens=np.concatenate([batch.mission_tokens for batch in batches], axis=0),
        mission_mask=np.concatenate([batch.mission_mask for batch in batches], axis=0),
        action=np.concatenate([batch.action for batch in batches], axis=0),
        done=np.concatenate([batch.done for batch in batches], axis=0),
        aux_targets={
            name: np.concatenate([batch.aux_targets[name] for batch in batches], axis=0)
            for name in aux_target_keys
        },
        aux_masks={
            name: np.concatenate([batch.aux_masks[name] for batch in batches], axis=0)
            for name in aux_mask_keys
        },
    )


def load_demo_files(paths: list[str | Path]) -> DemoBatch:
    return concatenate_demo_batches([load_demo_file(path) for path in paths])
