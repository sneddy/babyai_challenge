"""Behavior-cloning dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DemoBatch:
    image: np.ndarray
    mission_tokens: np.ndarray
    mission_mask: np.ndarray
    action: np.ndarray
    done: np.ndarray


def load_demo_file(path: str | Path) -> DemoBatch:
    payload = np.load(path, allow_pickle=False)
    return DemoBatch(
        image=payload["image"],
        mission_tokens=payload["mission_tokens"],
        mission_mask=payload["mission_mask"],
        action=payload["action"],
        done=payload["done"],
    )


def save_demo_file(path: str | Path, batch: DemoBatch) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image=batch.image,
        mission_tokens=batch.mission_tokens,
        mission_mask=batch.mission_mask,
        action=batch.action,
        done=batch.done,
    )


class DemoDataset:
    """Simple numpy-backed BC dataset."""

    def __init__(self, batch: DemoBatch) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return int(self.batch.action.shape[0])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {
            "image": self.batch.image[index],
            "mission_tokens": self.batch.mission_tokens[index],
            "mission_mask": self.batch.mission_mask[index],
            "action": self.batch.action[index],
            "done": self.batch.done[index],
        }


def concatenate_demo_batches(batches: list[DemoBatch]) -> DemoBatch:
    if not batches:
        raise ValueError("batches must be non-empty")
    if len(batches) == 1:
        return batches[0]
    return DemoBatch(
        image=np.concatenate([batch.image for batch in batches], axis=0),
        mission_tokens=np.concatenate([batch.mission_tokens for batch in batches], axis=0),
        mission_mask=np.concatenate([batch.mission_mask for batch in batches], axis=0),
        action=np.concatenate([batch.action for batch in batches], axis=0),
        done=np.concatenate([batch.done for batch in batches], axis=0),
    )


def load_demo_files(paths: list[str | Path]) -> DemoBatch:
    return concatenate_demo_batches([load_demo_file(path) for path in paths])
