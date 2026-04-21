"""Task curriculum helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.loader import load_curriculum_stage_payloads


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    env_names: tuple[str, ...]


CURRICULUM_STAGES = tuple(
    CurriculumStage(stage_payload["name"], tuple(stage_payload["env_names"]))
    for stage_payload in load_curriculum_stage_payloads()
)


def get_stage(name: str) -> CurriculumStage:
    for stage in CURRICULUM_STAGES:
        if stage.name == name:
            return stage
    raise KeyError(f"Unknown curriculum stage: {name}")


def list_stage_names() -> list[str]:
    return [stage.name for stage in CURRICULUM_STAGES]


def get_replay_envs(stage_name: str) -> list[str]:
    replay_envs: list[str] = []
    for stage in CURRICULUM_STAGES:
        if stage.name == stage_name:
            break
        replay_envs.extend(stage.env_names)
    return replay_envs
