"""Load JSON-compatible YAML config fragments from ./configs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"

DEFAULT_MODEL_PRESET_BY_CONFIG = {
    "minimalistic": "minimalistic",
    "minimalistic_finetune": "minimalistic",
}

LEGACY_MODEL_OVERRIDES = {
    "bootstrap_no_attn": {
        "attention": False,
    },
    "stable_multitask_no_attn": {
        "attention": False,
    },
}


def _load_json_yaml(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: dict, patch: dict) -> dict:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def list_model_presets() -> list[str]:
    return sorted(path.stem for path in (CONFIG_ROOT / "models").glob("*.yaml"))


def list_config_presets() -> list[str]:
    return sorted(path.stem for path in (CONFIG_ROOT / "presets").glob("*.yaml"))


def _resolve_model_preset(config_name: str, model_preset: str | None) -> str:
    resolved = model_preset or DEFAULT_MODEL_PRESET_BY_CONFIG.get(config_name, "base")
    if resolved not in list_model_presets():
        raise ValueError(f"Unknown model preset: {resolved}")
    return resolved


def get_config(config_name: str = "default", model_preset: str | None = None) -> dict:
    base = _load_json_yaml(CONFIG_ROOT / "base.yaml")
    preset_path = CONFIG_ROOT / "presets" / f"{config_name}.yaml"
    if not preset_path.exists():
        raise ValueError(f"Unknown config preset: {config_name}")
    config = _deep_merge(base, _load_json_yaml(preset_path))

    resolved_model_preset = _resolve_model_preset(config_name, model_preset)
    model_payload = _load_json_yaml(CONFIG_ROOT / "models" / f"{resolved_model_preset}.yaml")
    config["model"] = model_payload
    config["model_preset"] = resolved_model_preset

    if config_name in LEGACY_MODEL_OVERRIDES:
        config["model"] = _deep_merge(config["model"], LEGACY_MODEL_OVERRIDES[config_name])
        config["model_preset"] = f"{resolved_model_preset}+legacy_override"
    return config


def load_env_catalog() -> dict:
    return _load_json_yaml(CONFIG_ROOT / "envs" / "tiers.yaml")


def load_curriculum_stage_payloads() -> list[dict]:
    payload = _load_json_yaml(CONFIG_ROOT / "envs" / "curriculum.yaml")
    return list(payload["stages"])
