"""Leaderboard environment catalog loaded from config files."""

from __future__ import annotations

from ..config.loader import load_env_catalog


_CATALOG = load_env_catalog()

LEADERBOARD_WEIGHTS = dict(_CATALOG["weights"])
LEADERBOARD_TIERS = {tier_name: list(env_names) for tier_name, env_names in _CATALOG["tiers"].items()}
ALL_LEADERBOARD_ENVS = [
    env_name
    for env_names in LEADERBOARD_TIERS.values()
    for env_name in env_names
]
