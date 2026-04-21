"""Mission vocabulary utilities."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import gymnasium as gym

from ..envs.catalog import ALL_LEADERBOARD_ENVS
from ..envs.runtime import ensure_babyai_envs_registered


@dataclass(frozen=True)
class TokenizationConfig:
    max_mission_length: int = 16
    lowercase: bool = True
    regex_pattern: str = r"[a-z]+"


class MissionVocabulary:
    """Frozen or growing mission vocabulary with BabyAI-style tokenization."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(
        self,
        tokenization: TokenizationConfig | None = None,
        token_to_id: dict[str, int] | None = None,
        frozen: bool = False,
    ) -> None:
        self.tokenization = tokenization or TokenizationConfig()
        self._regex = re.compile(self.tokenization.regex_pattern)
        self.token_to_id = token_to_id or {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.frozen = frozen

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def to_dict(self) -> dict:
        return {
            "token_to_id": self.token_to_id,
            "frozen": self.frozen,
            "tokenization": {
                "max_mission_length": self.tokenization.max_mission_length,
                "lowercase": self.tokenization.lowercase,
                "regex_pattern": self.tokenization.regex_pattern,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MissionVocabulary":
        tokenization = TokenizationConfig(**payload.get("tokenization", {}))
        return cls(
            tokenization=tokenization,
            token_to_id=payload["token_to_id"],
            frozen=payload.get("frozen", True),
        )

    @classmethod
    def load(cls, path: str) -> "MissionVocabulary":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    def freeze(self) -> None:
        self.frozen = True

    def tokenize(self, mission: str) -> list[str]:
        text = mission.lower() if self.tokenization.lowercase else mission
        return self._regex.findall(text)

    def add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        if self.frozen:
            return self.unk_id
        next_id = len(self.token_to_id)
        self.token_to_id[token] = next_id
        return next_id

    def encode(self, mission: str) -> list[int]:
        return [self.add_token(token) for token in self.tokenize(mission)]

    def encode_padded(self, mission: str) -> tuple[list[int], list[int]]:
        token_ids = self.encode(mission)[: self.tokenization.max_mission_length]
        mask = [1] * len(token_ids)
        while len(token_ids) < self.tokenization.max_mission_length:
            token_ids.append(self.pad_id)
            mask.append(0)
        return token_ids, mask

    def build_from_missions(self, missions: Iterable[str]) -> None:
        for mission in missions:
            self.encode(mission)


def build_and_save_vocab(
    missions: Sequence[str],
    path: str,
    tokenization: TokenizationConfig | None = None,
) -> MissionVocabulary:
    vocab = MissionVocabulary(tokenization=tokenization, frozen=False)
    vocab.build_from_missions(missions)
    vocab.freeze()
    vocab.save(path)
    return vocab


def build_vocab_from_envs(
    env_names: Sequence[str],
    output_path: str,
    episodes_per_env: int = 128,
    seed: int = 0,
    tokenization: TokenizationConfig | None = None,
) -> MissionVocabulary:
    ensure_babyai_envs_registered()

    vocab = MissionVocabulary(tokenization=tokenization, frozen=False)

    for env_index, env_name in enumerate(env_names):
        env = gym.make(env_name)
        try:
            for episode_index in range(episodes_per_env):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    obs, _ = env.reset(seed=seed + env_index * episodes_per_env + episode_index)
                vocab.encode(obs["mission"])
        finally:
            env.close()

    vocab.freeze()
    vocab.save(output_path)
    return vocab


def build_global_vocab(
    output_path: str,
    episodes_per_env: int = 128,
    seed: int = 0,
    tokenization: TokenizationConfig | None = None,
    env_names: Sequence[str] | None = None,
) -> MissionVocabulary:
    return build_vocab_from_envs(
        env_names=env_names or ALL_LEADERBOARD_ENVS,
        output_path=output_path,
        episodes_per_env=episodes_per_env,
        seed=seed,
        tokenization=tokenization,
    )
