"""Observation wrappers and environment factories."""

from __future__ import annotations

import os
from itertools import cycle
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ..common.constants import DEFAULT_MAX_MISSION_LENGTH
from ..data.vocabulary import MissionVocabulary
from .runtime import ensure_babyai_envs_registered, suppress_noisy_env_output


def preprocess_observation(
    obs: dict,
    vocab: MissionVocabulary,
    max_mission_length: int | None = None,
) -> dict[str, np.ndarray]:
    """Convert raw evaluator observation into numeric arrays."""
    if max_mission_length is None or max_mission_length == vocab.tokenization.max_mission_length:
        mission_tokens, mission_mask = vocab.encode_padded(obs["mission"])
    else:
        token_config = vocab.tokenization.__class__(
            max_mission_length=max_mission_length,
            lowercase=vocab.tokenization.lowercase,
            regex_pattern=vocab.tokenization.regex_pattern,
        )
        temp_vocab = MissionVocabulary(tokenization=token_config, token_to_id=dict(vocab.token_to_id), frozen=True)
        mission_tokens, mission_mask = temp_vocab.encode_padded(obs["mission"])

    return {
        "image": np.asarray(obs["image"], dtype=np.int64),
        "mission_tokens": np.asarray(mission_tokens, dtype=np.int64),
        "mission_mask": np.asarray(mission_mask, dtype=np.int64),
    }


class TokenizedMissionWrapper(gym.ObservationWrapper):
    """Return numeric symbolic obs plus padded mission tokens."""

    def __init__(self, env, vocab: MissionVocabulary):
        super().__init__(env)
        self.vocab = vocab
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype=np.int64),
                "mission_tokens": spaces.Box(
                    low=0,
                    high=max(vocab.size - 1, 1),
                    shape=(vocab.tokenization.max_mission_length,),
                    dtype=np.int64,
                ),
                "mission_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(vocab.tokenization.max_mission_length,),
                    dtype=np.int64,
                ),
            }
        )

    def observation(self, obs):
        return preprocess_observation(obs, self.vocab)


class MultiTaskTokenizedEnv(gym.Env):
    """Sample one BabyAI task per episode while exposing a single observation space."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_names: list[str],
        vocab: MissionVocabulary,
        seed: int = 0,
        idx: int = 0,
        sampling_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if not env_names:
            raise ValueError("env_names must be non-empty")

        ensure_babyai_envs_registered()
        self.vocab = vocab
        self.env_names = list(env_names)
        with suppress_noisy_env_output():
            self._envs = {env_name: gym.make(env_name) for env_name in self.env_names}
        self._ordered_env_names = list(self.env_names)
        self._round_robin = cycle(self._ordered_env_names)
        self._sampling_weights = None
        if sampling_weights is not None:
            weights = np.asarray([float(sampling_weights.get(name, 0.0)) for name in self._ordered_env_names], dtype=np.float64)
            if np.any(weights < 0):
                raise ValueError("sampling_weights must be non-negative")
            if weights.sum() > 0:
                self._sampling_weights = weights / weights.sum()
        self._rng = np.random.default_rng(seed + idx)
        self._seed_cursor = seed + idx * 100_000
        self.current_env_name = self._ordered_env_names[0]
        self.current_env = self._envs[self.current_env_name]
        self.action_space = self.current_env.action_space
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=(7, 7, 3), dtype=np.int64),
                "mission_tokens": spaces.Box(
                    low=0,
                    high=max(vocab.size - 1, 1),
                    shape=(vocab.tokenization.max_mission_length,),
                    dtype=np.int64,
                ),
                "mission_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(vocab.tokenization.max_mission_length,),
                    dtype=np.int64,
                ),
            }
        )

    def _sample_env_name(self) -> str:
        if self._sampling_weights is not None:
            sampled_index = int(self._rng.choice(len(self._ordered_env_names), p=self._sampling_weights))
            return self._ordered_env_names[sampled_index]
        return next(self._round_robin)

    def set_sampling_weights(self, sampling_weights: dict[str, float] | None) -> None:
        self._sampling_weights = None
        if sampling_weights is None:
            return
        weights = np.asarray(
            [float(sampling_weights.get(name, 0.0)) for name in self._ordered_env_names],
            dtype=np.float64,
        )
        if np.any(weights < 0):
            raise ValueError("sampling_weights must be non-negative")
        if weights.sum() > 0:
            self._sampling_weights = weights / weights.sum()

    def get_sampling_weights(self) -> dict[str, float] | None:
        if self._sampling_weights is None:
            return None
        return {
            env_name: float(weight)
            for env_name, weight in zip(self._ordered_env_names, self._sampling_weights.tolist(), strict=False)
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed_cursor = seed
        self.current_env_name = self._sample_env_name()
        self.current_env = self._envs[self.current_env_name]
        with suppress_noisy_env_output():
            obs, info = self.current_env.reset(seed=self._seed_cursor, options=options)
        self._seed_cursor += 1
        info = dict(info)
        info["env_name"] = self.current_env_name
        return preprocess_observation(obs, self.vocab), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info = dict(info)
        info["env_name"] = self.current_env_name
        return preprocess_observation(obs, self.vocab), reward, terminated, truncated, info

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        super().close()


def make_env(
    env_name: str | list[str],
    vocab: MissionVocabulary,
    seed: int = 0,
    idx: int = 0,
    monitor_dir: str | None = None,
    sampling_weights: dict[str, float] | None = None,
) -> Callable:
    """Create a single-tokenized BabyAI env factory."""

    def _init():
        ensure_babyai_envs_registered()
        if isinstance(env_name, str):
            with suppress_noisy_env_output():
                env = gym.make(env_name)
                env.reset(seed=seed + idx)
            env = TokenizedMissionWrapper(env, vocab)
        else:
            env = MultiTaskTokenizedEnv(
                env_names=list(env_name),
                vocab=vocab,
                seed=seed,
                idx=idx,
                sampling_weights=sampling_weights,
            )

        if monitor_dir is not None:
            os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, os.path.join(monitor_dir, f"env_{idx}.monitor.csv"))

        return env

    return _init


def make_vec_env(
    env_name: str | list[str],
    vocab: MissionVocabulary,
    n_envs: int = 8,
    seed: int = 0,
    monitor_dir: str | None = None,
    sampling_weights: dict[str, float] | None = None,
):
    """Create vectorized envs for SB3."""
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return vec_env_cls(
        [
            make_env(
                env_name,
                vocab,
                seed=seed,
                idx=i,
                monitor_dir=monitor_dir,
                sampling_weights=sampling_weights,
            )
            for i in range(n_envs)
        ]
    )


def get_default_tokenization_length(vocab: MissionVocabulary | None) -> int:
    if vocab is None:
        return DEFAULT_MAX_MISSION_LENGTH
    return vocab.tokenization.max_mission_length
