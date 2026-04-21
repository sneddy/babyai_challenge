"""Submission runtime for BabyAI leaderboard evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

import models


class BabyAIAgent:
    """Evaluator-facing submission agent."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.checkpoint_path = self.model_dir / "agent.zip"
        self.vocab_path = self.model_dir / "vocab.json"
        self.config_path = self.model_dir / "config.json"
        self.device = "cpu"
        self._regex = re.compile(r"[a-z]+")
        self._state = None

        self.model = None
        self.vocab = None
        self.max_mission_length = 16
        self.recurrent = False

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {self.checkpoint_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Missing vocabulary: {self.vocab_path}")

        self.vocab = json.loads(self.vocab_path.read_text(encoding="utf-8"))
        config_payload = {}
        if self.config_path.exists():
            config_payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.recurrent = bool(config_payload.get("recurrent", False))
        tokenization = self.vocab.get("tokenization", {})
        self.max_mission_length = int(tokenization.get("max_mission_length", 16))

        self.model, _checkpoint = models.build_policy_from_checkpoint(str(self.checkpoint_path), map_location=self.device)
        if hasattr(self.model, "initial_state") and self.recurrent:
            self._state = self.model.initial_state(batch_size=1, device=torch.device(self.device))

    def _tokenize(self, mission: str) -> tuple[np.ndarray, np.ndarray]:
        text = mission.lower()
        tokens = self._regex.findall(text)
        token_to_id = self.vocab["token_to_id"]
        unk_id = token_to_id.get("<unk>", 1)
        pad_id = token_to_id.get("<pad>", 0)

        token_ids = [token_to_id.get(token, unk_id) for token in tokens[: self.max_mission_length]]
        mask = [1] * len(token_ids)
        while len(token_ids) < self.max_mission_length:
            token_ids.append(pad_id)
            mask.append(0)
        return self._to_array(token_ids), self._to_array(mask)

    def _to_array(self, values):
        return np.asarray(values, dtype=np.int64)

    def _prepare_obs(self, obs) -> dict[str, np.ndarray]:
        mission_tokens, mission_mask = self._tokenize(obs["mission"])
        return {
            "image": self._to_array(obs["image"]),
            "mission_tokens": mission_tokens,
            "mission_mask": mission_mask,
        }

    def predict(self, obs, deterministic=True):
        processed = self._prepare_obs(obs)
        tensor_obs = models.tensorize_observation(processed, device=self.device)

        with torch.no_grad():
            if self.recurrent:
                action, _value, self._state = self.model.act(
                    tensor_obs,
                    state=self._state,
                    deterministic=deterministic,
                )
            else:
                action, _value = self.model.act(tensor_obs, deterministic=deterministic)

        if isinstance(action, torch.Tensor):
            action = action.item()
        return int(action)

    def reset(self):
        if self.recurrent and hasattr(self.model, "initial_state"):
            self._state = self.model.initial_state(batch_size=1, device=torch.device(self.device))
        else:
            self._state = None
