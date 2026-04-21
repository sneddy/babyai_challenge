"""Shared pure-PyTorch model definitions used by training and submission."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.constants import DEFAULT_ACTION_DIM, DEFAULT_IMAGE_CHANNEL_VOCABS


@dataclass
class ModelConfig:
    action_dim: int = DEFAULT_ACTION_DIM
    image_embedding_dim: int = 128
    mission_embedding_dim: int = 32
    mission_hidden_dim: int = 128
    attention: bool = True
    features_dim: int = 256
    recurrent_hidden_dim: int = 256
    tile_vocab_sizes: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.tile_vocab_sizes is None:
            self.tile_vocab_sizes = dict(DEFAULT_IMAGE_CHANNEL_VOCABS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "image_embedding_dim": self.image_embedding_dim,
            "mission_embedding_dim": self.mission_embedding_dim,
            "mission_hidden_dim": self.mission_hidden_dim,
            "attention": self.attention,
            "features_dim": self.features_dim,
            "recurrent_hidden_dim": self.recurrent_hidden_dim,
            "tile_vocab_sizes": dict(self.tile_vocab_sizes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return cls(**payload)


class TileEmbeddingEncoder(nn.Module):
    """Encode symbolic 7x7x3 tiles using learned embeddings."""

    def __init__(self, embedding_dim: int, vocab_sizes: dict[str, int]) -> None:
        super().__init__()
        self.object_embedding = nn.Embedding(vocab_sizes["object"], embedding_dim)
        self.color_embedding = nn.Embedding(vocab_sizes["color"], embedding_dim)
        self.state_embedding = nn.Embedding(vocab_sizes["state"], embedding_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.long()
        object_ids = image[..., 0].clamp(min=0, max=self.object_embedding.num_embeddings - 1)
        color_ids = image[..., 1].clamp(min=0, max=self.color_embedding.num_embeddings - 1)
        state_ids = image[..., 2].clamp(min=0, max=self.state_embedding.num_embeddings - 1)

        tile_embeddings = (
            self.object_embedding(object_ids)
            + self.color_embedding(color_ids)
            + self.state_embedding(state_ids)
        ) / 3.0
        return tile_embeddings.permute(0, 3, 1, 2)


class MissionEncoder(nn.Module):
    """Embedding + BiGRU + attention-pooling encoder for BabyAI missions."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        pad_id: int = 0,
        attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_attention = attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1) if attention else None
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.embedding(tokens.long())

        if mask is None:
            lengths = torch.full(
                (tokens.shape[0],),
                tokens.shape[1],
                dtype=torch.long,
                device=tokens.device,
            )
        else:
            lengths = mask.long().sum(dim=1).clamp(min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=tokens.shape[1],
        )

        time_index = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        valid_mask = time_index < lengths.unsqueeze(1)
        if self.use_attention:
            attention_logits = self.attention(outputs).squeeze(-1)
            attention_logits = attention_logits.masked_fill(~valid_mask, torch.finfo(attention_logits.dtype).min)
            attention_weights = torch.softmax(attention_logits, dim=1)
            pooled = torch.sum(outputs * attention_weights.unsqueeze(-1), dim=1)
        else:
            valid_mask_f = valid_mask.unsqueeze(-1).to(outputs.dtype)
            pooled = torch.sum(outputs * valid_mask_f, dim=1) / valid_mask_f.sum(dim=1).clamp_min(1.0)
        return self.output_proj(pooled)


class FiLMResidualBlock(nn.Module):
    """Residual FiLM block inspired by the original BabyAI model."""

    def __init__(self, channels: int, mission_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gamma = nn.Linear(mission_dim, channels)
        self.beta = nn.Linear(mission_dim, channels)

    def forward(self, x: torch.Tensor, mission_embedding: torch.Tensor) -> torch.Tensor:
        residual = x
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        gamma = self.gamma(mission_embedding).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(mission_embedding).unsqueeze(-1).unsqueeze(-1)
        y = y * (1.0 + gamma) + beta
        return F.relu(y + residual)


class BabyAIEncoder(nn.Module):
    """BabyAI-1.1-style BOW + GRU + FiLM encoder."""

    def __init__(self, vocab_size: int, config: ModelConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.tile_encoder = TileEmbeddingEncoder(
            embedding_dim=config.image_embedding_dim,
            vocab_sizes=config.tile_vocab_sizes,
        )
        self.mission_encoder = MissionEncoder(
            vocab_size=vocab_size,
            embedding_dim=config.mission_embedding_dim,
            hidden_dim=config.mission_hidden_dim,
            pad_id=pad_id,
            attention=config.attention,
        )
        self.input_proj = nn.Sequential(
            nn.Conv2d(config.image_embedding_dim, config.image_embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.image_embedding_dim),
            nn.ReLU(),
        )
        self.film_blocks = nn.ModuleList(
            [
                FiLMResidualBlock(config.image_embedding_dim, config.mission_hidden_dim),
                FiLMResidualBlock(config.image_embedding_dim, config.mission_hidden_dim),
            ]
        )
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.output_proj = nn.Linear(config.image_embedding_dim, config.features_dim)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        image = obs["image"]
        mission_tokens = obs["mission_tokens"]
        mission_mask = obs.get("mission_mask")

        vision = self.tile_encoder(image)
        mission_embedding = self.mission_encoder(mission_tokens, mission_mask)

        vision = self.input_proj(vision)
        for block in self.film_blocks:
            vision = block(vision, mission_embedding)

        pooled = self.pool(vision).flatten(1)
        return self.output_proj(pooled)


class BabyAIFeedForwardPolicy(nn.Module):
    """Pure PyTorch actor-critic model for submission export."""

    def __init__(self, vocab_size: int, config: ModelConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.encoder = BabyAIEncoder(vocab_size=vocab_size, config=config, pad_id=pad_id)
        self.actor = nn.Linear(config.features_dim, config.action_dim)
        self.critic = nn.Linear(config.features_dim, 1)

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            probs = torch.distributions.Categorical(logits=logits)
            action = probs.sample()
        return action, value


class BabyAIRecurrentPolicy(nn.Module):
    """Pure PyTorch recurrent policy used for export-safe inference."""

    def __init__(self, vocab_size: int, config: ModelConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.encoder = BabyAIEncoder(vocab_size=vocab_size, config=config, pad_id=pad_id)
        self.core = nn.LSTMCell(config.features_dim, config.recurrent_hidden_dim)
        self.actor = nn.Linear(config.recurrent_hidden_dim, config.action_dim)
        self.critic = nn.Linear(config.recurrent_hidden_dim, 1)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or torch.device("cpu")
        hidden = torch.zeros(batch_size, self.config.recurrent_hidden_dim, device=device)
        cell = torch.zeros(batch_size, self.config.recurrent_hidden_dim, device=device)
        return hidden, cell

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        features = self.encoder(obs)
        if state is None:
            state = self.initial_state(features.shape[0], features.device)
        hidden, cell = self.core(features, state)
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        return logits, value, (hidden, cell)

    @torch.no_grad()
    def act(
        self,
        obs: dict[str, torch.Tensor],
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        logits, value, next_state = self.forward(obs, state=state)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            probs = torch.distributions.Categorical(logits=logits)
            action = probs.sample()
        return action, value, next_state


def tensorize_observation(obs: dict[str, Any], device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in obs.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        out[key] = tensor.to(device=device)
    return out


def save_exported_checkpoint(
    path: str,
    *,
    policy_state: dict[str, torch.Tensor],
    model_config: ModelConfig,
    vocab_payload: dict[str, Any],
    recurrent: bool,
) -> None:
    torch.save(
        {
            "policy_state": policy_state,
            "model_config": model_config.to_dict(),
            "vocab": vocab_payload,
            "recurrent": recurrent,
        },
        path,
    )


def load_exported_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def build_policy_from_checkpoint(
    path: str,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = load_exported_checkpoint(path, map_location=map_location)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    vocab_payload = checkpoint["vocab"]
    vocab_size = len(vocab_payload["token_to_id"])
    recurrent = checkpoint.get("recurrent", False)

    if recurrent:
        model = BabyAIRecurrentPolicy(vocab_size=vocab_size, config=config)
    else:
        model = BabyAIFeedForwardPolicy(vocab_size=vocab_size, config=config)
    model.load_state_dict(checkpoint["policy_state"])
    model.to(map_location)
    model.eval()
    return model, checkpoint


def checkpoint_to_json_summary(path: str) -> str:
    payload = load_exported_checkpoint(path, map_location="cpu")
    summary = {
        "recurrent": payload.get("recurrent", False),
        "model_config": payload["model_config"],
        "vocab_size": len(payload["vocab"]["token_to_id"]),
    }
    return json.dumps(summary, indent=2, sort_keys=True)
