"""Pure-PyTorch BabyAI submission models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    action_dim: int = 7
    image_embedding_dim: int = 128
    mission_embedding_dim: int = 32
    mission_hidden_dim: int = 128
    attention: bool = True
    features_dim: int = 256
    recurrent_hidden_dim: int = 256
    tile_vocab_sizes: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.tile_vocab_sizes is None:
            self.tile_vocab_sizes = {"object": 16, "color": 16, "state": 8}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        valid_fields = set(cls.__dataclass_fields__)
        filtered = {key: value for key, value in payload.items() if key in valid_fields}
        return cls(**filtered)


class TileEmbeddingEncoder(nn.Module):
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
            lengths = torch.full((tokens.shape[0],), tokens.shape[1], dtype=torch.long, device=tokens.device)
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
        return F.relu(y * (1.0 + gamma) + beta + residual)


class BabyAIEncoder(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.tile_encoder = TileEmbeddingEncoder(config.image_embedding_dim, config.tile_vocab_sizes)
        self.mission_encoder = MissionEncoder(
            vocab_size,
            config.mission_embedding_dim,
            config.mission_hidden_dim,
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
        return self.output_proj(self.pool(vision).flatten(1))


class BabyAIFeedForwardPolicy(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig, pad_id: int = 0) -> None:
        super().__init__()
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
            action = torch.distributions.Categorical(logits=logits).sample()
        return action, value


class BabyAIRecurrentPolicy(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.encoder = BabyAIEncoder(vocab_size=vocab_size, config=config, pad_id=pad_id)
        self.core = nn.LSTMCell(config.features_dim, config.recurrent_hidden_dim)
        self.actor = nn.Linear(config.recurrent_hidden_dim, config.action_dim)
        self.critic = nn.Linear(config.recurrent_hidden_dim, 1)

    def initial_state(self, batch_size: int, device: torch.device | None = None):
        device = device or torch.device("cpu")
        hidden = torch.zeros(batch_size, self.config.recurrent_hidden_dim, device=device)
        cell = torch.zeros(batch_size, self.config.recurrent_hidden_dim, device=device)
        return hidden, cell

    def forward(self, obs: dict[str, torch.Tensor], state=None):
        features = self.encoder(obs)
        if state is None:
            state = self.initial_state(features.shape[0], device=features.device)
        hidden, cell = self.core(features, state)
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        return logits, value, (hidden, cell)

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], state=None, deterministic: bool = True):
        logits, value, next_state = self.forward(obs, state=state)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(logits=logits).sample()
        return action, value, next_state


def tensorize_observation(obs: dict[str, Any], device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    out = {}
    for key, value in obs.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim in {1, 3}:
            tensor = tensor.unsqueeze(0)
        out[key] = tensor.to(device=device)
    return out


def load_exported_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def build_policy_from_checkpoint(path: str, map_location: str | torch.device = "cpu"):
    checkpoint = load_exported_checkpoint(path, map_location=map_location)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    vocab_payload = checkpoint["vocab"]
    recurrent = checkpoint.get("recurrent", False)
    vocab_size = len(vocab_payload["token_to_id"])

    if recurrent:
        model = BabyAIRecurrentPolicy(vocab_size=vocab_size, config=config)
    else:
        model = BabyAIFeedForwardPolicy(vocab_size=vocab_size, config=config)
    model.load_state_dict(checkpoint["policy_state"])
    model.eval()
    return model, checkpoint
