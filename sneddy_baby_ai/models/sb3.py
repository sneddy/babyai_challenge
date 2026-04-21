"""SB3 integration helpers."""

from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .core import BabyAIFeedForwardPolicy, BabyAIEncoder, ModelConfig


class BabyAIFeaturesExtractor(BaseFeaturesExtractor):
    """Custom SB3 feature extractor backed by the shared BabyAI encoder."""

    def __init__(self, observation_space, features_dim: int = 256, vocab_size: int = 100, model_config: dict | None = None):
        config = ModelConfig.from_dict(model_config or {"features_dim": features_dim})
        config.features_dim = features_dim
        super().__init__(observation_space, features_dim)
        self.core = BabyAIEncoder(vocab_size=vocab_size, config=config)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.core(observations)


def export_feedforward_policy_from_sb3(sb3_model, vocab_payload: dict, model_config: dict) -> BabyAIFeedForwardPolicy:
    """Extract a plain PyTorch feedforward policy from an SB3 PPO model.

    This path assumes `net_arch=[]`, so the SB3 policy head is a direct linear
    map from the feature extractor output to actions and value.
    """

    config = ModelConfig.from_dict(model_config)
    policy = BabyAIFeedForwardPolicy(
        vocab_size=len(vocab_payload["token_to_id"]),
        config=config,
    )

    feature_extractor = sb3_model.policy.features_extractor
    policy.encoder.load_state_dict(feature_extractor.core.state_dict())
    policy.actor.load_state_dict(sb3_model.policy.action_net.state_dict())
    policy.critic.load_state_dict(sb3_model.policy.value_net.state_dict())
    policy.eval()
    return policy
