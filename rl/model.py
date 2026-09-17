from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from hoola.action_codec import ActionCodec
from hoola.encoding import OBSERVATION_SIZE


@dataclass(frozen=True)
class ModelConfig:
    obs_size: int = OBSERVATION_SIZE
    action_size: int = ActionCodec().size
    hidden_sizes: tuple[int, ...] = (256, 256)


class MaskedActorCritic(nn.Module):
    """Small MLP actor-critic with exact legal-action masking."""

    def __init__(
        self,
        obs_size: int = OBSERVATION_SIZE,
        action_size: int | None = None,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if action_size is None:
            action_size = ActionCodec().size
        self.config = ModelConfig(
            obs_size=int(obs_size),
            action_size=int(action_size),
            hidden_sizes=tuple(int(x) for x in hidden_sizes),
        )

        layers: list[nn.Module] = []
        in_features = self.config.obs_size
        for hidden in self.config.hidden_sizes:
            layers.extend([nn.Linear(in_features, hidden), nn.ReLU()])
            in_features = hidden
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_features, self.config.action_size)
        self.value_head = nn.Linear(in_features, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        # PPO-style orthogonal initialization.  A smaller policy-head gain keeps
        # the initial legal-action distribution close to uniform.
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        features = self.backbone(observation.float())
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    @staticmethod
    def apply_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        legal = action_mask.to(dtype=torch.bool, device=logits.device)
        if logits.shape != legal.shape:
            raise ValueError(f"logits/mask shape mismatch: {logits.shape} vs {legal.shape}")
        if torch.any(legal.sum(dim=-1) == 0):
            raise ValueError("Each non-terminal policy row must contain at least one legal action")
        return logits.masked_fill(~legal, -1.0e9)

    def distribution_and_value(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        logits, value = self(observation)
        masked_logits = self.apply_action_mask(logits, action_mask)
        return Categorical(logits=masked_logits), value

    @torch.no_grad()
    def act(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.distribution_and_value(observation, action_mask)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def config_dict(self) -> dict:
        cfg = asdict(self.config)
        cfg["hidden_sizes"] = list(self.config.hidden_sizes)
        return cfg


def load_policy_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MaskedActorCritic, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    model = MaskedActorCritic(
        obs_size=config.get("obs_size", OBSERVATION_SIZE),
        action_size=config.get("action_size", ActionCodec().size),
        hidden_sizes=config.get("hidden_sizes", (256, 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
