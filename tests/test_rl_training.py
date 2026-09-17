from __future__ import annotations

import numpy as np
import torch

from rl.model import MaskedActorCritic
from rl.ppo import PPOConfig, collect_complete_episodes, ppo_update


def test_short_masked_ppo_update_runs_without_nan():
    device = torch.device("cpu")
    rng = np.random.default_rng(123)
    model = MaskedActorCritic(hidden_sizes=(32, 32)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    rollout = collect_complete_episodes(
        model,
        opponent="random",
        num_episodes=2,
        rng=rng,
        device=device,
        max_turns=100,
        seat_mode="alternate",
    )
    assert rollout.steps > 0
    assert rollout.observations.shape[0] == rollout.steps
    assert rollout.action_masks.shape[0] == rollout.steps

    metrics = ppo_update(
        model,
        optimizer,
        rollout,
        config=PPOConfig(ppo_epochs=1, minibatch_size=64),
        device=device,
        rng=rng,
    )
    assert all(np.isfinite(v) for v in metrics.values())
