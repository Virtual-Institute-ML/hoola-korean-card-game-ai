from __future__ import annotations

import numpy as np
import torch

from hoola.action_codec import ActionCodec
from hoola.encoding import OBSERVATION_SIZE
from rl.model import MaskedActorCritic


def test_masked_policy_never_samples_illegal_action():
    model = MaskedActorCritic(hidden_sizes=(32, 32))
    obs = torch.zeros((1, OBSERVATION_SIZE), dtype=torch.float32)
    mask = torch.zeros((1, ActionCodec().size), dtype=torch.bool)
    legal = [0, 17, 2399]
    mask[0, legal] = True

    for _ in range(50):
        action, _, _ = model.act(obs, mask)
        assert int(action.item()) in legal


def test_model_output_shapes():
    model = MaskedActorCritic(hidden_sizes=(32, 16))
    obs = torch.zeros((4, OBSERVATION_SIZE), dtype=torch.float32)
    logits, values = model(obs)
    assert logits.shape == (4, ActionCodec().size)
    assert values.shape == (4,)
