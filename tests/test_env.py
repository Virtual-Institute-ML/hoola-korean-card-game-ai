from __future__ import annotations

import numpy as np

from hoola.env import HoolaEnv
from hoola.encoding import OBSERVATION_SIZE


def test_env_reset_returns_training_observation_and_mask():
    env = HoolaEnv(opponent="random", learning_player=0)
    obs, info = env.reset(seed=777)
    assert obs["observation"].shape == (OBSERVATION_SIZE,)
    assert obs["action_mask"].shape == (env.codec.size,)
    assert obs["action_mask"].dtype == np.int8
    assert obs["action_mask"].sum() > 0
    assert info["seed"] == 777


def test_env_same_seed_reproduces_initial_tensor_and_mask():
    env = HoolaEnv(opponent="random", learning_player=0)
    obs1, _ = env.reset(seed=778)
    x1 = obs1["observation"].copy()
    m1 = obs1["action_mask"].copy()
    obs2, _ = env.reset(seed=778)
    np.testing.assert_array_equal(x1, obs2["observation"])
    np.testing.assert_array_equal(m1, obs2["action_mask"])


def test_masked_random_policy_can_finish_games():
    rng = np.random.default_rng(0)
    env = HoolaEnv(opponent="heuristic", learning_player=0, max_turns=200)
    for episode in range(10):
        obs, _ = env.reset(seed=1000 + episode)
        terminated = False
        steps = 0
        while not terminated:
            legal_ids = np.flatnonzero(obs["action_mask"])
            assert len(legal_ids) > 0
            action_id = int(rng.choice(legal_ids))
            obs, reward, terminated, truncated, info = env.step(action_id)
            assert not truncated
            assert reward in (-1.0, 0.0, 1.0)
            steps += 1
            assert steps < 1000
        assert obs["action_mask"].sum() == 0
        assert info["winner"] in (0, 1, None)


def test_learning_as_player_one_auto_advances_player_zero():
    env = HoolaEnv(opponent="random", learning_player=1)
    obs, info = env.reset(seed=779)
    assert env.engine.state.terminal or env.engine.state.current_player == 1
    if not env.engine.state.terminal:
        assert obs["action_mask"].sum() > 0
