"""Minimal masked-policy smoke test for HoolaEnv.

This is not learning yet.  It demonstrates exactly the API a PPO/DQN-style
trainer will consume: fixed observation tensor + legal action mask + action ID.
"""
from __future__ import annotations

import numpy as np

from hoola.env import HoolaEnv


def main(episodes: int = 5) -> None:
    rng = np.random.default_rng(2026)
    env = HoolaEnv(opponent="heuristic", learning_player=0)

    print(f"observation size : {env.observation_space.spaces['observation'].shape}")
    print(f"action space     : {env.action_space.n}")
    print(f"gymnasium found  : {env.gymnasium_available}")
    print()

    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        decisions = 0
        terminated = False

        while not terminated:
            legal_ids = np.flatnonzero(obs["action_mask"])
            action_id = int(rng.choice(legal_ids))
            obs, reward, terminated, truncated, info = env.step(action_id)
            total_reward += reward
            decisions += 1
            if truncated:
                break

        print(
            f"episode={episode + 1:02d} seed={info['seed']} "
            f"reward={total_reward:+.0f} winner={info['winner']} "
            f"decisions={decisions} turns={info['turns_completed']}"
        )


if __name__ == "__main__":
    main()
