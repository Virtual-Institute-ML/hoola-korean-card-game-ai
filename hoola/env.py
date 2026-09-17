from __future__ import annotations

import secrets
from typing import Any

import numpy as np

from agents import HeuristicAgent, RandomAgent

from .action_codec import ActionCodec
from .encoding import OBSERVATION_SIZE, encode_observation
from .engine import HoolaEngine
from .gym_compat import EnvBase, GYMNASIUM_AVAILABLE, spaces
from .observation import make_observation


class HoolaEnv(EnvBase):
    """Single-learning-player Gymnasium-compatible Hoola environment.

    The learning player chooses every one of its own micro-actions.  Whenever a
    discard passes control to the opponent, the configured opponent policy is
    automatically advanced until the learning player acts again or the game
    terminates.

    Observation is a dict:
      - ``observation``: fixed float32 feature vector
      - ``action_mask``: int8 mask over the global discrete action vocabulary
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        opponent: str | Any = "heuristic",
        learning_player: int = 0,
        seed: int | None = None,
        max_turns: int = 500,
    ) -> None:
        if learning_player not in (0, 1):
            raise ValueError("learning_player must be 0 or 1")
        self.learning_player = learning_player
        self.max_turns = max_turns
        self.codec = ActionCodec()
        self.action_space = spaces.Discrete(self.codec.size)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(OBSERVATION_SIZE,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.codec.size,),
                    dtype=np.int8,
                ),
            }
        )
        self._opponent_spec = opponent
        self._base_seed = seed
        self._episode_seed: int | None = None
        self.engine = HoolaEngine(seed=seed, max_turns=max_turns)
        self.opponent_agent = self._make_opponent(opponent, seed)

    @property
    def gymnasium_available(self) -> bool:
        return GYMNASIUM_AVAILABLE

    def _make_opponent(self, opponent: str | Any, seed: int | None):
        agent_seed = None if seed is None else seed + 10_000
        if opponent == "heuristic":
            return HeuristicAgent(seed=agent_seed)
        if opponent == "random":
            return RandomAgent(seed=agent_seed)
        if isinstance(opponent, str):
            raise ValueError(f"Unknown opponent: {opponent}")
        if not hasattr(opponent, "select_action"):
            raise TypeError("Custom opponent must provide select_action(observation, legal_actions)")
        return opponent

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Respect Gymnasium's seeding convention when available.
        try:
            super().reset(seed=seed)
        except TypeError:
            pass

        if seed is None:
            seed = secrets.randbits(63)
        self._episode_seed = int(seed)
        self.engine.reset(seed=self._episode_seed)
        if isinstance(self._opponent_spec, str):
            self.opponent_agent = self._make_opponent(self._opponent_spec, self._episode_seed)

        self._advance_opponent_if_needed()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        if self.engine.state.terminal:
            raise RuntimeError("Episode is finished; call reset() before step()")
        if self.engine.state.current_player != self.learning_player:
            raise RuntimeError("Internal error: opponent should have been auto-advanced")

        legal = self.engine.legal_actions()
        selected = self.codec.decode_legal(int(action), legal)
        self.engine.step(selected)

        if not self.engine.state.terminal:
            self._advance_opponent_if_needed()

        terminated = self.engine.state.terminal
        truncated = False
        reward = self._terminal_reward() if terminated else 0.0
        obs = self._get_obs()
        info = self._get_info()
        info["selected_action_id"] = int(action)
        info["selected_action"] = selected
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Mask compatible with common action-masking RL wrappers."""
        if self.engine.state.terminal:
            return np.zeros(self.codec.size, dtype=np.int8)
        if self.engine.state.current_player != self.learning_player:
            raise RuntimeError("Mask requested while opponent controls the state")
        return self.codec.action_mask(self.engine.legal_actions())

    def _advance_opponent_if_needed(self) -> None:
        while (
            not self.engine.state.terminal
            and self.engine.state.current_player != self.learning_player
        ):
            player = self.engine.state.current_player
            obs = make_observation(self.engine.state, player)
            legal = self.engine.legal_actions()
            if not legal:
                self.engine._end_draw()
                break
            action = self.opponent_agent.select_action(obs, legal)
            self.engine.step(action)

    def _get_obs(self) -> dict[str, np.ndarray]:
        if self.engine.state.terminal:
            # Preserve the learning player's final public view.  Mask is all 0.
            observation = make_observation(self.engine.state, self.learning_player)
            return {
                "observation": encode_observation(observation, max_turns=self.max_turns),
                "action_mask": np.zeros(self.codec.size, dtype=np.int8),
            }

        if self.engine.state.current_player != self.learning_player:
            raise RuntimeError("Observation requested before opponent auto-advance completed")
        observation = make_observation(self.engine.state, self.learning_player)
        return {
            "observation": encode_observation(observation, max_turns=self.max_turns),
            "action_mask": self.action_masks(),
        }

    def _terminal_reward(self) -> float:
        state = self.engine.state
        if not state.terminal or state.is_draw:
            return 0.0
        return 1.0 if state.winner == self.learning_player else -1.0

    def _get_info(self) -> dict[str, Any]:
        state = self.engine.state
        return {
            "seed": self._episode_seed,
            "learning_player": self.learning_player,
            "current_player": state.current_player,
            "winner": state.winner,
            "is_draw": state.is_draw,
            "turns_completed": state.turns_completed,
            "gymnasium_available": GYMNASIUM_AVAILABLE,
        }
