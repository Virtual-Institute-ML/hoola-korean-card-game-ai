from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .action import Action
from .action_codec import ActionCodec
from .encoding import OBSERVATION_SIZE, encode_observation
from .observation import Observation
from .state import GameState


class TrainingRecorder:
    """Compact fixed-shape recorder for imitation/value/RL preprocessing.

    Each decision stores the acting player's imperfect-information observation,
    the global legal-action mask, and a stable global action ID.  ``finalize``
    adds the final outcome from each acting player's perspective.
    """

    def __init__(
        self,
        out_path: str | Path,
        *,
        max_turns: int = 500,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.out_path = Path(out_path)
        self.max_turns = max_turns
        self.metadata = metadata or {}
        self.codec = ActionCodec()
        self.observations: list[np.ndarray] = []
        self.action_masks: list[np.ndarray] = []
        self.action_ids: list[int] = []
        self.players: list[int] = []
        self.agent_types: list[str] = []
        self.phases: list[str] = []
        self.turn_indices: list[int] = []
        self.terminal_after_action: list[bool] = []

    def record_decision(
        self,
        *,
        acting_player: int,
        acting_agent: str,
        observation: Observation,
        legal_actions: list[Action],
        selected_action: Action,
        state_after: GameState,
    ) -> None:
        self.observations.append(encode_observation(observation, max_turns=self.max_turns))
        self.action_masks.append(self.codec.action_mask(legal_actions))
        self.action_ids.append(self.codec.encode(selected_action))
        self.players.append(int(acting_player))
        self.agent_types.append(str(acting_agent))
        self.phases.append(observation.phase.name)
        self.turn_indices.append(int(observation.turns_completed))
        self.terminal_after_action.append(bool(state_after.terminal))

    def save(self, final_state: GameState) -> tuple[Path, Path]:
        n = len(self.action_ids)
        final_outcomes = np.zeros(n, dtype=np.float32)
        terminal_rewards = np.zeros(n, dtype=np.float32)

        if final_state.terminal and not final_state.is_draw and final_state.winner is not None:
            for i, player in enumerate(self.players):
                final_outcomes[i] = 1.0 if player == final_state.winner else -1.0
                if self.terminal_after_action[i]:
                    terminal_rewards[i] = final_outcomes[i]

        observations = (
            np.stack(self.observations).astype(np.float32)
            if self.observations
            else np.empty((0, OBSERVATION_SIZE), dtype=np.float32)
        )
        masks = (
            np.stack(self.action_masks).astype(np.int8)
            if self.action_masks
            else np.empty((0, self.codec.size), dtype=np.int8)
        )

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.out_path,
            observations=observations,
            action_masks=masks,
            action_ids=np.asarray(self.action_ids, dtype=np.int32),
            acting_players=np.asarray(self.players, dtype=np.int8),
            final_outcomes=final_outcomes,
            terminal_rewards=terminal_rewards,
            terminal_after_action=np.asarray(self.terminal_after_action, dtype=np.bool_),
            turn_indices=np.asarray(self.turn_indices, dtype=np.int32),
        )

        meta_path = self.out_path.with_suffix(".json")
        meta = {
            **self.metadata,
            "format": "hoola-training-v1",
            "num_steps": n,
            "observation_size": OBSERVATION_SIZE,
            "action_space_size": self.codec.size,
            "agent_types": self.agent_types,
            "phases": self.phases,
            "result": {
                "winner": final_state.winner,
                "is_draw": final_state.is_draw,
                "turns_completed": final_state.turns_completed,
            },
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.out_path, meta_path
