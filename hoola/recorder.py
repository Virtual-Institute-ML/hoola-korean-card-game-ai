from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .action import (
    Action,
    DiscardAction,
    DrawStockAction,
    LayoffAction,
    MeldAction,
    PassPlayAction,
    ThankYouAction,
)
from .card import Card
from .observation import Observation
from .state import GameState


class GameRecorder:
    """Record games as JSON for offline analysis or future imitation learning."""

    def __init__(
        self,
        out_path: str | Path,
        *,
        include_private_state: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.out_path = Path(out_path)
        self.include_private_state = include_private_state
        self.metadata = metadata or {}
        self.steps: list[dict[str, Any]] = []

    def record_step(
        self,
        *,
        acting_player: int,
        acting_agent: str,
        observation: Observation,
        legal_actions: list[Action],
        selected_action: Action,
        state_before: GameState,
        state_after: GameState,
    ) -> None:
        step = {
            "acting_player": acting_player,
            "acting_agent": acting_agent,
            "phase": state_before.phase.name,
            "turn_index": state_before.turns_completed,
            "observation": self._serialize_observation(observation),
            "legal_actions": [self._serialize_action(a) for a in legal_actions],
            "selected_action": self._serialize_action(selected_action),
            "public_state_before": self._serialize_public_state(state_before),
            "public_state_after": self._serialize_public_state(state_after),
        }
        if self.include_private_state:
            step["private_state_before"] = self._serialize_private_state(state_before)
            step["private_state_after"] = self._serialize_private_state(state_after)
        self.steps.append(step)

    def save(self, final_state: GameState) -> None:
        payload = {
            "metadata": self.metadata,
            "num_steps": len(self.steps),
            "steps": self.steps,
            "result": {
                "winner": final_state.winner,
                "is_draw": final_state.is_draw,
                "turns_completed": final_state.turns_completed,
            },
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _card(card: Card) -> dict[str, Any]:
        return {
            "id": card.id,
            "text": str(card),
            "suit": card.suit.name,
            "rank": int(card.rank),
        }

    @classmethod
    def _cards(cls, cards: list[Card] | tuple[Card, ...]) -> list[dict[str, Any]]:
        return [cls._card(card) for card in cards]

    @classmethod
    def _serialize_action(cls, action: Action) -> dict[str, Any]:
        if isinstance(action, DrawStockAction):
            return {"type": action.type.name}
        if isinstance(action, ThankYouAction):
            return {"type": action.type.name, "meld_cards": cls._cards(action.meld_cards)}
        if isinstance(action, MeldAction):
            return {"type": action.type.name, "cards": cls._cards(action.cards)}
        if isinstance(action, LayoffAction):
            return {
                "type": action.type.name,
                "meld_id": action.meld_id,
                "cards": cls._cards(action.cards),
            }
        if isinstance(action, PassPlayAction):
            return {"type": action.type.name}
        if isinstance(action, DiscardAction):
            return {"type": action.type.name, "card": cls._card(action.card)}
        raise TypeError(f"Unsupported action type: {type(action)!r}")

    @classmethod
    def _serialize_observation(cls, observation: Observation) -> dict[str, Any]:
        return {
            "player_id": observation.player_id,
            "my_hand": cls._cards(observation.my_hand),
            "opponent_hand_size": observation.opponent_hand_size,
            "stock_size": observation.stock_size,
            "discard_pile": cls._cards(observation.discard_pile),
            "melds": [
                {"owner": meld.owner, "cards": cls._cards(meld.cards)}
                for meld in observation.melds
            ],
            "has_registered": list(observation.has_registered),
            "current_player": observation.current_player,
            "phase": observation.phase.name,
            "turns_completed": observation.turns_completed,
        }

    @classmethod
    def _serialize_public_state(cls, state: GameState) -> dict[str, Any]:
        return {
            "current_player": state.current_player,
            "phase": state.phase.name,
            "turns_completed": state.turns_completed,
            "stock_size": len(state.stock),
            "discard_pile": cls._cards(tuple(state.discard_pile)),
            "melds": [
                {"owner": meld.owner, "cards": cls._cards(tuple(meld.cards))}
                for meld in state.melds
            ],
            "has_registered": list(state.has_registered),
            "hand_sizes": [len(hand) for hand in state.hands],
            "winner": state.winner,
            "is_draw": state.is_draw,
        }

    @classmethod
    def _serialize_private_state(cls, state: GameState) -> dict[str, Any]:
        return {
            **cls._serialize_public_state(state),
            "hands": [cls._cards(tuple(hand)) for hand in state.hands],
            "stock": cls._cards(tuple(state.stock)),
            "last_discarded_by": state.last_discarded_by,
        }
