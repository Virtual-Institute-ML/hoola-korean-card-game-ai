from __future__ import annotations

from dataclasses import dataclass

from .card import Card
from .state import GameState, TurnPhase


@dataclass(frozen=True, slots=True)
class PublicMeldView:
    owner: int
    cards: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class Observation:
    player_id: int
    my_hand: tuple[Card, ...]
    opponent_hand_size: int
    stock_size: int
    discard_pile: tuple[Card, ...]
    melds: tuple[PublicMeldView, ...]
    has_registered: tuple[bool, bool]
    current_player: int
    phase: TurnPhase
    turns_completed: int


def make_observation(state: GameState, player_id: int) -> Observation:
    if player_id not in (0, 1):
        raise ValueError("player_id must be 0 or 1")
    opponent = 1 - player_id
    return Observation(
        player_id=player_id,
        my_hand=tuple(sorted(state.hands[player_id], key=lambda c: c.id)),
        opponent_hand_size=len(state.hands[opponent]),
        stock_size=len(state.stock),
        discard_pile=tuple(state.discard_pile),
        melds=tuple(
            PublicMeldView(m.owner, tuple(sorted(m.cards, key=lambda c: c.id)))
            for m in state.melds
        ),
        has_registered=tuple(state.has_registered),
        current_player=state.current_player,
        phase=state.phase,
        turns_completed=state.turns_completed,
    )
