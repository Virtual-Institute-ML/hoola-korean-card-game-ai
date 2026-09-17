from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .card import Card


class TurnPhase(Enum):
    DRAW = auto()
    PLAY = auto()
    DISCARD = auto()
    TERMINAL = auto()


@dataclass(slots=True)
class TableMeld:
    owner: int
    cards: list[Card]


@dataclass(slots=True)
class GameState:
    hands: list[list[Card]]
    stock: list[Card]
    discard_pile: list[Card]
    melds: list[TableMeld] = field(default_factory=list)
    has_registered: list[bool] = field(default_factory=lambda: [False, False])
    current_player: int = 0
    phase: TurnPhase = TurnPhase.DRAW
    last_discarded_by: int | None = None
    winner: int | None = None
    is_draw: bool = False
    turns_completed: int = 0

    @property
    def terminal(self) -> bool:
        return self.phase == TurnPhase.TERMINAL
