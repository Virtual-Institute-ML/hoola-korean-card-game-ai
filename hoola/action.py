from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .card import Card


class ActionType(Enum):
    DRAW_STOCK = auto()
    THANK_YOU = auto()
    MELD = auto()
    LAYOFF = auto()
    PASS_PLAY = auto()
    DISCARD = auto()


@dataclass(frozen=True, slots=True)
class DrawStockAction:
    type: ActionType = ActionType.DRAW_STOCK


@dataclass(frozen=True, slots=True)
class ThankYouAction:
    """Take the latest opponent discard and immediately register meld_cards."""
    meld_cards: tuple[Card, ...]
    type: ActionType = ActionType.THANK_YOU


@dataclass(frozen=True, slots=True)
class MeldAction:
    cards: tuple[Card, ...]
    type: ActionType = ActionType.MELD


@dataclass(frozen=True, slots=True)
class LayoffAction:
    meld_id: int
    cards: tuple[Card, ...]
    type: ActionType = ActionType.LAYOFF


@dataclass(frozen=True, slots=True)
class PassPlayAction:
    type: ActionType = ActionType.PASS_PLAY


@dataclass(frozen=True, slots=True)
class DiscardAction:
    card: Card
    type: ActionType = ActionType.DISCARD


Action = (
    DrawStockAction
    | ThankYouAction
    | MeldAction
    | LayoffAction
    | PassPlayAction
    | DiscardAction
)
