from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from .action import (
    Action,
    DiscardAction,
    DrawStockAction,
    LayoffAction,
    MeldAction,
    PassPlayAction,
    ThankYouAction,
)
from .card import Card, Rank, Suit


def _canonical(cards: Iterable[Card]) -> tuple[Card, ...]:
    return tuple(sorted(cards, key=lambda c: c.id))


def build_meld_catalog() -> tuple[tuple[Card, ...], ...]:
    """Enumerate every possible meld in the project's 52-card ruleset.

    This avoids state-dependent action IDs.  A meld has the same integer ID in
    every game state, which is important for a conventional policy head.
    """
    melds: set[tuple[Card, ...]] = set()

    # Single-seven registrations.
    for suit in Suit:
        melds.add((Card(suit, Rank.SEVEN),))

    # Sets: same rank, 3 or 4 suits.
    for rank in Rank:
        rank_cards = [Card(suit, rank) for suit in Suit]
        for size in (3, 4):
            for chosen in combinations(rank_cards, size):
                melds.add(_canonical(chosen))

    # Cyclic same-suit runs, lengths 3..13 (K-A wrap is allowed).
    for suit in Suit:
        for length in range(3, 14):
            for start in range(1, 14):
                cards = [
                    Card(suit, Rank(((start - 1 + offset) % 13) + 1))
                    for offset in range(length)
                ]
                melds.add(_canonical(cards))

    return tuple(sorted(melds, key=lambda m: (len(m), tuple(c.id for c in m))))


MELD_CATALOG = build_meld_catalog()
MELD_TO_INDEX = {meld: i for i, meld in enumerate(MELD_CATALOG)}

# At most four singleton-seven melds can exist; all other newly registered melds
# consume at least three distinct cards.  Therefore 4 + floor(48/3) = 20 is a
# safe upper bound on simultaneously existing table melds.
MAX_TABLE_MELDS = 20

# The only atomic two-card layoff needed by the normalized RL action space is
# adding two additional sevens to a singleton seven to create a 3-card set.
SEVEN_PAIRS = tuple(
    _canonical(pair)
    for pair in combinations([Card(suit, Rank.SEVEN) for suit in Suit], 2)
)
SEVEN_PAIR_TO_INDEX = {pair: i for i, pair in enumerate(SEVEN_PAIRS)}


@dataclass(frozen=True, slots=True)
class ActionLayout:
    draw: int
    pass_play: int
    discard_start: int
    meld_start: int
    thank_you_start: int
    layoff_single_start: int
    layoff_pair_start: int
    size: int


class ActionCodec:
    """Stable integer encoding for all actions used by the RL interface.

    Layout
    ------
    0                     DRAW_STOCK
    1                     PASS_PLAY
    next 52               DISCARD(card_id)
    next |meld catalog|   MELD(cards)
    next |meld catalog|   THANK_YOU(cards)
    next 20*52            LAYOFF(meld_slot, one card)
    next 20*6             LAYOFF(single-seven meld_slot, two sevens)

    The engine normalizes layoff generation to these atomic actions, so every
    legal engine action has one stable global ID.
    """

    def __init__(self) -> None:
        discard_start = 2
        meld_start = discard_start + 52
        thank_you_start = meld_start + len(MELD_CATALOG)
        layoff_single_start = thank_you_start + len(MELD_CATALOG)
        layoff_pair_start = layoff_single_start + MAX_TABLE_MELDS * 52
        size = layoff_pair_start + MAX_TABLE_MELDS * len(SEVEN_PAIRS)
        self.layout = ActionLayout(
            draw=0,
            pass_play=1,
            discard_start=discard_start,
            meld_start=meld_start,
            thank_you_start=thank_you_start,
            layoff_single_start=layoff_single_start,
            layoff_pair_start=layoff_pair_start,
            size=size,
        )

    @property
    def size(self) -> int:
        return self.layout.size

    def encode(self, action: Action) -> int:
        if isinstance(action, DrawStockAction):
            return self.layout.draw
        if isinstance(action, PassPlayAction):
            return self.layout.pass_play
        if isinstance(action, DiscardAction):
            return self.layout.discard_start + action.card.id
        if isinstance(action, MeldAction):
            meld = _canonical(action.cards)
            try:
                return self.layout.meld_start + MELD_TO_INDEX[meld]
            except KeyError as exc:
                raise ValueError(f"Meld is not in global catalog: {meld}") from exc
        if isinstance(action, ThankYouAction):
            meld = _canonical(action.meld_cards)
            try:
                return self.layout.thank_you_start + MELD_TO_INDEX[meld]
            except KeyError as exc:
                raise ValueError(f"Thank You meld is not in global catalog: {meld}") from exc
        if isinstance(action, LayoffAction):
            if not 0 <= action.meld_id < MAX_TABLE_MELDS:
                raise ValueError(f"meld_id out of encodable range: {action.meld_id}")
            cards = _canonical(action.cards)
            if len(cards) == 1:
                return self.layout.layoff_single_start + action.meld_id * 52 + cards[0].id
            if len(cards) == 2 and cards in SEVEN_PAIR_TO_INDEX:
                return (
                    self.layout.layoff_pair_start
                    + action.meld_id * len(SEVEN_PAIRS)
                    + SEVEN_PAIR_TO_INDEX[cards]
                )
            raise ValueError(
                "RL action codec supports atomic layoffs: one card, or two sevens "
                "onto a singleton-seven meld"
            )
        raise TypeError(f"Unsupported action type: {type(action)!r}")

    def action_mask(self, legal_actions: list[Action]) -> np.ndarray:
        mask = np.zeros(self.size, dtype=np.int8)
        for action in legal_actions:
            mask[self.encode(action)] = 1
        return mask

    def decode_legal(self, action_id: int, legal_actions: list[Action]) -> Action:
        if not 0 <= int(action_id) < self.size:
            raise ValueError(f"action_id must be in [0, {self.size - 1}], got {action_id}")
        action_id = int(action_id)
        for action in legal_actions:
            if self.encode(action) == action_id:
                return action
        raise ValueError(f"Action id {action_id} is not legal in the current state")

    def describe_id(self, action_id: int) -> str:
        """Return a stable human-readable description of an action ID."""
        i = int(action_id)
        if i == self.layout.draw:
            return "DRAW_STOCK"
        if i == self.layout.pass_play:
            return "PASS_PLAY"
        if self.layout.discard_start <= i < self.layout.meld_start:
            return f"DISCARD card_id={i - self.layout.discard_start}"
        if self.layout.meld_start <= i < self.layout.thank_you_start:
            meld = MELD_CATALOG[i - self.layout.meld_start]
            return "MELD " + " ".join(str(c) for c in meld)
        if self.layout.thank_you_start <= i < self.layout.layoff_single_start:
            meld = MELD_CATALOG[i - self.layout.thank_you_start]
            return "THANK_YOU " + " ".join(str(c) for c in meld)
        if self.layout.layoff_single_start <= i < self.layout.layoff_pair_start:
            rel = i - self.layout.layoff_single_start
            meld_id, card_id = divmod(rel, 52)
            return f"LAYOFF card_id={card_id} -> meld_id={meld_id}"
        if self.layout.layoff_pair_start <= i < self.size:
            rel = i - self.layout.layoff_pair_start
            meld_id, pair_id = divmod(rel, len(SEVEN_PAIRS))
            return (
                "LAYOFF " + " ".join(str(c) for c in SEVEN_PAIRS[pair_id])
                + f" -> meld_id={meld_id}"
            )
        raise ValueError(f"Unknown action id: {action_id}")
