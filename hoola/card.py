from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13


SUIT_SYMBOLS = {
    Suit.CLUBS: "♣",
    Suit.DIAMONDS: "♦",
    Suit.HEARTS: "♥",
    Suit.SPADES: "♠",
}

RANK_SYMBOLS = {
    Rank.ACE: "A",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
}


@dataclass(frozen=True, order=True, slots=True)
class Card:
    suit: Suit
    rank: Rank

    @property
    def id(self) -> int:
        """Unique card id in [0, 51]."""
        return int(self.suit) * 13 + int(self.rank) - 1

    @classmethod
    def from_id(cls, card_id: int) -> "Card":
        if not 0 <= card_id < 52:
            raise ValueError(f"card_id must be in [0, 51], got {card_id}")
        suit = Suit(card_id // 13)
        rank = Rank(card_id % 13 + 1)
        return cls(suit=suit, rank=rank)

    def __str__(self) -> str:
        rank = RANK_SYMBOLS.get(self.rank, str(int(self.rank)))
        return f"{rank}{SUIT_SYMBOLS[self.suit]}"

    def __repr__(self) -> str:
        return f"Card({str(self)})"


def create_deck() -> list[Card]:
    return [Card(suit=suit, rank=rank) for suit in Suit for rank in Rank]


def cards_to_ids(cards: Iterable[Card]) -> tuple[int, ...]:
    return tuple(card.id for card in cards)
