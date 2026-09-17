from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from collections.abc import Iterable, Sequence

from .card import Card
from .meld import is_valid_meld


@dataclass(frozen=True, slots=True)
class RuleConfig:
    num_players: int = 2
    starting_hand_size: int = 7
    allow_single_seven: bool = True
    allow_thank_you: bool = True
    allow_meld_before_draw: bool = True
    allow_layoff_before_draw: bool = True
    allow_stop: bool = False


DEFAULT_RULES = RuleConfig()


def valid_thank_you_melds(hand: Sequence[Card], discarded_card: Card) -> list[tuple[Card, ...]]:
    """
    Return every new meld that can be formed by taking the opponent's
    most recent discard.

    Project rule:
    - the discarded card must be part of a NEW meld;
    - using it only to lay off onto an existing meld is not Thank You;
    - the taken discard is committed immediately to that meld.
    """
    if discarded_card in hand:
        raise ValueError("discarded_card must not already be in the player's hand")

    results: set[tuple[Card, ...]] = set()

    # Single-seven Thank You is intentionally not permitted: the discard must
    # combine with cards already in hand to make a new meld.
    # Therefore choose at least 2 hand cards.
    for n_hand_cards in range(2, len(hand) + 1):
        for chosen in combinations(hand, n_hand_cards):
            meld = tuple(chosen) + (discarded_card,)
            if is_valid_meld(meld):
                # Canonical ordering makes duplicate detection deterministic.
                results.add(tuple(sorted(meld, key=lambda c: c.id)))

    return sorted(results, key=lambda meld: (len(meld), tuple(card.id for card in meld)))


def can_thank_you(hand: Sequence[Card], discarded_card: Card) -> bool:
    return bool(valid_thank_you_melds(hand, discarded_card))


def validate_thank_you_choice(
    hand: Sequence[Card],
    discarded_card: Card,
    chosen_hand_cards: Iterable[Card],
) -> bool:
    chosen = tuple(chosen_hand_cards)
    if len(chosen) < 2:
        return False
    if len(set(chosen)) != len(chosen):
        return False
    if any(card not in hand for card in chosen):
        return False
    return is_valid_meld(chosen + (discarded_card,))
