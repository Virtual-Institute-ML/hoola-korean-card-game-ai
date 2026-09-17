from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, auto

from .card import Card, Rank


class MeldType(Enum):
    SET = auto()
    RUN = auto()
    SINGLE_SEVEN = auto()


def _has_duplicate_cards(cards: Sequence[Card]) -> bool:
    return len(set(cards)) != len(cards)


def is_set(cards: Sequence[Card]) -> bool:
    """Same rank, 3 or 4 cards."""
    if len(cards) not in (3, 4) or _has_duplicate_cards(cards):
        return False
    return len({card.rank for card in cards}) == 1


def _is_cyclic_consecutive(ranks: Sequence[int]) -> bool:
    """
    Check consecutiveness on the 13-rank cycle A-2-...-Q-K-A.

    This makes Q-K-A and K-A-2 valid runs, matching the ruleset chosen
    for this project. Ranks must be unique.
    """
    n = len(ranks)
    if n < 3 or n > 13:
        return False

    unique = sorted(set(ranks))
    if len(unique) != n:
        return False

    rank_set = set(unique)
    for start in unique:
        expected = {((start - 1 + offset) % 13) + 1 for offset in range(n)}
        if expected == rank_set:
            return True
    return False


def is_run(cards: Sequence[Card]) -> bool:
    """Same suit, at least 3 cards, consecutive with K-A wrap allowed."""
    if len(cards) < 3 or len(cards) > 13 or _has_duplicate_cards(cards):
        return False
    if len({card.suit for card in cards}) != 1:
        return False
    return _is_cyclic_consecutive([int(card.rank) for card in cards])


def is_single_seven(cards: Sequence[Card]) -> bool:
    return len(cards) == 1 and cards[0].rank == Rank.SEVEN


def get_meld_type(cards: Sequence[Card]) -> MeldType | None:
    if is_single_seven(cards):
        return MeldType.SINGLE_SEVEN
    if is_set(cards):
        return MeldType.SET
    if is_run(cards):
        return MeldType.RUN
    return None


def is_valid_meld(cards: Sequence[Card]) -> bool:
    return get_meld_type(cards) is not None



def is_valid_layoff_result(cards: Sequence[Card]) -> bool:
    """
    Validate the table meld after cards are laid off.

    Normally the resulting table group must be a legal meld.  Hoola's
    singleton-seven rule needs one extra transitional state: after a 7 has
    been registered alone, the adjacent same-suit 6 or 8 may be attached,
    producing a temporary two-card sequence (6-7 or 7-8).  That two-card
    table sequence can then be extended normally on later actions.
    """
    if is_valid_meld(cards):
        return True

    if len(cards) != 2 or _has_duplicate_cards(cards):
        return False
    if len({card.suit for card in cards}) != 1:
        return False

    ranks = {int(card.rank) for card in cards}
    return ranks in ({6, 7}, {7, 8})

def find_valid_melds(cards: Sequence[Card]) -> list[tuple[Card, ...]]:
    """Enumerate all valid meld subsets from a small hand."""
    from itertools import combinations

    results: set[tuple[Card, ...]] = set()
    n = len(cards)
    for size in range(1, n + 1):
        for chosen in combinations(cards, size):
            if is_valid_meld(chosen):
                canonical = tuple(sorted(chosen, key=lambda c: c.id))
                results.add(canonical)
    return sorted(results, key=lambda m: (len(m), tuple(c.id for c in m)))
