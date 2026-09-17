import pytest

from hoola.card import Card, Rank, Suit
from hoola.meld import MeldType, get_meld_type, is_run, is_set, is_single_seven, is_valid_meld


def C(suit, rank):
    return Card(suit, rank)


def test_valid_sets():
    assert is_set([C(Suit.SPADES, Rank.FIVE), C(Suit.HEARTS, Rank.FIVE), C(Suit.DIAMONDS, Rank.FIVE)])
    assert is_set([
        C(Suit.SPADES, Rank.QUEEN),
        C(Suit.HEARTS, Rank.QUEEN),
        C(Suit.DIAMONDS, Rank.QUEEN),
        C(Suit.CLUBS, Rank.QUEEN),
    ])


def test_invalid_sets():
    assert not is_set([C(Suit.SPADES, Rank.FIVE), C(Suit.HEARTS, Rank.FIVE)])
    assert not is_set([C(Suit.SPADES, Rank.FIVE), C(Suit.HEARTS, Rank.FIVE), C(Suit.DIAMONDS, Rank.SIX)])


@pytest.mark.parametrize(
    "cards",
    [
        [C(Suit.CLUBS, Rank.FOUR), C(Suit.CLUBS, Rank.FIVE), C(Suit.CLUBS, Rank.SIX)],
        [C(Suit.SPADES, Rank.JACK), C(Suit.SPADES, Rank.QUEEN), C(Suit.SPADES, Rank.KING)],
        [C(Suit.SPADES, Rank.QUEEN), C(Suit.SPADES, Rank.KING), C(Suit.SPADES, Rank.ACE)],
        [C(Suit.SPADES, Rank.KING), C(Suit.SPADES, Rank.ACE), C(Suit.SPADES, Rank.TWO)],
        [C(Suit.HEARTS, Rank.KING), C(Suit.HEARTS, Rank.ACE), C(Suit.HEARTS, Rank.TWO), C(Suit.HEARTS, Rank.THREE)],
    ],
)
def test_valid_runs(cards):
    assert is_run(cards)


@pytest.mark.parametrize(
    "cards",
    [
        [C(Suit.SPADES, Rank.FOUR), C(Suit.SPADES, Rank.FIVE)],
        [C(Suit.CLUBS, Rank.FOUR), C(Suit.CLUBS, Rank.FIVE), C(Suit.DIAMONDS, Rank.SIX)],
        [C(Suit.SPADES, Rank.KING), C(Suit.SPADES, Rank.ACE), C(Suit.SPADES, Rank.THREE)],
        [C(Suit.SPADES, Rank.FOUR), C(Suit.SPADES, Rank.FOUR), C(Suit.SPADES, Rank.FIVE)],
    ],
)
def test_invalid_runs(cards):
    assert not is_run(cards)


def test_single_seven():
    seven = [C(Suit.HEARTS, Rank.SEVEN)]
    six = [C(Suit.HEARTS, Rank.SIX)]
    assert is_single_seven(seven)
    assert is_valid_meld(seven)
    assert get_meld_type(seven) == MeldType.SINGLE_SEVEN
    assert not is_single_seven(six)
    assert not is_valid_meld(six)


def test_meld_type_detection():
    set_cards = [C(Suit.SPADES, Rank.FIVE), C(Suit.HEARTS, Rank.FIVE), C(Suit.DIAMONDS, Rank.FIVE)]
    run_cards = [C(Suit.CLUBS, Rank.FOUR), C(Suit.CLUBS, Rank.FIVE), C(Suit.CLUBS, Rank.SIX)]
    assert get_meld_type(set_cards) == MeldType.SET
    assert get_meld_type(run_cards) == MeldType.RUN


def test_single_seven_table_extension_accepts_adjacent_same_suit_card():
    from hoola.meld import is_valid_layoff_result

    assert is_valid_layoff_result([
        C(Suit.CLUBS, Rank.SIX),
        C(Suit.CLUBS, Rank.SEVEN),
    ])
    assert is_valid_layoff_result([
        C(Suit.CLUBS, Rank.SEVEN),
        C(Suit.CLUBS, Rank.EIGHT),
    ])


def test_single_seven_table_extension_rejects_wrong_two_card_pair():
    from hoola.meld import is_valid_layoff_result

    assert not is_valid_layoff_result([
        C(Suit.CLUBS, Rank.FIVE),
        C(Suit.CLUBS, Rank.SEVEN),
    ])
    assert not is_valid_layoff_result([
        C(Suit.CLUBS, Rank.SEVEN),
        C(Suit.HEARTS, Rank.EIGHT),
    ])
