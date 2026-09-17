from hoola.card import Card, Rank, Suit
from hoola.rules import can_thank_you, valid_thank_you_melds, validate_thank_you_choice


def C(suit, rank):
    return Card(suit, rank)


def test_thank_you_completes_run():
    hand = [
        C(Suit.SPADES, Rank.SIX),
        C(Suit.SPADES, Rank.SEVEN),
        C(Suit.HEARTS, Rank.KING),
    ]
    discard = C(Suit.SPADES, Rank.EIGHT)
    assert can_thank_you(hand, discard)
    assert validate_thank_you_choice(
        hand,
        discard,
        [C(Suit.SPADES, Rank.SIX), C(Suit.SPADES, Rank.SEVEN)],
    )


def test_thank_you_completes_set():
    hand = [
        C(Suit.SPADES, Rank.QUEEN),
        C(Suit.HEARTS, Rank.QUEEN),
        C(Suit.CLUBS, Rank.THREE),
    ]
    discard = C(Suit.DIAMONDS, Rank.QUEEN)
    assert can_thank_you(hand, discard)


def test_thank_you_supports_k_a_2_run():
    hand = [
        C(Suit.SPADES, Rank.KING),
        C(Suit.SPADES, Rank.ACE),
        C(Suit.CLUBS, Rank.THREE),
    ]
    discard = C(Suit.SPADES, Rank.TWO)
    assert can_thank_you(hand, discard)


def test_thank_you_not_possible_for_unrelated_discard():
    hand = [
        C(Suit.SPADES, Rank.SIX),
        C(Suit.SPADES, Rank.SEVEN),
        C(Suit.HEARTS, Rank.KING),
    ]
    discard = C(Suit.DIAMONDS, Rank.TWO)
    assert not can_thank_you(hand, discard)
    assert valid_thank_you_melds(hand, discard) == []


def test_discarded_seven_alone_is_not_thank_you():
    # In this project, Thank You must combine the discard with >=2 cards
    # from the hand to create a new meld. Single-seven registration alone
    # does not qualify as Thank You.
    hand = [
        C(Suit.SPADES, Rank.TWO),
        C(Suit.HEARTS, Rank.THREE),
    ]
    discard = C(Suit.DIAMONDS, Rank.SEVEN)
    assert not can_thank_you(hand, discard)
