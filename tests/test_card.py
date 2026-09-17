from hoola.card import Card, Rank, Suit, create_deck


def test_deck_has_52_unique_cards():
    deck = create_deck()
    assert len(deck) == 52
    assert len(set(deck)) == 52
    assert {card.id for card in deck} == set(range(52))


def test_card_id_round_trip():
    for card_id in range(52):
        card = Card.from_id(card_id)
        assert card.id == card_id


def test_card_string():
    assert str(Card(Suit.CLUBS, Rank.ACE)) == "A♣"
    assert str(Card(Suit.SPADES, Rank.KING)) == "K♠"
    assert str(Card(Suit.HEARTS, Rank.SEVEN)) == "7♥"
