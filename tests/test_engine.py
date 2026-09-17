from hoola.action import DiscardAction, DrawStockAction, MeldAction, PassPlayAction, ThankYouAction
from hoola.card import Card, Rank, Suit
from hoola.engine import HoolaEngine
from hoola.state import GameState, TurnPhase


def C(suit, rank):
    return Card(suit, rank)


def test_initial_game_shape():
    engine = HoolaEngine(seed=1)
    s = engine.state
    assert len(s.hands[0]) == 7
    assert len(s.hands[1]) == 7
    assert len(s.discard_pile) == 1
    assert len(s.stock) == 37
    assert s.phase == TurnPhase.DRAW
    assert s.last_discarded_by is None
    legal = engine.legal_actions()
    assert DrawStockAction() in legal
    # Turn 1 cannot have Thank You because the initial face-up discard was not
    # discarded by an opponent. A pre-existing hand meld (including a single
    # seven) may now be registered without drawing.
    assert not any(isinstance(a, ThankYouAction) for a in legal)


def test_draw_then_pass_then_discard_changes_player():
    engine = HoolaEngine(seed=2)
    engine.step(DrawStockAction())
    assert engine.state.phase == TurnPhase.PLAY
    assert len(engine.state.hands[0]) == 8

    engine.step(PassPlayAction())
    assert engine.state.phase == TurnPhase.DISCARD

    card = engine.state.hands[0][0]
    engine.step(DiscardAction(card))
    assert engine.state.current_player == 1
    assert engine.state.phase == TurnPhase.DRAW
    assert engine.state.last_discarded_by == 0


def test_thank_you_is_legal_after_opponent_discard():
    engine = HoolaEngine(seed=3)
    engine.state = GameState(
        hands=[
            [C(Suit.CLUBS, Rank.THREE)],
            [C(Suit.SPADES, Rank.SIX), C(Suit.SPADES, Rank.SEVEN), C(Suit.HEARTS, Rank.KING)],
        ],
        stock=[C(Suit.DIAMONDS, Rank.ACE)],
        discard_pile=[C(Suit.CLUBS, Rank.TWO), C(Suit.SPADES, Rank.EIGHT)],
        current_player=1,
        phase=TurnPhase.DRAW,
        last_discarded_by=0,
        has_registered=[False, False],
    )

    thank_yous = [a for a in engine.legal_actions() if isinstance(a, ThankYouAction)]
    assert thank_yous
    engine.step(thank_yous[0])
    assert engine.state.has_registered[1]
    assert len(engine.state.melds) == 1
    assert C(Suit.SPADES, Rank.EIGHT) in engine.state.melds[0].cards


def test_melding_entire_hand_wins_immediately():
    engine = HoolaEngine(seed=4)
    run = [
        C(Suit.HEARTS, Rank.FIVE),
        C(Suit.HEARTS, Rank.SIX),
        C(Suit.HEARTS, Rank.SEVEN),
    ]
    engine.state = GameState(
        hands=[run.copy(), [C(Suit.CLUBS, Rank.ACE)]],
        stock=[C(Suit.DIAMONDS, Rank.TWO)],
        discard_pile=[C(Suit.SPADES, Rank.THREE)],
        current_player=0,
        phase=TurnPhase.PLAY,
        has_registered=[False, False],
    )
    action = MeldAction(tuple(sorted(run, key=lambda c: c.id)))
    assert action in engine.legal_actions()
    engine.step(action)
    assert engine.state.terminal
    assert engine.state.winner == 0


def test_thank_you_entire_hand_wins_immediately():
    engine = HoolaEngine(seed=10)
    hand = [
        C(Suit.CLUBS, Rank.THREE),
        C(Suit.HEARTS, Rank.THREE),
    ]
    discarded = C(Suit.DIAMONDS, Rank.THREE)
    engine.state = GameState(
        hands=[hand.copy(), [C(Suit.SPADES, Rank.KING)]],
        stock=[C(Suit.DIAMONDS, Rank.ACE)],
        discard_pile=[C(Suit.CLUBS, Rank.TWO), discarded],
        current_player=0,
        phase=TurnPhase.DRAW,
        last_discarded_by=1,
        has_registered=[False, False],
    )

    action = next(a for a in engine.legal_actions() if isinstance(a, ThankYouAction))
    engine.step(action)
    assert engine.state.terminal
    assert engine.state.winner == 0
    assert engine.state.hands[0] == []


def test_layoff_entire_hand_wins_immediately():
    from hoola.action import LayoffAction
    from hoola.state import TableMeld

    engine = HoolaEngine(seed=11)
    last_card = C(Suit.HEARTS, Rank.EIGHT)
    engine.state = GameState(
        hands=[[last_card], [C(Suit.SPADES, Rank.KING)]],
        stock=[C(Suit.DIAMONDS, Rank.ACE)],
        discard_pile=[C(Suit.CLUBS, Rank.TWO)],
        melds=[
            TableMeld(
                owner=1,
                cards=[
                    C(Suit.HEARTS, Rank.FIVE),
                    C(Suit.HEARTS, Rank.SIX),
                    C(Suit.HEARTS, Rank.SEVEN),
                ],
            )
        ],
        current_player=0,
        phase=TurnPhase.PLAY,
        has_registered=[True, True],
    )

    action = LayoffAction(meld_id=0, cards=(last_card,))
    assert action in engine.legal_actions()
    engine.step(action)
    assert engine.state.terminal
    assert engine.state.winner == 0
    assert engine.state.hands[0] == []


def test_discard_entire_hand_wins_immediately():
    engine = HoolaEngine(seed=12)
    last_card = C(Suit.SPADES, Rank.FOUR)
    engine.state = GameState(
        hands=[[last_card], [C(Suit.HEARTS, Rank.KING)]],
        stock=[C(Suit.DIAMONDS, Rank.ACE)],
        discard_pile=[C(Suit.CLUBS, Rank.TWO)],
        current_player=0,
        phase=TurnPhase.DISCARD,
        has_registered=[False, False],
    )

    engine.step(DiscardAction(last_card))
    assert engine.state.terminal
    assert engine.state.winner == 0
    assert engine.state.hands[0] == []


def test_can_register_from_hand_without_drawing():
    """At DRAW phase, an existing hand meld can be registered instead of drawing."""
    engine = HoolaEngine(seed=30)
    meld = [
        C(Suit.CLUBS, Rank.FIVE),
        C(Suit.DIAMONDS, Rank.FIVE),
        C(Suit.HEARTS, Rank.FIVE),
    ]
    extra = C(Suit.SPADES, Rank.KING)
    stock = [C(Suit.DIAMONDS, Rank.ACE), C(Suit.CLUBS, Rank.TWO)]
    engine.state = GameState(
        hands=[meld + [extra], [C(Suit.SPADES, Rank.THREE)]],
        stock=stock.copy(),
        discard_pile=[C(Suit.HEARTS, Rank.NINE)],
        current_player=0,
        phase=TurnPhase.DRAW,
        has_registered=[False, False],
    )

    legal = engine.legal_actions()
    action = MeldAction(tuple(sorted(meld, key=lambda c: c.id)))
    assert DrawStockAction() in legal
    assert action in legal

    stock_before = engine.state.stock.copy()
    engine.step(action)

    assert engine.state.stock == stock_before  # no card was drawn
    assert engine.state.phase == TurnPhase.PLAY
    assert engine.state.hands[0] == [extra]
    assert engine.state.has_registered[0]
    assert len(engine.state.melds) == 1


def test_no_draw_phase_meld_option_when_hand_has_no_registration():
    engine = HoolaEngine(seed=31)
    engine.state = GameState(
        hands=[
            [
                C(Suit.CLUBS, Rank.TWO),
                C(Suit.DIAMONDS, Rank.FOUR),
                C(Suit.HEARTS, Rank.SIX),
                C(Suit.SPADES, Rank.EIGHT),
            ],
            [C(Suit.SPADES, Rank.THREE)],
        ],
        stock=[C(Suit.DIAMONDS, Rank.ACE)],
        discard_pile=[C(Suit.HEARTS, Rank.NINE)],
        current_player=0,
        phase=TurnPhase.DRAW,
        has_registered=[False, False],
    )

    legal = engine.legal_actions()
    assert DrawStockAction() in legal
    assert not any(isinstance(a, MeldAction) for a in legal)


def test_register_without_draw_can_win_immediately():
    engine = HoolaEngine(seed=32)
    meld = [
        C(Suit.HEARTS, Rank.FIVE),
        C(Suit.HEARTS, Rank.SIX),
        C(Suit.HEARTS, Rank.SEVEN),
    ]
    engine.state = GameState(
        hands=[meld.copy(), [C(Suit.SPADES, Rank.THREE)]],
        stock=[C(Suit.DIAMONDS, Rank.ACE)],
        discard_pile=[C(Suit.CLUBS, Rank.NINE)],
        current_player=0,
        phase=TurnPhase.DRAW,
        has_registered=[False, False],
    )

    action = MeldAction(tuple(sorted(meld, key=lambda c: c.id)))
    assert action in engine.legal_actions()
    engine.step(action)
    assert engine.state.terminal
    assert engine.state.winner == 0


def test_can_layoff_to_own_single_seven_without_drawing():
    """A registered 7 may receive same-suit 6/8 at turn start without a draw."""
    from hoola.action import LayoffAction
    from hoola.state import TableMeld

    engine = HoolaEngine(seed=40)
    six_clubs = C(Suit.CLUBS, Rank.SIX)
    seven_clubs = C(Suit.CLUBS, Rank.SEVEN)
    extra = C(Suit.HEARTS, Rank.KING)
    stock = [C(Suit.DIAMONDS, Rank.ACE), C(Suit.SPADES, Rank.TWO)]
    engine.state = GameState(
        hands=[[six_clubs, extra], [C(Suit.SPADES, Rank.THREE)]],
        stock=stock.copy(),
        discard_pile=[C(Suit.HEARTS, Rank.NINE)],
        melds=[TableMeld(owner=0, cards=[seven_clubs])],
        current_player=0,
        phase=TurnPhase.DRAW,
        has_registered=[True, False],
    )

    action = LayoffAction(meld_id=0, cards=(six_clubs,))
    legal = engine.legal_actions()
    assert DrawStockAction() in legal
    assert action in legal

    stock_before = engine.state.stock.copy()
    engine.step(action)

    assert engine.state.stock == stock_before
    assert engine.state.phase == TurnPhase.PLAY
    assert engine.state.hands[0] == [extra]
    assert set(engine.state.melds[0].cards) == {six_clubs, seven_clubs}


def test_can_layoff_eight_to_own_single_seven_without_drawing():
    from hoola.action import LayoffAction
    from hoola.state import TableMeld

    engine = HoolaEngine(seed=41)
    eight_clubs = C(Suit.CLUBS, Rank.EIGHT)
    seven_clubs = C(Suit.CLUBS, Rank.SEVEN)
    engine.state = GameState(
        hands=[[eight_clubs, C(Suit.HEARTS, Rank.KING)], [C(Suit.SPADES, Rank.THREE)]],
        stock=[C(Suit.DIAMONDS, Rank.ACE)],
        discard_pile=[C(Suit.HEARTS, Rank.NINE)],
        melds=[TableMeld(owner=0, cards=[seven_clubs])],
        current_player=0,
        phase=TurnPhase.DRAW,
        has_registered=[True, False],
    )

    action = LayoffAction(meld_id=0, cards=(eight_clubs,))
    assert action in engine.legal_actions()


def test_unregistered_player_cannot_layoff_to_opponent_meld_at_draw_or_play():
    from hoola.action import LayoffAction
    from hoola.state import TableMeld

    eight_hearts = C(Suit.HEARTS, Rank.EIGHT)
    table_run = [
        C(Suit.HEARTS, Rank.FIVE),
        C(Suit.HEARTS, Rank.SIX),
        C(Suit.HEARTS, Rank.SEVEN),
    ]
    engine = HoolaEngine(seed=42)
    engine.state = GameState(
        hands=[[eight_hearts, C(Suit.CLUBS, Rank.KING)], [C(Suit.SPADES, Rank.THREE)]],
        stock=[C(Suit.DIAMONDS, Rank.ACE)],
        discard_pile=[C(Suit.CLUBS, Rank.TWO)],
        melds=[TableMeld(owner=1, cards=table_run.copy())],
        current_player=0,
        phase=TurnPhase.DRAW,
        has_registered=[False, True],
    )

    action = LayoffAction(meld_id=0, cards=(eight_hearts,))
    assert action not in engine.legal_actions()

    engine.state.phase = TurnPhase.PLAY
    assert action not in engine.legal_actions()


def test_registered_player_can_layoff_to_opponent_meld_without_drawing():
    from hoola.action import LayoffAction
    from hoola.state import TableMeld

    eight_hearts = C(Suit.HEARTS, Rank.EIGHT)
    table_run = [
        C(Suit.HEARTS, Rank.FIVE),
        C(Suit.HEARTS, Rank.SIX),
        C(Suit.HEARTS, Rank.SEVEN),
    ]
    engine = HoolaEngine(seed=43)
    engine.state = GameState(
        hands=[[eight_hearts, C(Suit.CLUBS, Rank.KING)], [C(Suit.SPADES, Rank.THREE)]],
        stock=[C(Suit.DIAMONDS, Rank.ACE)],
        discard_pile=[C(Suit.CLUBS, Rank.TWO)],
        melds=[TableMeld(owner=1, cards=table_run.copy())],
        current_player=0,
        phase=TurnPhase.DRAW,
        has_registered=[True, True],
    )

    action = LayoffAction(meld_id=0, cards=(eight_hearts,))
    assert action in engine.legal_actions()
