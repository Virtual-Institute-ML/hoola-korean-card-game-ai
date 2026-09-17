from agents import HeuristicAgent
from hoola.action import (
    DiscardAction,
    DrawStockAction,
    MeldAction,
    ThankYouAction,
)
from hoola.card import Card, Rank, Suit
from hoola.observation import Observation
from hoola.state import TurnPhase


def C(suit, rank):
    return Card(suit, rank)


def make_obs(hand, phase, *, discard_pile=(), opponent_hand_size=7, registered=False):
    return Observation(
        player_id=0,
        my_hand=tuple(sorted(hand, key=lambda c: c.id)),
        opponent_hand_size=opponent_hand_size,
        stock_size=30,
        discard_pile=tuple(discard_pile),
        melds=(),
        has_registered=(registered, False),
        current_player=0,
        phase=phase,
        turns_completed=0,
    )


def test_heuristic_discards_isolated_card_and_keeps_adjacent_run_pair():
    five = C(Suit.SPADES, Rank.FIVE)
    six = C(Suit.SPADES, Rank.SIX)
    isolated = C(Suit.DIAMONDS, Rank.NINE)
    obs = make_obs([five, six, isolated], TurnPhase.DISCARD)
    legal = [DiscardAction(five), DiscardAction(six), DiscardAction(isolated)]

    agent = HeuristicAgent(seed=1)
    assert agent.select_action(obs, legal) == DiscardAction(isolated)


def test_heuristic_prefers_four_card_set_over_three_card_subset():
    set4 = [
        C(Suit.CLUBS, Rank.FIVE),
        C(Suit.DIAMONDS, Rank.FIVE),
        C(Suit.HEARTS, Rank.FIVE),
        C(Suit.SPADES, Rank.FIVE),
    ]
    extra = C(Suit.CLUBS, Rank.KING)
    obs = make_obs(set4 + [extra], TurnPhase.PLAY)
    meld3 = MeldAction(tuple(sorted(set4[:3], key=lambda c: c.id)))
    meld4 = MeldAction(tuple(sorted(set4, key=lambda c: c.id)))

    agent = HeuristicAgent(seed=2)
    assert agent.select_action(obs, [meld3, meld4]) == meld4


def test_heuristic_prefers_thank_you_over_blind_stock_draw():
    c3 = C(Suit.CLUBS, Rank.THREE)
    h3 = C(Suit.HEARTS, Rank.THREE)
    d3 = C(Suit.DIAMONDS, Rank.THREE)
    king = C(Suit.SPADES, Rank.KING)
    obs = make_obs(
        [c3, h3, king],
        TurnPhase.DRAW,
        discard_pile=(C(Suit.CLUBS, Rank.ACE), d3),
    )
    thank_you = ThankYouAction(tuple(sorted((c3, h3, d3), key=lambda c: c.id)))

    agent = HeuristicAgent(seed=3)
    assert agent.select_action(obs, [DrawStockAction(), thank_you]) == thank_you


def test_thank_you_risk_uses_only_public_card_availability():
    candidate = C(Suit.SPADES, Rank.SEVEN)
    hand = [candidate, C(Suit.CLUBS, Rank.KING)]
    agent = HeuristicAgent(seed=4)

    open_obs = make_obs(hand, TurnPhase.DISCARD, opponent_hand_size=7)
    open_risk = agent.estimate_thank_you_risk(open_obs, candidate)

    # Publicly exposing two potential partner cards can only reduce risk.
    blocked_obs = make_obs(
        hand,
        TurnPhase.DISCARD,
        opponent_hand_size=7,
        discard_pile=(C(Suit.SPADES, Rank.FIVE), C(Suit.SPADES, Rank.SIX)),
    )
    blocked_risk = agent.estimate_thank_you_risk(blocked_obs, candidate)

    assert 0.0 <= blocked_risk < open_risk <= 1.0


def test_public_explanation_is_available():
    from hoola.engine import HoolaEngine
    from hoola.observation import make_observation

    engine = HoolaEngine(seed=3)
    obs = make_observation(engine.state, 0)
    legal = engine.legal_actions()
    agent = HeuristicAgent(seed=3)
    action = agent.select_action(obs, legal)
    text = agent.explain_decision(obs, legal, action, mode="public")
    assert "Heuristic v1 chose" in text
    assert "score=" in text


def test_full_explanation_lists_candidates():
    from hoola.engine import HoolaEngine
    from hoola.observation import make_observation

    engine = HoolaEngine(seed=4)
    obs = make_observation(engine.state, 0)
    legal = engine.legal_actions()
    agent = HeuristicAgent(seed=4)
    action = agent.select_action(obs, legal)
    text = agent.explain_decision(obs, legal, action, mode="full")
    assert "FULL DEBUG" in text
    assert "top candidates" in text
