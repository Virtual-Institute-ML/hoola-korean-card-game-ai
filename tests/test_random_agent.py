from agents import RandomAgent
from hoola.action import MeldAction, PassPlayAction
from hoola.card import Card, Rank, Suit
from hoola.observation import Observation
from hoola.state import TurnPhase


def _obs_with_hand(hand):
    return Observation(
        player_id=0,
        my_hand=tuple(hand),
        opponent_hand_size=7,
        stock_size=30,
        discard_pile=(),
        melds=(),
        has_registered=(False, False),
        current_player=0,
        phase=TurnPhase.PLAY,
        turns_completed=0,
    )


def test_single_seven_can_be_suppressed_without_changing_legality():
    seven = Card(Suit.SPADES, Rank.SEVEN)
    single_seven = MeldAction((seven,))
    pass_action = PassPlayAction()

    agent = RandomAgent(seed=1, single_seven_weight=0.0)
    obs = _obs_with_hand([seven, Card(Suit.CLUBS, Rank.KING)])

    # The single-seven action is still in the legal-action list; only the
    # baseline agent's sampling policy de-prioritizes it.
    legal = [single_seven, pass_action]
    for _ in range(50):
        assert agent.select_action(obs, legal) == pass_action


def test_default_random_agent_does_not_always_choose_single_seven():
    seven = Card(Suit.HEARTS, Rank.SEVEN)
    single_seven = MeldAction((seven,))
    pass_action = PassPlayAction()
    agent = RandomAgent(seed=123)
    obs = _obs_with_hand([seven, Card(Suit.CLUBS, Rank.KING)])

    choices = [agent.select_action(obs, [single_seven, pass_action]) for _ in range(100)]
    assert pass_action in choices
    assert single_seven in choices
    assert choices.count(single_seven) < choices.count(pass_action)
