from __future__ import annotations

from hoola.engine import HoolaEngine
from hoola.logger import ConsoleGameLogger


def test_hidden_logger_does_not_print_hand(capsys):
    engine = HoolaEngine(seed=11)
    logger = ConsoleGameLogger(hidden_hand_players={0, 1})
    secret_cards = [str(card) for card in engine.state.hands[0]]
    logger.begin_turn(engine.state)
    out = capsys.readouterr().out
    assert "HAND       : [hidden]" in out
    for card in secret_cards:
        assert card not in out
