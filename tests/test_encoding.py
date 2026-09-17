from __future__ import annotations

import numpy as np

from hoola.encoding import (
    CARD_CHANNELS,
    MY_HAND,
    OBSERVATION_SIZE,
    UNKNOWN,
    encode_observation,
)
from hoola.engine import HoolaEngine
from hoola.observation import make_observation


def test_observation_encoding_shape_and_one_hot_cards():
    engine = HoolaEngine(seed=123)
    obs = make_observation(engine.state, 0)
    x = encode_observation(obs)
    assert x.shape == (OBSERVATION_SIZE,)
    cards = x[: 52 * CARD_CHANNELS].reshape(52, CARD_CHANNELS)
    np.testing.assert_allclose(cards.sum(axis=1), 1.0)
    for card in obs.my_hand:
        assert cards[card.id, MY_HAND] == 1.0


def test_hidden_opponent_cards_and_stock_are_not_distinguished():
    engine = HoolaEngine(seed=124)
    obs = make_observation(engine.state, 0)
    x = encode_observation(obs)
    cards = x[: 52 * CARD_CHANNELS].reshape(52, CARD_CHANNELS)
    hidden = list(engine.state.hands[1]) + list(engine.state.stock)
    for card in hidden:
        assert cards[card.id, UNKNOWN] == 1.0
