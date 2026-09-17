from __future__ import annotations

import numpy as np

from .observation import Observation
from .state import TurnPhase


CARD_CHANNELS = 6
GLOBAL_FEATURES = 10
OBSERVATION_SIZE = 52 * CARD_CHANNELS + GLOBAL_FEATURES

# Relative card-state channels.  Every card occupies exactly one channel.
UNKNOWN = 0
MY_HAND = 1
DISCARD_HISTORY = 2
TOP_DISCARD = 3
MY_MELD = 4
OPP_MELD = 5


def encode_observation(observation: Observation, *, max_turns: int = 500) -> np.ndarray:
    """Encode a player's imperfect-information observation as float32 vector.

    Hidden opponent cards and stock cards are deliberately indistinguishable
    (UNKNOWN).  No privileged information enters the training tensor.
    """
    card_state = np.zeros((52, CARD_CHANNELS), dtype=np.float32)
    card_state[:, UNKNOWN] = 1.0

    def mark(card_id: int, channel: int) -> None:
        card_state[card_id, :] = 0.0
        card_state[card_id, channel] = 1.0

    for card in observation.my_hand:
        mark(card.id, MY_HAND)

    if observation.discard_pile:
        for card in observation.discard_pile[:-1]:
            mark(card.id, DISCARD_HISTORY)
        mark(observation.discard_pile[-1].id, TOP_DISCARD)

    for meld in observation.melds:
        channel = MY_MELD if meld.owner == observation.player_id else OPP_MELD
        for card in meld.cards:
            mark(card.id, channel)

    you = observation.player_id
    opp = 1 - you
    phase_order = (TurnPhase.DRAW, TurnPhase.PLAY, TurnPhase.DISCARD, TurnPhase.TERMINAL)
    phase_onehot = [1.0 if observation.phase == phase else 0.0 for phase in phase_order]

    global_features = np.asarray(
        [
            min(observation.opponent_hand_size / 52.0, 1.0),
            min(observation.stock_size / 52.0, 1.0),
            min(len(observation.discard_pile) / 52.0, 1.0),
            float(observation.has_registered[you]),
            float(observation.has_registered[opp]),
            *phase_onehot,
            min(observation.turns_completed / max(1, max_turns), 1.0),
        ],
        dtype=np.float32,
    )

    encoded = np.concatenate([card_state.reshape(-1), global_features])
    if encoded.shape != (OBSERVATION_SIZE,):
        raise RuntimeError(f"Unexpected observation shape: {encoded.shape}")
    return encoded
