from __future__ import annotations

import numpy as np

from hoola.action_codec import ActionCodec, MELD_CATALOG, MAX_TABLE_MELDS
from hoola.engine import HoolaEngine


def test_action_vocabulary_is_stable_and_reasonable_size():
    codec = ActionCodec()
    assert len(MELD_CATALOG) == 593
    assert MAX_TABLE_MELDS == 20
    assert codec.size == 2400


def test_every_engine_legal_action_has_unique_global_id_across_random_games():
    codec = ActionCodec()
    for seed in range(20):
        engine = HoolaEngine(seed=seed)
        for _ in range(100):
            if engine.state.terminal:
                break
            legal = engine.legal_actions()
            ids = [codec.encode(a) for a in legal]
            assert len(ids) == len(set(ids))
            assert all(0 <= i < codec.size for i in ids)
            mask = codec.action_mask(legal)
            assert mask.dtype == np.int8
            assert int(mask.sum()) == len(legal)
            # Drive a deterministic legal path to visit multiple phases.
            engine.step(legal[0])
