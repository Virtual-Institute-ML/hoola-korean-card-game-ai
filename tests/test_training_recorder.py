from __future__ import annotations

import json
import numpy as np

from hoola.engine import HoolaEngine
from hoola.observation import make_observation
from hoola.training_recorder import TrainingRecorder


def test_training_recorder_saves_fixed_tensors_and_outcomes(tmp_path):
    engine = HoolaEngine(seed=88)
    rec = TrainingRecorder(tmp_path / "training_game.npz", metadata={"source": "test"})

    # Record a few legal decisions then force-save the unfinished state; shapes
    # are the important contract here, outcome is 0 until a terminal winner.
    for _ in range(5):
        if engine.state.terminal:
            break
        p = engine.state.current_player
        obs = make_observation(engine.state, p)
        legal = engine.legal_actions()
        action = legal[0]
        engine.step(action)
        rec.record_decision(
            acting_player=p,
            acting_agent="test",
            observation=obs,
            legal_actions=legal,
            selected_action=action,
            state_after=engine.state,
        )

    npz_path, meta_path = rec.save(engine.state)
    data = np.load(npz_path)
    assert data["observations"].shape[0] == len(rec.action_ids)
    assert data["action_masks"].shape == (len(rec.action_ids), rec.codec.size)
    assert data["action_ids"].shape == (len(rec.action_ids),)
    for i, action_id in enumerate(data["action_ids"]):
        assert data["action_masks"][i, action_id] == 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["format"] == "hoola-training-v1"
    assert meta["action_space_size"] == rec.codec.size
