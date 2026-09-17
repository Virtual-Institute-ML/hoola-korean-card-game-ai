from __future__ import annotations

import json

from hoola.engine import HoolaEngine
from hoola.observation import make_observation
from hoola.recorder import GameRecorder


def test_game_recorder_writes_json(tmp_path):
    engine = HoolaEngine(seed=1)
    obs = make_observation(engine.state, engine.state.current_player)
    legal = engine.legal_actions()
    action = legal[0]
    before = engine.clone_state()
    engine.step(action)

    out = tmp_path / "game.json"
    rec = GameRecorder(out, metadata={"test": True})
    rec.record_step(
        acting_player=0,
        acting_agent="Random",
        observation=obs,
        legal_actions=legal,
        selected_action=action,
        state_before=before,
        state_after=engine.clone_state(),
    )
    rec.save(engine.state)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metadata"]["test"] is True
    assert payload["num_steps"] == 1
    assert payload["steps"][0]["selected_action"]["type"] == action.type.name
