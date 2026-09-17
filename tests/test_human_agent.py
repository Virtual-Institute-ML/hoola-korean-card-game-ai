from __future__ import annotations

from agents.human_agent import HumanAgent
from hoola.engine import HoolaEngine
from hoola.observation import make_observation


def test_human_agent_can_choose_numbered_action():
    engine = HoolaEngine(seed=21)
    obs = make_observation(engine.state, 0)
    legal = engine.legal_actions()
    agent = HumanAgent(input_func=lambda prompt: "1")
    selected = agent.select_action(obs, legal)
    assert selected in legal
