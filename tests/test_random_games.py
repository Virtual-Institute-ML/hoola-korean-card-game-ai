from agents import RandomAgent
from hoola.engine import HoolaEngine
from hoola.observation import make_observation


def play_random_game(seed: int):
    engine = HoolaEngine(seed=seed, max_turns=500)
    agents = [RandomAgent(seed + 1000), RandomAgent(seed + 2000)]

    action_guard = 0
    while not engine.state.terminal:
        p = engine.state.current_player
        legal = engine.legal_actions()
        if not legal:
            engine._end_draw()
            break
        action = agents[p].select_action(make_observation(engine.state, p), legal)
        engine.step(action)
        action_guard += 1
        assert action_guard < 10000

    return engine.result()


def test_100_random_games_finish_without_crashing():
    results = [play_random_game(seed) for seed in range(100)]
    assert len(results) == 100
    assert all(r.turns_completed <= 500 for r in results)
