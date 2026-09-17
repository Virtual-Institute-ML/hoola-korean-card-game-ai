from __future__ import annotations

import argparse
from dataclasses import dataclass

from agents import HeuristicAgent, RandomAgent
from hoola.engine import HoolaEngine
from hoola.observation import make_observation


@dataclass(frozen=True)
class BenchmarkResult:
    games: int
    heuristic_wins: int
    random_wins: int
    draws: int
    average_turns: float

    @property
    def decisive_games(self) -> int:
        return self.heuristic_wins + self.random_wins

    @property
    def heuristic_win_rate(self) -> float:
        if self.decisive_games == 0:
            return 0.0
        return self.heuristic_wins / self.decisive_games


def play_game(seed: int, heuristic_seat: int) -> tuple[int | None, bool, int]:
    engine = HoolaEngine(seed=seed, max_turns=500)

    heuristic = HeuristicAgent(seed=seed + 10_000)
    random_agent = RandomAgent(seed=seed + 20_000)
    agents = [None, None]
    agents[heuristic_seat] = heuristic
    agents[1 - heuristic_seat] = random_agent

    action_guard = 0
    while not engine.state.terminal:
        p = engine.state.current_player
        legal = engine.legal_actions()
        if not legal:
            engine._end_draw()
            break
        obs = make_observation(engine.state, p)
        action = agents[p].select_action(obs, legal)
        engine.step(action)
        action_guard += 1
        if action_guard >= 10_000:
            raise RuntimeError("Action guard exceeded")

    result = engine.result()
    return result.winner, result.is_draw, result.turns_completed


def run_benchmark(games: int = 1000, seed: int = 12345) -> BenchmarkResult:
    if games <= 0:
        raise ValueError("games must be > 0")

    heuristic_wins = 0
    random_wins = 0
    draws = 0
    turns = 0

    # Alternate seats every game to cancel first-player advantage.
    for i in range(games):
        heuristic_seat = i % 2
        winner, is_draw, n_turns = play_game(seed + i, heuristic_seat)
        turns += n_turns

        if is_draw:
            draws += 1
        elif winner == heuristic_seat:
            heuristic_wins += 1
        else:
            random_wins += 1

    return BenchmarkResult(
        games=games,
        heuristic_wins=heuristic_wins,
        random_wins=random_wins,
        draws=draws,
        average_turns=turns / games,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HeuristicAgent v1 vs RandomAgent")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    result = run_benchmark(args.games, args.seed)
    print("=" * 64)
    print("HeuristicAgent v1 vs RandomAgent")
    print("Seats alternate every game")
    print("=" * 64)
    print(f"Games                 : {result.games}")
    print(f"Heuristic wins        : {result.heuristic_wins}")
    print(f"Random wins           : {result.random_wins}")
    print(f"Draws                 : {result.draws}")
    print(f"Heuristic decisive WR : {100.0 * result.heuristic_win_rate:.1f}%")
    print(f"Average turns         : {result.average_turns:.2f}")


if __name__ == "__main__":
    main()
