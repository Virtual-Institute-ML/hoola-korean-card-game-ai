from __future__ import annotations

import argparse

from rl.model import load_policy_checkpoint
from rl.ppo import evaluate_policy, resolve_device


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained Hoola RL checkpoint")
    p.add_argument("checkpoint", type=str)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--opponent", choices=("heuristic", "random"), default="heuristic")
    p.add_argument("--seed", type=int, default=900000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--stochastic", action="store_true", help="sample from policy instead of argmax")
    args = p.parse_args()

    device = resolve_device(args.device)
    model, checkpoint = load_policy_checkpoint(args.checkpoint, device=device)
    metrics = evaluate_policy(
        model,
        opponent=args.opponent,
        games=args.games,
        base_seed=args.seed,
        device=device,
        deterministic=not args.stochastic,
    )

    print("=" * 72)
    print(f"checkpoint : {args.checkpoint}")
    print(f"update     : {checkpoint.get('update', '?')}")
    print(f"opponent   : {args.opponent}")
    print(f"games      : {int(metrics['games'])}")
    print(f"W/L/D      : {int(metrics['wins'])}/{int(metrics['losses'])}/{int(metrics['draws'])}")
    print(f"win rate   : {100*metrics['win_rate']:.2f}% (decisive games)")
    print(f"avg turns  : {metrics['avg_turns']:.2f}")
    print(f"avg actions: {metrics['avg_decisions']:.2f} learner decisions/game")
    print("=" * 72)


if __name__ == "__main__":
    main()
