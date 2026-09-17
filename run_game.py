from __future__ import annotations

import argparse
import re
import secrets
from pathlib import Path

import torch

from agents import HeuristicAgent, HumanAgent, RandomAgent
from hoola.action_codec import ActionCodec
from hoola.encoding import encode_observation
from hoola.engine import HoolaEngine
from hoola.logger import ConsoleGameLogger
from hoola.observation import make_observation
from hoola.recorder import GameRecorder
from hoola.training_recorder import TrainingRecorder
from rl.model import load_policy_checkpoint
from rl.ppo import resolve_device


class RLCheckpointAgent:
    """Inference-only adapter around a trained masked actor-critic checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        stochastic: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.model, self.checkpoint = load_policy_checkpoint(
            checkpoint_path,
            device=self.device,
        )
        self.codec = ActionCodec()
        self.stochastic = stochastic
        self.checkpoint_path = str(checkpoint_path)

    def select_action(self, observation, legal_actions):
        if not legal_actions:
            raise RuntimeError("No legal actions available")

        obs_vec = encode_observation(observation)
        action_mask = self.codec.action_mask(legal_actions)

        obs_tensor = torch.as_tensor(
            obs_vec, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            action_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)

        action_id, _, _ = self.model.act(
            obs_tensor,
            mask_tensor,
            deterministic=not self.stochastic,
        )
        return self.codec.decode_legal(int(action_id.item()), legal_actions)


def make_agent(
    name: str,
    seed: int,
    *,
    rl_checkpoint: str | None = None,
    rl_device: str = "auto",
    rl_stochastic: bool = False,
):
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "heuristic":
        return HeuristicAgent(seed=seed)
    if name == "human":
        return HumanAgent()
    if name == "rl":
        if rl_checkpoint is None:
            raise ValueError("--rl-checkpoint is required when using an RL player")
        return RLCheckpointAgent(
            rl_checkpoint,
            device=rl_device,
            stochastic=rl_stochastic,
        )
    raise ValueError(f"Unknown agent: {name}")


def pretty_name(name: str) -> str:
    if name == "heuristic":
        return "Heuristic v1"
    if name == "random":
        return "Random"
    if name == "human":
        return "Human"
    if name == "rl":
        return "RL v1"
    return name


def resolve_seed(seed: int | None) -> int:
    """Use an explicit seed when supplied; otherwise create a fresh per-run seed."""
    return seed if seed is not None else secrets.randbits(63)

def resolve_training_record_path(path: str | Path) -> Path:
    """Return the next non-overwriting numbered training-record path.

    Examples
    --------
    If ``records/human_vs_heuristic_001.npz`` is requested and 001/002 already
    exist, this returns ``records/human_vs_heuristic_003.npz``.

    If no numeric suffix is supplied, ``_001`` is added automatically.
    Both .npz and its companion .json are considered occupied so the pair always
    receives the same fresh number.
    """
    requested = Path(path)
    if requested.suffix.lower() != ".npz":
        requested = requested.with_suffix(".npz")

    stem = requested.stem
    match = re.match(r"^(.*?)(\d+)$", stem)
    if match:
        prefix = match.group(1)
        requested_number = int(match.group(2))
        width = max(3, len(match.group(2)))
    else:
        prefix = stem if stem.endswith("_") else stem + "_"
        requested_number = 1
        width = 3

    directory = requested.parent
    max_seen = requested_number - 1
    if directory.exists():
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        for candidate in directory.iterdir():
            if candidate.suffix.lower() not in {".npz", ".json"}:
                continue
            m = pattern.match(candidate.stem)
            if m:
                max_seen = max(max_seen, int(m.group(1)))
                width = max(width, len(m.group(1)))

    next_number = max_seen + 1
    while True:
        numbered = directory / f"{prefix}{next_number:0{width}d}.npz"
        meta = numbered.with_suffix(".json")
        if not numbered.exists() and not meta.exists():
            return numbered
        next_number += 1


def main(
    seed: int | None = None,
    verbose: bool = True,
    p0: str = "heuristic",
    p1: str = "random",
    record_path: str | None = None,
    include_private_state: bool = False,
    explain_ai: str = "off",
    training_record_path: str | None = None,
    rl_checkpoint: str | None = None,
    rl_device: str = "auto",
    rl_stochastic: bool = False,
):
    seed = resolve_seed(seed)
    print(f"GAME SEED: {seed}  (replay with --seed {seed})")

    engine = HoolaEngine(seed=seed)
    if "rl" in (p0, p1) and rl_checkpoint is None:
        raise ValueError("--rl-checkpoint is required when --p0 rl or --p1 rl")

    agents = [
        make_agent(
            p0,
            seed + 100,
            rl_checkpoint=rl_checkpoint,
            rl_device=rl_device,
            rl_stochastic=rl_stochastic,
        ),
        make_agent(
            p1,
            seed + 200,
            rl_checkpoint=rl_checkpoint,
            rl_device=rl_device,
            rl_stochastic=rl_stochastic,
        ),
    ]
    agent_keys = [p0, p1]
    names = [pretty_name(p0), pretty_name(p1)]

    if "rl" in agent_keys:
        rl_agent = agents[agent_keys.index("rl")]
        update = getattr(rl_agent, "checkpoint", {}).get("update", "?")
        mode = "stochastic sampling" if rl_stochastic else "deterministic argmax"
        print(f"RL CHECKPOINT: {rl_checkpoint} | update={update} | {mode}")

    # In any human game, the normal logger must not reveal either player's
    # private hand. HumanAgent displays only the active human's own observation.
    human_game = "human" in agent_keys
    hidden = {0, 1} if human_game else set()
    logger = ConsoleGameLogger(player_names=names, hidden_hand_players=hidden) if verbose else None

    if explain_ai == "full" and human_game:
        print(
            "WARNING: --explain-ai full can reveal information about the AI's hidden hand. "
            "Use --explain-ai public for a fair human-vs-AI game.\n"
        )

    recorder = None
    if record_path is not None:
        recorder = GameRecorder(
            record_path,
            include_private_state=include_private_state,
            metadata={
                "seed": seed,
                "players": names,
                "player_types": agent_keys,
                "explain_ai": explain_ai,
                "rl_checkpoint": rl_checkpoint,
                "rl_stochastic": rl_stochastic,
            },
        )

    training_recorder = None
    resolved_training_record_path = None
    if training_record_path is not None:
        resolved_training_record_path = resolve_training_record_path(training_record_path)
        print(f"TRAINING RECORD: {resolved_training_record_path}")
        training_recorder = TrainingRecorder(
            resolved_training_record_path,
            metadata={
                "seed": seed,
                "players": names,
                "player_types": agent_keys,
                "rl_checkpoint": rl_checkpoint,
                "rl_stochastic": rl_stochastic,
            },
        )

    action_count = 0
    while not engine.state.terminal:
        player = engine.state.current_player
        obs = make_observation(engine.state, player)
        legal = engine.legal_actions()

        if not legal:
            engine._end_draw()
            break

        if logger is not None:
            logger.begin_turn(engine.state)

        action = agents[player].select_action(obs, legal)

        if agent_keys[player] == "heuristic" and explain_ai != "off":
            explanation = agents[player].explain_decision(
                obs,
                legal,
                action,
                mode=explain_ai,
            )
            print("\n" + explanation)

        snap = logger.snapshot(engine.state) if logger is not None else None
        state_before = engine.clone_state() if recorder is not None else None
        engine.step(action)
        action_count += 1

        if logger is not None and snap is not None:
            logger.log_action(snap, action, engine.state)

        if recorder is not None and state_before is not None:
            recorder.record_step(
                acting_player=player,
                acting_agent=names[player],
                observation=obs,
                legal_actions=legal,
                selected_action=action,
                state_before=state_before,
                state_after=engine.clone_state(),
            )

        if training_recorder is not None:
            training_recorder.record_decision(
                acting_player=player,
                acting_agent=agent_keys[player],
                observation=obs,
                legal_actions=legal,
                selected_action=action,
                state_after=engine.state,
            )

    result = engine.result()
    if logger is not None:
        logger.finish(engine.state, action_count)
    if recorder is not None:
        recorder.save(engine.state)
        print(f"Saved game record to {record_path}")
    if training_recorder is not None:
        npz_path, meta_path = training_recorder.save(engine.state)
        print(f"Saved training record to {npz_path} (+ {meta_path})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play one Hoola game")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="game seed; omit it for a fresh random game each run",
    )
    parser.add_argument("--p0", choices=("human", "random", "heuristic", "rl"), default="heuristic")
    parser.add_argument("--p1", choices=("human", "random", "heuristic", "rl"), default="random")
    parser.add_argument("--quiet", action="store_true", help="suppress turn-by-turn console logger")
    parser.add_argument("--record", type=str, default=None, help="optional output JSON path for game record")
    parser.add_argument(
        "--include-private-state",
        action="store_true",
        help="record hidden hands and stock in the JSON file (for analysis/debugging)",
    )
    parser.add_argument(
        "--training-record",
        type=str,
        default=None,
        help="optional compact NPZ path with fixed tensors/action masks for AI training",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        default=None,
        help="trained RL checkpoint (.pt), required when either player is 'rl'",
    )
    parser.add_argument(
        "--rl-device",
        type=str,
        default="auto",
        help="RL inference device: auto, cpu, cuda, mps, ...",
    )
    parser.add_argument(
        "--rl-stochastic",
        action="store_true",
        help="sample RL actions from the policy; default is deterministic argmax",
    )
    parser.add_argument(
        "--explain-ai",
        choices=("off", "public", "full"),
        default="off",
        help="Heuristic explanation mode: public is fair-play safe; full is debug and may reveal hidden information",
    )
    args = parser.parse_args()
    main(
        seed=args.seed,
        verbose=not args.quiet,
        p0=args.p0,
        p1=args.p1,
        record_path=args.record,
        include_private_state=args.include_private_state,
        explain_ai=args.explain_ai,
        training_record_path=args.training_record,
        rl_checkpoint=args.rl_checkpoint,
        rl_device=args.rl_device,
        rl_stochastic=args.rl_stochastic,
    )
