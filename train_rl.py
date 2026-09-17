from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch

from hoola.action_codec import ActionCodec
from hoola.encoding import OBSERVATION_SIZE, encode_observation
from rl.dataset import HumanDataset
from rl.model import MaskedActorCritic, load_policy_checkpoint
from rl.ppo import (
    PPOConfig,
    RolloutBatch,
    append_csv_row,
    collect_complete_episodes,
    evaluate_policy,
    human_auxiliary_update,
    ppo_update,
    resolve_device,
    save_checkpoint,
)


class FrozenRLAgent:
    """Inference-only RL opponent loaded from a fixed checkpoint.

    The model is never optimized.  It is used only to generate opponent actions
    during PPO rollout collection.  By default it uses deterministic argmax,
    which keeps the frozen opponent's strategy stable across training.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
        stochastic: bool = False,
        max_turns: int = 500,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.stochastic = bool(stochastic)
        self.max_turns = int(max_turns)
        self.codec = ActionCodec()
        self.model, self.checkpoint = load_policy_checkpoint(
            checkpoint_path,
            device=device,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def select_action(self, observation, legal_actions):
        if not legal_actions:
            raise RuntimeError("FrozenRLAgent received no legal actions")

        obs_vec = encode_observation(observation, max_turns=self.max_turns)
        action_mask = self.codec.action_mask(legal_actions)
        obs_t = torch.as_tensor(
            obs_vec,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask_t = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            action_t, _, _ = self.model.act(
                obs_t,
                mask_t,
                deterministic=not self.stochastic,
            )
        return self.codec.decode_legal(int(action_t.item()), legal_actions)


def parse_opponent_mix(spec: str | None) -> dict[str, float] | None:
    """Parse e.g. ``heuristic=0.7,rl=0.3`` and normalize the weights."""
    if spec is None:
        return None

    allowed = {"heuristic", "random", "rl"}
    weights: dict[str, float] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"Invalid --opponent-mix item {token!r}; expected name=weight"
            )
        name, raw_weight = token.split("=", 1)
        name = name.strip().lower()
        if name not in allowed:
            raise ValueError(
                f"Unknown opponent {name!r} in --opponent-mix; "
                f"choose from {sorted(allowed)}"
            )
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(
                f"Invalid weight {raw_weight!r} for opponent {name!r}"
            ) from exc
        if weight < 0:
            raise ValueError("Opponent-mix weights must be non-negative")
        weights[name] = weights.get(name, 0.0) + weight

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("--opponent-mix must contain at least one positive weight")

    return {name: weight / total for name, weight in weights.items() if weight > 0}


def merge_rollouts(parts: list[RolloutBatch]) -> RolloutBatch:
    """Merge complete-episode rollout batches collected from several opponents."""
    if not parts:
        raise ValueError("No rollout batches to merge")
    if len(parts) == 1:
        return parts[0]

    return RolloutBatch(
        observations=np.concatenate([part.observations for part in parts], axis=0),
        action_masks=np.concatenate([part.action_masks for part in parts], axis=0),
        actions=np.concatenate([part.actions for part in parts], axis=0),
        old_log_probs=np.concatenate([part.old_log_probs for part in parts], axis=0),
        old_values=np.concatenate([part.old_values for part in parts], axis=0),
        returns=np.concatenate([part.returns for part in parts], axis=0),
        outcomes=np.concatenate([part.outcomes for part in parts], axis=0),
        episode_lengths=tuple(
            length
            for part in parts
            for length in part.episode_lengths
        ),
        wins=sum(part.wins for part in parts),
        losses=sum(part.losses for part in parts),
        draws=sum(part.draws for part in parts),
        episodes=sum(part.episodes for part in parts),
    )


def collect_training_rollout(
    model: MaskedActorCritic,
    *,
    default_opponent: str,
    opponent_mix: dict[str, float] | None,
    frozen_rl_agent: FrozenRLAgent | None,
    num_episodes: int,
    rng: np.random.Generator,
    device: torch.device,
    max_turns: int,
    seat_mode: str,
    episode_offset: int,
) -> tuple[RolloutBatch, dict[str, int]]:
    """Collect PPO rollouts from either one opponent or an opponent mixture.

    For a mixture, the requested number of episodes is split as closely as
    possible to the requested fractions on every update.  For example, 64
    episodes with ``heuristic=0.7,rl=0.3`` becomes 45 heuristic + 19 RL.
    """
    if opponent_mix is None:
        rollout = collect_complete_episodes(
            model,
            opponent=default_opponent,
            num_episodes=num_episodes,
            rng=rng,
            device=device,
            max_turns=max_turns,
            seat_mode=seat_mode,
            episode_offset=episode_offset,
        )
        return rollout, {default_opponent: num_episodes}

    names = list(opponent_mix)
    probabilities = np.asarray([opponent_mix[name] for name in names], dtype=np.float64)
    raw_counts = probabilities * num_episodes
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = int(num_episodes - counts.sum())
    if remainder > 0:
        fractional = raw_counts - counts
        order = np.argsort(-fractional, kind="stable")
        for idx in order[:remainder]:
            counts[idx] += 1

    parts: list[RolloutBatch] = []
    count_map: dict[str, int] = {}
    offset = episode_offset

    for name, count in zip(names, counts.tolist()):
        if count <= 0:
            continue
        if name == "rl":
            if frozen_rl_agent is None:
                raise RuntimeError(
                    "RL opponent requested but no --rl-opponent-checkpoint was loaded"
                )
            opponent = frozen_rl_agent
        else:
            opponent = name

        part = collect_complete_episodes(
            model,
            opponent=opponent,
            num_episodes=int(count),
            rng=rng,
            device=device,
            max_turns=max_turns,
            seat_mode=seat_mode,
            episode_offset=offset,
        )
        parts.append(part)
        count_map[name] = int(count)
        offset += int(count)

    return merge_rollouts(parts), count_map


def format_mix(opponent_mix: dict[str, float] | None, default_opponent: str) -> str:
    if opponent_mix is None:
        return default_opponent
    return ", ".join(
        f"{name}={100.0 * weight:.1f}%"
        for name, weight in opponent_mix.items()
    )


def load_agent_dataset(
    path_spec: str | Path | None,
    *,
    agent_label: str,
) -> HumanDataset:
    """Load demonstration decisions for one agent label from training records.

    The file format is exactly the same NPZ + companion JSON format used by
    ``HumanDataset`` / ``TrainingRecorder``.  Only rows whose companion JSON
    ``agent_types`` entry matches ``agent_label`` are kept.

    This lets the trainer keep human and heuristic demonstrations physically
    separate while reusing the same tensor representation.
    """
    if path_spec is None:
        return HumanDataset.empty()

    spec = str(path_spec)
    path = Path(spec)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.rglob("*.npz"))
    else:
        files = [Path(x) for x in sorted(glob.glob(spec, recursive=True))]

    target = str(agent_label).strip().lower()
    action_size = ActionCodec().size
    obs_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    outcome_parts: list[np.ndarray] = []
    used_files: list[str] = []

    for npz_path in files:
        meta_path = npz_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            agent_types = meta.get("agent_types", [])
            with np.load(npz_path, allow_pickle=False) as data:
                observations = data["observations"]
                masks = data["action_masks"]
                actions = data["action_ids"]
                outcomes = data["final_outcomes"]
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue

        n = len(actions)
        if len(agent_types) != n:
            continue

        selector = np.asarray(
            [str(label).strip().lower() == target for label in agent_types],
            dtype=bool,
        )
        if not selector.any():
            continue

        selected_obs = np.asarray(observations[selector], dtype=np.float32)
        selected_masks = np.asarray(masks[selector], dtype=np.int8)
        selected_actions = np.asarray(actions[selector], dtype=np.int64)
        selected_outcomes = np.asarray(outcomes[selector], dtype=np.float32)

        if selected_obs.shape[1:] != (OBSERVATION_SIZE,):
            continue
        if selected_masks.shape[1:] != (action_size,):
            continue
        rows = np.arange(selected_actions.shape[0])
        if np.any(selected_actions < 0) or np.any(selected_actions >= action_size):
            raise ValueError(f"{agent_label} record has out-of-range action ID: {npz_path}")
        if np.any(selected_masks[rows, selected_actions] != 1):
            raise ValueError(
                f"{agent_label} record contains an illegal selected action: {npz_path}"
            )

        obs_parts.append(selected_obs)
        mask_parts.append(selected_masks)
        action_parts.append(selected_actions)
        outcome_parts.append(selected_outcomes)
        used_files.append(str(npz_path))

    if not obs_parts:
        return HumanDataset.empty()

    return HumanDataset(
        observations=np.concatenate(obs_parts, axis=0),
        action_masks=np.concatenate(mask_parts, axis=0),
        action_ids=np.concatenate(action_parts, axis=0),
        outcomes=np.concatenate(outcome_parts, axis=0),
        source_files=tuple(used_files),
    )


def renamed_aux_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """Rename the generic demonstration update metrics for CSV/logging."""
    return {
        f"{prefix}_bc_loss": float(metrics.get("human_bc_loss", 0.0)),
        f"{prefix}_value_loss": float(metrics.get("human_value_loss", 0.0)),
        f"{prefix}_updates": float(metrics.get("human_updates", 0.0)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Hoola RL v1 with masked PPO and optional opponent mixture"
    )
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--episodes-per-update", type=int, default=64)
    p.add_argument(
        "--opponent",
        choices=("heuristic", "random"),
        default="heuristic",
        help="single training opponent; ignored when --opponent-mix is supplied",
    )
    p.add_argument(
        "--opponent-mix",
        type=str,
        default=None,
        help="mixture such as heuristic=0.7,rl=0.3 (weights are normalized)",
    )
    p.add_argument(
        "--rl-opponent-checkpoint",
        type=str,
        default=None,
        help="frozen RL checkpoint used when 'rl' appears in --opponent-mix",
    )
    p.add_argument(
        "--rl-opponent-stochastic",
        action="store_true",
        help="sample frozen RL opponent actions; default is deterministic argmax",
    )
    p.add_argument(
        "--eval-opponent",
        choices=("heuristic", "random"),
        default="heuristic",
        help="fixed opponent used for best-checkpoint evaluation (default: heuristic)",
    )
    p.add_argument("--seat-mode", choices=("alternate", "random", "p0", "p1"), default="alternate")
    p.add_argument("--max-turns", type=int, default=500)

    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    p.add_argument("--human-data", type=str, default=None, help="NPZ, directory, or glob of recorded games")
    p.add_argument("--human-updates", type=int, default=2, help="auxiliary human minibatches after each PPO update")
    p.add_argument("--human-batch-size", type=int, default=128)
    p.add_argument("--human-bc-coef", type=float, default=0.10)
    p.add_argument("--human-value-coef", type=float, default=0.05)
    p.add_argument("--human-pretrain-updates", type=int, default=0)

    # Heuristic demonstrations use the exact same NPZ + JSON training-record
    # format as human demonstrations.  Only steps labelled "heuristic" are
    # selected from these files.
    p.add_argument(
        "--heuristic-data",
        type=str,
        default=None,
        help="NPZ, directory, or glob of training records containing heuristic decisions",
    )
    p.add_argument(
        "--heuristic-updates",
        type=int,
        default=2,
        help="heuristic demonstration minibatches after each PPO update",
    )
    p.add_argument("--heuristic-batch-size", type=int, default=256)
    p.add_argument(
        "--heuristic-bc-coef",
        type=float,
        default=0.10,
        help="weight of heuristic behavior-cloning auxiliary loss",
    )
    p.add_argument(
        "--heuristic-value-coef",
        type=float,
        default=0.0,
        help="optional value loss on heuristic-game final outcomes (default off)",
    )
    p.add_argument(
        "--heuristic-pretrain-updates",
        type=int,
        default=0,
        help="optional heuristic-BC warm-start minibatches before PPO",
    )

    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--eval-seed", type=int, default=900000)
    p.add_argument("--train-seed", type=int, default=20260911)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/rl_v1")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.episodes_per_update <= 0:
        raise ValueError("--updates and --episodes-per-update must be positive")

    opponent_mix = parse_opponent_mix(args.opponent_mix)
    if opponent_mix is not None and "rl" in opponent_mix and not args.rl_opponent_checkpoint:
        raise ValueError(
            "--rl-opponent-checkpoint is required when 'rl' appears in --opponent-mix"
        )

    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)

    device = resolve_device(args.device)
    rng = np.random.default_rng(args.train_seed)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "training_log.csv"

    model = MaskedActorCritic(
        obs_size=OBSERVATION_SIZE,
        action_size=ActionCodec().size,
        hidden_sizes=args.hidden_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    start_update = 1
    env_steps = 0
    episodes_seen = 0
    best_eval_win_rate = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_update = int(ckpt.get("update", 0)) + 1
        env_steps = int(ckpt.get("env_steps", 0))
        episodes_seen = int(ckpt.get("episodes", 0))
        best_eval_win_rate = float(ckpt.get("best_eval_win_rate", -1.0))
        print(f"Resumed from {args.resume} at update {start_update}")

    frozen_rl_agent = None
    if opponent_mix is not None and "rl" in opponent_mix:
        frozen_rl_agent = FrozenRLAgent(
            args.rl_opponent_checkpoint,
            device=device,
            stochastic=args.rl_opponent_stochastic,
            max_turns=args.max_turns,
        )

    human = HumanDataset.from_path(args.human_data)
    heuristic_demo = load_agent_dataset(
        args.heuristic_data,
        agent_label="heuristic",
    )
    print("=" * 78)
    print("Hoola RL v1 | Masked PPO")
    print(f"device              : {device}")
    print(f"network             : {OBSERVATION_SIZE} -> {' -> '.join(map(str, args.hidden_sizes))} -> policy {ActionCodec().size} + value 1")
    print(f"training opponents  : {format_mix(opponent_mix, args.opponent)}")
    if frozen_rl_agent is not None:
        frozen_update = frozen_rl_agent.checkpoint.get("update", "?")
        frozen_best = frozen_rl_agent.checkpoint.get("best_eval_win_rate", None)
        frozen_best_text = (
            "?" if frozen_best is None else f"{100.0 * float(frozen_best):.1f}%"
        )
        frozen_mode = "stochastic" if args.rl_opponent_stochastic else "deterministic argmax"
        print(
            f"frozen RL opponent  : {args.rl_opponent_checkpoint} "
            f"(update={frozen_update}, best={frozen_best_text}, {frozen_mode})"
        )
    print(f"evaluation opponent : {args.eval_opponent}")
    print(f"seat mode           : {args.seat_mode}")
    print(f"episodes/update     : {args.episodes_per_update}")
    print(f"human samples       : {len(human)} from {len(human.source_files)} file(s)")
    print(
        f"heuristic samples   : {len(heuristic_demo)} "
        f"from {len(heuristic_demo.source_files)} file(s)"
    )
    if len(heuristic_demo) > 0:
        print(
            f"heuristic aux       : BC coef={args.heuristic_bc_coef:g}, "
            f"value coef={args.heuristic_value_coef:g}, "
            f"{args.heuristic_updates} minibatch(es)/PPO update"
        )
    print(f"checkpoint dir      : {checkpoint_dir}")
    print("reward              : win=+1, loss=-1, draw=0 (undiscounted terminal outcome)")
    print("=" * 78)

    # Optional warm start from human demonstrations before on-policy PPO.
    if len(human) > 0 and args.human_pretrain_updates > 0:
        metrics = human_auxiliary_update(
            model,
            optimizer,
            human,
            updates=args.human_pretrain_updates,
            batch_size=args.human_batch_size,
            bc_coef=max(args.human_bc_coef, 1.0),
            value_coef=max(args.human_value_coef, 0.25),
            max_grad_norm=args.max_grad_norm,
            device=device,
            rng=rng,
        )
        print(
            f"Human warm-start: {args.human_pretrain_updates} minibatches | "
            f"BC={metrics['human_bc_loss']:.4f} V={metrics['human_value_loss']:.4f}"
        )

    # Optional teacher warm start.  Heuristic BC is intentionally separate
    # from human BC so its influence can be annealed/disabled independently.
    if len(heuristic_demo) > 0 and args.heuristic_pretrain_updates > 0:
        metrics = human_auxiliary_update(
            model,
            optimizer,
            heuristic_demo,
            updates=args.heuristic_pretrain_updates,
            batch_size=args.heuristic_batch_size,
            bc_coef=max(args.heuristic_bc_coef, 1.0),
            value_coef=max(args.heuristic_value_coef, 0.0),
            max_grad_norm=args.max_grad_norm,
            device=device,
            rng=rng,
        )
        print(
            f"Heuristic warm-start: {args.heuristic_pretrain_updates} minibatches | "
            f"BC={metrics['human_bc_loss']:.4f} V={metrics['human_value_loss']:.4f}"
        )

    ppo_config = PPOConfig(
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        max_grad_norm=args.max_grad_norm,
    )
    training_args = vars(args).copy()
    training_args["normalized_opponent_mix"] = opponent_mix

    # When resuming into a fresh checkpoint directory, preserve the starting
    # checkpoint as the baseline best.  Otherwise a run that never exceeds the
    # previous best score would finish without a best.pt in the new directory.
    if args.resume and not (checkpoint_dir / "best.pt").exists():
        baseline_path = save_checkpoint(
            checkpoint_dir / "best.pt",
            model=model,
            optimizer=optimizer,
            update=start_update - 1,
            env_steps=env_steps,
            episodes=episodes_seen,
            best_eval_win_rate=best_eval_win_rate,
            training_args=training_args,
        )
        print(f"Preserved resumed baseline best -> {baseline_path}")

    for update in range(start_update, args.updates + 1):
        rollout, opponent_counts = collect_training_rollout(
            model,
            default_opponent=args.opponent,
            opponent_mix=opponent_mix,
            frozen_rl_agent=frozen_rl_agent,
            num_episodes=args.episodes_per_update,
            rng=rng,
            device=device,
            max_turns=args.max_turns,
            seat_mode=args.seat_mode,
            episode_offset=episodes_seen,
        )
        episodes_seen += rollout.episodes
        env_steps += rollout.steps

        ppo_metrics = ppo_update(
            model,
            optimizer,
            rollout,
            config=ppo_config,
            device=device,
            rng=rng,
        )

        human_raw_metrics = human_auxiliary_update(
            model,
            optimizer,
            human,
            updates=args.human_updates if len(human) > 0 else 0,
            batch_size=args.human_batch_size,
            bc_coef=args.human_bc_coef,
            value_coef=args.human_value_coef,
            max_grad_norm=args.max_grad_norm,
            device=device,
            rng=rng,
        )
        human_metrics = renamed_aux_metrics(human_raw_metrics, "human")

        heuristic_raw_metrics = human_auxiliary_update(
            model,
            optimizer,
            heuristic_demo,
            updates=args.heuristic_updates if len(heuristic_demo) > 0 else 0,
            batch_size=args.heuristic_batch_size,
            bc_coef=args.heuristic_bc_coef,
            value_coef=args.heuristic_value_coef,
            max_grad_norm=args.max_grad_norm,
            device=device,
            rng=rng,
        )
        heuristic_metrics = renamed_aux_metrics(heuristic_raw_metrics, "heuristic")

        mean_ep_len = float(np.mean(rollout.episode_lengths)) if rollout.episode_lengths else 0.0
        mix_text = "/".join(
            f"{name}:{opponent_counts.get(name, 0)}"
            for name in ("heuristic", "rl", "random")
            if name in opponent_counts
        )
        aux_chunks = []
        if human_metrics["human_updates"] > 0:
            aux_chunks.append(
                f"HBC={human_metrics['human_bc_loss']:.3f} "
                f"HV={human_metrics['human_value_loss']:.3f}"
            )
        if heuristic_metrics["heuristic_updates"] > 0:
            heur_text = f"HeurBC={heuristic_metrics['heuristic_bc_loss']:.3f}"
            if args.heuristic_value_coef > 0:
                heur_text += f" HeurV={heuristic_metrics['heuristic_value_loss']:.3f}"
            aux_chunks.append(heur_text)
        aux_text = " | " + " | ".join(aux_chunks) if aux_chunks else ""
        print(
            f"update {update:04d}/{args.updates} | steps={env_steps:7d} | "
            f"opp[{mix_text}] | "
            f"train W/L/D={rollout.wins}/{rollout.losses}/{rollout.draws} "
            f"WR={100*rollout.win_rate:5.1f}% | ep_decisions={mean_ep_len:5.1f} | "
            f"pi={ppo_metrics['policy_loss']:+.4f} V={ppo_metrics['value_loss']:.4f} "
            f"H={ppo_metrics['entropy']:.3f} KL={ppo_metrics['approx_kl']:.5f}"
            f"{aux_text}"
        )

        eval_metrics = None
        if update == 1 or update % args.eval_every == 0 or update == args.updates:
            eval_metrics = evaluate_policy(
                model,
                opponent=args.eval_opponent,
                games=args.eval_games,
                base_seed=args.eval_seed,
                device=device,
                max_turns=args.max_turns,
                deterministic=True,
            )
            print(
                f"  EVAL vs {args.eval_opponent} fixed seeds | "
                f"W/L/D={int(eval_metrics['wins'])}/{int(eval_metrics['losses'])}/{int(eval_metrics['draws'])} "
                f"WR={100*eval_metrics['win_rate']:.1f}% | avg decisions={eval_metrics['avg_decisions']:.1f}"
            )

            if eval_metrics["win_rate"] > best_eval_win_rate:
                best_eval_win_rate = eval_metrics["win_rate"]
                best_path = save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    update=update,
                    env_steps=env_steps,
                    episodes=episodes_seen,
                    best_eval_win_rate=best_eval_win_rate,
                    training_args=training_args,
                )
                print(f"  new best checkpoint -> {best_path}")

        latest_path = save_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            update=update,
            env_steps=env_steps,
            episodes=episodes_seen,
            best_eval_win_rate=best_eval_win_rate,
            training_args=training_args,
        )

        row = {
            "update": update,
            "env_steps": env_steps,
            "episodes": episodes_seen,
            "train_vs_heuristic": opponent_counts.get("heuristic", 0),
            "train_vs_rl": opponent_counts.get("rl", 0),
            "train_vs_random": opponent_counts.get("random", 0),
            "train_wins": rollout.wins,
            "train_losses": rollout.losses,
            "train_draws": rollout.draws,
            "train_win_rate": rollout.win_rate,
            "mean_episode_decisions": mean_ep_len,
            **ppo_metrics,
            **human_metrics,
            **heuristic_metrics,
            "eval_opponent": args.eval_opponent,
            "eval_win_rate": "" if eval_metrics is None else eval_metrics["win_rate"],
            "eval_wins": "" if eval_metrics is None else int(eval_metrics["wins"]),
            "eval_losses": "" if eval_metrics is None else int(eval_metrics["losses"]),
            "eval_draws": "" if eval_metrics is None else int(eval_metrics["draws"]),
            "best_eval_win_rate": best_eval_win_rate,
        }
        append_csv_row(log_path, row)

    summary = {
        "updates": args.updates,
        "env_steps": env_steps,
        "episodes": episodes_seen,
        "best_eval_win_rate": best_eval_win_rate,
        "eval_opponent": args.eval_opponent,
        "opponent_mix": opponent_mix,
        "rl_opponent_checkpoint": args.rl_opponent_checkpoint,
        "latest_checkpoint": str(checkpoint_dir / "latest.pt"),
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "human_samples": len(human),
        "human_files": len(human.source_files),
        "heuristic_samples": len(heuristic_demo),
        "heuristic_files": len(heuristic_demo.source_files),
        "heuristic_bc_coef": args.heuristic_bc_coef,
        "heuristic_value_coef": args.heuristic_value_coef,
    }
    (checkpoint_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("=" * 78)
    print(
        f"Training complete. Best fixed-seed win rate vs {args.eval_opponent}: "
        f"{100*best_eval_win_rate:.1f}%"
    )
    print(f"Best checkpoint   : {checkpoint_dir / 'best.pt'}")
    print(f"Latest checkpoint : {checkpoint_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
