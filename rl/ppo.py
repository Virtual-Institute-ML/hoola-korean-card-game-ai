from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from hoola.env import HoolaEnv
from .dataset import HumanDataset
from .model import MaskedActorCritic


@dataclass
class RolloutBatch:
    observations: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    returns: np.ndarray
    outcomes: np.ndarray
    episode_lengths: tuple[int, ...]
    wins: int
    losses: int
    draws: int
    episodes: int

    @property
    def steps(self) -> int:
        return int(self.actions.shape[0])

    @property
    def win_rate(self) -> float:
        decisive = self.wins + self.losses
        return self.wins / decisive if decisive else 0.0


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _terminal_outcome_from_state(env: HoolaEnv) -> float:
    state = env.engine.state
    if not state.terminal or state.is_draw or state.winner is None:
        return 0.0
    return 1.0 if state.winner == env.learning_player else -1.0


def collect_complete_episodes(
    model: MaskedActorCritic,
    *,
    opponent: str,
    num_episodes: int,
    rng: np.random.Generator,
    device: torch.device,
    max_turns: int = 500,
    seat_mode: str = "alternate",
    episode_offset: int = 0,
) -> RolloutBatch:
    """Collect on-policy complete episodes against a fixed opponent.

    Hoola's turn consists of a variable number of micro-actions.  RL v1 uses an
    undiscounted terminal outcome target (+1/-1/0), so every decision in one
    completed episode receives the same Monte-Carlo return.  This avoids a
    per-micro-action discount bias.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if seat_mode not in {"alternate", "random", "p0", "p1"}:
        raise ValueError("seat_mode must be one of alternate/random/p0/p1")

    envs = {
        0: HoolaEnv(opponent=opponent, learning_player=0, max_turns=max_turns),
        1: HoolaEnv(opponent=opponent, learning_player=1, max_turns=max_turns),
    }

    obs_all: list[np.ndarray] = []
    mask_all: list[np.ndarray] = []
    action_all: list[int] = []
    logp_all: list[float] = []
    value_all: list[float] = []
    return_all: list[float] = []
    outcome_all: list[float] = []
    episode_lengths: list[int] = []
    wins = losses = draws = 0

    model.eval()
    for local_ep in range(num_episodes):
        global_ep = episode_offset + local_ep
        if seat_mode == "alternate":
            seat = global_ep % 2
        elif seat_mode == "random":
            seat = int(rng.integers(0, 2))
        elif seat_mode == "p0":
            seat = 0
        else:
            seat = 1

        env = envs[seat]
        episode_seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
        obs, _ = env.reset(seed=episode_seed)

        ep_obs: list[np.ndarray] = []
        ep_masks: list[np.ndarray] = []
        ep_actions: list[int] = []
        ep_logps: list[float] = []
        ep_values: list[float] = []

        if env.engine.state.terminal:
            outcome = _terminal_outcome_from_state(env)
        else:
            while True:
                obs_t = torch.as_tensor(obs["observation"], dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(obs["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t, logp_t, value_t = model.act(obs_t, mask_t, deterministic=False)
                action = int(action_t.item())

                ep_obs.append(np.asarray(obs["observation"], dtype=np.float32).copy())
                ep_masks.append(np.asarray(obs["action_mask"], dtype=np.int8).copy())
                ep_actions.append(action)
                ep_logps.append(float(logp_t.item()))
                ep_values.append(float(value_t.item()))

                obs, reward, terminated, truncated, _ = env.step(action)
                if truncated:
                    raise RuntimeError("HoolaEnv unexpectedly truncated an episode")
                if terminated:
                    outcome = float(reward)
                    break

        if outcome > 0:
            wins += 1
        elif outcome < 0:
            losses += 1
        else:
            draws += 1

        n = len(ep_actions)
        episode_lengths.append(n)
        obs_all.extend(ep_obs)
        mask_all.extend(ep_masks)
        action_all.extend(ep_actions)
        logp_all.extend(ep_logps)
        value_all.extend(ep_values)
        return_all.extend([outcome] * n)
        outcome_all.extend([outcome] * n)

    if not action_all:
        raise RuntimeError("Collected no learner decisions; increase episode count")

    return RolloutBatch(
        observations=np.stack(obs_all).astype(np.float32),
        action_masks=np.stack(mask_all).astype(np.int8),
        actions=np.asarray(action_all, dtype=np.int64),
        old_log_probs=np.asarray(logp_all, dtype=np.float32),
        old_values=np.asarray(value_all, dtype=np.float32),
        returns=np.asarray(return_all, dtype=np.float32),
        outcomes=np.asarray(outcome_all, dtype=np.float32),
        episode_lengths=tuple(episode_lengths),
        wins=wins,
        losses=losses,
        draws=draws,
        episodes=num_episodes,
    )


def ppo_update(
    model: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    *,
    config: PPOConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    obs = torch.as_tensor(rollout.observations, dtype=torch.float32, device=device)
    masks = torch.as_tensor(rollout.action_masks, dtype=torch.bool, device=device)
    actions = torch.as_tensor(rollout.actions, dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(rollout.old_log_probs, dtype=torch.float32, device=device)
    old_values = torch.as_tensor(rollout.old_values, dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout.returns, dtype=torch.float32, device=device)

    advantages = returns - old_values
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
    }
    n_minibatches = 0
    n = rollout.steps

    for _ in range(config.ppo_epochs):
        indices = rng.permutation(n)
        for start in range(0, n, config.minibatch_size):
            mb = indices[start : start + config.minibatch_size]
            mb_t = torch.as_tensor(mb, dtype=torch.long, device=device)

            dist, values = model.distribution_and_value(obs[mb_t], masks[mb_t])
            new_log_probs = dist.log_prob(actions[mb_t])
            entropy = dist.entropy().mean()
            log_ratio = new_log_probs - old_log_probs[mb_t]
            ratio = torch.exp(log_ratio)

            adv = advantages[mb_t]
            surrogate1 = ratio * adv
            surrogate2 = torch.clamp(
                ratio,
                1.0 - config.clip_eps,
                1.0 + config.clip_eps,
            ) * adv
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = nn.functional.mse_loss(values, returns[mb_t])
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip_eps).float().mean()

            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["entropy"] += float(entropy.item())
            totals["approx_kl"] += float(approx_kl.item())
            totals["clip_fraction"] += float(clip_fraction.item())
            totals["grad_norm"] += float(grad_norm.item())
            n_minibatches += 1

    return {k: v / max(1, n_minibatches) for k, v in totals.items()}


def human_auxiliary_update(
    model: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    dataset: HumanDataset,
    *,
    updates: int,
    batch_size: int,
    bc_coef: float,
    value_coef: float,
    max_grad_norm: float,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    if len(dataset) == 0 or updates <= 0 or (bc_coef <= 0 and value_coef <= 0):
        return {"human_bc_loss": 0.0, "human_value_loss": 0.0, "human_updates": 0.0}

    model.train()
    bc_total = 0.0
    value_total = 0.0
    for _ in range(updates):
        idx = dataset.sample_indices(batch_size, rng)
        obs = torch.as_tensor(dataset.observations[idx], dtype=torch.float32, device=device)
        masks = torch.as_tensor(dataset.action_masks[idx], dtype=torch.bool, device=device)
        actions = torch.as_tensor(dataset.action_ids[idx], dtype=torch.long, device=device)
        outcomes = torch.as_tensor(dataset.outcomes[idx], dtype=torch.float32, device=device)

        dist, values = model.distribution_and_value(obs, masks)
        bc_loss = -dist.log_prob(actions).mean()
        value_loss = nn.functional.mse_loss(values, outcomes)
        loss = bc_coef * bc_loss + value_coef * value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        bc_total += float(bc_loss.item())
        value_total += float(value_loss.item())

    return {
        "human_bc_loss": bc_total / updates,
        "human_value_loss": value_total / updates,
        "human_updates": float(updates),
    }


@torch.no_grad()
def evaluate_policy(
    model: MaskedActorCritic,
    *,
    opponent: str = "heuristic",
    games: int = 200,
    base_seed: int = 900_000,
    device: torch.device,
    max_turns: int = 500,
    deterministic: bool = True,
) -> dict[str, float]:
    if games <= 0:
        raise ValueError("games must be > 0")
    envs = {
        0: HoolaEnv(opponent=opponent, learning_player=0, max_turns=max_turns),
        1: HoolaEnv(opponent=opponent, learning_player=1, max_turns=max_turns),
    }
    wins = losses = draws = 0
    decisions = 0
    turns = 0
    model.eval()

    for i in range(games):
        seat = i % 2
        # Same deal seed is evaluated from both seats whenever possible.
        seed = int(base_seed + i // 2)
        env = envs[seat]
        obs, _ = env.reset(seed=seed)

        if env.engine.state.terminal:
            outcome = _terminal_outcome_from_state(env)
        else:
            while True:
                obs_t = torch.as_tensor(obs["observation"], dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(obs["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)
                action_t, _, _ = model.act(obs_t, mask_t, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(int(action_t.item()))
                decisions += 1
                if truncated:
                    raise RuntimeError("Evaluation episode truncated unexpectedly")
                if terminated:
                    outcome = float(reward)
                    break

        turns += env.engine.state.turns_completed
        if outcome > 0:
            wins += 1
        elif outcome < 0:
            losses += 1
        else:
            draws += 1

    decisive = wins + losses
    return {
        "games": float(games),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": wins / decisive if decisive else 0.0,
        "avg_decisions": decisions / games,
        "avg_turns": turns / games,
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    update: int,
    env_steps: int,
    episodes: int,
    best_eval_win_rate: float,
    training_args: dict,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "hoola-rl-v1",
            "model_config": model.config_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": int(update),
            "env_steps": int(env_steps),
            "episodes": int(episodes),
            "best_eval_win_rate": float(best_eval_win_rate),
            "training_args": training_args,
        },
        path,
    )
    return path


def append_csv_row(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
