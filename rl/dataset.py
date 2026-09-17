from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hoola.action_codec import ActionCodec
from hoola.encoding import OBSERVATION_SIZE


@dataclass
class HumanDataset:
    observations: np.ndarray
    action_masks: np.ndarray
    action_ids: np.ndarray
    outcomes: np.ndarray
    source_files: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.action_ids.shape[0])

    @classmethod
    def empty(cls) -> "HumanDataset":
        action_size = ActionCodec().size
        return cls(
            observations=np.empty((0, OBSERVATION_SIZE), dtype=np.float32),
            action_masks=np.empty((0, action_size), dtype=np.int8),
            action_ids=np.empty((0,), dtype=np.int64),
            outcomes=np.empty((0,), dtype=np.float32),
            source_files=(),
        )

    @classmethod
    def from_path(cls, path_spec: str | Path | None) -> "HumanDataset":
        """Load only decisions labelled ``human`` from training records.

        ``path_spec`` may be one NPZ file, a directory (recursive *.npz), or a
        glob expression.  Companion JSON metadata is used to identify which
        steps were made by a human player.
        """
        if path_spec is None:
            return cls.empty()

        spec = str(path_spec)
        path = Path(spec)
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(path.rglob("*.npz"))
        else:
            files = [Path(p) for p in sorted(glob.glob(spec, recursive=True))]

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
                data = np.load(npz_path, allow_pickle=False)
                observations = data["observations"]
                masks = data["action_masks"]
                actions = data["action_ids"]
                outcomes = data["final_outcomes"]
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                continue

            n = len(actions)
            if len(agent_types) != n:
                # Older/nonstandard records without per-step agent labels are
                # deliberately ignored rather than accidentally treating AI
                # moves as human demonstrations.
                continue
            selector = np.asarray([str(x).lower() == "human" for x in agent_types], dtype=bool)
            if not selector.any():
                continue

            selected_obs = np.asarray(observations[selector], dtype=np.float32)
            selected_masks = np.asarray(masks[selector], dtype=np.int8)
            selected_actions = np.asarray(actions[selector], dtype=np.int64)
            selected_outcomes = np.asarray(outcomes[selector], dtype=np.float32)

            if selected_obs.shape[1:] != (OBSERVATION_SIZE,):
                continue
            if selected_masks.shape[1:] != (ActionCodec().size,):
                continue
            rows = np.arange(selected_actions.shape[0])
            if np.any(selected_masks[rows, selected_actions] != 1):
                raise ValueError(f"Human record contains an illegal selected action: {npz_path}")

            obs_parts.append(selected_obs)
            mask_parts.append(selected_masks)
            action_parts.append(selected_actions)
            outcome_parts.append(selected_outcomes)
            used_files.append(str(npz_path))

        if not obs_parts:
            return cls.empty()
        return cls(
            observations=np.concatenate(obs_parts, axis=0),
            action_masks=np.concatenate(mask_parts, axis=0),
            action_ids=np.concatenate(action_parts, axis=0),
            outcomes=np.concatenate(outcome_parts, axis=0),
            source_files=tuple(used_files),
        )

    def sample_indices(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        if len(self) == 0:
            raise RuntimeError("Cannot sample from an empty HumanDataset")
        replace = len(self) < batch_size
        return rng.choice(len(self), size=batch_size, replace=replace)
