from __future__ import annotations

import json

import numpy as np

from hoola.action_codec import ActionCodec
from hoola.encoding import OBSERVATION_SIZE
from rl.dataset import HumanDataset


def test_human_dataset_filters_only_human_steps(tmp_path):
    n = 3
    observations = np.zeros((n, OBSERVATION_SIZE), dtype=np.float32)
    masks = np.zeros((n, ActionCodec().size), dtype=np.int8)
    actions = np.asarray([0, 1, 2], dtype=np.int32)
    masks[np.arange(n), actions] = 1
    outcomes = np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
    npz = tmp_path / "game_001.npz"
    np.savez_compressed(
        npz,
        observations=observations,
        action_masks=masks,
        action_ids=actions,
        final_outcomes=outcomes,
    )
    npz.with_suffix(".json").write_text(
        json.dumps({"agent_types": ["human", "heuristic", "human"]}),
        encoding="utf-8",
    )

    ds = HumanDataset.from_path(tmp_path)
    assert len(ds) == 2
    assert ds.action_ids.tolist() == [0, 2]
    assert ds.outcomes.tolist() == [1.0, 1.0]
