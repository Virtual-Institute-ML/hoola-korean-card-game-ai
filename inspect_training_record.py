from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(path: str) -> None:
    npz_path = Path(path)
    data = np.load(npz_path)
    meta_path = npz_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    print(f"file              : {npz_path}")
    print(f"steps             : {len(data['action_ids'])}")
    print(f"observations      : {data['observations'].shape} {data['observations'].dtype}")
    print(f"action masks      : {data['action_masks'].shape} {data['action_masks'].dtype}")
    print(f"action IDs        : {data['action_ids'].shape} {data['action_ids'].dtype}")
    if len(data["final_outcomes"]):
        unique, counts = np.unique(data["final_outcomes"], return_counts=True)
        print(f"final outcomes    : {dict(zip(unique.tolist(), counts.tolist()))}")
    if meta:
        print(f"seed              : {meta.get('seed')}")
        print(f"players           : {meta.get('players')}")
        print(f"result            : {meta.get('result')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    main(args.path)
