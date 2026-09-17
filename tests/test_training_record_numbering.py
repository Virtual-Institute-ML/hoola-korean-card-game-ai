from pathlib import Path

from run_game import resolve_training_record_path


def test_numbered_training_path_starts_at_requested_number(tmp_path):
    path = resolve_training_record_path(tmp_path / "human_vs_heuristic_001.npz")
    assert path.name == "human_vs_heuristic_001.npz"


def test_numbered_training_path_advances_without_overwrite(tmp_path):
    (tmp_path / "human_vs_heuristic_001.npz").touch()
    (tmp_path / "human_vs_heuristic_001.json").touch()
    (tmp_path / "human_vs_heuristic_002.npz").touch()
    (tmp_path / "human_vs_heuristic_002.json").touch()

    path = resolve_training_record_path(tmp_path / "human_vs_heuristic_001.npz")
    assert path.name == "human_vs_heuristic_003.npz"


def test_json_alone_reserves_number(tmp_path):
    (tmp_path / "human_vs_heuristic_001.json").touch()
    path = resolve_training_record_path(tmp_path / "human_vs_heuristic_001.npz")
    assert path.name == "human_vs_heuristic_002.npz"


def test_unnumbered_training_path_gets_001_suffix(tmp_path):
    path = resolve_training_record_path(tmp_path / "human_vs_heuristic.npz")
    assert path.name == "human_vs_heuristic_001.npz"


def test_unnumbered_path_increments_from_existing_files(tmp_path):
    (tmp_path / "human_vs_heuristic_001.npz").touch()
    (tmp_path / "human_vs_heuristic_002.json").touch()
    path = resolve_training_record_path(tmp_path / "human_vs_heuristic.npz")
    assert path.name == "human_vs_heuristic_003.npz"
