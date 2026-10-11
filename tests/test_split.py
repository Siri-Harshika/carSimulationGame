"""Check the user-based split and its leakage guards (DS.5).

Owner: Shobita. Covers the Shobita test_split.py row of docs/workflow.md section 9.
Each test builds a folder of tiny synthetic sessions in tmp_path.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.capture.landmarks import HandLandmarks
from src.config import CORE_CONDITIONS, LABELS
from src.data.collect import landmarks_to_row, save_rows
from src.data.split import load_split, load_splits, main, make_splits, save_splits

USERS = ("u01", "u02", "u03", "u04", "u05", "u06")


def make_folder(root: Path, users: tuple[str, ...] = USERS,
                skip: tuple[str, str] | None = None) -> Path:
    """Write one 1-row-per-label session per user × core condition (minus ``skip``)."""
    rng = np.random.default_rng(218)
    for i, user in enumerate(users):
        for j, condition in enumerate(CORE_CONDITIONS):
            if (user, condition) == skip:
                continue
            rows = [landmarks_to_row(user, condition, label,
                                     HandLandmarks(rng.uniform(0.2, 0.8, (21, 3)), "Right", 1.0))
                    for label in LABELS]
            save_rows(root / f"{user}_{condition}_20261014-1{i}{j}000.csv", rows)
    return root


def test_no_user_or_file_in_two_splits(tmp_path: Path) -> None:
    splits = make_splits(make_folder(tmp_path))
    assert not set(splits.train_users) & set(splits.test_users)
    assert not set(splits.files["train"]) & set(splits.files["test"])
    assert len(splits.test_users) == 2 and len(splits.train_users) == 4


def test_every_file_of_a_user_is_in_that_users_split(tmp_path: Path) -> None:
    splits = make_splits(make_folder(tmp_path))
    for name, users in (("train", splits.train_users), ("test", splits.test_users)):
        for user in users:
            own = sorted(p.name for p in tmp_path.glob(f"{user}_*.csv"))
            assert own == sorted(f for f in splits.files[name] if f.startswith(user))
    assert len(splits.files["train"]) + len(splits.files["test"]) == 24


def test_excluded_files_in_no_split(tmp_path: Path) -> None:
    make_folder(tmp_path)
    bad = sorted(p.name for p in tmp_path.glob("u03_*.csv"))[0]
    splits = make_splits(tmp_path, test_users=["u01", "u02"], excluded_files={bad: "wrong lamp"})
    assert bad not in splits.files["train"] + splits.files["test"]
    assert splits.excluded_files == {bad: "wrong lamp"}


def test_explicit_test_users_respected(tmp_path: Path) -> None:
    splits = make_splits(make_folder(tmp_path), test_users=["u05", "u02"])
    assert splits.test_users == ("u02", "u05")
    assert splits.train_users == ("u01", "u03", "u04", "u06")


def test_same_seed_same_split(tmp_path: Path) -> None:
    make_folder(tmp_path)
    assert make_splits(tmp_path, seed=7) == make_splits(tmp_path, seed=7)


def test_unknown_test_user_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="u09"):
        make_splits(make_folder(tmp_path), test_users=["u09"])


def test_incomplete_test_user_raises(tmp_path: Path) -> None:
    make_folder(tmp_path, skip=("u02", "dim_far"))
    with pytest.raises(ValueError, match="missing"):
        make_splits(tmp_path, test_users=["u02"])
    # A random pick never chooses the incomplete user.
    for seed in range(20):
        assert "u02" not in make_splits(tmp_path, seed=seed).test_users


def test_fewer_than_3_train_users_raises(tmp_path: Path) -> None:
    make_folder(tmp_path, users=USERS[:4])
    with pytest.raises(ValueError, match="at least 3"):
        make_splits(tmp_path, test_users=["u01", "u02"])


def test_save_load_round_trip(tmp_path: Path) -> None:
    data = make_folder(tmp_path / "raw")
    splits = make_splits(data, test_users=["u02", "u05"])
    path = tmp_path / "splits.json"
    save_splits(splits, path)
    loaded = load_splits(path)
    assert loaded.data_dir.resolve() == data.resolve()
    assert loaded == splits.__class__(**{**splits.__dict__, "data_dir": loaded.data_dir})
    assert json.loads(path.read_text())["validation"] == "leave_one_user_out"


def test_hand_edited_overlap_raises_on_load(tmp_path: Path) -> None:
    data = make_folder(tmp_path / "raw")
    path = tmp_path / "splits.json"
    save_splits(make_splits(data, test_users=["u02", "u05"]), path)
    raw = json.loads(path.read_text())
    raw["train_users"].append("u02")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="u02"):
        load_splits(path)


def test_load_split_test_returns_only_test_users(tmp_path: Path) -> None:
    data = make_folder(tmp_path / "raw")
    path = tmp_path / "splits.json"
    save_splits(make_splits(data, test_users=["u02", "u05"]), path)
    test = load_split("test", path)
    assert set(test.groups) == {"u02", "u05"}
    assert len(test.y) == 2 * len(CORE_CONDITIONS) * len(LABELS)
    assert set(load_split("train", path).groups) == {"u01", "u03", "u04", "u06"}
    with pytest.raises(ValueError):
        load_split("validation", path)  # type: ignore[arg-type]


def test_cli_writes_and_keeps_exclusions(tmp_path: Path) -> None:
    data = make_folder(tmp_path)
    bad = sorted(p.name for p in data.glob("u04_*.csv"))[0]
    main(["--data", str(data), "--test", "u01", "u02", "--exclude", f"{bad}=wrong lamp"])
    main(["--data", str(data), "--test", "u01", "u02"])      # re-run without --exclude
    assert load_splits(data / "splits.json").excluded_files == {bad: "wrong lamp"}
