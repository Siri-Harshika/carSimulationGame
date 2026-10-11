"""Check loading, every cleaning rule and the quality report (DS.3, DS.4).

Owner: Shobita. Covers the Shobita test_dataset.py row of docs/workflow.md section 9.
Data is synthetic and written to tmp_path, so no webcam or real recordings are needed.
"""

from pathlib import Path

import numpy as np
import pytest

from src.capture.landmarks import HandLandmarks
from src.config import LABELS, MIN_ROWS_PER_CELL, SAMPLE_DIR
from src.data.collect import landmarks_to_row, save_rows
from src.data.dataset import load_dataset, parse_file_name, quality_report

RNG_SEED = 218


def good_rows(user: str, condition: str, per_label: int, seed: int = RNG_SEED) -> list[list]:
    """Return per_label distinct, valid rows for every label."""
    rng = np.random.default_rng(seed)
    rows = []
    for label in LABELS:
        for _ in range(per_label):
            hand = HandLandmarks(rng.uniform(0.2, 0.8, size=(21, 3)), "Right", 0.9)
            rows.append(landmarks_to_row(user, condition, label, hand))
    return rows


def write(tmp_path: Path, user: str, condition: str, rows: list[list],
          stamp: str = "20261014-161502") -> Path:
    """Write a session CSV with the recorder's own writer."""
    path = tmp_path / f"{user}_{condition}_{stamp}.csv"
    save_rows(path, rows)
    return path


def test_loads_sample_data() -> None:
    files = sorted(SAMPLE_DIR.glob("*.csv"))
    if not files:
        pytest.skip("data/sample/ has not been recorded yet")
    dataset, report = load_dataset(files)
    assert dataset.X.shape[1] == 63
    assert set(dataset.sessions) <= {f.name for f in files}
    assert report.rows_kept == len(dataset.y)


def test_shape_groups_and_sessions(tmp_path: Path) -> None:
    a = write(tmp_path, "u01", "bright_near", good_rows("u01", "bright_near", 3))
    b = write(tmp_path, "u02", "dim_far", good_rows("u02", "dim_far", 2, seed=1))
    dataset, report = load_dataset([a, b])
    assert dataset.X.shape == (20, 63) and dataset.X.dtype == np.float32
    assert report.rows_read == report.rows_kept == 20
    assert set(dataset.groups) == {"u01", "u02"}
    assert set(dataset.sessions) == {a.name, b.name}
    assert (dataset.groups[dataset.sessions == b.name] == "u02").all()
    assert set(dataset.conditions) == {"bright_near", "dim_far"}


def test_wrong_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "u01_bright_near_20261014-161502.csv"
    path.write_text("user,label\nu01,fist\n", encoding="utf-8")
    with pytest.raises(ValueError, match=path.name):
        load_dataset([path])


def test_user_id_differing_from_file_name_raises(tmp_path: Path) -> None:
    path = write(tmp_path, "u01", "bright_near", good_rows("u02", "bright_near", 1))
    with pytest.raises(ValueError, match="u01"):
        load_dataset([path])


def test_bad_file_name_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        parse_file_name(tmp_path / "shobita_test.csv")


def test_no_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(sorted(tmp_path.glob("*.csv")))


def _planted(reason: str, base: list) -> list:
    """Return a copy of a valid 'fist' row broken in exactly one way."""
    row = list(base)
    if reason == "bad_label":
        row[2] = "thumbs_up"
    elif reason == "non_finite":
        row[10] = float("nan")
    elif reason == "bad_handedness":
        row[-1] = "Both"
    elif reason == "degenerate":
        row[3:-1] = [0.5] * 63              # every point identical → palm size 0
    elif reason == "out_of_frame":
        row[3:-1] = [v + 0.9 if i % 3 == 0 else v for i, v in enumerate(row[3:-1])]
    return row


@pytest.mark.parametrize(
    "reason", ["bad_label", "non_finite", "bad_handedness", "degenerate", "out_of_frame",
               "duplicate"],
)
def test_each_cleaning_rule_drops_planted_row(tmp_path: Path, reason: str) -> None:
    rows = good_rows("u01", "bright_near", 2)
    fist = rows[0]
    rows.append(list(fist) if reason == "duplicate" else _planted(reason, fist))
    dataset, report = load_dataset([write(tmp_path, "u01", "bright_near", rows)])
    assert report.rows_read == 9
    assert report.rows_kept == len(dataset.y) == 8
    assert report.dropped[reason] == 1
    assert sum(report.dropped.values()) == 1


def test_out_of_frame_kept_for_none(tmp_path: Path) -> None:
    rows = good_rows("u01", "bright_near", 1)
    edge = _planted("out_of_frame", rows[-1])
    assert edge[2] == "none"
    rows.append(edge)
    dataset, report = load_dataset([write(tmp_path, "u01", "bright_near", rows)])
    assert report.dropped["out_of_frame"] == 0
    assert report.rows_kept == 5


def test_excluded_files_not_loaded(tmp_path: Path) -> None:
    a = write(tmp_path, "u01", "bright_near", good_rows("u01", "bright_near", 1))
    b = write(tmp_path, "u01", "dim_near", good_rows("u01", "dim_near", 1, seed=2))
    dataset, report = load_dataset([a, b], excluded_files={b.name: "wrong lamp"})
    assert set(dataset.sessions) == {a.name}
    assert report.excluded_files == {b.name: "wrong lamp"}


def test_quality_report_flags_small_cells(tmp_path: Path) -> None:
    path = write(tmp_path, "u01", "bright_near", good_rows("u01", "bright_near", 2))
    dataset, cleaning = load_dataset([path])
    text = quality_report(dataset, cleaning, {"train": ["u01"], "test": []})
    assert f"below {MIN_ROWS_PER_CELL}" in text
    line = next(line for line in text.splitlines() if line.startswith("u01   bright_near"))
    assert "2*" in line
    # Missing core conditions are listed as 0 and flagged too.
    assert any(line.startswith("u01   dim_far") and "0*" in line for line in text.splitlines())
    assert "Cells below 60: 16" in text
