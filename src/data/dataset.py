"""Load the recorded CSVs, remove invalid rows and write the quality report.

Owner: Shobita (Dataset).
Spec: docs/interfaces.md section 4.5, cleaning rules in docs/data-eval.md section 7.5.

Usage (from the repo root):
    python -m src.data.dataset --report
    python -m src.data.dataset --report --data data/sample
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from src.capture.preprocess import preprocess, row_to_landmarks
from src.config import (
    CONDITIONS,
    CORE_CONDITIONS,
    CSV_COLUMNS,
    DATA_DIR,
    LABELS,
    MAX_OUT_OF_FRAME_LANDMARKS,
    MIN_ROWS_PER_CELL,
    OUT_OF_FRAME_MARGIN,
    RESULTS_DIR,
    SPLITS_PATH,
)

DROP_REASONS = ("bad_label", "non_finite", "bad_handedness", "degenerate", "out_of_frame",
                "duplicate")


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray           # float32 (n, 63), preprocess() output
    y: np.ndarray           # labels
    groups: np.ndarray      # user_id of each row
    conditions: np.ndarray  # condition of each row
    sessions: np.ndarray    # CSV file name of each row
    handedness: np.ndarray  # "Left" / "Right"


@dataclass(frozen=True)
class CleaningReport:
    rows_read: int
    rows_kept: int
    dropped: dict[str, int]          # reason -> count
    excluded_files: dict[str, str]   # file name -> reason


def parse_file_name(path: Path | str) -> tuple[str, str]:
    """Get (user_id, condition) from a name like u03_dim_far_20261014-161502.csv."""
    name = Path(path).name
    match = re.fullmatch(r"(u\d{2})_([a-z]+_[a-z]+)_\d{8}-\d{6}\.csv", name)
    if not match or match.group(2) not in CONDITIONS:
        raise ValueError(f"{name}: expected a name like u03_dim_far_20261014-161502.csv")
    return match.group(1), match.group(2)


def check_row(row: list[str], seen: set) -> tuple[str | None, np.ndarray | None]:
    """Apply the cleaning rules to one CSV row.

    Returns (reason, None) if the row should be dropped, else (None, features).
    """
    label, handedness = row[2], row[-1]
    if label not in LABELS:
        return "bad_label", None
    try:
        values = np.array(row[3:-1], dtype=float)
    except ValueError:
        return "non_finite", None
    if not np.isfinite(values).all():
        return "non_finite", None
    if handedness not in ("Left", "Right"):
        return "bad_handedness", None
    try:
        features = preprocess(row_to_landmarks(values, handedness))
    except ValueError:
        return "degenerate", None

    # Points far outside the image are guesses by MediaPipe. Only check the real
    # gestures, half-visible hands are on purpose for "none".
    if label != "none":
        xy = values.reshape(21, 3)[:, :2]
        outside = ((xy < -OUT_OF_FRAME_MARGIN) | (xy > 1 + OUT_OF_FRAME_MARGIN)).any(axis=1)
        if outside.sum() > MAX_OUT_OF_FRAME_LANDMARKS:
            return "out_of_frame", None

    key = tuple(values)
    if key in seen:
        return "duplicate", None
    seen.add(key)
    return None, features


def load_dataset(paths=None, excluded_files=None) -> tuple[Dataset, CleaningReport]:
    """Read the CSVs (default data/raw/*.csv), drop invalid rows and preprocess the rest.

    Raises:
        FileNotFoundError: no CSV files to read.
        ValueError: a file has the wrong header, or its rows don't match its file name.
    """
    excluded = dict(excluded_files or {})
    if paths is None:
        paths = sorted(DATA_DIR.glob("*.csv"))
    paths = [Path(p) for p in paths if Path(p).name not in excluded]
    if not paths:
        raise FileNotFoundError("No CSV files found, record some data first.")

    X, meta = [], []
    dropped = dict.fromkeys(DROP_REASONS, 0)
    rows_read = 0

    for path in paths:
        user_id, condition = parse_file_name(path)
        seen = set()   # for duplicates, which are checked per session
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            if tuple(next(reader, ())) != CSV_COLUMNS:
                raise ValueError(f"{path.name}: header does not match CSV_COLUMNS")
            for row in reader:
                if not row:
                    continue
                rows_read += 1
                if len(row) != len(CSV_COLUMNS):
                    dropped["non_finite"] += 1
                    continue
                if (row[0], row[1]) != (user_id, condition):
                    raise ValueError(f"{path.name}: contains rows for {row[0]} {row[1]}")
                reason, features = check_row(row, seen)
                if reason:
                    dropped[reason] += 1
                else:
                    X.append(features)
                    meta.append((row[2], user_id, condition, path.name, row[-1]))

    columns = list(zip(*meta)) or [()] * 5
    y, groups, conditions, sessions, hands = (np.array(col, dtype=str) for col in columns)
    dataset = Dataset(np.array(X, dtype=np.float32).reshape(-1, 63),
                      y, groups, conditions, sessions, hands)
    return dataset, CleaningReport(rows_read, len(X), dropped, excluded)


def quality_report(dataset: Dataset, cleaning: CleaningReport, split_users=None) -> str:
    """Text report: kept/dropped rows, counts per user x condition x label, label balance.

    split_users is an optional {"train": [...], "test": [...]} for balance per split.
    Cells below MIN_ROWS_PER_CELL are marked with * and need re-recording.
    """
    lines = [
        "DATASET QUALITY REPORT",
        "",
        f"Rows read: {cleaning.rows_read}",
        f"Rows kept: {cleaning.rows_kept}",
        "Rows dropped:",
    ]
    lines += [f"  {reason:<15}{count}" for reason, count in cleaning.dropped.items()]

    cells = Counter(zip(dataset.groups, dataset.conditions, dataset.y))
    users = sorted(set(dataset.groups))
    lines += ["", f"Counts per user x condition x label (* = below {MIN_ROWS_PER_CELL})",
              f"{'user':<6}{'condition':<14}" + "".join(f"{label:>11}" for label in LABELS)]
    flagged = 0
    for user in users:
        recorded = {c for (u, c, _) in cells if u == user}
        # Always show the 4 core conditions, so a missing session shows up as 0.
        for condition in [c for c in CONDITIONS if c in CORE_CONDITIONS or c in recorded]:
            line = f"{user:<6}{condition:<14}"
            for label in LABELS:
                n = cells[(user, condition, label)]
                low = n < MIN_ROWS_PER_CELL
                flagged += low
                line += f"{n:>10}{'*' if low else ' '}"
            lines.append(line)
    lines.append(f"Cells below {MIN_ROWS_PER_CELL}: {flagged}")

    def balance(labels: np.ndarray) -> str:
        counts = Counter(labels)
        return "  ".join(f"{label} {counts[label]}" for label in LABELS)

    lines += ["", f"Label balance ({len(users)} users): {balance(dataset.y)}"]
    for name, members in (split_users or {}).items():
        mask = np.isin(dataset.groups, list(members))
        lines.append(f"  {name} {list(members)}: {balance(dataset.y[mask])}")

    lines += ["", f"Excluded files: {len(cleaning.excluded_files)}"]
    lines += [f"  {name}: {reason}" for name, reason in cleaning.excluded_files.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """CLI: --report [--data DIR] [--out FILE]. Uses the folder's splits.json if there is one."""
    parser = argparse.ArgumentParser(description="Clean the data and write the quality report.")
    parser.add_argument("--report", action="store_true", required=True)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "dataset_report.txt")
    args = parser.parse_args(argv)

    # data/raw uses data/splits.json, any other folder its own splits.json.
    is_raw = args.data.resolve() == DATA_DIR.resolve()
    splits_path = SPLITS_PATH if is_raw else args.data / "splits.json"
    excluded, split_users = {}, None
    if splits_path.exists():
        from src.data.split import load_splits   # imported here, split.py imports this file
        splits = load_splits(splits_path)
        excluded = splits.excluded_files
        split_users = {"train": splits.train_users, "test": splits.test_users}

    dataset, cleaning = load_dataset(sorted(args.data.glob("*.csv")), excluded)
    report = quality_report(dataset, cleaning, split_users)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
