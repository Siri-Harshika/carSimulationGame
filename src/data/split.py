"""Split the sessions by user into train and test, saved as data/splits.json.

Owner: Shobita (Dataset).
Spec: docs/interfaces.md section 4.5, split rules in docs/data-eval.md section 7.5.

Splitting by user (not by random rows) means the test score shows how the model
does on people it has never seen. Siri does the validation (leave-one-user-out)
on the train users.

Usage (from the repo root):
    python -m src.data.split                    # pick 2 random test users
    python -m src.data.split --test u02 u05     # choose the test users
    python -m src.data.split --exclude u04_dim_far_20261014-180001.csv="wrong lamp"
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from src.config import CORE_CONDITIONS, DATA_DIR, N_TEST_USERS, ROOT_DIR, SEED, SPLITS_PATH
from src.data.dataset import Dataset, load_dataset, parse_file_name

MIN_TRAIN_USERS = 3   # leave-one-user-out needs at least 3 train users


@dataclass(frozen=True)
class Splits:
    data_dir: Path
    train_users: tuple[str, ...]
    test_users: tuple[str, ...]
    validation: str                      # always "leave_one_user_out"
    files: dict[str, tuple[str, ...]]    # "train"/"test" -> file names
    excluded_files: dict[str, str]       # file name -> reason
    seed: int = SEED


def check_splits(splits: Splits) -> None:
    """Raise ValueError if a user or file is in both splits (data leakage)."""
    if set(splits.files) != {"train", "test"}:
        raise ValueError("files must have exactly 'train' and 'test'")
    both = set(splits.train_users) & set(splits.test_users)
    if both:
        raise ValueError(f"users in both train and test: {sorted(both)}")
    if len(splits.train_users) < MIN_TRAIN_USERS:
        raise ValueError(f"need at least {MIN_TRAIN_USERS} train users, "
                         f"got {len(splits.train_users)}")
    if not splits.test_users:
        raise ValueError("need at least 1 test user")

    users = {"train": splits.train_users, "test": splits.test_users}
    for name in ("train", "test"):
        for file_name in splits.files[name]:
            if file_name in splits.excluded_files:
                raise ValueError(f"{file_name} is excluded but is in {name}")
            if parse_file_name(file_name)[0] not in users[name]:
                raise ValueError(f"{file_name} is in {name} but its user is not")


def make_splits(data_dir: Path = DATA_DIR, test_users=None, n_test_users: int = N_TEST_USERS,
                seed: int = SEED, excluded_files=None) -> Splits:
    """Put every file in train or test depending on its user.

    If test_users is None, n_test_users are picked at random (with seed) from the
    users who recorded all 4 core conditions.

    """
    data_dir = Path(data_dir)
    excluded = dict(excluded_files or {})
    names = sorted(p.name for p in data_dir.glob("*.csv") if p.name not in excluded)
    if not names:
        raise FileNotFoundError(f"no CSV files in {data_dir}")

    files_of, conditions_of = {}, {}
    for name in names:
        user, condition = parse_file_name(name)
        files_of.setdefault(user, []).append(name)
        conditions_of.setdefault(user, set()).add(condition)

    # Test users need all core conditions so test accuracy can be compared per condition.
    complete = sorted(u for u in files_of if set(CORE_CONDITIONS) <= conditions_of[u])
    if test_users is None:
        if len(complete) < n_test_users:
            raise ValueError(f"only {len(complete)} users have all core conditions, "
                             f"need {n_test_users}")
        test_users = random.Random(seed).sample(complete, n_test_users)
    else:
        for user in test_users:
            if user not in files_of:
                raise ValueError(f"test user {user} has no files")
            if user not in complete:
                raise ValueError(f"test user {user} is missing a core condition")

    test = tuple(sorted(set(test_users)))
    train = tuple(u for u in sorted(files_of) if u not in test)
    splits = Splits(
        data_dir=data_dir,
        train_users=train,
        test_users=test,
        validation="leave_one_user_out",
        files={"train": tuple(f for u in train for f in files_of[u]),
               "test": tuple(f for u in test for f in files_of[u])},
        excluded_files=excluded,
        seed=seed,
    )
    check_splits(splits)
    return splits


def save_splits(splits: Splits, path: Path = SPLITS_PATH) -> None:
    """Save the split as JSON (format in docs/data-eval.md section 7.5)."""
    data_dir = Path(splits.data_dir).resolve()
    # Store the folder relative to the repo so the file works on everyone's laptop.
    if data_dir.is_relative_to(ROOT_DIR):
        data_dir = data_dir.relative_to(ROOT_DIR)
    data = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": splits.seed,
        "data_dir": data_dir.as_posix(),
        "test_users": list(splits.test_users),
        "train_users": list(splits.train_users),
        "validation": splits.validation,
        "files": {name: list(files) for name, files in splits.files.items()},
        "excluded_files": splits.excluded_files,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_splits(path: Path = SPLITS_PATH) -> Splits:
    """Read splits.json and check it again, so a hand-edited file can't leak users."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run python -m src.data.split first")
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        splits = Splits(
            data_dir=ROOT_DIR / data["data_dir"],   # an absolute path stays absolute
            train_users=tuple(data["train_users"]),
            test_users=tuple(data["test_users"]),
            validation=data["validation"],
            files={name: tuple(data["files"][name]) for name in ("train", "test")},
            excluded_files=data.get("excluded_files", {}),
            seed=data.get("seed", SEED),
        )
    except KeyError as missing:
        raise ValueError(f"{path} is missing {missing}") from None
    check_splits(splits)
    return splits


def load_split(name: str, path: Path = SPLITS_PATH) -> Dataset:
    """Load the cleaned, preprocessed data of the "train" or "test" split."""
    if name not in ("train", "test"):
        raise ValueError(f"name must be 'train' or 'test', got {name!r}")
    splits = load_splits(path)
    paths = [splits.data_dir / f for f in splits.files[name]]
    return load_dataset(paths, splits.excluded_files)[0]


def main(argv: list[str] | None = None) -> None:
    """CLI: [--test u02 u05] [--exclude FILE=REASON ...] [--data DIR] [--out FILE]."""
    parser = argparse.ArgumentParser(description="Write the user-based train/test split.")
    parser.add_argument("--test", nargs="+")
    parser.add_argument("--exclude", nargs="+", default=[], metavar="FILE=REASON")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    out = args.out
    if out is None:
        is_raw = args.data.resolve() == DATA_DIR.resolve()
        out = SPLITS_PATH if is_raw else args.data / "splits.json"

    # Keep exclusions from an existing file so a bad session is never put back by accident.
    excluded = {}
    if out.exists():
        excluded = json.loads(out.read_text(encoding="utf-8")).get("excluded_files", {})
        print(f"Replacing {out} (keep it frozen once Siri starts tuning)")
    for item in args.exclude:
        file_name, _, reason = item.partition("=")
        if not reason:
            parser.error(f"--exclude needs FILE=REASON, got {item!r}")
        excluded[file_name] = reason

    splits = make_splits(args.data, test_users=args.test, excluded_files=excluded)
    save_splits(splits, out)
    print(f"Saved {out}")
    print(f"  train: {', '.join(splits.train_users)} ({len(splits.files['train'])} files)")
    print(f"  test:  {', '.join(splits.test_users)} ({len(splits.files['test'])} files)")
    print(f"  excluded: {len(excluded)} files")


if __name__ == "__main__":
    main()
