"""Every module in src/ imports, and config.py obeys its invariants.

Owner: Lead.
Specification: docs/workflow.md §9 (Lead row), docs/game-logic.md §6.5.

This is the Iteration 1 exit check: if it fails, someone broke an import or a
constant that other people's code relies on.
"""

from importlib import import_module
from pathlib import Path

import pytest

from src import config

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def _all_modules() -> list[str]:
    """Return dotted names of every module in the src package tree.

    Folders without __init__.py (e.g. src/capture/check/, Triet's webcam scripts
    that open the camera at import time) are not packages and are skipped.
    """
    names = []
    for path in sorted(SRC_DIR.rglob("*.py")):
        if not (path.parent / "__init__.py").exists():
            continue
        parts = path.relative_to(SRC_DIR.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


@pytest.mark.parametrize("module", _all_modules())
def test_module_imports(module: str) -> None:
    """Importing a module must not fail or open hardware."""
    import_module(module)


@pytest.mark.parametrize(("package", "names"), [
    ("src.capture", ["Camera", "Frame", "HandTracker", "HandLandmarks", "preprocess"]),
    ("src.data", ["Dataset", "load_dataset", "Splits", "load_splits", "load_split"]),
    ("src.game", ["GameController", "GameState"]),
    ("src.model", ["GestureClassifier", "Classifier"]),
])
def test_package_reexports(package: str, names: list[str]) -> None:
    """Each package re-exports what docs/interfaces.md §3 promises."""
    mod = import_module(package)
    for name in names:
        assert getattr(mod, name) is not None


def test_labels() -> None:
    """Labels are unique, NO_HAND is never a classifier label, CSV columns are I2 + handedness."""
    assert len(set(config.LABELS)) == len(config.LABELS) == 4
    assert config.LABEL_NONE in config.LABELS
    assert config.NO_HAND not in config.LABELS
    assert config.NUM_FEATURES == config.NUM_LANDMARKS * config.NUM_COORDS == 63
    assert config.CSV_COLUMNS[:3] == ("user_id", "condition", "label")
    assert config.CSV_COLUMNS[-1] == "handedness"
    assert len(config.CSV_COLUMNS) == 3 + config.NUM_FEATURES + 1


def test_conditions() -> None:
    """Core conditions are a subset of all conditions."""
    assert set(config.CORE_CONDITIONS) <= set(config.CONDITIONS)
    assert len(config.CONDITIONS) == len(config.LIGHTING_CONDITIONS) * len(config.DISTANCE_CONDITIONS)


def test_window_and_lanes() -> None:
    """Lanes tile the window exactly, and car/obstacle fit inside a lane and the window."""
    assert config.LANE_WIDTH * config.NUM_LANES == config.WINDOW_WIDTH
    assert config.CAR_WIDTH < config.LANE_WIDTH
    assert config.OBSTACLE_WIDTH < config.LANE_WIDTH
    assert config.HUD_HEIGHT < config.CAR_Y
    assert config.CAR_Y + config.CAR_HEIGHT <= config.WINDOW_HEIGHT


def test_gameplay_is_fair() -> None:
    """A gap between rows always leaves room to change lane (docs/risks.md #15)."""
    assert config.SPAWN_GAP_PX > config.OBSTACLE_HEIGHT + config.CAR_HEIGHT
    assert 0.0 <= config.DOUBLE_OBSTACLE_PROB < 1.0
    assert 0 < config.BASE_SPEED <= config.MAX_SPEED
    assert config.SPEED_INCREASE_PER_SECOND >= 0
    assert 0 < config.MAX_DT <= 0.1


def test_pipeline_ranges() -> None:
    """Smoothing/debounce/threshold values are within the ranges their constructors accept."""
    assert config.SMOOTHING_WINDOW_S > 0
    assert 0.5 < config.SMOOTHING_MIN_FRACTION <= 1.0
    assert config.SMOOTHING_MIN_FRAMES >= 1
    assert config.ACTION_COOLDOWN_S >= 0
    assert 0.0 <= config.CONFIDENCE_THRESHOLD <= 1.0
    assert config.RESUME_COUNTDOWN_S > 0
    assert config.GRACE_PERIOD_S >= 0


def test_data_settings() -> None:
    """Recording settings agree with each other."""
    assert config.MIN_ROWS_PER_CELL <= config.RECORD_TARGET_PER_LABEL
    assert config.RECORD_SAMPLE_HZ > 0
    assert config.N_TEST_USERS >= 1
    assert config.SVM_C in config.SVM_GRID_C
    assert config.SVM_GAMMA in config.SVM_GRID_GAMMA


def test_paths_are_inside_repo() -> None:
    """Every path constant points inside the repository."""
    for path in (config.DATA_DIR, config.SAMPLE_DIR, config.SPLITS_PATH,
                 config.MODEL_PATH, config.RESULTS_DIR):
        assert config.ROOT_DIR in path.parents
