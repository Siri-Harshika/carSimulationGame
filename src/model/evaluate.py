"""One-time test-set evaluation, confusion matrix, per-condition tables.

Owner: Siri (Classifier).
Provides: results/ files of docs/data-eval.md §8.5.
Specification: docs/interfaces.md §4.5.

Usage (from the repo root):
    python -m src.model.evaluate [--splits data/splits.json] [--timing] [--nice]

STUB (Lead, Iteration 1): signatures are frozen; Siri replaces every
`raise NotImplementedError` with the real implementation.
"""

from pathlib import Path

from src.config import MODEL_PATH, SPLITS_PATH
from src.data.dataset import Dataset


def validation_predictions(train: Dataset, C: float, gamma: float | str) -> list[dict]:
    """Out-of-fold predictions via leave-one-user-out over the TRAIN users.

    Used for threshold tuning and per-user spread.

    Args:
        train: The train split.
        C: SVM regularisation parameter.
        gamma: RBF kernel width.

    Returns:
        Records {user_id, condition, true, pred, confidence}.
    """
    raise NotImplementedError


def test_predictions(bundle_path: Path = MODEL_PATH,
                     splits_path: Path = SPLITS_PATH) -> list[dict]:
    """Predictions of the saved model on the TEST users.

    Args:
        bundle_path: The saved model bundle.
        splits_path: The frozen splits.json.

    Returns:
        Records {user_id, condition, true, pred, confidence}.

    Raises:
        ValueError: If any test user is in bundle["trained_on_users"] or
            bundle["test_users"] != splits.test_users (leakage guard).
    """
    raise NotImplementedError


# pytest collects any module-level function named test_*; this one is not a test.
test_predictions.__test__ = False  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> None:
    """CLI: [--splits data/splits.json] [--timing] [--nice]; writes the §8.5 Siri files.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
