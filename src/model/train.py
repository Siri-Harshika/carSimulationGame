"""build_pipeline(), tune (LOUO over train users), train_model(), save.

Owner: Siri (Classifier).
Provides: the model bundle models/gesture_svm.joblib — consumed by predict.py and evaluate.py.
Consumes: load_split("train") from src.data.split. NEVER loads the test split.
Specification: docs/interfaces.md §4.5.

Usage (from the repo root):
    python -m src.model.train [--grid] [--splits data/splits.json] [--out models/gesture_svm.joblib]

STUB (Lead, Iteration 1): signatures are frozen; Siri replaces every
`raise NotImplementedError` with the real implementation.
"""

from pathlib import Path

from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, SVM_C, SVM_GAMMA
from src.data.dataset import Dataset
from src.data.split import Splits


def build_pipeline(C: float = SVM_C, gamma: float | str = SVM_GAMMA) -> Pipeline:
    """Build StandardScaler → SVC(rbf, probability=True, class_weight='balanced', random_state=SEED).

    Args:
        C: SVM regularisation parameter.
        gamma: RBF kernel width ("scale" or a float).

    Returns:
        An unfitted sklearn Pipeline.
    """
    raise NotImplementedError


def tune(train: Dataset) -> tuple[dict, list[dict]]:
    """Grid-search SVM_GRID_C × SVM_GRID_GAMMA with LeaveOneGroupOut over train users (macro-F1).

    Args:
        train: The train split (groups = user_id).

    Returns:
        (best_params, per-fold/per-param records).
    """
    raise NotImplementedError


def train_model(train: Dataset, C: float = SVM_C, gamma: float | str = SVM_GAMMA) -> Pipeline:
    """Fit the pipeline on ALL train users.

    Args:
        train: The train split.
        C: SVM regularisation parameter.
        gamma: RBF kernel width.

    Returns:
        The fitted Pipeline.
    """
    raise NotImplementedError


def save_model(pipeline: Pipeline, splits: Splits, path: Path = MODEL_PATH) -> None:
    """joblib.dump the model bundle dict (fields listed in docs/interfaces.md §4.5).

    Args:
        pipeline: Fitted pipeline.
        splits: The splits it was trained with (records train/test users for the leakage guard).
        path: Output file.
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    """CLI: [--grid] [--splits data/splits.json] [--out models/gesture_svm.joblib].

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
