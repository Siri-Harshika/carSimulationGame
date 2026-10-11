"""Classifier Protocol + GestureClassifier (load model, predict()).

Owner: Siri (Classifier).
Provides: Classifier, GestureClassifier — consumed by Evan (worker.py, main.py).
Specification: docs/interfaces.md §4.5.

STUB (Lead, Iteration 1): the Protocol is complete; GestureClassifier signatures
are frozen and Siri replaces every `raise NotImplementedError`.
"""

from pathlib import Path
from typing import Protocol

import numpy as np

from src.config import CONFIDENCE_THRESHOLD, MODEL_PATH


class Classifier(Protocol):
    """Anything the live loop can use: real SVM or keyboard mock (LI.8)."""

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Classify one preprocessed feature vector.

        Args:
            features: preprocess() output, shape (63,).

        Returns:
            (label, confidence) with label in LABELS and confidence in [0, 1].
        """
        ...


class GestureClassifier:
    """The trained SVM, loaded once. Called only from the GestureWorker thread."""

    def __init__(self, model_path: Path = MODEL_PATH,
                 threshold: float = CONFIDENCE_THRESHOLD) -> None:
        """Load the model bundle.

        Args:
            model_path: The joblib bundle written by train.py.
            threshold: Confidence below this returns "none". Use 0.0 for evaluation.

        Raises:
            FileNotFoundError: If the bundle is missing (suggests running src.model.train).
            ValueError: If bundle["preprocess_version"] != config.PREPROCESS_VERSION.
        """
        raise NotImplementedError

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Classify one feature vector (< 2 ms on a laptop CPU).

        Args:
            features: preprocess() output, float32/float64, shape (63,).

        Returns:
            (label, confidence): argmax of predict_proba and its probability;
            ("none", confidence) if confidence < threshold.

        Raises:
            ValueError: If the shape is not (63,) or any value is non-finite.
        """
        raise NotImplementedError

    def predict_proba(self, features: np.ndarray) -> dict[str, float]:
        """[added] Return every class probability, for evaluation/debug.

        Args:
            features: preprocess() output, shape (63,).

        Returns:
            label → probability.
        """
        raise NotImplementedError
