"""Adapt Siri's saved SVM pipeline to Evan's live prediction contract.

Siri's GestureClassifier.predict() returns a label only. Live integration needs
(label, confidence), plus rejection of uncertain predictions. This adapter uses
the fitted pipeline's predict_proba() without modifying training/evaluation.
"""
from pathlib import Path
import math
import numpy as np

from src.config import CONFIDENCE_THRESHOLD, LABELS, LABEL_NONE, MODEL_PATH, NUM_FEATURES
from src.model.predict import GestureClassifier


class LiveGestureClassifier:
    """Load once and provide validated, thresholded predictions to the worker."""

    def __init__(self, model_path: Path = MODEL_PATH,
                 threshold: float = CONFIDENCE_THRESHOLD):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be finite and in [0, 1]")
        self.threshold = threshold
        self._classifier = GestureClassifier(model_path=model_path)
        self._model = self._classifier.model
        if getattr(self._model, "n_features_in_", None) != NUM_FEATURES:
            raise ValueError(f"Model must accept {NUM_FEATURES} preprocessed features")
        self._labels = tuple(str(label) for label in self._model.classes_)
        if len(self._labels) != len(LABELS) or set(self._labels) != set(LABELS):
            raise ValueError("Model classes do not match the project's gesture labels")
        if not callable(getattr(self._model, "predict_proba", None)):
            raise ValueError("Model must provide class probabilities")
        # The current file stores a bare Pipeline, without preprocessing-version
        # metadata. Shape/classes can be checked, but version compatibility cannot.

    def predict_proba(self, features: np.ndarray) -> dict[str, float]:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (NUM_FEATURES,) or not np.isfinite(values).all():
            raise ValueError(f"Expected {NUM_FEATURES} finite features of shape ({NUM_FEATURES},)")
        probabilities = np.asarray(self._model.predict_proba(values.reshape(1, -1))[0])
        if (probabilities.shape != (len(self._labels),) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.isclose(probabilities.sum(), 1)):
            raise ValueError("Model returned invalid probabilities")
        return dict(zip(self._labels, map(float, probabilities)))

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Use the highest class probability; uncertain frames become 'none'."""
        probabilities = self.predict_proba(features)
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]
        return (LABEL_NONE if confidence < self.threshold else label), confidence
