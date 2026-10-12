"""Integration compatibility checks using the committed SVM, without retraining."""
import numpy as np
import pytest
from src.app.model_adapter import LiveGestureClassifier
from src.config import LABELS, MODEL_PATH
from src.data.split import load_split


@pytest.fixture(scope="module")
def classifier():
    return LiveGestureClassifier()


@pytest.fixture(scope="module")
def train_example():
    # Use training data for this wiring test; do not retune on reserved test users.
    data = load_split("train")
    return data.X[0]


def test_committed_model_loads_and_returns_probabilities(classifier, train_example):
    assert MODEL_PATH.name == "classifier.pkl"
    probabilities = classifier.predict_proba(train_example)
    assert set(probabilities) == set(LABELS)
    assert sum(probabilities.values()) == pytest.approx(1)
    label, confidence = classifier.predict(train_example)
    assert label in LABELS
    assert 0 <= confidence <= 1
    assert confidence == pytest.approx(max(probabilities.values()))


def test_threshold_rejects_uncertain_predictions(classifier, monkeypatch):
    monkeypatch.setattr(classifier, "predict_proba", lambda features: {
        "fist": .4, "v_sign": .3, "open_palm": .2, "none": .1})
    assert classifier.predict(np.zeros(63)) == ("none", .4)


@pytest.mark.parametrize("features", [np.zeros(62), np.zeros((1, 63)),
                                      np.full(63, np.nan), np.full(63, np.inf)])
def test_invalid_features(classifier, features):
    with pytest.raises(ValueError):
        classifier.predict(features)


@pytest.mark.parametrize("threshold", [-.1, 1.1, float("nan")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        LiveGestureClassifier(threshold=threshold)


def test_probability_argmax_and_boundary(monkeypatch):
    import src.app.model_adapter as module
    class Pipeline:
        n_features_in_ = 63
        classes_ = np.array(["fist", "none", "open_palm", "v_sign"])
        def predict_proba(self, features):
            return np.array([[.7, .1, .1, .1]])
    class Loaded:
        def __init__(self, **kwargs):
            self.model = Pipeline()
    monkeypatch.setattr(module, "GestureClassifier", Loaded)
    assert LiveGestureClassifier().predict(np.zeros(63)) == ("fist", .7)


def test_real_classifier_application_loop(capsys):
    from src.app.main import main
    main(["--camera", "mock", "--duration", ".2", "--no-preview"])
    assert "classifier=svm" in capsys.readouterr().out
