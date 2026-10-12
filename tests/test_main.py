"""Evan: application smoke test, mock selection, and error reporting."""
import argparse
import pygame
import pytest
from src.app.main import main, build_components
from src.app.mock_classifier import KeyboardState, MockCamera, MockClassifier, MockHandTracker


def test_component_selection():
    args = argparse.Namespace(classifier="mock", camera="mock")
    camera, factory, classifier = build_components(args, KeyboardState())
    assert isinstance(camera, MockCamera)
    assert isinstance(factory(), MockHandTracker)
    assert isinstance(classifier, MockClassifier)
    camera.close()


def test_mock_app_applies_held_prediction_once(monkeypatch, capsys):
    # Feed a held key through the entire application, not directly to the controller.
    pressed = [False]*512
    pressed[pygame.K_1] = True
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: pressed)
    main(["--classifier", "mock", "--camera", "mock", "--duration", ".3"])
    output = capsys.readouterr().out
    assert output.count("fist -> resume_and_left") == 1
    assert "'count': 1" in output
    assert not pygame.get_init()


def test_missing_model_fails_without_camera(monkeypatch, tmp_path, capsys):
    import src.app.main as app
    monkeypatch.setattr(app, "MODEL_PATH", tmp_path / "missing.joblib")
    def forbidden():
        raise AssertionError("Camera must not be opened if model is missing")
    monkeypatch.setattr(app, "Camera", forbidden)
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 1
    assert "--classifier mock" in capsys.readouterr().err
