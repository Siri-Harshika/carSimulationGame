"""Mock keyboard snapshots must preserve Pygame key-code indexing."""
import numpy as np
import pygame
import pytest
from src.app.mock_classifier import KeyboardState, MockClassifier, MockHandTracker


class ScancodeLikeKeys:
    """Iteration exposes scancodes, while indexed lookup translates key codes."""
    def __init__(self, held):
        self.held = held
    def __len__(self):
        return 512
    def __iter__(self):
        # Simulate the SDL position for the 1 key, which differs from K_1.
        return iter([i == 30 for i in range(512)])
    def __getitem__(self, key):
        return key in self.held


@pytest.mark.parametrize("key, label", [(pygame.K_1, "fist"),
    (pygame.K_2, "v_sign"), (pygame.K_3, "open_palm")])
def test_snapshot_uses_key_codes_not_iteration(key, label):
    keys = KeyboardState()
    keys.update(ScancodeLikeKeys({key}))
    assert MockClassifier(keys.get).predict(np.zeros(63)) == (label, 1.0)
    keys.update(ScancodeLikeKeys(set()))
    assert MockClassifier(keys.get).predict(np.zeros(63)) == ("none", 1.0)


def test_missing_hand_key_survives_snapshot():
    keys = KeyboardState()
    tracker = MockHandTracker(keys.get)
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    keys.update(ScancodeLikeKeys({pygame.K_h}))
    assert tracker.detect(image) is None
    keys.update(ScancodeLikeKeys(set()))
    assert tracker.detect(image) is not None
