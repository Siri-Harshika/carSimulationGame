"""Shared pytest setup: headless Pygame and a fake HandLandmarks fixture.

Owner: Lead.
Specification: docs/workflow.md §9.

The SDL variables must be set BEFORE anything imports pygame, which is why
they live at module level here: pytest imports conftest.py before any test module.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # no window: tests run headless / in CI
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")   # no sound device needed
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402  (must come after the SDL settings)
import pytest  # noqa: E402

from src.capture.landmarks import HandLandmarks  # noqa: E402
from src.config import MIDDLE_MCP_IDX, NUM_COORDS, NUM_LANDMARKS, SEED, WRIST_IDX  # noqa: E402


def make_hand(handedness: str = "Right", seed: int = SEED) -> HandLandmarks:
    """Build a repeatable, non-degenerate raw hand (no webcam needed).

    Args:
        handedness: "Left" or "Right".
        seed: RNG seed for the non-anchor landmarks.

    Returns:
        HandLandmarks with points of shape (21, 3) inside the frame.
    """
    points = np.random.default_rng(seed).uniform(0.2, 0.8, size=(NUM_LANDMARKS, NUM_COORDS))
    # Pin the wrist and middle MCP apart so the palm size is never ~0.
    points[WRIST_IDX] = (0.4, 0.7, 0.0)
    points[MIDDLE_MCP_IDX] = (0.5, 0.4, -0.03)
    return HandLandmarks(points=points, handedness=handedness, score=0.95)


@pytest.fixture
def fake_hand() -> HandLandmarks:
    """A fresh right hand for each test (tests may mutate it)."""
    return make_hand()


@pytest.fixture
def fake_left_hand() -> HandLandmarks:
    """A fresh left hand for each test."""
    return make_hand("Left")
