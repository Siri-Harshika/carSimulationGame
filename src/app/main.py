"""Entry point: Pygame loop (60 FPS), CLI flags, keyboard fallback, preview.

Owner: Evan (Live integration).
Specification: docs/interfaces.md §4.6.

Usage (from the repo root):
    python -m src.app.main                                    # real webcam + SVM
    python -m src.app.main --classifier mock --camera mock    # no webcam at all
    python -m src.app.main --log-latency                      # also write results/latency_*.csv

STUB (Lead, Iteration 1): signatures are frozen; Evan replaces every
`raise NotImplementedError` with the real implementation.
"""

import argparse
from collections.abc import Callable

from src.app.mock_classifier import KeyboardState, MockCamera, MockHandTracker
from src.capture.camera import Camera
from src.capture.landmarks import HandTracker
from src.model.predict import Classifier


def build_components(args: argparse.Namespace, keys: KeyboardState) -> tuple[
        Camera | MockCamera, Callable[[], HandTracker | MockHandTracker], Classifier]:
    """Create the camera, tracker factory and classifier selected by the CLI flags.

    Args:
        args: Parsed CLI arguments.
        keys: Keyboard snapshot the mocks read.

    Returns:
        (camera, tracker_factory, classifier).
    """
    raise NotImplementedError


def run(args: argparse.Namespace) -> None:
    """Run the game loop until the window is closed or the worker fails.

    Args:
        args: Parsed CLI arguments.
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    """CLI: --classifier {svm,mock} --camera {webcam,mock} --log-latency --no-preview.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
