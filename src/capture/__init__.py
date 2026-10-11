"""Re-exports Camera, Frame, HandTracker, HandLandmarks, preprocess.

Owner: Triet (Hand tracking).
Specification: docs/interfaces.md §4.2.

Imports are lazy so `import src.capture` does not load OpenCV/MediaPipe until a name is used.
"""

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "Camera": "src.capture.camera",
    "Frame": "src.capture.camera",
    "HandTracker": "src.capture.landmarks",
    "HandLandmarks": "src.capture.landmarks",
    "preprocess": "src.capture.preprocess",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import a re-exported name from its submodule on first use."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.capture' has no attribute {name!r}")
    return getattr(import_module(_EXPORTS[name]), name)
