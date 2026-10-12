"""KeyboardState, MockClassifier, MockCamera, MockHandTracker.

Owner: Evan (Live integration), written with the Lead in Iteration 1.
Provides: drop-in replacements for Camera, HandTracker and GestureClassifier
(LI.8), so the live loop runs with no webcam and no trained model.
Specification: docs/interfaces.md §4.6.

Keys: 1 → fist, 2 → v_sign, 3 → open_palm, otherwise none; holding H = no hand.
key_state callables return a pygame.key.get_pressed()-style sequence indexed
by pygame key codes. Tests inject a plain function instead of pygame.
"""

from collections.abc import Callable, Sequence
import threading
import time

import numpy as np
import pygame

from src.capture.camera import Frame
from src.capture.landmarks import HandLandmarks
from src.config import (
    CAMERA_FPS,
    CAMERA_READ_TIMEOUT_S,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LABEL_FIST,
    LABEL_NONE,
    LABEL_OPEN_PALM,
    LABEL_V_SIGN,
    NUM_FEATURES,
    NUM_LANDMARKS,
)

# Checked in order, so holding 1 and 2 together gives fist.
_KEY_LABELS: tuple[tuple[int, str], ...] = (
    (pygame.K_1, LABEL_FIST),
    (pygame.K_2, LABEL_V_SIGN),
    (pygame.K_3, LABEL_OPEN_PALM),
)


def _is_pressed(keys: Sequence[bool], key: int) -> bool:
    """Return keys[key], treating a too-short sequence as 'not pressed'."""
    return key < len(keys) and bool(keys[key])


class KeyboardState:
    """Thread-safe snapshot of pressed keys.

    main.py writes it every frame (pygame may only be touched on the main
    thread); the mocks read it from the worker thread.
    """

    def __init__(self) -> None:
        """Start with no keys pressed."""
        self._lock = threading.Lock()
        self._pressed: tuple[bool, ...] = ()

    def update(self, pressed: Sequence[bool]) -> None:
        """Store a copy of pygame.key.get_pressed(). Main thread only.

        Args:
            pressed: Sequence indexed by pygame key code.
        """
        # Pygame's ScancodeWrapper indexes by key code but iterates by SDL
        # scancode. Converting it directly to a tuple loses that translation.
        # Read the mock keys explicitly while still on the Pygame main thread.
        mock_keys = (*[key for key, _ in _KEY_LABELS], pygame.K_h)
        snapshot_values = [False] * (max(mock_keys) + 1)
        for key in mock_keys:
            snapshot_values[key] = _is_pressed(pressed, key)
        snapshot = tuple(snapshot_values)
        with self._lock:
            self._pressed = snapshot

    def get(self) -> Sequence[bool]:
        """Return the last snapshot written. Any thread."""
        with self._lock:
            return self._pressed


class MockClassifier:
    """Keyboard-driven Classifier: 1/2/3 → fist/v_sign/open_palm, else none."""

    def __init__(self, key_state: Callable[[], Sequence[bool]]) -> None:
        """Create the mock.

        Args:
            key_state: Returns the current pressed-keys sequence (e.g. KeyboardState.get).
        """
        self._key_state = key_state

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Return the label of the held key with confidence 1.0.

        Args:
            features: preprocess() output; values are ignored but the shape is
                checked exactly like GestureClassifier, so bugs upstream still show.

        Returns:
            (label, 1.0).

        Raises:
            ValueError: If features.shape != (63,).
        """
        if np.shape(features) != (NUM_FEATURES,):
            raise ValueError(f"expected features of shape ({NUM_FEATURES},), "
                             f"got {np.shape(features)}")
        keys = self._key_state()
        for key, label in _KEY_LABELS:
            if _is_pressed(keys, key):
                return label, 1.0
        return LABEL_NONE, 1.0


class MockCamera:
    """Same public API as Camera (read, latest, fps, close); black 640×480 frames at CAMERA_FPS."""

    def __init__(self, fps: float = CAMERA_FPS) -> None:
        """Create the mock camera.

        Args:
            fps: Frames per second that read() is paced to.
        """
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self._period = 1.0 / fps
        self._lock = threading.Lock()
        self._image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self._latest: Frame | None = None
        self._next_index = 0
        self._next_due = time.perf_counter()
        self._times: list[float] = []
        self._closed = False

    def read(self, timeout: float = CAMERA_READ_TIMEOUT_S) -> Frame:
        """Wait for the next frame slot and return a new black Frame.

        Args:
            timeout: Unused except for validation; a mock frame is never late.

        Returns:
            A Frame with a strictly increasing index and t_capture.

        Raises:
            ValueError: If timeout is not positive.
            RuntimeError: If the camera is closed.
        """
        if timeout <= 0:
            raise ValueError("timeout must be a positive number.")
        with self._lock:
            if self._closed:
                raise RuntimeError("MockCamera is closed.")
            wait = self._next_due - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        with self._lock:
            now = time.perf_counter()
            frame = Frame(image=self._image, t_capture=now, index=self._next_index)
            self._next_index += 1
            # Pace from the due time, not from now, so the average rate stays CAMERA_FPS.
            self._next_due = max(self._next_due + self._period, now)
            self._times = [t for t in self._times if now - t <= 1.0] + [now]
            self._latest = frame
            return frame

    def latest(self) -> Frame | None:
        """Return the newest frame without waiting (None before the first read)."""
        with self._lock:
            return self._latest

    @property
    def fps(self) -> float:
        """Frames delivered per second over the last second."""
        with self._lock:
            if len(self._times) < 2:
                return 0.0
            elapsed = self._times[-1] - self._times[0]
            return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0

    def close(self) -> None:
        """Mark closed; idempotent."""
        with self._lock:
            self._closed = True

    def __enter__(self) -> "MockCamera":
        """Support `with MockCamera() as cam:` like Camera."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close on leaving the with block."""
        self.close()


def _fixed_hand_points() -> np.ndarray:
    """A valid, non-degenerate right hand: wrist low, fingers fanned upwards."""
    points = np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)
    points[0] = (0.5, 0.8, 0.0)                      # wrist
    for i in range(1, NUM_LANDMARKS):
        finger, joint = divmod(i - 1, 4)             # 5 fingers × 4 joints
        points[i] = (0.35 + 0.075 * finger, 0.65 - 0.06 * joint, -0.01 * joint)
    return points


class MockHandTracker:
    """Same public API as HandTracker; a fixed valid hand unless H is held."""

    def __init__(self, key_state: Callable[[], Sequence[bool]]) -> None:
        """Create the mock.

        Args:
            key_state: Returns the current pressed-keys sequence.
        """
        self._key_state = key_state
        self._points = _fixed_hand_points()

    def detect(self, image_bgr: np.ndarray) -> HandLandmarks | None:
        """Return the fixed hand, or None while H is held (simulates "no hand").

        Args:
            image_bgr: Ignored apart from validation.

        Raises:
            ValueError: If image_bgr is not a (h, w, 3) uint8 array.
        """
        if (not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
                or image_bgr.ndim != 3 or image_bgr.shape[2] != 3):
            raise ValueError("Expected a BGR uint8 image of shape (height, width, 3).")
        if _is_pressed(self._key_state(), pygame.K_h):
            return None
        # Copy so a caller mutating the points cannot change later detections.
        return HandLandmarks(points=self._points.copy(), handedness="Right", score=1.0)

    def draw(self, image_bgr: np.ndarray, hand: HandLandmarks) -> None:
        """No-op: the mock has nothing worth drawing."""

    def close(self) -> None:
        """No-op; present for API parity."""
