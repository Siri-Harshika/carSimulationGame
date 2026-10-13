"""GestureSmoother (time-window majority vote) + Debouncer.

Owner: Evan (Live integration).
Provides: GestureSmoother, Debouncer — consumed by worker.py.
Specification: docs/interfaces.md §4.6; algorithms in docs/game-logic.md §6.2 and §6.3.

Pure logic: no I/O, no threads.

Implementation: Evan live integration.
"""

from src.config import (
    ACTION_COOLDOWN_S,
    SMOOTHING_MIN_FRACTION,
    SMOOTHING_MIN_FRAMES,
    SMOOTHING_WINDOW_S,
)


import math
from collections import Counter, deque

from src.config import LABELS, LABEL_NONE, LABEL_FIST, LABEL_V_SIGN, NO_HAND


def _validate_token(token: str) -> None:
    if token not in (*LABELS, NO_HAND):
        raise ValueError(f"Unknown gesture token: {token!r}")


class GestureSmoother:
    """Time-window majority vote with hysteresis; capture times drive the window."""

    def __init__(self, window_s=SMOOTHING_WINDOW_S,
                 min_fraction=SMOOTHING_MIN_FRACTION,
                 min_frames=SMOOTHING_MIN_FRAMES):
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("window_s must be finite and positive")
        if not math.isfinite(min_fraction) or not 0.5 < min_fraction <= 1:
            raise ValueError("min_fraction must be in (0.5, 1]")
        if not isinstance(min_frames, int) or isinstance(min_frames, bool) or min_frames < 1:
            raise ValueError("min_frames must be a positive integer")
        self.window_s, self.min_fraction, self.min_frames = window_s, min_fraction, min_frames
        self.reset()

    def update(self, token: str, t: float) -> str:
        _validate_token(token)
        if not math.isfinite(t) or (self._last_time is not None and t < self._last_time):
            raise ValueError("Capture time must be finite and cannot go backwards")
        self._last_time = t
        self._history.append((t, token))
        # Preserve at least three samples at low FPS, even if older than the window.
        while len(self._history) > self.min_frames and self._history[0][0] < t - self.window_s:
            self._history.popleft()
        if len(self._history) >= self.min_frames:
            winner, count = Counter(token for _, token in self._history).most_common(1)[0]
            if count / len(self._history) >= self.min_fraction:
                self.stable = winner
        return self.stable

    def reset(self) -> None:
        self._history = deque()
        self._last_time = None
        self.stable = NO_HAND


class Debouncer:
    """Emit token changes only; never queue steering edges rejected by cooldown."""

    def __init__(self, cooldown_s=ACTION_COOLDOWN_S):
        if not math.isfinite(cooldown_s) or cooldown_s < 0:
            raise ValueError("cooldown_s must be finite and nonnegative")
        self.cooldown_s = cooldown_s
        self.reset()

    def update(self, stable: str, now: float) -> str | None:
        _validate_token(stable)
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError("Capture time must be finite and cannot go backwards")
        self._last_time = now
        if stable == self._last_stable:
            return None
        self._last_stable = stable
        if stable == LABEL_NONE:
            return None
        if stable in (LABEL_FIST, LABEL_V_SIGN):
            if now - self._last_emit < self.cooldown_s:
                return None
            self._last_emit = now
        return stable

    def reset(self) -> None:
        self._last_stable = NO_HAND
        self._last_emit = -math.inf
        self._last_time = None
