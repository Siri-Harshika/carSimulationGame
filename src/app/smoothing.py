"""GestureSmoother (time-window majority vote) + Debouncer.

Owner: Evan (Live integration).
Provides: GestureSmoother, Debouncer — consumed by worker.py.
Specification: docs/interfaces.md §4.6; algorithms in docs/game-logic.md §6.2 and §6.3.

Pure logic: no I/O, no threads.

STUB (Lead, Iteration 1): signatures are frozen; Evan replaces every
`raise NotImplementedError` with the real implementation.
"""

from src.config import (
    ACTION_COOLDOWN_S,
    SMOOTHING_MIN_FRACTION,
    SMOOTHING_MIN_FRAMES,
    SMOOTHING_WINDOW_S,
)


class GestureSmoother:
    """Majority vote over a window of capture time, with hysteresis.

    Attributes:
        stable: Current stable token; initially NO_HAND.
    """

    stable: str

    def __init__(self, window_s: float = SMOOTHING_WINDOW_S,
                 min_fraction: float = SMOOTHING_MIN_FRACTION,
                 min_frames: int = SMOOTHING_MIN_FRAMES) -> None:
        """Create an empty smoother.

        Args:
            window_s: s of capture time in the window (> 0).
            min_fraction: Share a token needs to become stable (0.5 < x ≤ 1).
            min_frames: The window never shrinks below this many frames (≥ 1).

        Raises:
            ValueError: If any argument is out of range.
        """
        raise NotImplementedError

    def update(self, token: str, t: float) -> str:
        """Add one raw token and return the current stable token.

        Args:
            token: Raw token (LABELS value or NO_HAND).
            t: Frame.t_capture in s.

        Returns:
            The stable token.

        Raises:
            ValueError: If t goes backwards.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Forget all history; stable becomes NO_HAND."""
        raise NotImplementedError


class Debouncer:
    """Turns a stable-token stream into discrete events, rate-limiting lane changes."""

    def __init__(self, cooldown_s: float = ACTION_COOLDOWN_S) -> None:
        """Create a debouncer.

        Args:
            cooldown_s: s between repeated lane-change events (≥ 0).

        Raises:
            ValueError: If cooldown_s < 0.
        """
        raise NotImplementedError

    def update(self, stable: str, now: float) -> str | None:
        """Feed one stable token.

        Args:
            stable: GestureSmoother output.
            now: Capture time in s.

        Returns:
            An event token (never "none"), or None.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Forget the last event."""
        raise NotImplementedError
