"""LatencyLogger: timestamps → results/latency_*.csv + summary.

Owner: Evan (Live integration).
Provides: ActionTiming, LatencyLogger — consumed by main.py (--log-latency).
Specification: docs/interfaces.md §4.6; metrics in docs/data-eval.md §8.

time.perf_counter() is one process-wide clock, so timestamps from the camera,
worker and main threads can be compared directly.

STUB (Lead, Iteration 1): ActionTiming is complete (plain data); LatencyLogger
signatures are frozen and Evan replaces every `raise NotImplementedError`.
"""

from dataclasses import dataclass
from pathlib import Path

from src.app.actions import Action


@dataclass
class ActionTiming:
    """Timestamps (s, perf_counter) of one action that changed the game."""

    action: Action
    token: str
    t_onset: float        # from GestureEvent
    t_trigger: float      # from GestureEvent
    t_landmarks: float    # from GestureEvent
    t_predict: float      # from GestureEvent
    t_action: float       # right after the GameController call returned, main thread


class LatencyLogger:
    """Collects ActionTimings and optionally writes them to CSV. Main thread only."""

    def __init__(self, out_path: Path | None) -> None:
        """Create a logger.

        Args:
            out_path: CSV to write; None keeps the timings in memory only.
        """
        raise NotImplementedError

    def log_action(self, timing: ActionTiming) -> None:
        """Record one action.

        Args:
            timing: The action's timestamps.
        """
        raise NotImplementedError

    def summary(self) -> dict[str, float]:
        """Return median/p95/mean/max of each latency metric, in ms."""
        raise NotImplementedError

    def close(self) -> None:
        """Flush and close the CSV (if any)."""
        raise NotImplementedError
