"""LatencyLogger: timestamps → results/latency_*.csv + summary.

Owner: Evan (Live integration).
Provides: ActionTiming, LatencyLogger — consumed by main.py (--log-latency).
Specification: docs/interfaces.md §4.6; metrics in docs/data-eval.md §8.

time.perf_counter() is one process-wide clock, so timestamps from the camera,
worker and main threads can be compared directly.

Implementation: Evan live integration.
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


import csv
import math
import numpy as np


class LatencyLogger:
    """Store changed actions only; flush each CSV row so crashes lose less data."""

    _METRICS = ("response_ms", "processing_ms", "landmarks_ms", "predict_ms", "dispatch_ms")

    def __init__(self, out_path: Path | None):
        self._rows = []
        self._file = None
        self._closed = False
        self._writer = None
        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = out_path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=[
                "action", "token", "t_onset", "t_trigger", "t_landmarks", "t_predict", "t_action",
                *self._METRICS])
            self._writer.writeheader()
            self._file.flush()

    def log_action(self, timing: ActionTiming) -> None:
        if self._closed:
            raise RuntimeError("Latency logger is closed")
        stamps = [timing.t_onset, timing.t_trigger, timing.t_landmarks, timing.t_predict, timing.t_action]
        if not all(math.isfinite(t) for t in stamps) or stamps != sorted(stamps):
            raise ValueError("Action timestamps must be finite and chronological")
        row = dict(action=timing.action.value, token=timing.token,
                   t_onset=timing.t_onset, t_trigger=timing.t_trigger,
                   t_landmarks=timing.t_landmarks, t_predict=timing.t_predict, t_action=timing.t_action,
                   response_ms=(timing.t_action-timing.t_onset)*1000,
                   processing_ms=(timing.t_action-timing.t_trigger)*1000,
                   landmarks_ms=(timing.t_landmarks-timing.t_trigger)*1000,
                   predict_ms=(timing.t_predict-timing.t_landmarks)*1000,
                   dispatch_ms=(timing.t_action-timing.t_predict)*1000)
        self._rows.append(row)
        if self._writer is not None:
            self._writer.writerow(row)
            self._file.flush()

    def summary(self) -> dict[str, float]:
        result = {"count": len(self._rows)}
        if self._rows:
            for metric in self._METRICS:
                values = [row[metric] for row in self._rows]
                for stat, value in (("median", np.median(values)), ("p95", np.percentile(values, 95)),
                                    ("mean", np.mean(values)), ("max", np.max(values))):
                    result[f"{metric}_{stat}"] = float(value)
        return result

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._file is not None:
                self._file.close()
