"""GestureWorker thread: frame → landmarks → predict → smoothing → events.

Owner: Evan (Live integration).
Provides: GestureObservation, GestureEvent, GestureWorker — consumed by main.py.
Specification: docs/interfaces.md §4.6; thread rules in docs/DESIGN.md §2.1.

STUB (Lead, Iteration 1): the two dataclasses are complete (plain data);
GestureWorker signatures are frozen and Evan replaces every `raise NotImplementedError`.
"""

from collections.abc import Callable
from dataclasses import dataclass

from src.app.mock_classifier import MockCamera, MockHandTracker
from src.app.smoothing import Debouncer, GestureSmoother
from src.capture.camera import Camera
from src.capture.landmarks import HandTracker
from src.model.predict import Classifier


@dataclass(frozen=True)
class GestureObservation:
    """What the worker saw on its latest frame (for the HUD)."""

    token: str             # raw: classifier label (after threshold) or NO_HAND
    confidence: float      # [0, 1]; 0.0 for NO_HAND
    stable: str            # GestureSmoother output
    hand_present: bool
    frame_index: int
    t_capture: float       # Frame.t_capture (s)
    t_landmarks: float     # perf_counter() after HandTracker.detect (s)
    t_predict: float       # perf_counter() after Classifier.predict (= t_landmarks if no hand) (s)


@dataclass(frozen=True)
class GestureEvent:
    """One debounced gesture event for the main thread."""

    token: str             # ∈ LABELS ∪ {NO_HAND}, never "none"
    t_onset: float         # t_capture of the FIRST frame of the raw-token streak (s)
    t_trigger: float       # t_capture of the frame whose processing emitted it (s)
    t_landmarks: float
    t_predict: float


class GestureWorker:
    """Background thread: camera → HandTracker → preprocess → Classifier → smoother → debouncer.

    Shared between the worker thread and the main thread; internally locked.
    """

    def __init__(self, camera: Camera | MockCamera,
                 tracker_factory: Callable[[], HandTracker | MockHandTracker],
                 classifier: Classifier,
                 smoother: GestureSmoother | None = None,
                 debouncer: Debouncer | None = None) -> None:
        """Create a stopped worker.

        Args:
            camera: Frame source.
            tracker_factory: Called INSIDE the worker thread, so MediaPipe lives on one thread.
            classifier: Real GestureClassifier or MockClassifier.
            smoother: Defaults to GestureSmoother().
            debouncer: Defaults to Debouncer().
        """
        raise NotImplementedError

    def start(self) -> None:
        """Start the daemon thread."""
        raise NotImplementedError

    def stop(self, timeout: float = 1.0) -> None:
        """Signal the thread and join it; idempotent.

        Args:
            timeout: s to wait for the join.
        """
        raise NotImplementedError

    def latest(self) -> GestureObservation | None:
        """Return the newest observation without blocking (None before the first frame)."""
        raise NotImplementedError

    def poll_events(self) -> list[GestureEvent]:
        """Drain the event queue. Main thread."""
        raise NotImplementedError

    @property
    def fps(self) -> float:
        """Processed frames per second, rolling 1 s."""
        raise NotImplementedError

    @property
    def error(self) -> BaseException | None:
        """The exception that killed the thread, or None."""
        raise NotImplementedError
