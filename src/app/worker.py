"""GestureWorker thread: frame → landmarks → predict → smoothing → events.

Owner: Evan (Live integration).
Provides: GestureObservation, GestureEvent, GestureWorker — consumed by main.py.
Specification: docs/interfaces.md §4.6; thread rules in docs/DESIGN.md §2.1.

Implementation: Evan live integration.
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


import math
import queue
import threading
import time
from collections import deque

from src.capture.preprocess import preprocess
from src.config import LABELS, NO_HAND


class GestureWorker:
    """Own tracking on a daemon thread; publish snapshots and discrete events.

    The camera is owned/closed by main.py. The tracker is created, used and
    closed here, on the same thread. Game methods are never called here.
    """

    def __init__(self, camera, tracker_factory, classifier, smoother=None, debouncer=None):
        self.camera, self.tracker_factory, self.classifier = camera, tracker_factory, classifier
        self.smoother = smoother if smoother is not None else GestureSmoother()
        self.debouncer = debouncer if debouncer is not None else Debouncer()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self._error = None
        self._events = queue.Queue()
        self._times = deque()

    def start(self) -> None:
        if self._thread is not None:
            if self._thread.is_alive():
                return
            raise RuntimeError("Create a new GestureWorker after it has stopped")
        self._thread = threading.Thread(target=self._run, name="GestureWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError("Gesture worker did not stop; close the camera and retry stop()")

    def latest(self):
        with self._lock:
            return self._latest

    def poll_events(self) -> list[GestureEvent]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    @property
    def fps(self) -> float:
        with self._lock:
            now = time.perf_counter()
            while self._times and self._times[0] < now - 1:
                self._times.popleft()
            if len(self._times) < 2:
                return 0.0
            span = self._times[-1] - self._times[0]
            return (len(self._times) - 1) / span if span > 0 else 0.0

    @property
    def error(self):
        with self._lock:
            return self._error

    def _set_error(self, exc):
        with self._lock:
            if self._error is None:
                self._error = exc

    def _run(self):
        tracker = None
        previous_raw = None
        onset = None
        streak_onsets = {}
        last_index = -1
        try:
            tracker = self.tracker_factory()
            while not self._stop.is_set():
                frame = self.camera.read()
                if self._stop.is_set():
                    break
                if frame.index <= last_index:
                    raise ValueError("Camera must deliver increasing frame indices")
                last_index = frame.index
                hand = tracker.detect(frame.image)
                t_landmarks = time.perf_counter()
                if hand is None:
                    token, confidence = NO_HAND, 0.0
                    t_predict = t_landmarks
                else:
                    token, confidence = self.classifier.predict(preprocess(hand))
                    t_predict = time.perf_counter()
                    if token not in LABELS or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                        raise ValueError("Classifier returned an invalid label or confidence")
                # Measure onset from the uninterrupted raw streak, not stability time.
                if token != previous_raw:
                    onset, previous_raw = frame.t_capture, token
                    streak_onsets[token] = onset
                stable = self.smoother.update(token, frame.t_capture)
                event = self.debouncer.update(stable, frame.t_capture)
                observation = GestureObservation(token, confidence, stable, hand is not None,
                                                 frame.index, frame.t_capture, t_landmarks, t_predict)
                with self._lock:
                    self._latest = observation
                    self._times.append(time.perf_counter())
                    while self._times and self._times[0] < self._times[-1] - 1:
                        self._times.popleft()
                if event is not None:
                    self._events.put(GestureEvent(event, streak_onsets[event], frame.t_capture,
                                                  t_landmarks, t_predict))
        except Exception as exc:
            if not self._stop.is_set():
                self._set_error(exc)
        finally:
            if tracker is not None:
                try:
                    tracker.close()
                except Exception as exc:
                    self._set_error(exc)
