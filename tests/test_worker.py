"""Evan: scripted predictions test the actual thread and event pipeline."""
import threading
import time
import numpy as np
import pytest
from src.app.worker import GestureWorker
from src.app.smoothing import GestureSmoother, Debouncer
from src.capture.camera import Frame
from src.app.actions import apply_action, gesture_to_action
from src.game.controller import GameController
from src.game.states import GameState


class ScriptCamera:
    def __init__(self, count):
        self.count, self.index = count, 0
        self.done = threading.Event()
        self.closed = False
    def read(self):
        if self.index >= self.count:
            self.done.set()
            while not self.closed:
                time.sleep(.001)
            raise RuntimeError("closed")
        time.sleep(.002)
        frame = Frame(np.zeros((12, 12, 3), dtype=np.uint8), time.perf_counter(), self.index)
        self.index += 1
        return frame
    def close(self):
        self.closed = True


class Tracker:
    def __init__(self, hand, missing=()):
        self.hand, self.missing, self.index = hand, missing, 0
        self.created_thread = threading.get_ident()
        self.closed_thread = None
    def detect(self, image):
        assert threading.get_ident() == self.created_thread
        result = None if self.index in self.missing else self.hand
        self.index += 1
        return result
    def close(self):
        self.closed_thread = threading.get_ident()


class Classifier:
    def __init__(self, labels):
        self.labels = iter(labels)
        self.calls = 0
    def predict(self, features):
        assert features.shape == (63,)
        self.calls += 1
        return next(self.labels), 1.0


def test_pipeline_held_gesture_missing_hand_and_resume(fake_hand):
    camera = ScriptCamera(15)
    trackers = []
    def factory():
        tracker = Tracker(fake_hand, missing=range(6, 9))
        trackers.append(tracker)
        return tracker
    classifier = Classifier(["fist"]*6 + ["v_sign"]*6)
    worker = GestureWorker(camera, factory, classifier,
                           GestureSmoother(min_frames=3, window_s=.0001), Debouncer(0))
    worker.start()
    assert camera.done.wait(2)
    assert worker.error is None
    events = worker.poll_events()
    assert [e.token for e in events] == ["fist", "no_hand", "v_sign"]
    assert classifier.calls == 12  # missing hands bypass classification
    assert all(e.t_onset <= e.t_trigger <= e.t_landmarks <= e.t_predict for e in events)
    assert events[0].t_onset < events[0].t_trigger
    c = GameController()
    for event in events:
        apply_action(c, gesture_to_action(event.token, c.state))
    assert c.state == GameState.COUNTDOWN
    assert c.lane == 1
    assert c.calls == ["resume", "move_left", "pause", "resume", "move_right"]
    assert worker.latest().stable == "v_sign"
    assert worker.fps > 0
    assert worker.poll_events() == []
    # Signal stop before closing the camera so normal shutdown isn't recorded as failure.
    with pytest.raises(RuntimeError):
        worker.stop(timeout=0)
    camera.close()
    worker.stop()
    worker.stop()
    assert trackers[0].closed_thread == trackers[0].created_thread
    assert trackers[0].closed_thread != threading.get_ident()


@pytest.mark.parametrize("where", ["tracker", "classifier", "factory", "close"])
def test_worker_reports_failures_and_closes_tracker(fake_hand, where):
    camera = ScriptCamera(3)
    tracker = Tracker(fake_hand)
    def fail(*args):
        raise ValueError("deliberate failure")
    if where == "tracker":
        tracker.detect = fail
    if where == "close":
        tracker.close = fail
    classifier = Classifier(["fist"]*3)
    if where == "classifier":
        classifier.predict = fail
    def factory():
        tracker.created_thread = threading.get_ident()
        return tracker
    worker = GestureWorker(camera, fail if where == "factory" else factory, classifier)
    worker.start()
    if where == "close":
        assert camera.done.wait(2)
        with pytest.raises(RuntimeError):
            worker.stop(0)
        camera.close()
    deadline = time.perf_counter()+2
    while worker.error is None and time.perf_counter() < deadline:
        time.sleep(.005)
    assert isinstance(worker.error, ValueError)
    camera.close()
    worker.stop()
    if where in ("tracker", "classifier"):
        assert tracker.closed_thread is not None
