"""Evan: noisy streams, frame-rate variation and one command per gesture."""
import pytest
from src.app.smoothing import GestureSmoother, Debouncer
from src.config import NO_HAND


def test_majority_and_hysteresis():
    s = GestureSmoother()
    assert s.update("fist", 0) == NO_HAND
    assert s.update("fist", .03) == NO_HAND
    assert s.update("fist", .06) == "fist"
    assert s.update("v_sign", .09) == "fist"
    assert s.update("v_sign", .12) == "fist"
    assert s.update("v_sign", .15) == "fist"  # tie: retain previous output
    assert s.update("v_sign", .18) == "fist"  # 4/7 below 60%
    assert s.update("v_sign", .21) == "v_sign"


@pytest.mark.parametrize("fps", [8, 15, 30])
def test_missing_hand_smoothed_at_multiple_frame_rates(fps):
    s = GestureSmoother()
    outputs = [s.update("fist", i/fps) for i in range(10)]
    assert outputs[-1] == "fist"
    assert s.update(NO_HAND, 10/fps) == "fist"  # one missed detection
    for i in range(11, 21):
        s.update(NO_HAND, i/fps)
    assert s.stable == NO_HAND
    s.reset()
    assert s.stable == NO_HAND


def test_held_gesture_release_and_cooldown():
    d = Debouncer()
    assert d.update("fist", 0) == "fist"
    assert d.update("fist", 5) is None
    assert d.update("none", 5.01) is None
    assert d.update("fist", 5.02) == "fist"
    assert d.update("v_sign", 5.1) is None
    assert d.update("v_sign", 6) is None  # rejected edge is never queued
    assert d.update("open_palm", 6.01) == "open_palm"
    assert d.update(NO_HAND, 6.02) == NO_HAND
    assert d.update("v_sign", 6.03) == "v_sign"


@pytest.mark.parametrize("kwargs", [{"window_s": 0}, {"window_s": float("nan")},
                                    {"min_fraction": .5}, {"min_frames": 0}])
def test_invalid_smoother_options(kwargs):
    with pytest.raises(ValueError):
        GestureSmoother(**kwargs)


def test_invalid_token_time_and_reset():
    for obj in (GestureSmoother(), Debouncer()):
        obj.update("fist", 2)
        with pytest.raises(ValueError):
            obj.update("fist", 1)
        with pytest.raises(ValueError):
            obj.update("bad", 3)
        obj.reset()
        obj.update("fist", 0)
    with pytest.raises(ValueError):
        Debouncer(-1)
