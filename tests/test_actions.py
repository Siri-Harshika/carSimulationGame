"""Evan: all 20 mapping cells and controller interactions."""
import pytest
from src.app.actions import Action, apply_action, gesture_to_action
from src.game.controller import GameController
from src.game.states import GameState


@pytest.mark.parametrize("token, expected", [
    ("fist", ["move_left", "move_left", "resume_and_left", "noop"]),
    ("v_sign", ["move_right", "move_right", "resume_and_right", "noop"]),
    ("open_palm", ["pause", "pause", "noop", "restart"]),
    ("none", ["noop"]*4), ("no_hand", ["pause", "pause", "noop", "noop"]),
])
@pytest.mark.parametrize("index, state", list(enumerate([
    GameState.PLAYING, GameState.COUNTDOWN, GameState.PAUSED, GameState.GAME_OVER])))
def test_mapping(token, expected, index, state):
    assert gesture_to_action(token, state).value == expected[index]


def test_resume_move_pause_boundaries_and_restart():
    c = GameController()
    assert apply_action(c, Action.RESUME_AND_LEFT)
    assert c.calls == ["resume", "move_left"]
    assert c.lane == 0
    assert not apply_action(c, Action.MOVE_LEFT)
    assert apply_action(c, Action.PAUSE)
    assert c.is_paused()
    assert not apply_action(c, Action.PAUSE)
    assert not apply_action(c, Action.NOOP)
    assert apply_action(c, Action.RESTART)
    assert c.lane == 1
    assert c.state == GameState.COUNTDOWN


def test_invalid_mapping():
    with pytest.raises(ValueError):
        gesture_to_action("bad", GameState.PLAYING)
