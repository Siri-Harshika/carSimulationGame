"""Action enum, gesture_to_action() (THE mapping), apply_action().

Owner: Evan (Live integration).
Provides: Action, gesture_to_action, apply_action — consumed by main.py.
Specification: docs/interfaces.md §4.6 (the 5 × 4 mapping table).

Implementation: Evan live integration.
"""

from enum import Enum

from src.game.controller import GameController
from src.game.states import GameState


class Action(Enum):
    """What the live loop asks the GameController to do."""

    NOOP = "noop"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    PAUSE = "pause"
    RESUME_AND_LEFT = "resume_and_left"
    RESUME_AND_RIGHT = "resume_and_right"
    RESTART = "restart"


# Column order matches the frozen state table in docs/interfaces.md.
_STATES = (GameState.PLAYING, GameState.COUNTDOWN, GameState.PAUSED, GameState.GAME_OVER)
_MAPPING = {
    "fist": (Action.MOVE_LEFT, Action.MOVE_LEFT, Action.RESUME_AND_LEFT, Action.NOOP),
    "v_sign": (Action.MOVE_RIGHT, Action.MOVE_RIGHT, Action.RESUME_AND_RIGHT, Action.NOOP),
    "open_palm": (Action.PAUSE, Action.PAUSE, Action.NOOP, Action.RESTART),
    "none": (Action.NOOP,) * 4,
    "no_hand": (Action.PAUSE, Action.PAUSE, Action.NOOP, Action.NOOP),
}


def gesture_to_action(token: str, state: GameState) -> Action:
    """Look up the sole gesture-to-action mapping, without changing the game."""
    if token not in _MAPPING or state not in _STATES:
        raise ValueError(f"Invalid gesture/state: {token!r}, {state!r}")
    return _MAPPING[token][_STATES.index(state)]


def apply_action(controller: GameController, action: Action) -> bool:
    """Use only controller methods; resume before movement during countdown."""
    if not isinstance(action, Action):
        raise ValueError(f"Invalid action: {action!r}")
    if action == Action.NOOP:
        return False
    if action == Action.MOVE_LEFT:
        return controller.move_left()
    if action == Action.MOVE_RIGHT:
        return controller.move_right()
    before = controller.state
    if action == Action.PAUSE:
        controller.pause()
    elif action == Action.RESTART:
        controller.restart()
        return True  # A reset changes the run even if already in countdown.
    else:
        controller.resume()
        moved = (controller.move_left() if action == Action.RESUME_AND_LEFT
                 else controller.move_right())
        return moved or controller.state != before
    return controller.state != before
