"""Action enum, gesture_to_action() (THE mapping), apply_action().

Owner: Evan (Live integration).
Provides: Action, gesture_to_action, apply_action — consumed by main.py.
Specification: docs/interfaces.md §4.6 (the 5 × 4 mapping table).

STUB (Lead, Iteration 1): Action is complete (plain data); function signatures
are frozen and Evan replaces every `raise NotImplementedError`.
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


def gesture_to_action(token: str, state: GameState) -> Action:
    """THE single gesture→action mapping (I6). Pure: no side effects, no I/O.

    Args:
        token: A LABELS value or NO_HAND.
        state: Current game state.

    Returns:
        The Action from the docs/interfaces.md §4.6 table.

    Raises:
        ValueError: For any other token.
    """
    raise NotImplementedError


def apply_action(controller: GameController, action: Action) -> bool:
    """Execute an Action via GameController methods only. Main thread only.

    Args:
        controller: The game.
        action: What to do. RESUME_AND_LEFT = resume() then move_left().

    Returns:
        True if the game changed.
    """
    raise NotImplementedError
