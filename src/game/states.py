"""GameState enum + StateMachine (pure, no Pygame).

Owner: Uy (Car game).
Provides: GameState (also read by Evan's actions.py), StateMachine (used by GameController).
Specification: docs/interfaces.md §4.4, state diagram in docs/game-logic.md §5.

STUB (Lead, Iteration 1): GameState is complete (it is plain data). StateMachine
signatures are frozen; Uy replaces every `raise NotImplementedError`.
"""

from enum import Enum


class GameState(Enum):
    """The four game states."""

    PAUSED = "paused"        # initial state; also after open palm / no hand
    COUNTDOWN = "countdown"  # after resume/restart: obstacles frozen, car CAN change lanes
    PLAYING = "playing"
    GAME_OVER = "game_over"


class StateMachine:
    """Pure state transitions with the resume countdown and grace period.

    Attributes:
        state: Current state (initially PAUSED).
        countdown_left: s remaining while in COUNTDOWN, else 0.0.
        grace_left: s of collision immunity remaining after COUNTDOWN ends.
    """

    state: GameState
    countdown_left: float
    grace_left: float

    def __init__(self) -> None:
        """Start in PAUSED with no countdown and no grace."""
        raise NotImplementedError

    def pause(self) -> bool:
        """PLAYING/COUNTDOWN → PAUSED.

        Returns:
            True if a transition happened.
        """
        raise NotImplementedError

    def resume(self) -> bool:
        """PAUSED → COUNTDOWN (countdown_left = RESUME_COUNTDOWN_S).

        Returns:
            True if a transition happened.
        """
        raise NotImplementedError

    def restart(self) -> bool:
        """GAME_OVER (or any state) → COUNTDOWN.

        Returns:
            True if a transition happened.
        """
        raise NotImplementedError

    def crash(self) -> bool:
        """PLAYING → GAME_OVER.

        Returns:
            True if a transition happened.
        """
        raise NotImplementedError

    def tick(self, dt: float) -> None:
        """Count down; COUNTDOWN → PLAYING with grace_left = GRACE_PERIOD_S.

        Args:
            dt: Seconds since the last tick.
        """
        raise NotImplementedError
