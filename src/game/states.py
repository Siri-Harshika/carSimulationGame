"""GameState enum + StateMachine (pure, no Pygame).

Owner: Uy (Car game).
Specification: docs/interfaces.md §4.4, state diagram in docs/game-logic.md §5.

Provides: ``GameState`` — read by Evan's ``src/app/actions.py`` to decide what a
gesture means in the current situation — and ``StateMachine``, used only by
``GameController``.

Like ``gameplay.py`` this module is pure: no pygame, no I/O, no timers of its own.
It is told how much time passed and answers what the round's state is now. That is
what makes every transition in docs/game-logic.md §5 testable headless.
"""
from enum import Enum

from src.config import GRACE_PERIOD_S, RESUME_COUNTDOWN_S


class GameState(Enum):
    """The four situations a round can be in.

    Attributes:
        PAUSED: Nothing moves. The initial state, and where the game goes on an open
            palm or when the player's hand leaves the camera.
        COUNTDOWN: A short run-up after resuming or restarting. Obstacles are frozen
            but the car **can** still change lanes, so the player can steer out of
            the way before the road starts moving again.
        PLAYING: Normal play.
        GAME_OVER: The car was hit. Score and gameplay are frozen until a restart.
    """

    PAUSED = "paused"
    COUNTDOWN = "countdown"
    PLAYING = "playing"
    GAME_OVER = "game_over"


class StateMachine:
    """Tracks which ``GameState`` a round is in and how much of its timers is left.

    Attributes:
        state: The current state. Starts at ``GameState.PAUSED`` (docs/game-logic.md
            §5 rule 1: nothing moves until the player shows a steering gesture).
        countdown_left: Seconds remaining in ``COUNTDOWN``; 0.0 in every other state.
        grace_left: Seconds of collision immunity remaining after a countdown ends;
            0.0 in every other state.

    Not thread-safe: used only on the main thread, like everything else the
    controller owns (docs/DESIGN.md §2.1 rule 7).
    """

    def __init__(self) -> None:
        """Create a state machine sitting in ``PAUSED`` with both timers at zero."""
        self.state: GameState = GameState.PAUSED
        self.countdown_left: float = 0.0
        self.grace_left: float = 0.0

    def pause(self) -> bool:
        """Pause an active round.

        Valid from ``PLAYING`` and ``COUNTDOWN``. Calling it in ``PAUSED`` or
        ``GAME_OVER`` does nothing, which is what makes the live loop simple: Evan
        can fire ``pause()`` on every open-palm event without first checking the
        state.

        Returns:
            True if the state actually changed, False if the call was a no-op.
        """
        if self.state not in (GameState.PLAYING, GameState.COUNTDOWN):
            return False
        self.state = GameState.PAUSED
        self.countdown_left = 0.0
        # Immunity is not banked across a pause: resuming always grants a fresh
        # countdown followed by a fresh GRACE_PERIOD_S, so keeping the remainder
        # here would only ever be double-counted.
        self.grace_left = 0.0
        return True

    def resume(self) -> bool:
        """Leave ``PAUSED`` and begin the run-up countdown.

        Returns:
            True if the state actually changed, False if the round was not paused.
        """
        if self.state is not GameState.PAUSED:
            return False
        self.state = GameState.COUNTDOWN
        self.countdown_left = RESUME_COUNTDOWN_S
        self.grace_left = 0.0
        return True

    def restart(self) -> bool:
        """Start a fresh round from any state, beginning with the countdown.

        Restart is a transition, not a state (docs/game-logic.md §5 rule 6). The
        caller is responsible for resetting ``Gameplay``; this method only moves the
        state. ``GameController.restart()`` does both, which is why integration code
        is required to go through the controller.

        Returns:
            Always True: a restart always lands in ``COUNTDOWN``.
        """
        self.state = GameState.COUNTDOWN
        self.countdown_left = RESUME_COUNTDOWN_S
        self.grace_left = 0.0
        return True

    def crash(self) -> bool:
        """End the round after a collision.

        Only valid from ``PLAYING``. A crash cannot happen during ``COUNTDOWN``
        (obstacles are frozen) or during the grace period (the controller disables
        collisions), so a call from any other state means the caller tested for a
        collision when it should not have.

        Returns:
            True if the state actually changed, False if the round was not playing.
        """
        if self.state is not GameState.PLAYING:
            return False
        self.state = GameState.GAME_OVER
        self.countdown_left = 0.0
        self.grace_left = 0.0
        return True

    def tick(self, dt: float) -> None:
        """Advance the countdown and grace timers by ``dt`` seconds.

        ``COUNTDOWN`` becomes ``PLAYING`` the moment the countdown reaches zero, and
        the grace period starts at its **full** ``GRACE_PERIOD_S`` in that same call:
        the ``elif`` below means the tick that ends the countdown never also eats
        into the immunity it just granted.

        Args:
            dt: Elapsed time in seconds. Must not be negative.

        Raises:
            ValueError: If ``dt`` is negative.
        """
        if dt < 0:
            raise ValueError(f"dt must be non-negative, got {dt!r}")

        if self.state is GameState.COUNTDOWN:
            self.countdown_left -= dt
            if self.countdown_left <= 0.0:
                self.countdown_left = 0.0
                self.state = GameState.PLAYING
                self.grace_left = GRACE_PERIOD_S
        elif self.state is GameState.PLAYING:
            self.grace_left = max(0.0, self.grace_left - dt)
