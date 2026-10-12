"""GameController: the ONLY API integration calls.

Owner: Uy (Car game).
Provides: GameController — consumed by Evan (actions.py, main.py) and tests.
Specification: docs/interfaces.md §4.4.

STUB (Lead, Iteration 1): this is the "logs calls and tracks paused" stub from
docs/workflow.md §11, so Evan can build the live loop before Uy's controller
exists. It keeps a lane and a GameState but has no obstacles, score or
countdown timer. Uy replaces the whole body; the signatures are frozen.
"""

import logging
import random

import pygame

from src.config import LABELS, NO_HAND, NUM_LANES
from src.game.states import GameState

logger = logging.getLogger(__name__)


class GameController:
    """Facade over Gameplay + StateMachine + rendering. Main thread only (not thread-safe).

    Stub attributes (not part of the interface, only for early tests/debugging):
        calls: Names of every interface method called, in order.
        lane: Current car lane.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        """Create a controller in PAUSED.

        Args:
            rng: Random source passed to Gameplay. Unused by the stub.
        """
        self.calls: list[str] = []
        self.lane: int = 1
        self._state = GameState.PAUSED
        self._gesture: tuple[str, float, bool] = (NO_HAND, 0.0, False)

    def _log(self, name: str) -> None:
        """Record one call so Evan can see what the live loop does."""
        self.calls.append(name)
        logger.info("GameController.%s (state=%s, lane=%d)", name, self._state.value, self.lane)

    # ---- Fixed interface (I5) ----
    def move_left(self) -> bool:
        """Move one lane left in PLAYING or COUNTDOWN.

        Returns:
            True if the lane actually changed. Never raises.
        """
        self._log("move_left")
        if self._state in (GameState.PLAYING, GameState.COUNTDOWN) and self.lane > 0:
            self.lane -= 1
            return True
        return False

    def move_right(self) -> bool:
        """Move one lane right in PLAYING or COUNTDOWN.

        Returns:
            True if the lane actually changed. Never raises.
        """
        self._log("move_right")
        if self._state in (GameState.PLAYING, GameState.COUNTDOWN) and self.lane < NUM_LANES - 1:
            self.lane += 1
            return True
        return False

    def pause(self) -> None:
        """PLAYING/COUNTDOWN → PAUSED; otherwise no-op."""
        self._log("pause")
        if self._state in (GameState.PLAYING, GameState.COUNTDOWN):
            self._state = GameState.PAUSED

    def resume(self) -> None:
        """PAUSED → COUNTDOWN; otherwise no-op.

        The stub has no timer: the next update() moves COUNTDOWN → PLAYING.
        """
        self._log("resume")
        if self._state is GameState.PAUSED:
            self._state = GameState.COUNTDOWN

    def is_paused(self) -> bool:
        """Return True only in PAUSED (COUNTDOWN is not paused: the car can move)."""
        return self._state is GameState.PAUSED

    # ---- [added] needed by the live loop ----
    @property
    def state(self) -> GameState:
        """Current game state."""
        return self._state

    @property
    def score(self) -> int:
        """Current score as int (always 0 in the stub)."""
        return 0

    def restart(self) -> None:
        """Reset the run and go to COUNTDOWN."""
        self._log("restart")
        self.lane = 1
        self._state = GameState.COUNTDOWN

    def update(self, dt: float) -> None:
        """Advance the game by dt seconds.

        Args:
            dt: Seconds since the last frame.

        Raises:
            ValueError: If dt < 0.
        """
        if dt < 0:
            raise ValueError(f"dt must be >= 0, got {dt}")
        if self._state is GameState.COUNTDOWN:
            self._state = GameState.PLAYING

    def render(self, surface: pygame.Surface) -> None:
        """Draw the game. The stub only clears the surface.

        Args:
            surface: Target surface (main thread only).
        """
        surface.fill((40, 40, 40))

    def set_gesture_info(self, label: str, confidence: float, hand_present: bool) -> None:
        """Store the latest gesture for the HUD (GM.9/LI.6).

        Args:
            label: One of LABELS or NO_HAND.
            confidence: Classifier confidence in [0, 1].
            hand_present: Whether MediaPipe found a hand.

        Raises:
            ValueError: If confidence is outside [0, 1] or label is unknown.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")
        if label not in LABELS and label != NO_HAND:
            raise ValueError(f"unknown label {label!r}; expected one of {LABELS} or {NO_HAND!r}")
        self._gesture = (label, confidence, hand_present)
