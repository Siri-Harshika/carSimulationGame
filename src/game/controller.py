"""GameController: the ONLY API integration calls.

Owner: Uy (Car game).
Specification: docs/interfaces.md §4.4.

Provides: ``GameController`` — consumed by Evan's ``src/app/main.py`` and by
``keyboard_demo.py --controller``.

This class is the whole of requirement GM.10. Integration code never touches
``Gameplay`` or ``StateMachine``; it calls the methods here and nothing else
(docs/DESIGN.md §2.1 rule 6). That boundary is what lets the game and the gesture
pipeline be developed and tested separately: ``src/game`` knows nothing about
cameras, landmarks or classifiers, and only ever receives a ``(label, confidence)``
pair to print on the HUD.

Not thread-safe. Pygame and this controller are used only on the main thread
(docs/DESIGN.md §2.1 rule 7).
"""
import random

import pygame

from src.config import LABELS, NO_HAND
from src.game.gameplay import Gameplay
from src.game.render import draw_world
from src.game.states import GameState, StateMachine
from src.game.ui import draw_hud, draw_overlay

# Everything set_gesture_info() will accept: the four classifier labels plus the
# pipeline-only "no hand" token, which is not a label the model can ever output.
_VALID_GESTURE_TOKENS: frozenset[str] = frozenset((*LABELS, NO_HAND))


class GameController:
    """Owns one round and exposes the only interface the live loop may use.

    The controller decides what a collision *means* (``Gameplay`` only reports one)
    and which calls are legal in which state, so the integration layer can fire
    ``move_left()`` or ``pause()`` on any gesture event without first inspecting the
    game.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        """Create a controller sitting in ``GameState.PAUSED``.

        Args:
            rng: Random source passed straight to ``Gameplay``. Seed it to make a
                test deterministic.
        """
        self._gameplay = Gameplay(rng)
        self._states = StateMachine()
        # HUD values, overwritten by the integration layer once per rendered frame.
        # The defaults describe the truthful situation before the first camera frame
        # arrives: no hand seen yet.
        self._label: str = NO_HAND
        self._confidence: float = 0.0
        self._hand_present: bool = False

    # ------------------------------------------------------------------
    # Fixed interface (I5)
    # ------------------------------------------------------------------

    def move_left(self) -> bool:
        """Move the car one lane left, if the current state allows it.

        Allowed in ``PLAYING`` and in ``COUNTDOWN`` — during the countdown the
        obstacles are frozen but steering already works, so the player can get out
        of the way before the road starts moving (docs/game-logic.md §5 rule 3).
        Ignored in ``PAUSED`` and ``GAME_OVER``.

        Returns:
            True only if the car actually changed lane. False covers both "the move
            was not allowed here" and "the car was already in the leftmost lane".
            Evan's latency logger uses this to time only actions that had a visible
            effect (docs/data-eval.md §8.3). Never raises.
        """
        if self._states.state not in (GameState.PLAYING, GameState.COUNTDOWN):
            return False
        return self._gameplay.move_left()

    def move_right(self) -> bool:
        """Move the car one lane right, if the current state allows it.

        Returns:
            True only if the car actually changed lane. See :meth:`move_left`.
        """
        if self._states.state not in (GameState.PLAYING, GameState.COUNTDOWN):
            return False
        return self._gameplay.move_right()

    def pause(self) -> None:
        """Pause the round. A no-op unless the state is ``PLAYING`` or ``COUNTDOWN``."""
        self._states.pause()

    def resume(self) -> None:
        """Resume a paused round, starting the countdown. A no-op in other states."""
        self._states.resume()

    def is_paused(self) -> bool:
        """Report whether the round is paused.

        Returns:
            True only in ``GameState.PAUSED``. ``COUNTDOWN`` counts as *not* paused,
            because the car can already be steered there.
        """
        return self._states.state is GameState.PAUSED

    # ------------------------------------------------------------------
    # Added for the live loop (docs/interfaces.md §4.4)
    # ------------------------------------------------------------------

    @property
    def state(self) -> GameState:
        """The current game state, read by Evan's ``gesture_to_action()``."""
        return self._states.state

    @property
    def score(self) -> int:
        """The score as whole points. Fractional points are an internal detail."""
        return int(self._gameplay.score)

    def restart(self) -> None:
        """Start a fresh round: reset the simulation and go to ``COUNTDOWN`` (GM.8).

        Works from any state, so the player can restart from the game-over screen
        with an open palm, and the keyboard ``R`` works at any time.
        """
        self._gameplay.reset()
        self._states.restart()

    def update(self, dt: float) -> None:
        """Advance the round by ``dt`` seconds.

        Timers always advance; the simulation only advances in ``PLAYING``, which is
        what freezes the obstacles and the score during ``PAUSED``, ``COUNTDOWN`` and
        ``GAME_OVER``.

        Collisions are disabled while ``grace_left`` is positive. The state machine
        is ticked **before** the simulation so that the tick which ends the countdown
        grants the full grace period before the first moving frame is simulated —
        otherwise an obstacle sitting on top of the car when the game paused would
        kill the player immediately on resume.

        Args:
            dt: Elapsed time in seconds since the previous update.

        Raises:
            ValueError: If ``dt`` is negative (propagated from ``StateMachine.tick``
                and ``Gameplay.update``).
        """
        if dt < 0:
            raise ValueError(f"dt must be non-negative, got {dt!r}")

        self._states.tick(dt)

        if self._states.state is GameState.PLAYING:
            hit = self._gameplay.update(
                dt, collisions_enabled=self._states.grace_left <= 0.0
            )
            if hit:
                self._states.crash()

    def render(self, surface: pygame.Surface, fps: float | None = None) -> None:
        """Draw the whole frame: world, then HUD, then the state overlay.

        Args:
            surface: Target surface, normally the pygame display.
            fps: Frames per second to show on the HUD, or None to omit it.

                **Interface note.** docs/interfaces.md §4.4 specifies
                ``render(surface)`` with no second parameter, but the same section
                gives ``ui.draw_hud()`` an ``fps`` argument and docs/workflow.md §11
                requires Evan's HUD to display the frame rate. Since integration may
                only call ``GameController`` (GM.10), there was no path for that
                value to reach the HUD. This optional, backward-compatible parameter
                closes the gap — an existing ``render(screen)`` call is unaffected.
                Approved by the Lead; the design doc still needs the matching edit.

        Returns:
            None.
        """
        # The car blinks exactly while it is immune, so the visual and the rule
        # cannot drift apart.
        draw_world(surface, self._gameplay, blink_car=self._states.grace_left > 0.0)
        draw_hud(surface, self.score, self._label, self._confidence,
                 self._hand_present, fps)
        draw_overlay(surface, self._states.state, self._states.countdown_left,
                     self.score)

    def set_gesture_info(self, label: str, confidence: float, hand_present: bool) -> None:
        """Store what the classifier last saw, for display on the HUD (GM.9, LI.6).

        This only feeds the HUD. It never moves the car: actions arrive through
        ``move_left``/``move_right``/``pause``/``resume``, already smoothed and
        debounced by Evan's worker.

        Args:
            label: One of ``src.config.LABELS`` or ``src.config.NO_HAND``.
            confidence: Classifier confidence, in [0, 1].
            hand_present: Whether MediaPipe detected a hand in the latest frame.

        Raises:
            ValueError: If ``confidence`` is outside [0, 1], or ``label`` is not a
                known token. The valid set is closed and known at import time, so an
                unknown value is always an upstream bug; raising makes it fail in
                Evan's ``test_worker.py`` rather than quietly painting nonsense on
                the HUD during the demo. The live loop in ``src/app/main.py`` should
                guard its frame loop so a bad frame cannot kill the whole session.
        """
        if label not in _VALID_GESTURE_TOKENS:
            raise ValueError(
                f"label must be one of {sorted(_VALID_GESTURE_TOKENS)}, got {label!r}"
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence!r}")

        self._label = label
        self._confidence = confidence
        self._hand_present = hand_present
