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
from collections import deque

import pygame

from src.config import LABELS, NO_HAND
from src.game.gameplay import Gameplay
from src.game.render import draw_world
from src.game.states import GameState, StateMachine
from src.game.ui import draw_hud, draw_overlay

# Everything set_gesture_info() will accept: the four classifier labels plus the
# pipeline-only "no hand" token, which is not a label the model can ever output.
_VALID_GESTURE_TOKENS: frozenset[str] = frozenset((*LABELS, NO_HAND))

# How many entries ``calls`` keeps. The log exists for assertions and debugging, not
# for gameplay, so it is capped: a long demo session fires an action roughly once a
# second and would otherwise grow the list for as long as the program runs. The
# longest sequence any test checks is five entries, so the cap is never reached
# during a test.
_CALL_LOG_MAXLEN: int = 256


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
        # Append-only log of the action methods integration called; see ``calls``.
        self._calls: deque[str] = deque(maxlen=_CALL_LOG_MAXLEN)

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
        self._calls.append("move_left")
        if self._states.state not in (GameState.PLAYING, GameState.COUNTDOWN):
            return False
        return self._gameplay.move_left()

    def move_right(self) -> bool:
        """Move the car one lane right, if the current state allows it.

        Returns:
            True only if the car actually changed lane. See :meth:`move_left`.
        """
        self._calls.append("move_right")
        if self._states.state not in (GameState.PLAYING, GameState.COUNTDOWN):
            return False
        return self._gameplay.move_right()

    def pause(self) -> None:
        """Pause the round. A no-op unless the state is ``PLAYING`` or ``COUNTDOWN``."""
        self._calls.append("pause")
        self._states.pause()

    def resume(self) -> None:
        """Resume a paused round, starting the countdown. A no-op in other states."""
        self._calls.append("resume")
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

    @property
    def lane(self) -> int:
        """The lane the car is in: 0 = left … ``NUM_LANES - 1`` = right.

        Read-only and derived, so it cannot drift from ``Gameplay``'s own state. It
        exists because GM.10 forbids integration code from reaching into
        ``Gameplay``, yet two consumers legitimately need the lane: Evan's debug
        overlay in ``src/app/main.py`` prints it, and ``tests/test_worker.py``
        asserts on it to check that a gesture event actually steered the car.

        **Interface note.** docs/interfaces.md §4.4 does not list this property; the
        interface stubs and those tests do. §4.4 needs the matching edit.
        """
        return self._gameplay.car.lane

    @property
    def calls(self) -> list[str]:
        """Names of the action methods called on this controller, oldest first.

        One entry per call to ``move_left``, ``move_right``, ``pause``, ``resume`` or
        ``restart``, recorded on entry — so a call that had no effect, such as
        ``move_left()`` in the leftmost lane or ``pause()`` while already paused, is
        still logged. The log answers "what did the integration layer ask for?", which
        is a different question from "what changed?"; the return value of each method
        already answers the second. ``update``, ``render`` and ``set_gesture_info`` are
        deliberately not logged: they run every frame and would bury the actions.

        Returns:
            A fresh list, so a caller cannot mutate the controller's own log, and
            comparing it to a list literal works. Capped at ``_CALL_LOG_MAXLEN``
            entries; ``restart()`` adds to the log rather than clearing it, because the
            log belongs to the controller object, not to one round.

        This exists for ``tests/test_actions.py`` and ``tests/test_worker.py``, which
        assert on the exact call sequence an action produces — ``RESUME_AND_LEFT``, for
        instance, must call ``resume()`` *before* ``move_left()`` or the move is
        swallowed in ``PAUSED``, and no return value can prove that ordering. It is a
        test seam in production code, which is a real cost; the cleaner alternative is
        for those tests to wrap the controller in ``MagicMock(spec=GameController)``.
        If the team prefers that, this property can be deleted unchanged.

        **Interface note.** docs/interfaces.md §4.4 does not list this property; the
        interface stubs and those tests do. §4.4 needs the matching edit.
        """
        return list(self._calls)

    def restart(self) -> None:
        """Start a fresh round: reset the simulation and go to ``COUNTDOWN`` (GM.8).

        Works from any state, so the player can restart from the game-over screen
        with an open palm, and the keyboard ``R`` works at any time.
        """
        self._calls.append("restart")
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
