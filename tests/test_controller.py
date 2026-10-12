"""Unit tests for controller (cases listed in the spec below).

Owner: Uy (Car game).
Specification: docs/workflow.md §9.

``GameController`` is the only API the integration layer may call (GM.10), so these
tests are the contract Evan's ``src/app/main.py`` relies on. They exercise it the
way the live loop does: ``update(dt)`` then ``render(surface)``, with gestures
arriving only through ``set_gesture_info()``.

Rendering is tested on a plain offscreen ``pygame.Surface``. No display is opened
and no ``SDL_VIDEODRIVER`` is needed: ``pygame.Surface`` and ``pygame.font`` both
work without one, which keeps these tests independent of ``tests/conftest.py``
(Lead-owned, currently empty).

A few tests reach into ``controller._gameplay.obstacles``. That is deliberate and
unavoidable: the controller exposes no public way to place an obstacle, and the
grace-period rules cannot be tested without one. It is the only private access
here, and it is read/append only — the state machine is never touched directly.
"""
import random

import pygame
import pytest

from src.config import (GRACE_PERIOD_S, LABEL_FIST, NO_HAND, SEED,
                        WINDOW_HEIGHT, WINDOW_WIDTH)
from src.game.controller import GameController
from src.game.entities import Obstacle
from src.game.states import GameState

_FRAME_DT: float = 1.0 / 60.0


@pytest.fixture()
def controller() -> GameController:
    """A fresh controller with a seeded RNG, sitting in ``PAUSED``."""
    return GameController(random.Random(SEED))


@pytest.fixture()
def surface() -> pygame.Surface:
    """An offscreen surface the size of the real window. No display is created."""
    return pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))


def _reach_playing(controller: GameController) -> None:
    """Resume and tick through the countdown, leaving the round in ``PLAYING``."""
    controller.resume()
    while controller.state is GameState.COUNTDOWN:
        controller.update(0.1)
    assert controller.state is GameState.PLAYING


def _run_out_grace(controller: GameController) -> None:
    """Tick until the collision immunity has expired, still in ``PLAYING``.

    Freshly spawned obstacles start above the window and need seconds to reach the
    car, so no random obstacle can end the round inside the grace window.
    """
    while controller._states.grace_left > 0.0:
        controller.update(0.1)
    assert controller.state is GameState.PLAYING


def _put_obstacle_on_the_car(controller: GameController) -> None:
    """Clear the road and park one obstacle exactly on the car.

    Clearing first makes the next ``update()`` deterministic: the only thing that
    can be hit is the obstacle this helper placed.
    """
    car = controller._gameplay.car
    controller._gameplay.obstacles.clear()
    controller._gameplay.obstacles.append(Obstacle(lane=car.lane, y=car.y))


# ----------------------------------------------------------------------
# Pause / resume
# ----------------------------------------------------------------------


def test_a_new_controller_is_paused(controller: GameController) -> None:
    """The round waits for the player, and the HUD starts out truthful: no hand yet."""
    assert controller.is_paused() is True
    assert controller.state is GameState.PAUSED
    assert controller.score == 0


def test_resume_leaves_the_paused_state(controller: GameController) -> None:
    """``is_paused()`` is False during COUNTDOWN, because steering already works.

    Evan's ``gesture_to_action()`` branches on this, so reporting COUNTDOWN as
    "paused" would make a steering gesture resume a round that is already resuming.
    """
    controller.resume()

    assert controller.is_paused() is False
    assert controller.state is GameState.COUNTDOWN


def test_pause_and_resume_round_trip(controller: GameController) -> None:
    """PLAYING → PAUSED → COUNTDOWN, i.e. resuming always costs a fresh run-up."""
    _reach_playing(controller)

    controller.pause()
    assert controller.is_paused() is True

    controller.resume()
    assert controller.state is GameState.COUNTDOWN


def test_pause_and_resume_are_safe_to_repeat(controller: GameController) -> None:
    """The gesture layer can deliver the same event repeatedly; neither call raises.

    ``pause()``/``resume()`` return nothing by design, so the only thing to assert
    is that the state ends up where the contract says it does.
    """
    controller.pause()        # already paused
    assert controller.is_paused() is True

    controller.resume()
    controller.resume()       # already resuming
    assert controller.state is GameState.COUNTDOWN


# ----------------------------------------------------------------------
# Steering is state-dependent
# ----------------------------------------------------------------------


def test_steering_is_refused_while_paused(controller: GameController) -> None:
    """No lane change while paused, and the car does not move behind the overlay."""
    lane_before = controller._gameplay.car.lane

    assert controller.move_left() is False
    assert controller.move_right() is False
    assert controller._gameplay.car.lane == lane_before


def test_steering_works_during_the_countdown(controller: GameController) -> None:
    """Obstacles are frozen but the car can already move (docs/game-logic.md §5 rule 3).

    This is what makes resuming fair: the player can steer out of the way before
    the road starts moving again.
    """
    controller.resume()
    assert controller.state is GameState.COUNTDOWN

    assert controller.move_left() is True
    assert controller._gameplay.car.lane == 0


def test_steering_is_refused_after_game_over(controller: GameController) -> None:
    """A dead car cannot be driven; only ``restart()`` leaves GAME_OVER."""
    _reach_playing(controller)
    _run_out_grace(controller)
    _put_obstacle_on_the_car(controller)
    controller.update(0.001)
    assert controller.state is GameState.GAME_OVER

    assert controller.move_left() is False
    assert controller.move_right() is False


def test_steering_reports_false_at_the_edge_of_the_road(controller: GameController) -> None:
    """A refused move and an impossible move both return False.

    Evan's latency logger only times actions with a visible effect, so "allowed but
    already at the wall" has to be False too.
    """
    _reach_playing(controller)

    assert controller.move_left() is True    # 1 -> 0
    assert controller.move_left() is False   # already leftmost


# ----------------------------------------------------------------------
# The simulation only runs in PLAYING
# ----------------------------------------------------------------------


def test_the_game_is_frozen_while_paused(controller: GameController) -> None:
    """One second of updates while paused changes nothing at all.

    This is the auto-pause guarantee from LI.4: losing the hand to the camera must
    not cost the player score or kill them.
    """
    obstacles_before = list(controller._gameplay.obstacles)

    for _ in range(60):
        controller.update(_FRAME_DT)

    assert controller.score == 0
    assert controller._gameplay.obstacles == obstacles_before


def test_the_game_is_frozen_during_the_countdown(controller: GameController) -> None:
    """The score does not tick up during the run-up either."""
    controller.resume()

    controller.update(0.1)

    assert controller.state is GameState.COUNTDOWN
    assert controller.score == 0


def test_the_score_rises_while_playing(controller: GameController) -> None:
    """Sanity check in the other direction: the simulation does run in PLAYING.

    Without this, every "frozen" test above would also pass on a controller whose
    ``update()`` does nothing at all.
    """
    _reach_playing(controller)

    for _ in range(60):
        controller.update(_FRAME_DT)

    assert controller.score > 0


def test_the_game_is_frozen_after_game_over(controller: GameController) -> None:
    """The final score stays readable on the game-over screen."""
    _reach_playing(controller)
    _run_out_grace(controller)
    _put_obstacle_on_the_car(controller)
    controller.update(0.001)
    assert controller.state is GameState.GAME_OVER

    final_score = controller.score
    for _ in range(60):
        controller.update(_FRAME_DT)

    assert controller.score == final_score


# ----------------------------------------------------------------------
# Grace period
# ----------------------------------------------------------------------


def test_a_collision_during_the_grace_period_does_not_end_the_round(
    controller: GameController,
) -> None:
    """Immunity right after the countdown, so resuming cannot be instantly fatal.

    Without it, an obstacle that happened to be sitting on the car when the player
    paused would kill them on the first moving frame after they resume — a death
    with no possible input.
    """
    _reach_playing(controller)
    assert controller._states.grace_left > 0.0

    _put_obstacle_on_the_car(controller)
    controller.update(0.001)

    assert controller.state is GameState.PLAYING


def test_a_collision_after_the_grace_period_ends_the_round(controller: GameController) -> None:
    """Once immunity expires a hit is a hit: PLAYING → GAME_OVER."""
    _reach_playing(controller)
    _run_out_grace(controller)

    _put_obstacle_on_the_car(controller)
    controller.update(0.001)

    assert controller.state is GameState.GAME_OVER


def test_the_grace_period_is_not_banked_across_a_pause(controller: GameController) -> None:
    """Pausing mid-immunity does not stack a second grace period on resume.

    Otherwise a player could hold an open palm to top up their immunity and sit on
    top of an obstacle indefinitely.
    """
    _reach_playing(controller)
    controller.pause()
    assert controller._states.grace_left == 0.0

    _reach_playing(controller)

    assert controller._states.grace_left == pytest.approx(GRACE_PERIOD_S)


# ----------------------------------------------------------------------
# restart
# ----------------------------------------------------------------------


def test_restart_resets_the_score_and_enters_the_countdown(controller: GameController) -> None:
    """GM.8: a restart starts a genuinely fresh round, from any state."""
    _reach_playing(controller)
    for _ in range(60):
        controller.update(_FRAME_DT)
    assert controller.score > 0

    controller.restart()

    assert controller.score == 0
    assert controller.state is GameState.COUNTDOWN
    assert controller._gameplay.car.lane == 1
    assert controller._gameplay.obstacles == []


def test_restart_works_from_game_over(controller: GameController) -> None:
    """The player restarts with an open palm from the game-over screen."""
    _reach_playing(controller)
    _run_out_grace(controller)
    _put_obstacle_on_the_car(controller)
    controller.update(0.001)
    assert controller.state is GameState.GAME_OVER

    controller.restart()

    assert controller.state is GameState.COUNTDOWN
    assert controller.score == 0


# ----------------------------------------------------------------------
# update() input validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad_dt", [-0.001, -1.0])
def test_update_rejects_a_negative_dt(controller: GameController, bad_dt: float) -> None:
    """A clock that went backwards is a bug in the live loop, not something to absorb."""
    with pytest.raises(ValueError):
        controller.update(bad_dt)


# ----------------------------------------------------------------------
# set_gesture_info()
# ----------------------------------------------------------------------


def test_set_gesture_info_accepts_every_valid_token(controller: GameController) -> None:
    """All four labels plus ``no_hand`` are accepted, at both ends of the range."""
    controller.set_gesture_info(LABEL_FIST, 0.0, True)
    controller.set_gesture_info(LABEL_FIST, 1.0, True)
    controller.set_gesture_info(NO_HAND, 0.0, False)


@pytest.mark.parametrize("bad_confidence", [1.5, -0.1])
def test_set_gesture_info_rejects_a_confidence_outside_zero_to_one(
    controller: GameController, bad_confidence: float
) -> None:
    """Confidence is a probability. 1.5 means something upstream is broken.

    Raising makes it fail in Evan's ``test_worker.py`` instead of quietly painting
    "(1.50)" on the HUD during the presentation.
    """
    with pytest.raises(ValueError):
        controller.set_gesture_info(LABEL_FIST, bad_confidence, True)


@pytest.mark.parametrize("bad_label", ["thumbs_up", "", "FIST"])
def test_set_gesture_info_rejects_an_unknown_label(
    controller: GameController, bad_label: str
) -> None:
    """The valid token set is closed and known at import time, so anything else is a bug."""
    with pytest.raises(ValueError):
        controller.set_gesture_info(bad_label, 0.9, True)


def test_set_gesture_info_never_moves_the_car(controller: GameController) -> None:
    """GM.9: this feeds the HUD only.

    Actions reach the controller through ``move_left``/``move_right``/``pause``/
    ``resume``, already smoothed and debounced. If a label could also steer, a
    single noisy frame would move the car.
    """
    _reach_playing(controller)
    lane_before = controller._gameplay.car.lane
    state_before = controller.state

    controller.set_gesture_info(LABEL_FIST, 0.99, True)

    assert controller._gameplay.car.lane == lane_before
    assert controller.state is state_before


# ----------------------------------------------------------------------
# render()
# ----------------------------------------------------------------------


def test_render_works_in_every_state(controller: GameController, surface: pygame.Surface) -> None:
    """Each state draws a different overlay, so each one needs its own pass.

    No pixels are asserted on: this test exists to catch a crash (a missing font, a
    bad rect, a ``None`` reaching a draw call) on a frame the demo would show.
    """
    # PAUSED
    controller.render(surface)

    # COUNTDOWN
    controller.resume()
    controller.render(surface)

    # PLAYING
    _reach_playing(controller)
    controller.render(surface)

    # GAME_OVER
    _run_out_grace(controller)
    _put_obstacle_on_the_car(controller)
    controller.update(0.001)
    assert controller.state is GameState.GAME_OVER
    controller.render(surface)


@pytest.mark.parametrize("fps", [None, 0.0, 9.0, 60.0])
def test_render_accepts_any_fps_value(
    controller: GameController, surface: pygame.Surface, fps: float | None
) -> None:
    """The optional ``fps`` argument covers both branches of the low-FPS warning.

    ``None`` omits the field, 9.0 is below ``LOW_FPS_WARNING`` and 60.0 is above it.
    """
    _reach_playing(controller)

    controller.render(surface, fps=fps)


def test_render_does_not_change_the_game_state(
    controller: GameController, surface: pygame.Surface
) -> None:
    """Drawing is read-only: the live loop renders once per frame, with no side effect."""
    _reach_playing(controller)
    for _ in range(10):
        controller.update(_FRAME_DT)

    state_before = (controller.state, controller.score, controller._gameplay.car.lane,
                    [(o.lane, o.y) for o in controller._gameplay.obstacles])

    controller.render(surface, fps=60.0)

    assert (controller.state, controller.score, controller._gameplay.car.lane,
            [(o.lane, o.y) for o in controller._gameplay.obstacles]) == state_before


def test_render_works_on_a_smaller_surface(controller: GameController) -> None:
    """A surface smaller than the window clips instead of raising.

    Worth pinning down: the window size is a config value, and a mismatch during
    integration should not take the whole demo down.
    """
    controller.render(pygame.Surface((100, 100)), fps=30.0)


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


def test_two_controllers_with_the_same_seed_behave_identically() -> None:
    """Dependency-injected RNG: the same seed gives the same round.

    This is what makes a difficulty complaint reproducible, and it is why
    ``Gameplay`` takes a ``random.Random`` instead of calling the module-level
    ``random`` functions.
    """
    a, b = GameController(random.Random(SEED)), GameController(random.Random(SEED))
    for each in (a, b):
        _reach_playing(each)

    for _ in range(300):
        a.update(_FRAME_DT)
        b.update(_FRAME_DT)

    assert a.score == b.score
    assert [(o.lane, o.y) for o in a._gameplay.obstacles] == \
           [(o.lane, o.y) for o in b._gameplay.obstacles]


def test_the_default_controller_needs_no_rng() -> None:
    """``GameController()`` works with no arguments, as the live loop may construct it."""
    assert GameController().state is GameState.PAUSED
