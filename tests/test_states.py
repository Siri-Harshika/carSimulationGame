"""Unit tests for states (cases listed in the spec below).

Owner: Uy (Car game).
Specification: docs/workflow.md §9, state diagram in docs/game-logic.md §5.

``StateMachine`` has no timer of its own — it is told how much time passed — so
every transition in the diagram is testable headless and instantly, with no sleeps.
A test that slept for ``RESUME_COUNTDOWN_S`` would be both slower and less reliable.
"""
import pytest

from src.config import GRACE_PERIOD_S, RESUME_COUNTDOWN_S
from src.game.states import GameState, StateMachine


@pytest.fixture()
def machine() -> StateMachine:
    """A fresh state machine, i.e. a round that has not started yet."""
    return StateMachine()


def _advance(machine: StateMachine, seconds: float, step: float = 0.1) -> None:
    """Tick ``machine`` through ``seconds`` in realistic frame-sized steps.

    One big tick would also work, but stepping is closer to what the live loop does
    and catches a transition that only fires on an exact boundary.
    """
    remaining = seconds
    while remaining > 0.0:
        dt = min(step, remaining)
        machine.tick(dt)
        remaining -= dt


# ----------------------------------------------------------------------
# Initial state
# ----------------------------------------------------------------------


def test_a_new_round_starts_paused(machine: StateMachine) -> None:
    """Nothing moves until the player shows a steering gesture (§5 rule 1).

    Starting in PLAYING would mean the player loses points, or crashes, while they
    are still getting their hand into the camera frame.
    """
    assert machine.state is GameState.PAUSED
    assert machine.countdown_left == 0.0
    assert machine.grace_left == 0.0


# ----------------------------------------------------------------------
# resume / countdown / grace
# ----------------------------------------------------------------------


def test_resume_enters_the_countdown(machine: StateMachine) -> None:
    """PAUSED → COUNTDOWN, with the full run-up on the clock."""
    assert machine.resume() is True

    assert machine.state is GameState.COUNTDOWN
    assert machine.countdown_left == pytest.approx(RESUME_COUNTDOWN_S)


def test_resume_is_a_no_op_when_not_paused(machine: StateMachine) -> None:
    """Firing ``resume()`` on an already-running round changes nothing.

    This is what lets the live loop call ``resume()`` on every steering gesture
    without first inspecting the state.
    """
    machine.resume()
    assert machine.resume() is False
    assert machine.state is GameState.COUNTDOWN


def test_the_countdown_ends_in_playing_with_a_full_grace_period(machine: StateMachine) -> None:
    """After ``RESUME_COUNTDOWN_S`` the round is PLAYING and immune for the full period.

    The grace period must start at its full value on the very tick that ends the
    countdown. If that tick also decremented it, an obstacle that was sitting on the
    car when the game paused could kill the player on the first moving frame.
    """
    machine.resume()

    _advance(machine, RESUME_COUNTDOWN_S)

    assert machine.state is GameState.PLAYING
    assert machine.countdown_left == 0.0
    assert machine.grace_left == pytest.approx(GRACE_PERIOD_S)


def test_the_grace_period_runs_down_during_play_and_stops_at_zero(machine: StateMachine) -> None:
    """Immunity decays while PLAYING and never goes negative."""
    machine.resume()
    _advance(machine, RESUME_COUNTDOWN_S)

    machine.tick(0.1)
    assert machine.grace_left == pytest.approx(GRACE_PERIOD_S - 0.1)

    _advance(machine, GRACE_PERIOD_S + 1.0)
    assert machine.grace_left == 0.0
    assert machine.state is GameState.PLAYING


def test_the_countdown_does_not_overshoot_into_negative_time(machine: StateMachine) -> None:
    """A single oversized tick still lands on exactly 0.0, not below it.

    ``countdown_left`` is shown on screen, so a negative value would render as a
    wrong digit in the overlay.
    """
    machine.resume()

    machine.tick(RESUME_COUNTDOWN_S * 5)

    assert machine.state is GameState.PLAYING
    assert machine.countdown_left == 0.0


# ----------------------------------------------------------------------
# pause
# ----------------------------------------------------------------------


def test_pause_works_from_playing(machine: StateMachine) -> None:
    """PLAYING → PAUSED, with both timers cleared."""
    machine.resume()
    _advance(machine, RESUME_COUNTDOWN_S)
    assert machine.state is GameState.PLAYING

    assert machine.pause() is True

    assert machine.state is GameState.PAUSED
    assert machine.countdown_left == 0.0
    # Immunity is not banked across a pause: resuming grants a fresh countdown and
    # a fresh grace period, so keeping the remainder would be double-counted.
    assert machine.grace_left == 0.0


def test_pause_works_from_countdown(machine: StateMachine) -> None:
    """The player can pause again during the run-up, before the road starts moving."""
    machine.resume()
    assert machine.state is GameState.COUNTDOWN

    assert machine.pause() is True

    assert machine.state is GameState.PAUSED
    assert machine.countdown_left == 0.0


def test_pause_while_already_paused_reports_no_change(machine: StateMachine) -> None:
    """An open palm held in front of the camera must not do anything odd.

    The gesture pipeline can deliver repeated ``open_palm`` events, so ``pause()``
    has to be idempotent and say so in its return value.
    """
    assert machine.pause() is False
    assert machine.state is GameState.PAUSED


def test_pause_after_game_over_reports_no_change(machine: StateMachine) -> None:
    """GAME_OVER is left only by a restart: a pause gesture cannot resurrect a round."""
    machine.resume()
    _advance(machine, RESUME_COUNTDOWN_S)
    machine.crash()

    assert machine.pause() is False
    assert machine.state is GameState.GAME_OVER


# ----------------------------------------------------------------------
# crash
# ----------------------------------------------------------------------


def test_crash_ends_a_playing_round(machine: StateMachine) -> None:
    """PLAYING → GAME_OVER, timers cleared."""
    machine.resume()
    _advance(machine, RESUME_COUNTDOWN_S)

    assert machine.crash() is True

    assert machine.state is GameState.GAME_OVER
    assert machine.countdown_left == 0.0
    assert machine.grace_left == 0.0


@pytest.mark.parametrize("prepare", ["paused", "countdown", "game_over"])
def test_crash_is_refused_outside_playing(machine: StateMachine, prepare: str) -> None:
    """A crash reported from any other state means the caller tested too early.

    Obstacles are frozen during COUNTDOWN and the controller disables collisions
    during the grace period, so there is no legitimate path to a collision outside
    PLAYING. Returning False keeps that assumption checkable.
    """
    if prepare == "countdown":
        machine.resume()
    elif prepare == "game_over":
        machine.resume()
        _advance(machine, RESUME_COUNTDOWN_S)
        machine.crash()

    state_before = machine.state

    assert machine.crash() is False
    assert machine.state is state_before


# ----------------------------------------------------------------------
# restart
# ----------------------------------------------------------------------


@pytest.mark.parametrize("prepare", ["paused", "countdown", "playing", "game_over"])
def test_restart_always_enters_the_countdown(machine: StateMachine, prepare: str) -> None:
    """Restart is a transition available from every state (§5 rule 6).

    The player has to be able to restart from the game-over screen with an open
    palm, and the keyboard ``R`` has to work at any time during the demo.
    """
    if prepare == "countdown":
        machine.resume()
    elif prepare == "playing":
        machine.resume()
        _advance(machine, RESUME_COUNTDOWN_S)
    elif prepare == "game_over":
        machine.resume()
        _advance(machine, RESUME_COUNTDOWN_S)
        machine.crash()

    assert machine.restart() is True

    assert machine.state is GameState.COUNTDOWN
    assert machine.countdown_left == pytest.approx(RESUME_COUNTDOWN_S)
    assert machine.grace_left == 0.0


# ----------------------------------------------------------------------
# tick input validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad_dt", [-0.001, -1.0])
def test_tick_rejects_a_negative_dt(machine: StateMachine, bad_dt: float) -> None:
    """Negative time would run the countdown backwards instead of forwards."""
    with pytest.raises(ValueError):
        machine.tick(bad_dt)


def test_tick_does_nothing_in_paused_or_game_over(machine: StateMachine) -> None:
    """Timers only belong to COUNTDOWN and PLAYING; elsewhere a tick is inert."""
    machine.tick(1.0)
    assert machine.state is GameState.PAUSED

    machine.resume()
    _advance(machine, RESUME_COUNTDOWN_S)
    machine.crash()

    machine.tick(1.0)
    assert machine.state is GameState.GAME_OVER
