"""Unit tests for gameplay (cases listed in the spec below).

Owner: Uy (Car game).
Specification: docs/workflow.md §9.

``Gameplay`` is pure: no pygame, no display, no wall-clock timer. Every test here
drives it by handing it a ``dt`` and asserting on its public attributes, which is
exactly how ``GameController`` uses it. The RNG is injected (``random.Random(SEED)``)
so the spawn tests are reproducible — a flaky spawn test would be worse than no
spawn test at all.
"""
import random

import pytest

from src.config import (BASE_SPEED, MAX_DT, MAX_SPEED, NUM_LANES,
                        OBSTACLE_HEIGHT, SCORE_PER_SECOND, SEED,
                        SPEED_INCREASE_PER_SECOND, WINDOW_HEIGHT)
from src.game.entities import Obstacle
from src.game.gameplay import Gameplay

# The y a freshly spawned obstacle sits at: fully above the window. Duplicated from
# gameplay._SPAWN_Y rather than imported, because importing a private name would let
# a wrong value in the implementation agree with the test and pass.
_SPAWN_Y: float = -OBSTACLE_HEIGHT


@pytest.fixture()
def gameplay() -> Gameplay:
    """A fresh round with a seeded RNG, so spawn order is identical on every run."""
    return Gameplay(random.Random(SEED))


# ----------------------------------------------------------------------
# Lanes and steering
# ----------------------------------------------------------------------


def test_car_starts_in_the_middle_lane(gameplay: Gameplay) -> None:
    """Lane 1 of 3: the player can dodge either way on the first obstacle."""
    assert gameplay.car.lane == 1


def test_move_left_at_the_leftmost_lane_is_refused(gameplay: Gameplay) -> None:
    """At lane 0 the move returns False and the car does not leave the road.

    The return value matters beyond correctness: Evan's latency logger only times
    actions that had a visible effect (docs/data-eval.md §8.3).
    """
    assert gameplay.move_left() is True   # 1 -> 0
    assert gameplay.car.lane == 0

    assert gameplay.move_left() is False
    assert gameplay.car.lane == 0


def test_move_right_at_the_rightmost_lane_is_refused(gameplay: Gameplay) -> None:
    """Mirror image of the left boundary, at lane ``NUM_LANES - 1``."""
    assert gameplay.move_right() is True   # 1 -> 2
    assert gameplay.car.lane == NUM_LANES - 1

    assert gameplay.move_right() is False
    assert gameplay.car.lane == NUM_LANES - 1


# ----------------------------------------------------------------------
# Collisions
# ----------------------------------------------------------------------


def test_update_reports_a_collision_in_the_cars_lane(gameplay: Gameplay) -> None:
    """An obstacle placed on the car is reported as a hit.

    ``dt`` is tiny so the obstacle barely moves: the test is about the collision
    check, not about how far the obstacle travelled first.
    """
    gameplay.obstacles.append(Obstacle(lane=gameplay.car.lane, y=gameplay.car.y))

    assert gameplay.update(0.001) is True


def test_update_reports_no_collision_in_an_adjacent_lane(gameplay: Gameplay) -> None:
    """Same y, neighbouring lane: dodging works."""
    gameplay.obstacles.append(Obstacle(lane=gameplay.car.lane - 1, y=gameplay.car.y))

    assert gameplay.update(0.001) is False


def test_update_reports_no_collision_when_collisions_are_disabled(gameplay: Gameplay) -> None:
    """``collisions_enabled=False`` is how the controller implements the grace period.

    The obstacle still moves and the score still rises — only the verdict is
    suppressed — so immunity cannot be used to freeze the game.
    """
    gameplay.obstacles.append(Obstacle(lane=gameplay.car.lane, y=gameplay.car.y))

    assert gameplay.update(0.001, collisions_enabled=False) is False


# ----------------------------------------------------------------------
# Score and speed
# ----------------------------------------------------------------------


def test_score_increases_by_score_per_second_times_dt(gameplay: Gameplay) -> None:
    """Score is time-based, so it cannot be farmed by steering or by a high frame rate.

    Two unequal steps are used rather than one: a bug that adds a flat amount per
    call instead of scaling with ``dt`` would survive a single-step test.
    """
    gameplay.update(0.05)
    assert gameplay.score == pytest.approx(SCORE_PER_SECOND * 0.05)

    gameplay.update(0.02)
    assert gameplay.score == pytest.approx(SCORE_PER_SECOND * 0.07)


def test_speed_increases_over_time(gameplay: Gameplay) -> None:
    """Difficulty ramps with elapsed simulated time, starting from ``BASE_SPEED``."""
    assert gameplay.speed == pytest.approx(BASE_SPEED)

    gameplay.update(MAX_DT)

    assert gameplay.speed == pytest.approx(BASE_SPEED + SPEED_INCREASE_PER_SECOND * MAX_DT)
    assert gameplay.speed > BASE_SPEED


def test_speed_is_capped_at_max_speed(gameplay: Gameplay) -> None:
    """The ramp is clamped, otherwise the game becomes unplayable rather than hard.

    The number of steps is derived from the config values, so re-tuning the ramp
    does not silently turn this into a test that stops before reaching the cap.
    """
    seconds_to_cap = (MAX_SPEED - BASE_SPEED) / SPEED_INCREASE_PER_SECOND
    steps = int(seconds_to_cap / MAX_DT) + 10

    for _ in range(steps):
        gameplay.update(MAX_DT)

    assert gameplay.speed == pytest.approx(MAX_SPEED)


# ----------------------------------------------------------------------
# dt handling
# ----------------------------------------------------------------------


def test_update_clamps_dt_to_max_dt(gameplay: Gameplay) -> None:
    """A long stall must not teleport the obstacles through the car.

    Without the clamp, a 1 s freeze (window drag, GC pause, a slow first camera
    frame) would move obstacles ~220 px in one step and the player would die to a
    car they never saw move.
    """
    gameplay.update(1.0)

    assert gameplay.elapsed == pytest.approx(MAX_DT)
    assert gameplay.score == pytest.approx(SCORE_PER_SECOND * MAX_DT)


@pytest.mark.parametrize("bad_dt", [-0.001, -1.0])
def test_update_rejects_a_negative_dt(gameplay: Gameplay, bad_dt: float) -> None:
    """Negative time is never physical: it means the caller's clock went backwards."""
    with pytest.raises(ValueError):
        gameplay.update(bad_dt)


# ----------------------------------------------------------------------
# Obstacle lifecycle
# ----------------------------------------------------------------------


def test_obstacles_below_the_window_are_removed(gameplay: Gameplay) -> None:
    """Passed obstacles are dropped, so the list cannot grow without bound.

    A round can last minutes; keeping every obstacle ever spawned would make the
    per-frame collision scan slower the longer the player survives.
    """
    passed = Obstacle(lane=0, y=WINDOW_HEIGHT - 1.0)
    gameplay.obstacles.append(passed)

    gameplay.update(MAX_DT)

    # Identity, not equality: ``Obstacle`` is a dataclass, so a newly spawned
    # obstacle in the same lane could compare equal to the one we are tracking.
    assert all(o is not passed for o in gameplay.obstacles)


def test_no_spawned_row_ever_fills_every_lane() -> None:
    """The core playability invariant, over 10,000 seeded spawns.

    If a row occupied all three lanes the player would be dead with no possible
    input, which is a bug no amount of skill can work around.

    This drives the public ``update()`` rather than the private ``_spawn_row()``.
    It costs more iterations, but it also proves spawning is actually wired into
    the game loop, and it does not break the day someone renames a private helper.

    A row is identified by its spawn y: ``update()`` moves the existing obstacles
    *before* spawning, so anything still sitting exactly at ``_SPAWN_Y`` after the
    call was created during that call.
    """
    gameplay = Gameplay(random.Random(SEED))
    target_rows = 10_000
    rows_seen = 0
    guard = 0

    # Generous upper bound on iterations (~46k are expected at the capped speed);
    # it exists only so a spawning bug fails the test instead of hanging the suite.
    max_iterations = target_rows * 40

    while rows_seen < target_rows:
        guard += 1
        assert guard <= max_iterations, (
            f"only {rows_seen} rows after {guard} updates - spawning looks stuck"
        )
        # Collisions off: the car is parked in lane 1 and would be hit constantly,
        # and this test is about spawning, not about the crash rule.
        gameplay.update(MAX_DT, collisions_enabled=False)

        row = [o for o in gameplay.obstacles if o.y == _SPAWN_Y]
        if not row:
            continue

        rows_seen += 1
        lanes = [o.lane for o in row]
        assert len(lanes) < NUM_LANES, f"row {rows_seen} filled every lane: {lanes}"
        assert len(set(lanes)) == len(lanes), f"row {rows_seen} stacked a lane: {lanes}"

    assert rows_seen == target_rows
