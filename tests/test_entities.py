"""Unit tests for entities (cases listed in the spec below).

Owner: Uy (Car game).
Specification: docs/workflow.md §9.

``src.game.entities`` is pure geometry, so nothing here needs pygame, a display
or a seeded RNG. Every expected number is written out literally (80 / 240 / 400
rather than a formula) on purpose: a test that recomputes the value with the same
expression as the implementation would pass even if the expression itself is wrong.
"""
import pytest

from src.config import NUM_LANES
from src.game.entities import Car, Obstacle, collides, lane_center_x


# ----------------------------------------------------------------------
# lane_center_x()
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lane", "expected_x"),
    [(0, 80.0), (1, 240.0), (2, 400.0)],
)
def test_lane_center_x_returns_the_centre_of_each_lane(lane: int, expected_x: float) -> None:
    """The three lane centres are 80, 240 and 400 px (docs/game-logic.md §6.1)."""
    assert lane_center_x(lane) == pytest.approx(expected_x)


@pytest.mark.parametrize("bad_lane", [NUM_LANES, -1, 99])
def test_lane_center_x_rejects_lanes_outside_the_road(bad_lane: int) -> None:
    """An out-of-range lane is a caller bug, so it raises instead of extrapolating.

    Returning a plausible-looking x for lane 3 would silently draw the car off the
    road; raising makes the mistake land in the test suite instead.
    """
    with pytest.raises(ValueError):
        lane_center_x(bad_lane)


def test_car_and_obstacle_x_are_centred_in_their_lane() -> None:
    """``x`` is the left edge, derived from the lane centre minus half the width."""
    car = Car(lane=1)
    obstacle = Obstacle(lane=1, y=0.0)

    assert car.x == pytest.approx(240.0 - car.width / 2)
    assert obstacle.x == pytest.approx(240.0 - obstacle.width / 2)


# ----------------------------------------------------------------------
# collides()
# ----------------------------------------------------------------------


def test_collides_is_true_when_the_boxes_overlap() -> None:
    """Same lane, obstacle sitting on top of the car: a real hit."""
    car = Car(lane=1)
    obstacle = Obstacle(lane=1, y=car.y)

    assert collides(car, obstacle) is True


def test_collides_is_false_for_different_lanes() -> None:
    """Lanes 0 and 2 never touch: 160 px of lane 1 sits between them.

    This is the case the player relies on — dodging has to actually work — so it is
    checked for both neighbours of the car rather than just one.
    """
    car = Car(lane=1)

    for lane in (0, 2):
        assert collides(car, Obstacle(lane=lane, y=car.y)) is False


def test_collides_is_false_when_the_boxes_only_touch_edges() -> None:
    """Touching edges are not a collision: the comparisons are strict (``<``).

    The obstacle's bottom edge is placed exactly on the car's top edge. One frame
    later it overlaps and the hit is reported, so the player loses nothing by this
    choice — but it keeps the rule unambiguous instead of depending on float noise
    at the boundary.
    """
    car = Car(lane=1)
    obstacle = Obstacle(lane=1, y=car.y - Obstacle(lane=1, y=0.0).height)

    assert obstacle.y + obstacle.height == pytest.approx(car.y)
    assert collides(car, obstacle) is False


def test_collides_is_true_one_pixel_past_the_touching_edge() -> None:
    """The counterpart of the test above: 1 px of overlap already counts.

    Without this, an implementation that never reports a collision would still pass
    the "touching edges" test.
    """
    car = Car(lane=1)
    obstacle = Obstacle(lane=1, y=car.y - Obstacle(lane=1, y=0.0).height + 1.0)

    assert collides(car, obstacle) is True


def test_collides_does_not_mutate_its_arguments() -> None:
    """Collision testing is a query. It runs every frame, so it must stay read-only."""
    car = Car(lane=1)
    obstacle = Obstacle(lane=1, y=car.y)
    before = (car.lane, car.y, obstacle.lane, obstacle.y)

    collides(car, obstacle)

    assert (car.lane, car.y, obstacle.lane, obstacle.y) == before
