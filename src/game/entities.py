"""Car, Obstacle dataclasses + collides() (AABB).

Owner: Uy (Car game).
Provides: Car, Obstacle, lane_center_x, collides — consumed inside src/game only
(gameplay.py, render.py, ui.py).
Specification: docs/interfaces.md §4.3.

STUB (Lead, Iteration 1): signatures are frozen; Uy replaces every
`raise NotImplementedError` with the real implementation.
"""

from dataclasses import dataclass

from src.config import CAR_HEIGHT, CAR_WIDTH, CAR_Y, OBSTACLE_HEIGHT, OBSTACLE_WIDTH


@dataclass
class Car:
    """The player's car. Only the lane changes; y is fixed near the bottom.

    Attributes:
        lane: 0 = left, 1 = middle, 2 = right.
        width: px.
        height: px.
        y: px, top edge (fixed).
    """

    lane: int
    width: float = CAR_WIDTH
    height: float = CAR_HEIGHT
    y: float = CAR_Y

    @property
    def x(self) -> float:
        """Left edge in px: lane_center_x(lane) - width / 2."""
        raise NotImplementedError


@dataclass
class Obstacle:
    """One falling obstacle.

    Attributes:
        lane: 0..NUM_LANES-1.
        y: px, top edge; increases as it falls.
        width: px.
        height: px.
    """

    lane: int
    y: float
    width: float = OBSTACLE_WIDTH
    height: float = OBSTACLE_HEIGHT

    @property
    def x(self) -> float:
        """Left edge in px: lane_center_x(lane) - width / 2."""
        raise NotImplementedError


def lane_center_x(lane: int) -> float:
    """Return the centre x (px) of a lane.

    Args:
        lane: Lane index in range(NUM_LANES).

    Returns:
        Centre x in px (80 / 240 / 400 for the default 480 px window).

    Raises:
        ValueError: If lane is not in range(NUM_LANES).
    """
    raise NotImplementedError


def collides(car: Car, obstacle: Obstacle) -> bool:
    """Axis-aligned bounding-box overlap test.

    Touching edges do NOT count as a collision.

    Args:
        car: The player's car.
        obstacle: One obstacle.

    Returns:
        True if the two boxes overlap.
    """
    raise NotImplementedError
