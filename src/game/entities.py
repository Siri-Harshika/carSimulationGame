"""Geometry primitives of the car game: the player car, falling obstacles, lane maths.

Owner: Uy (Car game).
Specification: docs/interfaces.md §4.3.

Provides: ``Car``, ``Obstacle``, ``lane_center_x()``, ``collides()`` — consumed by
``src/game/gameplay.py`` (simulation), ``src/game/render.py`` and ``src/game/ui.py``
(drawing).

This module is deliberately free of pygame and of any I/O: it is pure geometry, so
the collision rule can be unit-tested without a display (see docs/workflow.md §9).
All sizes are in pixels in the game's own coordinate system, where y grows downwards
(the pygame convention) and y = 0 is the top of the window.
"""
from dataclasses import dataclass

from src.config import (CAR_HEIGHT, CAR_WIDTH, CAR_Y, LANE_WIDTH, NUM_LANES,
                        OBSTACLE_HEIGHT, OBSTACLE_WIDTH)


def lane_center_x(lane: int) -> float:
    """Return the horizontal centre of a lane.

    The road spans the full window width and is divided into ``NUM_LANES`` equal
    lanes, so lane ``i`` is centred half a lane width into its own slot. With the
    values in ``src/config.py`` (3 lanes of 160 px) this yields 80, 240 and 400.

    Args:
        lane: Lane index, 0 = left … ``NUM_LANES - 1`` = right.

    Returns:
        The centre x coordinate of that lane, in pixels.

    Raises:
        ValueError: If ``lane`` is outside ``range(NUM_LANES)``. We raise instead of
            clamping because an out-of-range lane always means a bug in the caller's
            lane bookkeeping, and silently clamping would hide it until the car
            visibly teleported.
    """
    if not 0 <= lane < NUM_LANES:
        raise ValueError(f"lane must be in range(0, {NUM_LANES}), got {lane!r}")
    return (lane + 0.5) * LANE_WIDTH


@dataclass
class Car:
    """The player-controlled car.

    The car only ever moves between discrete lanes, so its horizontal position is
    derived from ``lane`` rather than stored. That makes an inconsistent state
    (lane says 2, x says lane 0) impossible to represent.

    Attributes:
        lane: Current lane index, 0 = left … ``NUM_LANES - 1`` = right.
        width: Car width in pixels.
        height: Car height in pixels.
        y: Top edge of the car in pixels. Fixed: the car never moves vertically,
            the world scrolls past it.
    """

    lane: int
    width: float = CAR_WIDTH
    height: float = CAR_HEIGHT
    y: float = CAR_Y

    @property
    def x(self) -> float:
        """Left edge of the car in pixels, centred in its lane.

        Raises:
            ValueError: If ``lane`` is outside ``range(NUM_LANES)`` (propagated
                from :func:`lane_center_x`).
        """
        return lane_center_x(self.lane) - self.width / 2


@dataclass
class Obstacle:
    """One falling obstacle occupying a single lane.

    Attributes:
        lane: Lane index the obstacle falls in.
        y: Top edge in pixels. Increases over time as the obstacle falls; obstacles
            are spawned at a negative y, just above the visible window.
        width: Obstacle width in pixels.
        height: Obstacle height in pixels.
    """

    lane: int
    y: float
    width: float = OBSTACLE_WIDTH
    height: float = OBSTACLE_HEIGHT

    @property
    def x(self) -> float:
        """Left edge of the obstacle in pixels, centred in its lane.

        Raises:
            ValueError: If ``lane`` is outside ``range(NUM_LANES)`` (propagated
                from :func:`lane_center_x`).
        """
        return lane_center_x(self.lane) - self.width / 2


def collides(car: Car, obstacle: Obstacle) -> bool:
    """Test whether the car and an obstacle overlap (axis-aligned bounding boxes).

    Touching edges do **not** count as a collision: the comparison is strict on
    every side. The reason is that the car sits at a fixed y while obstacles move
    down in steps of ``speed * dt``; with a non-strict test, an obstacle that landed
    exactly on the car's top edge in one frame would be a hit, which feels unfair to
    the player and makes the outcome depend on the frame rate.

    With the sizes in ``src/config.py`` (lane 160 px, car 70 px, obstacle 90 px) two
    different lanes can never overlap horizontally: 35 + 45 = 80 px of combined
    half-width is less than one lane width. The x test below is therefore normally
    redundant, but it is kept so the function stays correct if those sizes change.

    Args:
        car: The player car.
        obstacle: One falling obstacle.

    Returns:
        True if the two rectangles overlap by a non-zero area.

    Raises:
        ValueError: If either object has an out-of-range lane (propagated from
            :func:`lane_center_x`).
    """
    car_x, obstacle_x = car.x, obstacle.x
    return (car_x < obstacle_x + obstacle.width
            and obstacle_x < car_x + car.width
            and car.y < obstacle.y + obstacle.height
            and obstacle.y < car.y + car.height)
