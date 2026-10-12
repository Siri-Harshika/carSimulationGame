"""Gameplay: lanes, spawning, movement, collisions, score, speed.

Owner: Uy (Car game).
Provides: Gameplay — consumed only inside src/game (GameController, keyboard_demo.py).
Specification: docs/interfaces.md §4.3.

Pure logic: no pygame import, so it is fully unit-testable with a seeded
random.Random.

STUB (Lead, Iteration 1): signatures are frozen; Uy replaces every
`raise NotImplementedError` with the real implementation.
"""

import random

from src.game.entities import Car, Obstacle


class Gameplay:
    """The simulation of one run: car lane, obstacles, score and speed.

    Attributes:
        car: The player's car.
        obstacles: Obstacles currently on screen.
        score: Points; += SCORE_PER_SECOND * dt while advancing.
        speed: px/s obstacle fall speed.
        elapsed: s of active (non-paused) play since reset.
    """

    car: Car
    obstacles: list[Obstacle]
    score: float
    speed: float
    elapsed: float

    def __init__(self, rng: random.Random | None = None) -> None:
        """Create a game in its reset state.

        Args:
            rng: Random source for spawning. Pass a seeded one in tests.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset to lane 1, no obstacles, score 0, speed BASE_SPEED, elapsed 0."""
        raise NotImplementedError

    def move_left(self) -> bool:
        """Move the car one lane left.

        Returns:
            True if the lane changed; False at lane 0.
        """
        raise NotImplementedError

    def move_right(self) -> bool:
        """Move the car one lane right.

        Returns:
            True if the lane changed; False at lane NUM_LANES-1.
        """
        raise NotImplementedError

    def update(self, dt: float, collisions_enabled: bool = True) -> bool:
        """Advance the simulation by dt seconds (algorithm: docs/interfaces.md §4.3).

        Args:
            dt: Seconds since the last update; clamped to MAX_DT.
            collisions_enabled: False during the grace period.

        Returns:
            True if the car collides with any obstacle (and collisions are enabled).

        Raises:
            ValueError: If dt < 0.
        """
        raise NotImplementedError
