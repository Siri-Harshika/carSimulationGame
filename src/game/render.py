"""draw_world(): road, lane lines, car, obstacles (Pygame).

Owner: Uy (Car game).
Provides: draw_world — consumed by GameController.render and keyboard_demo.py.
Specification: docs/interfaces.md §4.3.

STUB (Lead, Iteration 1): signatures are frozen; Uy replaces every
`raise NotImplementedError` with the real implementation.
"""

import pygame

from src.game.gameplay import Gameplay


def draw_world(surface: pygame.Surface, gameplay: Gameplay, blink_car: bool = False) -> None:
    """Draw road, lane dividers, obstacles and the car.

    Main thread only (Pygame).

    Args:
        surface: Target surface, WINDOW_WIDTH × WINDOW_HEIGHT.
        gameplay: The simulation to draw.
        blink_car: True draws the car semi-transparent (grace period).
    """
    raise NotImplementedError
