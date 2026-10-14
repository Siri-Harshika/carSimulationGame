"""draw_world(): road, lane lines, car, obstacles (Pygame).

Owner: Uy (Car game).
Specification: docs/interfaces.md §4.3.

Provides: ``draw_world()`` — called by ``GameController.render()``, and by
``keyboard_demo.py`` when it drives ``Gameplay`` directly.

This is the first module in ``src/game`` that touches pygame. Everything it needs
to know arrives as a ``Gameplay`` object; it never mutates that object, so drawing
can be skipped entirely (as the tests do) without changing the simulation.

Note on constants: docs/workflow.md §13 asks for every tunable value to live in
``src/config.py``, but ``config.py`` is owned by the Lead and holds no palette. The
colours below are therefore module-private. They are presentation-only — no
gameplay number depends on them — so moving them into ``config.py`` later is a
mechanical change.
"""
import pygame

from src.config import HUD_HEIGHT, LANE_WIDTH, NUM_LANES, WINDOW_HEIGHT
from src.game.entities import Car, Obstacle
from src.game.gameplay import Gameplay

# ---- Palette (RGB) ----
_COLOR_ROAD: tuple[int, int, int] = (38, 40, 46)
_COLOR_LANE_LINE: tuple[int, int, int] = (90, 94, 104)
_COLOR_CAR: tuple[int, int, int] = (72, 190, 255)
_COLOR_CAR_TRIM: tuple[int, int, int] = (18, 96, 140)
_COLOR_OBSTACLE: tuple[int, int, int] = (235, 84, 84)
_COLOR_OBSTACLE_TRIM: tuple[int, int, int] = (150, 40, 40)

# ---- Shape constants (px) ----
_LANE_LINE_WIDTH: int = 3
_LANE_DASH_LENGTH: int = 36      # length of one painted dash
_LANE_DASH_GAP: int = 28         # unpainted gap between dashes
_CORNER_RADIUS: int = 10
_TRIM_WIDTH: int = 3             # outline thickness on the car and obstacles

# Alpha applied to the car while it is immune during the grace period. Low enough to
# read as "blinking/invulnerable" at a glance, high enough that the player can still
# see which lane they are in.
_CAR_GRACE_ALPHA: int = 110


def draw_world(surface: pygame.Surface, gameplay: Gameplay, blink_car: bool = False) -> None:
    """Draw the road, the lane dividers, every obstacle and the car.

    Nothing else is drawn here: the HUD and the state overlays belong to
    ``src/game/ui.py`` and are painted on top by ``GameController.render()``.

    Args:
        surface: Target surface, expected to be ``WINDOW_WIDTH`` × ``WINDOW_HEIGHT``.
            A smaller surface simply clips; nothing raises.
        gameplay: The round to draw. Read-only here.
        blink_car: True while the car is immune during the grace period, which draws
            it semi-transparent so the player can see the immunity is still active.

    Returns:
        None.
    """
    surface.fill(_COLOR_ROAD)
    _draw_lane_lines(surface)
    for obstacle in gameplay.obstacles:
        _draw_obstacle(surface, obstacle)
    _draw_car(surface, gameplay.car, blink_car)


def _draw_lane_lines(surface: pygame.Surface) -> None:
    """Draw dashed dividers between the lanes.

    The dashes are static rather than scrolling. A scrolling road would need the
    distance travelled to be threaded through from ``Gameplay``, and the obstacles
    already give the player a clear sense of speed.
    """
    # Only the inner boundaries: the outer two are the window edges.
    for lane in range(1, NUM_LANES):
        x = int(lane * LANE_WIDTH)
        y = HUD_HEIGHT
        while y < WINDOW_HEIGHT:
            pygame.draw.line(
                surface, _COLOR_LANE_LINE,
                (x, y), (x, min(y + _LANE_DASH_LENGTH, WINDOW_HEIGHT)),
                _LANE_LINE_WIDTH,
            )
            y += _LANE_DASH_LENGTH + _LANE_DASH_GAP


def _draw_obstacle(surface: pygame.Surface, obstacle: Obstacle) -> None:
    """Draw one obstacle as a rounded rectangle with an outline."""
    rect = pygame.Rect(
        round(obstacle.x), round(obstacle.y),
        round(obstacle.width), round(obstacle.height),
    )
    pygame.draw.rect(surface, _COLOR_OBSTACLE, rect, border_radius=_CORNER_RADIUS)
    pygame.draw.rect(surface, _COLOR_OBSTACLE_TRIM, rect, _TRIM_WIDTH,
                     border_radius=_CORNER_RADIUS)


def _draw_car(surface: pygame.Surface, car: Car, blink: bool) -> None:
    """Draw the player car, semi-transparent while it is immune.

    Transparency needs a separate per-pixel-alpha surface: ``pygame.draw`` ignores
    the alpha channel of a colour when drawing straight onto an opaque surface.
    """
    width, height = round(car.width), round(car.height)
    local = pygame.Rect(0, 0, width, height)
    car_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    pygame.draw.rect(car_surface, _COLOR_CAR, local, border_radius=_CORNER_RADIUS)
    pygame.draw.rect(car_surface, _COLOR_CAR_TRIM, local, _TRIM_WIDTH,
                     border_radius=_CORNER_RADIUS)
    # A darker band across the upper third reads as a windscreen and, more usefully,
    # tells the player which way the car is facing.
    pygame.draw.rect(
        car_surface, _COLOR_CAR_TRIM,
        pygame.Rect(_TRIM_WIDTH * 2, height // 5, width - _TRIM_WIDTH * 4, height // 5),
        border_radius=_CORNER_RADIUS // 2,
    )

    if blink:
        car_surface.set_alpha(_CAR_GRACE_ALPHA)
    surface.blit(car_surface, (round(car.x), round(car.y)))
