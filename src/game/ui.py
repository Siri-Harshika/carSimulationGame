"""HUD + pause/countdown/game-over overlays.

Owner: Uy (Car game); Lead/Triet backup (docs/workflow.md §10).
Provides: draw_hud, draw_overlay — consumed by GameController.render.
Specification: docs/interfaces.md §4.4.

STUB (Lead, Iteration 1): signatures are frozen; Uy replaces every
`raise NotImplementedError` with the real implementation.
"""

import pygame

from src.game.states import GameState


def draw_hud(surface: pygame.Surface, score: int, label: str, confidence: float,
             hand_present: bool, fps: float | None = None) -> None:
    """Draw the 48 px top bar: score, current gesture + confidence, optional FPS.

    Main thread only (Pygame).

    Args:
        surface: Target surface.
        score: Current score (points).
        label: A LABELS value or NO_HAND.
        confidence: Classifier confidence in [0, 1].
        hand_present: False shows "NO HAND" in red.
        fps: Camera FPS; shown in orange below LOW_FPS_WARNING. None hides it.
    """
    raise NotImplementedError


def draw_overlay(surface: pygame.Surface, state: GameState, countdown_left: float,
                 score: int) -> None:
    """Draw the overlay for the current state (nothing for PLAYING).

    Main thread only (Pygame).

    Args:
        surface: Target surface.
        state: Current game state.
        countdown_left: s remaining in COUNTDOWN (shown as a large 2, 1).
        score: Final score shown on GAME_OVER.
    """
    raise NotImplementedError
