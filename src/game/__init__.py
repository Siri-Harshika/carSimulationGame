"""Re-exports GameController, GameState.

Owner: Uy (Car game).
Specification: docs/interfaces.md §4.4.
"""
from src.game.controller import GameController
from src.game.states import GameState

__all__ = ["GameController", "GameState"]
