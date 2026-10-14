"""Standalone keyboard-only loop driving Gameplay (dev tool).

Owner: Uy (Car game).
Specification: docs/interfaces.md §4.3.

Run it with::

    python -m src.game.keyboard_demo               # raw Gameplay, arrow keys only
    python -m src.game.keyboard_demo --controller  # full GameController: states, HUD, overlays

This is requirement GM.6: the game must be playable and demonstrable by keyboard
before the camera, the classifier or the integration layer exist. It is also the
fallback if anything in the gesture pipeline fails during the presentation.

The two modes are deliberately different in scope. The default mode drives
``Gameplay`` directly and is the quickest way to feel out the difficulty curve. The
``--controller`` mode drives ``GameController`` exactly the way Evan's ``main.py``
will, which is what makes every state, overlay and HUD field demonstrable on its
own (required by the Iteration 2 acceptance check in docs/workflow.md §11).

This module is a development tool, not a dependency of anything: no other module
imports it.
"""
import argparse
import random
import sys

import pygame

from src.config import (LABEL_FIST, LABEL_NONE, LABEL_OPEN_PALM, LABEL_V_SIGN,
                        MAX_DT, NO_HAND, SEED, TARGET_FPS, WINDOW_HEIGHT,
                        WINDOW_WIDTH)
from src.game.controller import GameController
from src.game.gameplay import Gameplay
from src.game.render import draw_world

# Number keys to fake classifier output in --controller mode, so the HUD can be
# demonstrated before a real model exists. Mirrors the keys Evan's MockClassifier
# uses (docs/interfaces.md §4.6), which keeps the demo muscle memory the same.
_FAKE_GESTURE_KEYS: dict[int, tuple[str, float, bool]] = {
    pygame.K_1: (LABEL_FIST, 0.95, True),
    pygame.K_2: (LABEL_V_SIGN, 0.91, True),
    pygame.K_3: (LABEL_OPEN_PALM, 0.88, True),
    pygame.K_0: (LABEL_NONE, 0.40, True),
    pygame.K_h: (NO_HAND, 0.0, False),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line options.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        A namespace with ``controller`` (bool) and ``seed`` (int).
    """
    parser = argparse.ArgumentParser(
        description="Keyboard demo for the car game (no webcam required)."
    )
    parser.add_argument(
        "--controller", action="store_true",
        help="drive GameController (states, HUD, overlays) instead of raw Gameplay",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="RNG seed for obstacle spawning (default: src.config.SEED)",
    )
    return parser.parse_args(argv)


def _new_window() -> pygame.Surface:
    """Initialise pygame and open the game window.

    Returns:
        The display surface, ``WINDOW_WIDTH`` × ``WINDOW_HEIGHT``.
    """
    pygame.init()
    pygame.display.set_caption("Car game - keyboard demo")
    return pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))


def run_gameplay_demo(seed: int = SEED) -> None:
    """Run the raw-``Gameplay`` demo: arrow keys steer, a crash prints and resets.

    There is no state machine here on purpose. This loop exercises exactly the
    gameplay layer, so a difficulty problem seen here cannot be blamed on the
    countdown, the grace period or the controller.

    Args:
        seed: RNG seed for obstacle spawning.

    Returns:
        None.
    """
    screen = _new_window()
    clock = pygame.time.Clock()
    gameplay = Gameplay(random.Random(seed))

    running = True
    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    gameplay.move_left()
                elif event.key == pygame.K_RIGHT:
                    gameplay.move_right()

        # dt is clamped inside Gameplay.update too; clamping here as well keeps the
        # printed score honest after a stall.
        if gameplay.update(min(dt, MAX_DT)):
            print(f"crash - score {int(gameplay.score)}")
            gameplay.reset()

        draw_world(screen, gameplay)
        pygame.display.flip()

    pygame.quit()


def run_controller_demo(seed: int = SEED) -> None:
    """Run the full-``GameController`` demo.

    Keys: ← / → steer, ``P`` pauses or resumes, ``R`` restarts, ``Esc`` quits, and
    ``1`` / ``2`` / ``3`` / ``0`` / ``H`` push fake gesture values onto the HUD.

    ``P`` mirrors the gesture contract rather than being a plain toggle: in
    ``PAUSED`` it resumes, everywhere else it pauses (docs/game-logic.md §5 rule 7).

    Args:
        seed: RNG seed for obstacle spawning.

    Returns:
        None.
    """
    screen = _new_window()
    clock = pygame.time.Clock()
    controller = GameController(random.Random(seed))

    running = True
    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    controller.move_left()
                elif event.key == pygame.K_RIGHT:
                    controller.move_right()
                elif event.key == pygame.K_p:
                    controller.resume() if controller.is_paused() else controller.pause()
                elif event.key == pygame.K_r:
                    controller.restart()
                elif event.key in _FAKE_GESTURE_KEYS:
                    controller.set_gesture_info(*_FAKE_GESTURE_KEYS[event.key])

        controller.update(min(dt, MAX_DT))
        # get_fps() is the real render rate of this loop, which is what the HUD's
        # FPS field is for.
        controller.render(screen, fps=clock.get_fps())
        pygame.display.flip()

    pygame.quit()


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m src.game.keyboard_demo``.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        Process exit code: 0 on a clean quit.
    """
    args = _parse_args(argv)
    if args.controller:
        run_controller_demo(args.seed)
    else:
        run_gameplay_demo(args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
