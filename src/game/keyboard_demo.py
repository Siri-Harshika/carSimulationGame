"""Standalone keyboard-only loop driving Gameplay (dev tool, GM.6).

Owner: Uy (Car game).
Specification: docs/interfaces.md §4.3.

Usage (from the repo root):
    python -m src.game.keyboard_demo                # ←/→ drive Gameplay directly
    python -m src.game.keyboard_demo --controller   # drive GameController (P, R, 1/2/3/0/H too)

STUB (Lead, Iteration 1): signatures are frozen; Uy replaces every
`raise NotImplementedError` with the real implementation.
"""


def main(argv: list[str] | None = None) -> None:
    """Run the 60 FPS demo loop until the window is closed.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]). Accepts --controller.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
