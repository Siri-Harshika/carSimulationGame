"""Gameplay: lanes, spawning, movement, collisions, score, speed.

Owner: Uy (Car game).
Specification: docs/interfaces.md §4.3.

Provides: ``Gameplay`` — the pure simulation of one round. Consumed **only** inside
``src/game`` (by ``GameController`` and ``keyboard_demo.py``). Integration code must
never touch this class directly; it goes through ``GameController`` (GM.10,
docs/DESIGN.md §2.1 rule 6).

This module imports neither pygame nor anything outside ``src.config`` and
``src.game.entities``. It has no I/O, no threads and no global random state: the RNG
is injected, so every test can replay the exact same sequence of obstacle rows.
``Gameplay`` also makes no decision about what a collision *means* — it only reports
one. Whether the round ends is the state machine's business.
"""
import random

from src.config import (BASE_SPEED, DOUBLE_OBSTACLE_PROB, MAX_DT, MAX_SPEED,
                        NUM_LANES, OBSTACLE_HEIGHT, SCORE_PER_SECOND, SEED,
                        SPAWN_GAP_PX, SPEED_INCREASE_PER_SECOND, WINDOW_HEIGHT)
from src.game.entities import Car, Obstacle, collides

# Lane the car starts in. Derived rather than hard-coded so the game still starts in
# the middle if the Lead ever changes NUM_LANES in src/config.py. With NUM_LANES = 3
# this is lane 1.
_START_LANE: int = NUM_LANES // 2

# y coordinate a freshly spawned row is placed at: fully above the visible window, so
# an obstacle never "appears" mid-screen. Also used by the tests to recognise the
# obstacles created during the most recent update() call.
_SPAWN_Y: float = -OBSTACLE_HEIGHT


class Gameplay:
    """The simulation of a single round: one car, falling obstacles, score and speed.

    Attributes:
        car: The player car. Its lane is the only thing the player controls.
        obstacles: Every obstacle currently in play, ordered oldest-first.
        score: Points accumulated this round. Grows at ``SCORE_PER_SECOND`` per
            second of *advancing* time (R2/GM.4).
        speed: Current fall speed of every obstacle, in px/s.
        elapsed: Seconds of active (non-paused) play since the last ``reset()``.
            Paused time is not counted, because ``update()`` simply is not called
            while the game is paused.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        """Create a round, ready to play.

        Args:
            rng: Random source for obstacle spawning. Pass a seeded
                ``random.Random`` to make a test deterministic. When omitted, a
                private generator seeded from ``src.config.SEED`` is used. We never
                use the module-level ``random`` functions: that global state is
                shared with every other library in the process, so a test could not
                rely on it.
        """
        self._rng: random.Random = rng if rng is not None else random.Random(SEED)
        # Declared here (not only in reset) so the attributes exist for type checkers
        # and so __init__ alone is enough to see the full shape of the object.
        self.car: Car = Car(lane=_START_LANE)
        self.obstacles: list[Obstacle] = []
        self.score: float = 0.0
        self.speed: float = BASE_SPEED
        self.elapsed: float = 0.0
        self.reset()

    def reset(self) -> None:
        """Return the round to its starting condition.

        The car goes back to the middle lane, the road is cleared, and score and
        speed go back to zero and ``BASE_SPEED``. The RNG is deliberately **not**
        reseeded: a restart should give a new obstacle pattern, not a replay of the
        one the player just lost to.
        """
        self.car = Car(lane=_START_LANE)
        self.obstacles = []
        self.score = 0.0
        self.speed = BASE_SPEED
        self.elapsed = 0.0

    def move_left(self) -> bool:
        """Move the car one lane to the left, if there is one.

        Returns:
            True if the lane actually changed, False if the car was already in the
            leftmost lane. The boolean matters upstream: ``GameController`` passes it
            on so Evan's latency logger only times actions that had a visible effect
            (docs/interfaces.md §4.4).
        """
        if self.car.lane <= 0:
            return False
        self.car.lane -= 1
        return True

    def move_right(self) -> bool:
        """Move the car one lane to the right, if there is one.

        Returns:
            True if the lane actually changed, False if the car was already in the
            rightmost lane.
        """
        if self.car.lane >= NUM_LANES - 1:
            return False
        self.car.lane += 1
        return True

    def update(self, dt: float, collisions_enabled: bool = True) -> bool:
        """Advance the simulation by ``dt`` seconds.

        The steps, in the order required by docs/interfaces.md §4.3:
        move obstacles, drop the ones that left the screen, spawn a new row if the
        road has travelled far enough, add score, then test for a collision.

        Args:
            dt: Elapsed time in seconds since the previous update. Clamped to
                ``MAX_DT``.
            collisions_enabled: When False, collisions are not reported at all. The
                controller passes False during the post-countdown grace period
                (docs/game-logic.md §5 rule 3).

        Returns:
            True if the car currently overlaps at least one obstacle and
            ``collisions_enabled`` is True. ``Gameplay`` does not act on this; the
            caller decides whether the round ends.

        Raises:
            ValueError: If ``dt`` is negative. Time running backwards is always a
                caller bug (a mis-ordered timestamp), and silently accepting it would
                let the score and the speed ramp go backwards too.
        """
        if dt < 0:
            raise ValueError(f"dt must be non-negative, got {dt!r}")

        # A long stall (window dragged, laptop resumed from sleep) would otherwise
        # move obstacles hundreds of pixels in one step and teleport them straight
        # through the car without ever overlapping it. Clamping costs a little
        # real-time accuracy and buys a collision test that cannot be skipped.
        dt = min(dt, MAX_DT)

        self.elapsed += dt
        # Difficulty ramp (GM.5): linear in time, hard-capped so the game stays
        # winnable. With the config values the cap is reached after ~72 s.
        self.speed = min(MAX_SPEED, BASE_SPEED + SPEED_INCREASE_PER_SECOND * self.elapsed)

        distance = self.speed * dt
        for obstacle in self.obstacles:
            obstacle.y += distance

        # Anything whose top edge has passed the bottom of the window can never
        # matter again. Dropping it keeps the list at ~4 items instead of growing
        # without bound over a long round.
        self.obstacles = [o for o in self.obstacles if o.y < WINDOW_HEIGHT]

        self._maybe_spawn_row()

        self.score += SCORE_PER_SECOND * dt

        if not collisions_enabled:
            return False
        return any(collides(self.car, o) for o in self.obstacles)

    def _maybe_spawn_row(self) -> None:
        """Spawn a new obstacle row once the previous one has travelled far enough.

        Spawning is driven by **distance, not time**. If it were time-based, the gap
        between rows would shrink as the speed ramps up and the game would become
        impossible rather than merely harder. Distance-based spawning keeps the road
        looking the same at every speed; the challenge comes from the rows arriving
        sooner, which is exactly the intended difficulty curve.
        """
        if self.obstacles:
            # Every obstacle falls at the same speed, so the row with the smallest y
            # is always the most recently spawned one.
            newest_y = min(o.y for o in self.obstacles)
            if newest_y < _SPAWN_Y + SPAWN_GAP_PX:
                return
        # Empty road (first update after a reset): spawn straight away. The row needs
        # about three seconds to reach the car, and the round always begins with a
        # COUNTDOWN state, so the player is never ambushed at the start.
        self._spawn_row()

    def _spawn_row(self) -> None:
        """Append one row of 1 or 2 obstacles in distinct random lanes.

        A row never occupies all ``NUM_LANES`` lanes. This is an **invariant, not a
        tuning choice**: with three lanes blocked the player would have no legal move
        and the round would be lost to the RNG rather than to their own reaction
        time. docs/workflow.md §9 requires a test that checks it over 10,000 spawns.
        """
        # Capped at NUM_LANES - 1 so the invariant above holds for any lane count.
        count = 2 if self._rng.random() < DOUBLE_OBSTACLE_PROB else 1
        count = min(count, NUM_LANES - 1)
        for lane in self._rng.sample(range(NUM_LANES), count):
            self.obstacles.append(Obstacle(lane=lane, y=_SPAWN_Y))
