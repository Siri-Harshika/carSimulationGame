# Evan - Live integration technical notes

## Summary for presentation

The integration layer connects camera frames, MediaPipe hand landmarks, shared
preprocessing, gesture prediction and the game controller. A time-based majority
vote stabilises predictions. A debouncer emits one command per held gesture,
with a separate steering cooldown. Missing hands pause play after smoothing;
a fist or V sign resumes and performs its movement. Camera/tracking work runs
in the background, while Pygame and game commands run on the main thread.
Simulated predictions allow these behaviours to be tested before the classifier
and complete game are available.

## Current scope and dependencies

Implemented: `src/app/smoothing.py`, `actions.py`, `worker.py`, `main.py`, and
`latency.py`. Existing mocks and the temporary GameController are used unchanged.
The real camera, tracker and preprocessing implementations can be selected now.
The real SVM implementation/model and Uy's gameplay/rendering/state machine
remain unfinished. Mock mode displays state, lane, predictions, FPS and last
changed action; it does not demonstrate obstacles, collisions or scoring.
Game-over palm restart is covered by mapping tests; actual collision-to-restart
play must be checked once the complete controller arrives.

## Pipeline and thread ownership

1. Camera supplies a BGR frame, capture timestamp and increasing frame index.
2. The worker creates its tracker on its own thread and detects a hand.
3. With a hand, the shared `preprocess()` produces 63 normalised coordinates and
   `classifier.predict()` supplies a label/confidence. Without a hand, prediction
   is bypassed and the raw token becomes `no_hand` with confidence zero.
4. Smoothing and debouncing turn raw tokens into discrete events.
5. An event queue transfers commands to the main thread, which selects an action
   from the current game state and applies it through GameController methods.

`none` means an unrecognised detected pose and causes no action. `no_hand` means
tracking found no hand and pauses PLAYING/COUNTDOWN. Keeping these distinct
prevents an uncertain pose from being treated as a missing hand.
Snapshots and FPS histories are protected with a lock. Queue operations are
thread-safe. Pygame keyboard state is copied on the main thread; mocks read that
copy. The preview copies the latest frame and shows camera pixels; it does not
create a second tracker or draw landmark overlays. The tracker closes on its
own thread. Shutdown signals the worker, closes the camera to wake reads, joins
the worker, closes the CSV and quits Pygame. Worker errors are reported and cause
an unsuccessful application exit rather than silent fallback.

## Prediction stabilisation

The defaults in `src/config.py` are a 0.25-second window, a 60% majority and a
minimum of three frames. Old samples are removed while retaining at least three
samples at low FPS. Before three samples arrive, stability stays at `no_hand`.
When no token wins 60%, the previous stable token remains (hysteresis).
The time window reduces differences between bright and dim camera frame rates.
A short missed detection is filtered; persistent missing hands eventually pause.

The debouncer emits only when the stable token changes. Holding a fist for five
seconds therefore produces one movement. To repeat, relax long enough for
`none` to become stable, then show the gesture again. Steering events must be
at least 0.30 seconds apart. A rejected edge is discarded, not queued for later.
Palm and missing-hand pause events bypass the steering cooldown. Constructors
reject invalid parameters and update methods reject backwards/nonfinite times.

## Actions and state transitions

Fist/V sign move left/right in PLAYING and COUNTDOWN. In PAUSED, they call
`resume()` first, then perform the movement. Palm and missing hands pause active
play. Palm in GAME_OVER restarts; other gestures in GAME_OVER do nothing.
`none` never produces an action. The temporary controller has no real countdown,
obstacles or collision system. The integration does not reach into Gameplay or
StateMachine; the finished GameController can replace the stub via the same API.

## Latency measurements

All stages use the process-wide `time.perf_counter()` clock. An event records the
start of the latest uninterrupted raw streak for its triggering token, the
trigger frame capture time, detection completion and prediction completion.
Main records action completion only when the action changes the game.

- `response_ms`: action completion minus raw-streak onset.
- `processing_ms`: action completion minus trigger-frame capture.
- `landmarks_ms`: detection completion minus trigger capture.
- `predict_ms`: prediction/preprocessing completion minus detection completion.
- `dispatch_ms`: action completion minus prediction completion (includes
  smoothing/debouncing execution, queue wait and the controller call).

CSV rows flush immediately. On exit, the console reports count and median,
p95, mean and maximum for every metric. Mock measurements validate the logger;
they must not be reported as real gesture-recognition performance. Sensor
exposure/driver buffering before capture timestamps and screen refresh after
controller calls are excluded. Final bright/dim measurements, tuning and
presentation plots still require the real classifier and complete game.

## Run and manually test

A local Python 3.12 environment has been created at `.venv` and dependencies
installed from `requirements.lock`. From the repository root, use:

```powershell
.\.venv\Scripts\python.exe -m src.app.main --classifier mock --camera mock --log-latency
```

Click the window so it receives keys. Hold each simulated gesture for at least
half a second, and release for at least half a second between repeated gestures.

1. Initial state should be PAUSED, lane 1.
2. Hold `1`: state becomes PLAYING (via the stub COUNTDOWN), lane becomes 0.
   Continue holding for several seconds: it must not move again.
3. Release, then hold `2`: lane becomes 1. Keep holding: lane remains 1.
4. Release and hold `2` again: lane becomes 2. Release/repeat: lane stays 2,
   demonstrating the boundary check.
5. Hold `3`: state becomes PAUSED. Hold it longer: no repeated commands.
6. Release, then hold `1`: resume and move left together.
7. Hold `H`: persistent simulated missing hand pauses. Release H: raw/stable
   `none` does not resume; show `1` or `2` to resume.
8. Arrow keys provide direct fallback steering; `P` pauses/resumes and `R`
   restarts. These bypass classifier smoothing and are not gesture-latency rows.
9. Press Esc or close the window. Check the printed latency summary and the
   timestamped CSV in `results/`.

To test actual webcam tracking with simulated classification:

```powershell
.\.venv\Scripts\python.exe -m src.app.main --classifier mock --log-latency
```

A real hand must be visible for keys 1/2/3 to classify. Remove it to test automatic
pause; H only simulates no hand with `--camera mock`. This mode verifies webcam
integration but does not recognise actual fist/V/palm poses; keys supply labels.

After Siri and Uy deliver their components:

```powershell
.\.venv\Scripts\python.exe -m src.app.main --log-latency
```

Then verify actual gestures, countdown/grace behaviour, collision/game-over palm
restart and gesture-only play in bright and dim light. Record at least 50 changed
actions per lighting condition before reporting real median/p95 response times.

## Automated verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Integration tests exercise all 20 mapping cells, resume-before-move ordering,
lane boundaries, smoothing/hysteresis at multiple FPS values, held gestures,
cooldown, missing-hand bypass, threaded tracker ownership, worker failures,
CSV metrics and the actual mock application loop. A bounded smoke run is:

```powershell
.\.venv\Scripts\python.exe -m src.app.main --classifier mock --camera mock --duration 3 --no-preview
```

Existing tests in other components may be empty or skipped, so a passing suite
does not prove the unfinished classifier or full game is complete.

## Verification on this checkout

Full pytest suite: 180 passed, 1 skipped. The skipped dataset test requires recordings in
`data/sample/`, which is currently empty. Dependency consistency passed (`pip check`). A headless mock application
run emitted one resume-and-left action for a held fist, wrote a latency CSV,
and exited cleanly. The rendered status display was visually inspected.
Actual webcam operation and real recognition were not exercised in this run.

Keyboard regression fix: Pygame key-state iteration uses SDL scancodes;
explicit key-code lookups preserve 1/2/3/H when copying the snapshot. Tests
cover this distinction, which a plain list of pressed keys does not reproduce.
