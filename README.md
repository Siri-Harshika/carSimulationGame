# carSimulationGame

Gesture-controlled car dodging game (CSCI218 group project). A webcam + MediaPipe
reads your hand, an SVM classifies the gesture, and the car changes lanes:

| Gesture | Playing | Paused | Game over |
|---|---|---|---|
| Fist | move left | resume + move left | – |
| V sign | move right | resume + move right | – |
| Open palm | pause | – | restart |
| No hand | pause | – | – |

The full design is in [`docs/DESIGN.md`](docs/DESIGN.md). Interfaces are in
[`docs/interfaces.md`](docs/interfaces.md), and the team workflow is in
[`docs/workflow.md`](docs/workflow.md).

## Setup

You need **Python 3.12, 64-bit**. Check it with
`python -c "import struct,sys; print(struct.calcsize('P')*8, sys.version)"`.

```bash
py -3.12 -m venv .venv          # Windows   |  python3.12 -m venv .venv   (macOS/Linux)
.venv\Scripts\activate           # Windows   |  source .venv/bin/activate
pip install -r requirements.lock
python -c "import mediapipe as mp, cv2; print(mp.__version__, cv2.__version__)"   # expect 0.10.21 4.11.0
```

Always install from `requirements.lock`, not `requirements.txt`, and never
`pip install opencv-python`: it conflicts with the `opencv-contrib-python` that
MediaPipe uses. See [docs/interfaces.md §3.1](docs/interfaces.md#31-requirementstxt-and-requirementslock).

## How to run

Always run from the repository root, so that `src` is importable.

| Command | What it does | Owner |
|---|---|---|
| `python -m src.app.main` | Full game, real webcam, real SVM | Evan |
| `python -m src.app.main --classifier mock` | Real webcam + MediaPipe, gestures typed on keys 1/2/3 | Evan |
| `python -m src.app.main --classifier mock --camera mock` | No webcam at all (pure keyboard dev mode) | Evan |
| `python -m src.app.main --log-latency` | Same as default, plus writes `results/latency_<timestamp>.csv` | Evan |
| `python -m src.game.keyboard_demo` | Game only, arrow keys | Uy |
| `python -m src.data.collect --user u03 --condition bright_near` | Recording tool | Shobita |
| `python -m src.data.dataset --report` | Cleaning + quality report → `results/dataset_report.txt` | Shobita |
| `python -m src.data.split --test u02 u05` | Write the user-based split `data/splits.json` | Shobita |
| `python -m src.model.train` | Tune on train users (LOUO), train, save `models/gesture_svm.joblib` | Siri |
| `python -m src.model.evaluate` | One-time evaluation on the test users, writes `results/` | Siri |
| `pytest` | All unit tests (headless, no webcam or model needed) | everyone |

## Contributing

- Never push to `main`. Work on your personal branch (`lead-setup`, `triet-tracking`,
  `uy-game`, `shobita-data`, `siri-classifier`, `evan-integration`) and open a PR.
  The Lead merges it once `pytest` passes.
- Commit messages follow Conventional Commits, e.g. `feat(game): add lane change`.
- Every public signature is frozen in [`docs/interfaces.md`](docs/interfaces.md).
  To change an interface, open a PR against the doc first.
- Stub modules raise `NotImplementedError`. Replace the bodies, but keep the signatures.

Full rules: [docs/workflow.md §12–13](docs/workflow.md#12-git-workflow).

## Integration development (Evan)

The integration layer is implemented. The real classifier and full game remain
unfinished, so start with the existing mocks and temporary controller:

```powershell
.\.venv\Scripts\python.exe -m src.app.main --classifier mock --camera mock --log-latency
```

Hold 1/2/3 for fist/V sign/palm, or H for no hand. Release between repeated
commands. The status panel shows state, lane, predictions and FPS. There are
currently no obstacles or collisions. See [Evan's technical notes](docs/notes/evan.md)
for the manual test checklist, architecture, latency definitions and limitations.
