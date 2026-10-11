# Shobita – Dataset collection and preparation — Technical notes
Author: Shobita   |   Files owned: `src/data/collect.py`, `src/data/dataset.py`, `src/data/split.py`, `tests/test_collect.py`, `tests/test_dataset.py`, `tests/test_split.py`, `data/`   |   Requirement IDs: DS.1–DS.5, I2, R5, R10

## 1. What it does
I built the tool we used to record hand gestures, recorded everyone's data, cleaned it, and split it into training and test sets for Siri's classifier. `collect.py` opens the webcam, finds the hand with Triet's `HandTracker` and saves the 21 hand landmarks (63 numbers) to a CSV file while a label key is pressed. `dataset.py` loads the CSVs, removes bad rows, runs Triet's `preprocess()` and writes a quality report. `split.py` divides the data **by user** into train and test sets and saves `data/splits.json`. In the pipeline ([DESIGN.md §2](../DESIGN.md#2-system-architecture)) this sits between hand tracking (Triet) and the classifier (Siri).

## 2. Design decisions
| Decision | Alternatives considered | Why we chose it |
|---|---|---|
| Save landmarks, not images | Save webcam images (as in the anantSinghCross project) | Smaller files, no privacy problem, and the classifier only needs the 63 numbers |
| Save **raw** landmarks | Save preprocessed features (as in the thinkall project) | If `preprocess()` changes, we only retrain; nobody has to record again |
| Press a key to start, press again (or Space) to stop | Hold the key down | OpenCV only reports key presses, not held keys |
| 1 s "GET READY" delay, max 10 rows/s, auto-stop at 80 | Record every frame | The hand is in position before recording starts, and rows are not near-copies of each other |
| A different label key is ignored while recording | Switch label immediately | A wrong key press can never mislabel samples |
| Save the file after every stop, with Backspace to undo the last burst | Save once at the end | A crash loses at most one burst, and mistakes are easy to fix |
| Clean the data every time it is loaded, never edit the raw files | Edit or delete bad rows by hand | Raw data stays untouched and every dropped row is counted by reason |
| Split by **user** (whole users in train or test) | Random split of rows | A random split puts the same person in train and test, so the test score would be too optimistic |
| 6 team members + 1 outside volunteer (`u07`) | Team members only | `u07` is an extra unseen person in the test set and is never used for training |

## 3. Interfaces provided / consumed
Provided (as in [interfaces.md §4.5](../interfaces.md#45-dataset--classifier--srcdata-shobita-srcmodel-siri)):
```python
# collect.py
key_to_label(key: int) -> str | None
landmarks_to_row(user_id, condition, label, hand: HandLandmarks) -> list[str | float]
class RecordingSession: handle_key(key, now) -> str; offer(hand, now) -> bool; undo_last_burst() -> int
# dataset.py
load_dataset(paths=None, excluded_files=None) -> tuple[Dataset, CleaningReport]
quality_report(dataset, cleaning, split_users=None) -> str
# split.py
make_splits(data_dir, test_users=None, n_test_users=2, seed=218, excluded_files=None) -> Splits
save_splits(splits, path); load_splits(path) -> Splits; load_split("train" | "test", path) -> Dataset
```
CSV columns: `user_id, condition, label, f0 … f62, handedness`.

Consumed: `Camera` (`src/capture/camera.py`), `HandTracker` and `HandLandmarks` (`src/capture/landmarks.py`), `preprocess()` and `row_to_landmarks()` (`src/capture/preprocess.py`) from Triet.

Small additions:
- `save_rows()` writes a session CSV. The tests use it too.
- `parse_file_name()` reads the user and condition from a file name. `split.py` uses it, so it never has to open the files.
- `quality_report()` takes an optional `split_users` to show the label balance per split.
- `Splits` has a `seed` field so the seed is stored in `splits.json`.

## 4. Challenges and how we solved them
- **Repeated samples.** Recording every frame (30 FPS) gives almost identical rows. → I limited recording to 10 rows per second and asked recorders to keep moving their hand slightly. → The `duplicate` rule dropped 0 rows.
- **False starts while recording.** One session (`u06` dim_far) was started twice, so the first file had only 32 `fist` rows. The quality report flagged it. → I deleted the incomplete file and kept the full re-recording. → Every cell has 80 rows.
- **Keeping test users unseen.** → `make_splits` and `load_splits` raise an error if a user or file appears in both splits, and the tests check this (`tests/test_split.py`).
- **Dim light.** MediaPipe still finds the hand, but the landmarks are noisier and the camera FPS drops. → We recorded `dim_near` and `dim_far` for every user, so the model learns noisy hands and we can compare accuracy between bright and dim light.

## 5. Results
- **[Must] Quality report:** `results/dataset_report.txt`.
  - 8,960 rows read, 8,960 kept, 0 dropped by any cleaning rule.
  - 7 users × 4 core conditions × 4 labels × 80 rows = 112 sets of  labels, **0 cells below 60**.
  - Labels are perfectly balanced: 2,240 rows each.
- **[Must] Split:** `data/splits.json`. The test users were named with `--test`.

  | Split | Users | Rows |
  |---|---|---|
  | Train (LOUO validation inside) | u01, u02, u03, u04 | 5,120 (1,280 per label) |
  | Test (never trained on) | u05, u06 + volunteer u07 | 3,840 (960 per label) |

- **Conditions:** `bright_near`, `bright_far`, `dim_near` and `dim_far` for every user. No `backlit` data was recorded ([Nice]).
- **Hands:** 6,751 right-hand rows and 2,209 left-hand rows. Handedness is the label MediaPipe gives.
- **Tests:** `tests/test_collect.py`, `test_dataset.py` and `test_split.py`: 39 passed, 1 skipped (the `data/sample/` check, skipped because no sample data is committed).

## 6. Limitations & future work
- Only 7 people, all recorded with one laptop webcam, so the data may not cover other cameras sizes well.
- The plan asked for half of the `bright_near` samples to use the other hand. Only 346 of 2,240 `bright_near` rows are left-hand, so the left hand is under-represented there.
- `none` covers only the poses we thought of. Live play-testing may find other poses that trigger actions by mistake, which would need more `none` data from train users.
- No `backlit` stress-test data and no sample dataset (`data/sample/`) yet.

## 7. Slide summary (will add more)
**Text:** "We recorded 8,960 labelled hand-landmark samples from 7 people (6 team members + 1 unseen volunteer) in bright/dim light at near/far distance. The data is split by user: 4 train users, 3 held-out test users. No images are stored."
**Figure:** the counts table from `results/dataset_report.txt`.

---

## How to record (guide for the team)
1. Activate the environment: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS).
2. Run one command per condition with your user ID:
   ```
   python -m src.data.collect --user u03 --condition bright_near
   ```
   Conditions: `bright_near`, `bright_far`, `dim_near`, `dim_far`.
3. Press **1** fist, **2** V sign, **3** open palm or **0** none. Wait for "GET READY", then hold the gesture and keep moving your hand slightly. Recording stops by itself at 80 rows; **Space** stops early.
4. For `none`, use any other pose: thumbs up, pointing, OK sign, relaxed hand, hand partly out of frame.
5. **Backspace** deletes the last burst. **Q** saves and quits.
6. Check the data with `python -m src.data.dataset --report`.

## Recording conditions
| Condition | Setup |
|---|---|
| `bright` | Room lights on plus daylight or a desk lamp in front of the user |
| `dim` | Main lights off, only the monitor or one lamp; the face is still visible but grainy |
| `near` | Hand about 40 cm from the camera |
| `far` | Hand about 100 cm from the camera |

## References
- Project design docs: [data-eval.md §7](../data-eval.md#7-dataset-plan), [interfaces.md §4.5](../interfaces.md#45-dataset--classifier--srcdata-shobita-srcmodel-siri).
- anantSinghCross, *realtime-hand-gesture-recognition*, `captureHand.py`. Captures gesture training data with OpenCV, but saves thresholded hand **images**; we save landmarks instead. https://github.com/anantSinghCross/realtime-hand-gesture-recognition/blob/master/captureHand.py
- thinkall, *hand-gesture-recognition-mediapipe*. Records MediaPipe keypoints to a CSV by pressing number keys, the same idea as our label keys. https://github.com/thinkall/hand-gesture-recognition-mediapipe
- haneure, *Hand-Tracking-Capture*, `main.py`. A webcam loop with MediaPipe that saves hand landmarks to CSV. https://github.com/haneure/Hand-Tracking-Capture/blob/master/main.py
- gabang2, MediaPipe hand gesture gist. Recognises gestures from the 21 MediaPipe hand landmarks. https://gist.github.com/gabang2/81ab1ac27bc4e1e41a8756a0b2b9a2b6
- MediaPipe Hands documentation: https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
- OpenCV `waitKey`, `imshow` and `putText`: https://docs.opencv.org/4.x/
- Python `csv`, `argparse` and `random` modules: https://docs.python.org/3.12/library/
