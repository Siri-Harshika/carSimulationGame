"""Recording tool: label key starts, same key/Space stops -> data/raw/*.csv.

Owner: Shobita (Dataset).
Spec: docs/interfaces.md section 4.5, docs/data-eval.md section 7.

Usage (from the repo root):
    python -m src.data.collect --user u03 --condition bright_near

Keys: 1 fist, 2 v_sign, 3 open_palm, 0 none, Space stop, Backspace undo, Q quit.
Only landmark numbers are saved, never images.
"""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import re
import time

import cv2

from src.capture.camera import Camera
from src.capture.landmarks import HandLandmarks, HandTracker
from src.config import (
    CONDITIONS,
    CSV_COLUMNS,
    DATA_DIR,
    LABELS,
    RECORD_SAMPLE_HZ,
    RECORD_START_DELAY_S,
    RECORD_TARGET_PER_LABEL,
)

KEY_SPACE = 32
KEY_BACKSPACE = (8, 127)          # Windows/Linux, macOS
KEY_QUIT = (ord("q"), ord("Q"), 27)
KEY_LABELS = {ord("1"): "fist", ord("2"): "v_sign", ord("3"): "open_palm", ord("0"): "none"}
WINDOW = "Recorder"


def key_to_label(key: int) -> str | None:
    """Return the label for a waitKey code ('1', '2', '3', '0'), or None."""
    return KEY_LABELS.get(key)


def landmarks_to_row(user_id: str, condition: str, label: str,
                     hand: HandLandmarks) -> list[str | float]:
    """Return one CSV row: user_id, condition, label, f0..f62 (raw x, y, z), handedness."""
    return [user_id, condition, label, *map(float, hand.points.reshape(-1)), hand.handedness]


class RecordingSession:
    """Start/stop recording logic without OpenCV, so it can be unit tested.

    A "burst" is everything recorded between one start and the next stop.
    """

    def __init__(self, user_id: str, condition: str,
                 target_per_label: int = RECORD_TARGET_PER_LABEL,
                 sample_hz: float = RECORD_SAMPLE_HZ,
                 start_delay_s: float = RECORD_START_DELAY_S) -> None:
        self.user_id = user_id
        self.condition = condition
        self.target_per_label = target_per_label
        self.period = 1.0 / sample_hz
        self.start_delay_s = start_delay_s
        self.active_label: str | None = None
        self.counts: dict[str, int] = {label: 0 for label in LABELS}
        self.rows: list[list[str | float]] = []
        self.bursts: list[tuple[str, int]] = []   # (label, index of first row)
        self.ready_at = 0.0
        self.next_sample_at = 0.0

    def is_getting_ready(self, now: float) -> bool:
        """True during the start delay after a label key was pressed."""
        return self.active_label is not None and now < self.ready_at

    def handle_key(self, key: int, now: float) -> str:
        """Start, stop or undo depending on the key. Returns a message for the screen."""
        label = key_to_label(key)

        if self.active_label is not None:
            if key == KEY_SPACE or label == self.active_label:
                return self.stop()
            if label is not None:
                # Never switch label mid-burst, a stray key would mislabel samples.
                return f"Ignored {label}, stop {self.active_label} first"
            return ""

        if key in KEY_BACKSPACE:
            return f"Undid {self.undo_last_burst()} rows"
        if label is None:
            return ""
        if self.counts[label] >= self.target_per_label:
            return f"{label} already has {self.target_per_label} rows"

        self.active_label = label
        self.ready_at = now + self.start_delay_s
        self.next_sample_at = self.ready_at
        self.bursts.append((label, len(self.rows)))
        return f"GET READY: {label}"

    def offer(self, hand: HandLandmarks | None, now: float) -> bool:
        """Record the hand if recording is on and enough time has passed. Returns True if kept."""
        if self.active_label is None or hand is None or now < self.next_sample_at:
            return False

        self.rows.append(landmarks_to_row(self.user_id, self.condition, self.active_label, hand))
        self.counts[self.active_label] += 1
        # Step the schedule by one period to keep ~10 rows/s, but never schedule the
        # next sample right after a camera stall (that would give near-duplicates).
        self.next_sample_at = max(self.next_sample_at + self.period, now + self.period / 2)

        if self.counts[self.active_label] >= self.target_per_label:
            self.stop()
        return True

    def stop(self) -> str:
        """Stop the current burst."""
        label, start = self.bursts[-1]
        kept = len(self.rows) - start
        if kept == 0:
            self.bursts.pop()   # nothing to undo later
        self.active_label = None
        return f"Stopped {label}: {kept} rows"

    def undo_last_burst(self) -> int:
        """Delete the last finished burst. Returns the number of rows removed."""
        if self.active_label is not None or not self.bursts:
            return 0
        label, start = self.bursts.pop()
        removed = len(self.rows) - start
        del self.rows[start:]
        self.counts[label] -= removed
        return removed


def save_rows(path: Path, rows: list[list[str | float]]) -> None:
    """Write the header and all rows to path (floats with 6 decimals)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first so a crash never leaves a half-written CSV.
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in row])
    os.replace(tmp, path)


def draw_status(image, session: RecordingSession, hand_found: bool, message: str) -> None:
    """Draw the counts and recording state on the preview image."""
    def text(s: str, y: int, color=(255, 255, 255), size: float = 0.5) -> None:
        cv2.putText(image, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)

    text(f"{session.user_id} {session.condition}", 20)
    text("  ".join(f"{k} {v}/{session.target_per_label}" for k, v in session.counts.items()), 45)
    if session.active_label is not None:
        if session.is_getting_ready(time.perf_counter()):
            text(f"GET READY: {session.active_label}", 90, (0, 165, 255), 1.0)
        else:
            n = session.counts[session.active_label]
            text(f"REC {session.active_label} {n}/{session.target_per_label}", 90, (0, 0, 255), 1.0)
    if not hand_found:
        text("NO HAND", 130, (0, 0, 255), 0.9)
    text(message, image.shape[0] - 40, (0, 255, 0))
    text("1 fist  2 v_sign  3 open_palm  0 none | Space stop | Backspace undo | Q quit",
         image.shape[0] - 15, size=0.45)


def main(argv: list[str] | None = None) -> None:
    """CLI: --user uXX --condition <lighting>_<distance> [--target 80] [--out-dir data/raw]."""
    parser = argparse.ArgumentParser(description="Record hand landmarks to CSV.")
    parser.add_argument("--user", required=True)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--target", type=int, default=RECORD_TARGET_PER_LABEL)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"u\d{2}", args.user):
        parser.error("--user must look like u03")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out_dir / f"{args.user}_{args.condition}_{stamp}.csv"
    session = RecordingSession(args.user, args.condition, target_per_label=args.target)
    saved = 0
    message = "Press 1/2/3/0 to start"

    tracker = HandTracker()
    try:
        with Camera() as camera:
            while True:
                frame = camera.read()
                hand = tracker.detect(frame.image)
                session.offer(hand, frame.t_capture)

                preview = frame.image.copy()   # don't draw on the shared frame
                if hand is not None:
                    tracker.draw(preview, hand)
                draw_status(preview, session, hand is not None, message)
                cv2.imshow(WINDOW, preview)

                key = cv2.waitKey(1) & 0xFF
                if key in KEY_QUIT or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key != 0xFF:
                    message = session.handle_key(key, time.perf_counter()) or message

                # Save after every stop or undo, so a crash loses at most one burst.
                if session.active_label is None and len(session.rows) != saved:
                    save_rows(out_path, session.rows)
                    saved = len(session.rows)
    finally:
        if session.active_label is not None:
            session.stop()
        if len(session.rows) != saved:
            save_rows(out_path, session.rows)
        tracker.close()
        cv2.destroyAllWindows()

    if session.rows:
        print(f"Saved {len(session.rows)} rows to {out_path}: {session.counts}")
    else:
        print("Nothing recorded.")


if __name__ == "__main__":
    main()
