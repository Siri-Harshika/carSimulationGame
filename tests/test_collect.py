"""Check the recording tool's pure logic (DS.1) without a webcam.

Owner: Shobita. Covers the Shobita test_collect.py row of docs/workflow.md section 9.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from src.capture.landmarks import HandLandmarks
from src.config import CSV_COLUMNS, RECORD_SAMPLE_HZ, RECORD_START_DELAY_S
from src.data.collect import (
    KEY_BACKSPACE,
    KEY_SPACE,
    RecordingSession,
    key_to_label,
    landmarks_to_row,
    main,
    save_rows,
)

KEY_FIST, KEY_V, KEY_PALM, KEY_NONE = ord("1"), ord("2"), ord("3"), ord("0")
FRAME_DT = 1 / 30   # a 30 FPS camera


@pytest.fixture
def hand() -> HandLandmarks:
    """Use a repeatable nondegenerate raw hand."""
    points = np.random.default_rng(218).uniform(0.2, 0.8, size=(21, 3))
    return HandLandmarks(points, "Right", 0.95)


def feed(session: RecordingSession, hand: HandLandmarks | None, start: float,
         seconds: float) -> float:
    """Offer 30 FPS frames for ``seconds`` starting at ``start``; return the end time."""
    t = start
    for _ in range(round(seconds / FRAME_DT)):
        session.offer(hand, t)
        t += FRAME_DT
    return t


def test_key_to_label() -> None:
    assert key_to_label(KEY_FIST) == "fist"
    assert key_to_label(KEY_V) == "v_sign"
    assert key_to_label(KEY_PALM) == "open_palm"
    assert key_to_label(KEY_NONE) == "none"
    assert key_to_label(ord("x")) is None
    assert key_to_label(KEY_SPACE) is None


def test_landmarks_to_row_order(hand: HandLandmarks) -> None:
    row = landmarks_to_row("u03", "dim_far", "fist", hand)
    assert len(row) == len(CSV_COLUMNS) == 67
    assert row[:3] == ["u03", "dim_far", "fist"]
    assert row[-1] == "Right"
    # f(3i)=x_i, f(3i+1)=y_i, f(3i+2)=z_i
    np.testing.assert_allclose(row[3:-1], hand.points.reshape(-1))


def test_label_key_starts_after_delay(hand: HandLandmarks) -> None:
    session = RecordingSession("u01", "bright_near")
    assert "GET READY" in session.handle_key(KEY_FIST, now=0.0)
    assert session.active_label == "fist"
    assert session.is_getting_ready(RECORD_START_DELAY_S / 2)
    feed(session, hand, 0.0, RECORD_START_DELAY_S - 0.05)
    assert session.rows == []                # nothing during GET READY
    assert session.offer(hand, RECORD_START_DELAY_S)
    assert session.counts["fist"] == 1


def test_same_key_stops(hand: HandLandmarks) -> None:
    session = RecordingSession("u01", "bright_near", start_delay_s=0)
    session.handle_key(KEY_FIST, 0.0)
    end = feed(session, hand, 0.0, 1.0)
    assert "Stopped" in session.handle_key(KEY_FIST, end)
    assert session.active_label is None
    n = len(session.rows)
    feed(session, hand, end, 1.0)
    assert len(session.rows) == n           # stopped → no more rows


def test_space_stops(hand: HandLandmarks) -> None:
    session = RecordingSession("u01", "bright_near", start_delay_s=0)
    session.handle_key(KEY_V, 0.0)
    session.handle_key(KEY_SPACE, 0.5)
    assert session.active_label is None


def test_different_label_key_is_ignored(hand: HandLandmarks) -> None:
    session = RecordingSession("u01", "bright_near", start_delay_s=0)
    session.handle_key(KEY_FIST, 0.0)
    message = session.handle_key(KEY_V, 0.2)
    assert message.startswith("Ignored")
    assert session.active_label == "fist"
    feed(session, hand, 0.0, 1.0)
    assert {row[2] for row in session.rows} == {"fist"}


def test_auto_stop_at_target(hand: HandLandmarks) -> None:
    session = RecordingSession("u01", "bright_near", target_per_label=5, start_delay_s=0)
    session.handle_key(KEY_PALM, 0.0)
    feed(session, hand, 0.0, 3.0)
    assert session.counts["open_palm"] == 5
    assert session.active_label is None
    # A full label cannot be started again (Backspace undoes instead).
    assert "already has" in session.handle_key(KEY_PALM, 4.0)
    assert session.active_label is None


def test_at_most_sample_hz_rows_per_second(hand: HandLandmarks) -> None:
    session = RecordingSession("u01", "bright_near", start_delay_s=0)
    session.handle_key(KEY_NONE, 0.0)
    feed(session, hand, 0.0, 4.0)
    assert len(session.rows) <= 4 * RECORD_SAMPLE_HZ
    assert len(session.rows) >= 4 * RECORD_SAMPLE_HZ - 1   # and not much fewer


def test_offer_none_adds_nothing() -> None:
    session = RecordingSession("u01", "bright_near", start_delay_s=0)
    session.handle_key(KEY_FIST, 0.0)
    assert not session.offer(None, 0.5)
    assert session.rows == [] and session.counts["fist"] == 0


def test_undo_removes_exactly_last_burst(hand: HandLandmarks) -> None:
    session = RecordingSession("u01", "bright_near", start_delay_s=0)
    session.handle_key(KEY_FIST, 0.0)
    t = feed(session, hand, 0.0, 1.0)
    session.handle_key(KEY_FIST, t)
    first = len(session.rows)
    session.handle_key(KEY_V, t)
    t = feed(session, hand, t, 0.5)
    session.handle_key(KEY_SPACE, t)
    second = len(session.rows) - first
    assert second > 0
    assert session.undo_last_burst() == second
    assert len(session.rows) == first
    assert session.counts["v_sign"] == 0 and session.counts["fist"] == first
    # Backspace (key) undoes the next burst back; then there is nothing left.
    assert "Undid" in session.handle_key(KEY_BACKSPACE[0], t)
    assert session.rows == []
    assert session.undo_last_burst() == 0


def test_save_rows_writes_header_and_6_decimals(tmp_path: Path, hand: HandLandmarks) -> None:
    path = tmp_path / "u01_bright_near_20261014-161502.csv"
    save_rows(path, [landmarks_to_row("u01", "bright_near", "fist", hand)])
    with path.open(newline="", encoding="utf-8") as handle:
        header, row = list(csv.reader(handle))
    assert tuple(header) == CSV_COLUMNS
    assert len(row[3].split(".")[1]) == 6


@pytest.mark.parametrize("argv", [
    ["--user", "bob", "--condition", "bright_near"],
    ["--user", "u01", "--condition", "sunny_near"],
])
def test_cli_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
