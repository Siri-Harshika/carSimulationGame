"""Evan: timestamp metrics and durable CSV output."""
import csv
import pytest
from src.app.actions import Action
from src.app.latency import ActionTiming, LatencyLogger


def test_latency_csv_summary_and_close(tmp_path):
    path = tmp_path / "latency.csv"
    logger = LatencyLogger(path)
    logger.log_action(ActionTiming(Action.MOVE_LEFT, "fist", 1, 1.15, 1.16, 1.17, 1.2))
    row = list(csv.DictReader(path.open()))[0]
    assert float(row["response_ms"]) == pytest.approx(200)
    assert float(row["processing_ms"]) == pytest.approx(50)
    assert logger.summary()["response_ms_p95"] == pytest.approx(200)
    logger.close()
    logger.close()
    with pytest.raises(RuntimeError):
        logger.log_action(ActionTiming(Action.PAUSE, "open_palm", 0, 0, 0, 0, 0))


def test_empty_and_invalid_timestamps():
    logger = LatencyLogger(None)
    assert logger.summary() == {"count": 0}
    with pytest.raises(ValueError):
        logger.log_action(ActionTiming(Action.PAUSE, "open_palm", 2, 1, 3, 4, 5))
