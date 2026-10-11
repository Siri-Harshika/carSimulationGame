"""Prepare the same landmark features for model training and live prediction.

Owner: Triet (Hand tracking).
Provides: preprocess(), row_to_landmarks() and PREPROCESS_VERSION.
Consumes: HandLandmarks from detection or raw CSV rows; never accesses a camera.
Contract: docs/interfaces.md section 4.2 and docs/game-logic.md section 6.1.
"""

from collections.abc import Sequence

import numpy as np

from src.capture.landmarks import HandLandmarks
from src.config import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MIDDLE_MCP_IDX,
    MIN_PALM_SIZE,
    NUM_COORDS,
    NUM_FEATURES,
    NUM_LANDMARKS,
    PREPROCESS_VERSION,
    WRIST_IDX,
)

"""
1: Validate: check the coordinate shape, values and handedness
2: Correct image proportions: adjust x and z using the image width to height ratio
3: Centre on the wrist: subtract the wrist coordinates so the write becomes (0,0,0)
4: Mirror left-hand coordinates: reverse their x values so both hands use a consistent orientation.
5: divide coordinates by the wrist-to-middle-finger-base distance measured in x and y.
6: Flatten: turn the (21,3) coordinate array into 63 numbers
"""


def preprocess(landmarks: HandLandmarks) -> np.ndarray:
    """
    Validates and normalises 21 hand landmarks, then returns a float32 array containing 63 features. 
    Leaves the original coordinates unchanged.
    """
    if landmarks.handedness not in ("Left", "Right"):
        raise ValueError('handedness must be "Left" or "Right".')
    try:
        points = np.array(landmarks.points, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Landmark coordinates must be numeric.") from exc
    if points.shape != (NUM_LANDMARKS, NUM_COORDS):
        raise ValueError(f"Expected landmark shape (21, 3), got {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError("Landmark coordinates must be finite; remove NaN/inf samples.")

    # MediaPipe normalizes x by width and y by height. z uses x's scale too.
    # Converting x/z to height units avoids distorting a 640x480 hand shape.
    with np.errstate(over="ignore", invalid="ignore"):
        points[:, [0, 2]] *= FRAME_WIDTH / FRAME_HEIGHT
        points -= points[WRIST_IDX].copy()
        if landmarks.handedness == "Left":
            points[:, 0] *= -1

        # Use x-y palm length: finger extension varies by gesture and z is noisy.
        palm = float(np.linalg.norm(points[MIDDLE_MCP_IDX, :2]))
        if not np.isfinite(points).all() or not np.isfinite(palm):
            raise ValueError("Landmark magnitude is too large to normalize safely.")
        if palm < MIN_PALM_SIZE:
            raise ValueError("Degenerate hand: wrist and middle-finger MCP are too close.")
        points /= palm
        features = points.reshape(NUM_FEATURES).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Normalized landmark values exceed the float32 range.")
    # Rotation is deliberately retained (spec section 6.1). Changing this
    # algorithm later requires a PREPROCESS_VERSION bump and model retraining.
    return features


def row_to_landmarks(features_raw: Sequence[float], handedness: str) -> HandLandmarks:
    """
    Takes 63 raw coordinate values already loaded from a CSV, plus a Left/Right label, and rebuilds a HandLandmarks object.
    """
    if handedness not in ("Left", "Right"):
        raise ValueError('handedness must be "Left" or "Right".')
    try:
        features = np.array(features_raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV landmark fields must be numeric.") from exc
    if features.shape != (NUM_FEATURES,) or not np.isfinite(features).all():
        raise ValueError("Expected exactly 63 finite raw CSV landmark values.")
    return HandLandmarks(
        points=features.reshape(NUM_LANDMARKS, NUM_COORDS),
        handedness=handedness,
        score=1.0,
    )
