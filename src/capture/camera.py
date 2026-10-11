"""Capture timestamped webcam frames without building up a queue of old images.

Owner: Triet (Hand tracking).
Provides: Frame and Camera for Shobita's collector and Evan's live worker/preview.
Consumes: OpenCV and camera constants from src.config.
Contract: docs/interfaces.md section 4.2; tests: docs/workflow.md section 9.
"""

from collections import deque
from dataclasses import dataclass
import math
import threading
import time

import cv2
import numpy as np

from src.config import (
    CAMERA_FPS,
    CAMERA_INDEX,
    CAMERA_MAX_FAILED_READS,
    CAMERA_READ_TIMEOUT_S,
    CAMERA_THREADED,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MIRROR_FRAME,
)


"""
A thread lets the program carry out another task alongside its main processing loop.
While the main program detects a hand in one image, the camera thread continues capturing newer images.
1: __init__() starts _capture_loop(), which repeatedly calls _capture_frame() 
followed by _publish_frame() to stores the newest fram updates timestamp history.
Both threads access shared information, such as self._latest_frame. 
A lock allows only one thread at a time to access the protected information.
2: Other code calls read() or latest() to obtain the stored frame. 
The fps property uses its timestamp history.
3: close() stops background capture, and the worker calls _release_capture() to prevent repeated clenup. 
A with block triggers this automatically through __exit__().
4: With background capture disabled, read() uses _read_sync() to capture and publish a frame directly.
"""

@dataclass(frozen=True)
class Frame:
    """Bundle one image with its capture time and sequence number.

    Attributes:
        image: BGR uint8 pixels, shape (height, width, 3), resized and mirrored.
        t_capture: perf_counter() time in seconds, immediately after device.read().
        index: Zero-based successful-capture number; gaps mean skipped frames.

    Frozen fields prevent reassignment, but NumPy pixels remain mutable. Preview
    code must copy image before drawing so it does not change a shared frame.
    """

    image: np.ndarray
    t_capture: float
    index: int


class Camera:
    """Share the newest webcam frame safely between capture, processing and UI.

    Background capture is the default. read() consumes each index at most once
    across callers; latest() observes the saved frame without consuming it.
    Synchronous mode is for debugging: a blocking driver call cannot be forcibly
    interrupted by Python, so its deadline is checked before and after capture.
    """

    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        mirror: bool = MIRROR_FRAME,
        threaded: bool = CAMERA_THREADED, #A thread lets the program carry out another task alongside its main processing loop.
    ) -> None:
        """Open the device and optionally start its background capture worker.

        Args:
            index: Nonnegative OpenCV device index.
            width: Positive output width in pixels.
            height: Positive output height in pixels.
            mirror: Whether to flip horizontally for a selfie-style preview.
            threaded: Whether a daemon worker captures continuously.

        Raises:
            TypeError: An option has the wrong type.
            ValueError: A device index or dimension is invalid.
            RuntimeError: The camera cannot be opened/configured or started.
        """
        for name, value in (("index", index), ("width", width), ("height", height)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer.")
        if index < 0 or width <= 0 or height <= 0:
            raise ValueError("Camera index must be >= 0 and dimensions must be > 0.")
        if not isinstance(mirror, bool) or not isinstance(threaded, bool):
            raise TypeError("mirror and threaded must be bool values.")

        self._camera_index = index
        self._width = width
        self._height = height
        self._mirror = mirror
        self._threaded = threaded
        self._latest_frame: Frame | None = None
        self._next_index = 0
        self._last_read_index = -1  # No index has been delivered, including zero.
        self._failed_reads = 0
        self._error: RuntimeError | None = None
        self._closed = False
        self._released = False
        self._capture_times: deque[float] = deque()

        # Shared metadata uses a Condition; slow synchronous device I/O uses a
        # different lock so latest() never waits for that I/O to finish.
        self._condition = threading.Condition()
        self._capture_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        try:
            self._cap = cv2.VideoCapture(index)
        except cv2.error as exc:
            raise RuntimeError(f"Cannot open camera index {index}: {exc}") from exc
        try:
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Cannot open camera index {index} – "
                    "close other apps using the webcam or set CAMERA_INDEX"
                )
            # These are requests to the driver. _capture_frame() enforces size,
            # while fps measures the real rate rather than trusting the request.
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            if threaded:
                self._thread = threading.Thread(
                    target=self._capture_loop, name="CameraCapture", daemon=True
                )
                self._thread.start()
        except Exception as exc:
            self._cap.release()
            self._released = True
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"Cannot configure/start camera index {index}: {exc}") from exc

    def read(self, timeout: float = CAMERA_READ_TIMEOUT_S) -> Frame:
        """Return a frame newer than the last one delivered by read().
            Returns a frame newer than the last delivered one, waiting if necessary.
        """
        if not math.isfinite(timeout) or timeout <= 0: #Timeout
            raise ValueError("timeout must be a finite, positive number.")
        deadline = time.perf_counter() + timeout
        if not self._threaded:
            return self._read_sync(deadline)

        with self._condition:
            while True:
                self._check_available() #Check if the frame is available
                frame = self._latest_frame #Check the latest frame
                if frame is not None and frame.index > self._last_read_index:
                    self._last_read_index = frame.index 
                    return frame
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise RuntimeError("Timed out waiting for a new frame – check the webcam.")
                # wait releases the lock, so capture can publish. A wake-up can
                # also mean shutdown/error, which is why every check is repeated.
                self._condition.wait(timeout=remaining)

    def latest(self) -> Frame | None:
        """Return the newest saved frame, or None before the first capture.

        This briefly locks metadata but never waits for a fresh camera frame.
        It can return the same index repeatedly and does not consume read().
        """
        with self._condition:
            return self._latest_frame #Return the latest frame

    @property
    def fps(self) -> float:
        """Return measured capture FPS from intervals within the last second.

        Returns:
            Zero until at least two recent timestamps exist, otherwise the
            number of intervals divided by their elapsed seconds.
        """
        with self._condition:
            self._trim_capture_times(time.perf_counter())
            if len(self._capture_times) < 2:
                return 0.0
            elapsed = self._capture_times[-1] - self._capture_times[0]
            return (len(self._capture_times) - 1) / elapsed if elapsed > 0 else 0.0

    def close(self) -> None:
        """Signal shutdown, wake readers and release the camera; repeatable.

        Raises:
            RuntimeError: A native capture call prevents shutdown within one
                second. The capture owner will release the device when it exits;
                never release a webcam underneath an active native read call.
        """
        with self._condition:
            self._closed = True
            self._stop_event.set()
            self._condition.notify_all()

        if self._thread is not None:
            # Never hold _condition while joining: the worker needs it to exit.
            if self._thread is not threading.current_thread():
                self._thread.join(timeout=1.0)
                if self._thread.is_alive():
                    raise RuntimeError(
                        "Camera capture did not stop within 1 second; "
                        "check the webcam/driver and retry close()."
                    )
        else:
            if not self._capture_lock.acquire(timeout=1.0):
                raise RuntimeError("A camera read is still blocked; retry close() when it ends.")
            try:
                self._release_capture()
            finally:
                self._capture_lock.release()

    def __enter__(self) -> "Camera":
        """Return this open Camera for use with a with block."""
        with self._condition:
            self._check_available()
        return self

    def __exit__(self, *exc: object) -> None:
        """Close on normal exit or an exception; do not suppress that exception."""
        self.close()

    def _check_available(self) -> None:
        """Raise a stored error or closed-state error; caller holds _condition."""
        if self._closed:
            raise RuntimeError("Camera is closed.")
        if self._error is not None:
            raise self._error

    def _capture_frame(self) -> Frame | None:
        """
        Reads an image, records its timestamp, validates it, 
        resizes/mirrors it, and assigns a frame number.
        """
        success, image = self._cap.read() #Read the image
        t_capture = time.perf_counter()  # Before resize/flip.
        #If reading the image not success
        if not success:
            self._failed_reads += 1
            if self._failed_reads >= CAMERA_MAX_FAILED_READS:
                raise RuntimeError("Repeated camera read failures – check the webcam connection.")
            return None
        self._failed_reads = 0
        if ( #Return invadlid BGR uint8 image
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8 
            or image.ndim != 3
            or image.shape[2] != 3
            or 0 in image.shape
        ):
            raise RuntimeError("Camera returned an invalid BGR uint8 image; check its driver.")
        if image.shape[:2] != (self._height, self._width): #Resizeing the image
            image = cv2.resize(image, (self._width, self._height))
        if self._mirror: #mirroing the image
            image = cv2.flip(image, 1) 
        #Assign number to the frame
        frame = Frame(image=image, t_capture=t_capture, index=self._next_index)
        self._next_index += 1
        return frame

    def _publish_frame(self, frame: Frame) -> None:
        
        """Replace the saved frame, update FPS history,timestamp history and wake waiting readers."""
        #Both threads access shared information, such as self._latest_frame. 
        #A lock allows only one thread at a time to access the protected information.
        with self._condition:
            if self._closed:
                return  # Discard a capture that completed during shutdown.
            self._latest_frame = frame #Capture the latest frame
            self._capture_times.append(frame.t_capture) #capture the time
            self._trim_capture_times(frame.t_capture)
            self._condition.notify_all()


    def _trim_capture_times(self, now: float) -> None:
        """Keep the specified one-second history; caller holds _condition."""
        while self._capture_times and self._capture_times[0] < now - 1.0:
            self._capture_times.popleft()


    def _capture_loop(self) -> None:
        """Repeatedly captures and publishes frames until stopped;
          handles errors and cleanup."""
        """Own background device I/O, reporting failures to read() before exit."""
        try:
            while not self._stop_event.is_set():
                frame = self._capture_frame() #Calling capture frame, repeat
                if frame is not None:
                    self._publish_frame(frame)
        except Exception as exc:
            # Thread exceptions do not propagate to callers on their own.
            with self._condition:
                self._error = RuntimeError(f"Camera capture failed: {exc}")
                self._error.__cause__ = exc #Escape the program
                self._condition.notify_all()
        finally:
            with self._capture_lock:
                self._release_capture() # Release capture after done

    def _release_capture(self) -> None:
        """Release once only; caller holds _capture_lock and owns device I/O."""
        if not self._released:
            self._cap.release()
            self._released = True

    def _read_sync(self, deadline: float) -> Frame:
        """
        Capture in the caller, serializing I/O without blocking latest().
        """
        remaining = deadline - time.perf_counter()
        if remaining <= 0 or not self._capture_lock.acquire(timeout=remaining):
            raise RuntimeError("Timed out waiting to access the webcam.")
        try:
            while time.perf_counter() < deadline:
                with self._condition:
                    self._check_available()
                frame = self._capture_frame()
                if frame is None:
                    continue
                self._publish_frame(frame)
                with self._condition:
                    self._check_available()
                    if time.perf_counter() >= deadline:
                        break
                    self._last_read_index = frame.index
                    return frame
            raise RuntimeError("Timed out waiting for a new frame.")
        except cv2.error as exc:
            raise RuntimeError(f"Camera capture failed: {exc}") from exc
        finally:
            try:
                # close() can time out while a driver is stuck. The I/O owner
                # still performs eventual cleanup as soon as that call returns.
                with self._condition:
                    closed = self._closed
                if closed:
                    self._release_capture()
            finally:
                self._capture_lock.release()
