
# <Name> – <role, e.g. Hand tracking> — Technical notes
Author: <name>   |   Files owned: <list>   |   Requirement IDs: <e.g. HT.1–HT.4, R3, R4>

## 1. What it does (3–5 sentences)

This component turns live webcam images into hand-coordinate data for the gesture-controlled car simulation.
`camera.py` uses OpenCV to capture, resize and mirror images while measuring capture FPS.
`landmarks.py` uses MediaPipe to detect detect 21 landmarks of the hand and draw the landmarks on the preview.
`preprocess.py` normalises these coordinates into 63 features for model training and live gesture prediction.

Camera.py:
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

landmark.py
1:Create the tracker: __init__() prepares MediaPipe in video mode.
2: Process each camera frame: detect() calls _validate_image() to check that input is a nonempty
NumPy image with 3 colour channels and the expected pixel type, then extracts hand coordinates.
3:Use the result: when a hand is found, the caller can pass it to draw() for the preview and preprocess() 
for feature preparation. draw() also calls _validate_image().
4:Finish: the caller calls close() to release resources.

preprocess.py
1: Validate: check the coordinate shape, values and handedness
2: Correct image proportions: adjust x and z using the image width to height ratio
3: Centre on the wrist: subtract the wrist coordinates so the write becomes (0,0,0)
4: Mirror left-hand coordinates: reverse their x values so both hands use a consistent orientation.
5: divide coordinates by the wrist-to-middle-finger-base distance measured in x and y.
6: Flatten: turn the (21,3) coordinate array into 63 numbers

## 2. Design decisions
camera.py
| Decision | Alternatives considered | Why we chose it |
|---|---|---|
| Keep only the newest frame | Queue every captured frame | Avoids processing a backlog of outdated images. |
| Calculate FPS from capture timestamps | Use the camera’s configured FPS | Reports the measured capture rate. |
| Protect shared data with a condition lock | Access shared data without coordination | Keeps updates safe and lets readers wait for new frames. |
| Support automatic cleanup with `with` | Require callers to close manually | Helps release the webcam after normal use or an error. |

landmarks.py
| Decision | Alternatives considered | Why we chose it |
|---|---|---|
| Use MediaPipe for hand detection | Build and train our own detector | Provides an existing model for detecting 21 hand landmarks. |
| Reuse one detector in video mode | Create a detector for every frame | Avoids repeated setup and preserves tracking state between frames. |
| Return the first detected hand | Return multiple hands | Matches the project’s single-hand control interface. |
| Store results in `HandLandmarks` | Return MediaPipe’s output directly | Gives other modules a simple interface for coordinates, handedness, and confidence. |
| Separate `detect()` from `draw()` | Always draw during detection | Allows detection without a preview and keeps the original image unchanged. |

preprocess.py
| Decision | Alternatives considered | Why we chose it |
|---|---|---|
| Correct for image proportions | Use raw normalised coordinates directly | Prevents unequal width and height scales from distorting the hand shape. |
| Centre coordinates on the wrist | Keep absolute image coordinates | Reduces dependence on where the hand appears in the image. |
| Scale using palm length | Use finger length or bounding-box size | Palm length changes less when fingers bend or extend. |
| Mirror left-hand coordinates | Keep separate left/right representations | Makes corresponding gestures from either hand more comparable. |
| Preserve hand rotation | Rotate every hand into a fixed orientation | Keeps orientation information available to the classifier. |



## 3. Interfaces provided / consumed
| Module | Consumes | Provides to |
|---|---|---|
| `camera.py` | OpenCV webcam images and camera settings from `src.config` | Shobita’s collector and Evan’s worker/preview |
| `landmarks.py` | Mirrored BGR images, MediaPipe, and tracking settings | Shobita’s collector and Evan’s worker |
| `preprocess.py` | Raw `HandLandmarks` or CSV coordinates, NumPy, and preprocessing settings | Shobita’s dataset loader and Evan’s live prediction pipeline |


## 4. Challenges and how we solved them
| Challenge | What we implemented | Evidence |
|---|---|---|
| Two visible hands create ambiguity about which hand should control the game. | Set `MP_MAX_NUM_HANDS = 1` and made `detect()` return only the first detected hand. | A test supplies multiple simulated detection entries and confirms that only the first hand’s 21 landmarks are returned. |
| Left hand produces mirrored coordinates. | `preprocess()` uses the handedness label to reverse the left hand’s `x` coordinates. | `test_mirrored_left_matches_right` confirms that synthetic mirrored left/right inputs produce matching feature vectors. |

## 5. Results

| Priority | Result | Evidence file | Interpretation |
|---|---|---|---|
| **[Must]** Automated tests | **55/55 passed (100%)**: 25 camera, 8 landmarks, 22 preprocessing | [Test report](../../results/triet-tests.xml) | All 55 tests in the refreshed report passed, with no failures or errors. |
| **[Must]** Bright versus dim FPS | Bright: **30.07 FPS**, 301 frames; dim: **29.94 FPS**, 300 frames | [Bright measurements](../../results/triet-fps-bright.txt), [dim measurements](../../results/triet-fps-dim.txt) | Both runs measured approximately 30 FPS under the tested lighting conditions. |
| **[Must]** Live webcam verification | Landmarks and **63 features** displayed; mirroring, no-hand recovery, and shutdown manually confirmed; increasing frame numbers and timestamps logged | [Preview log](../../results/triet-preview-log.txt), [manual observations](../../results/triet-webcam-check.txt) | The log supports the frame-timing checks; the observation note records Triet's earlier visual confirmations. |


## 6. Limitations & future work
Single-hand control: When both hands are visible, the selected hand may change. Future work could add an explicit hand-selection rule.
Limited device testing: Current webcam evidence comes from my webcam. Repeat verification on other laptops, including macOS.
Full-system testing: These checks cover capture, landmarks, and preprocessing. Gesture accuracy and response time still require testing with the classifier and game.

## 7. Slide summary (max 40 words + 1 figure)
This component turns live webcam images into hand-coordinate data for the gesture-controlled car simulation.
`camera.py` uses OpenCV to capture, resize and mirror images while measuring capture FPS.
`landmarks.py` uses MediaPipe to detect detect 21 landmarks of the hand and draw the landmarks on the preview.
`preprocess.py` normalises these coordinates into 63 features for model training and live gesture prediction.


Note: We must include the evidence 

| Challenge | What we implemented | 
|---|---|
| Two visible hands create ambiguity about which hand should control the game. | Set `MP_MAX_NUM_HANDS = 1` and made `detect()` return only the first detected hand. | 
| Left hand produces mirrored coordinates. | `preprocess()` uses the handedness label to reverse the left hand’s `x` coordinates. | `test_mirrored_left_matches_right` confirms that synthetic mirrored left/right inputs produce matching feature vectors. |


