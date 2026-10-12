"""Evan's live integration entry point. Pygame and game calls stay on this thread.

Run with --classifier mock --camera mock until the real classifier/game arrive.
The small status panel is integration diagnostics; the controller owns gameplay.
"""
import argparse
import math
from datetime import datetime
import sys
import time

import cv2
import pygame

from src.app.actions import Action, apply_action, gesture_to_action
from src.app.latency import ActionTiming, LatencyLogger
from src.app.mock_classifier import KeyboardState, MockCamera, MockClassifier, MockHandTracker
from src.app.worker import GestureWorker
from src.capture.camera import Camera
from src.capture.landmarks import HandTracker
from src.config import (LOW_FPS_WARNING, MAX_DT, MODEL_PATH, PREVIEW_SIZE, RESULTS_DIR,
                        SHOW_CAMERA_PREVIEW, TARGET_FPS, WINDOW_HEIGHT, WINDOW_WIDTH)
from src.game.controller import GameController
from src.game.states import GameState


def build_components(args, keys):
    # Load the model before opening hardware, so a missing model cannot leak a camera.
    if args.classifier == "mock":
        classifier = MockClassifier(keys.get)
    else:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"{MODEL_PATH} not found; Siri must train the model. "
                                    "Use --classifier mock to test integration.")
        from src.model.predict import GestureClassifier
        try:
            classifier = GestureClassifier()
        except NotImplementedError as exc:
            raise RuntimeError("Real classifier is unfinished. Use --classifier mock.") from exc
    if args.camera == "mock":
        camera, factory = MockCamera(), lambda: MockHandTracker(keys.get)
    else:
        camera, factory = Camera(), HandTracker
    return camera, factory, classifier


def _keyboard(controller, key):
    """KEYDOWN is a discrete fallback, separate from keys used for mock predictions."""
    if key in (pygame.K_LEFT, pygame.K_RIGHT):
        token = "fist" if key == pygame.K_LEFT else "v_sign"
        apply_action(controller, gesture_to_action(token, controller.state))
    elif key == pygame.K_p:
        if controller.state == GameState.PAUSED:
            controller.resume()
        else:
            apply_action(controller, gesture_to_action("open_palm", controller.state))
    elif key == pygame.K_r:
        controller.restart()


def _status(surface, font, controller, observation, worker, camera, args, last_action):
    """Make stub behaviour observable without changing Uy's game implementation."""
    lines = [f"State: {controller.state.value} | score: {controller.score}",
             f"Classifier: {args.classifier} | camera: {args.camera}"]
    if hasattr(controller, "calls") and hasattr(controller, "lane"):
        lines += ["TEMPORARY CONTROLLER: no obstacles/collisions", f"Lane: {controller.lane} (0=left, 2=right)"]
    if observation is None:
        lines.append("Waiting for the first prediction...")
    else:
        lines += [f"Raw: {observation.token} | confidence: {observation.confidence:.2f}",
                  f"Stable: {observation.stable} | hand: {observation.hand_present}"]
    fps = worker.fps
    lines += [f"Camera FPS: {camera.fps:.1f} | worker FPS: {fps:.1f}",
              f"Last changed action: {last_action}",
              "1=fist  2=V sign  3=palm  H=no hand (mock)",
              "Arrows=steer  P=pause/resume  R=restart  Esc=quit"]
    panel = pygame.Surface((WINDOW_WIDTH, len(lines)*23+12))
    panel.fill((18, 23, 31))
    for i, line in enumerate(lines):
        color = (255, 180, 70) if "FPS" in line and fps < LOW_FPS_WARNING else (235, 240, 245)
        panel.blit(font.render(line, True, color), (8, 6+i*23))
    surface.blit(panel, (0, 0))
    # This is a copied camera frame, so rendering cannot mutate worker input.
    if SHOW_CAMERA_PREVIEW and not args.no_preview:
        frame = camera.latest()
        if frame is not None:
            rgb = cv2.cvtColor(frame.image.copy(), cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, PREVIEW_SIZE)
            preview = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            surface.blit(preview, (WINDOW_WIDTH-PREVIEW_SIZE[0]-8, panel.get_height()+8))


def run(args):
    keys = KeyboardState()
    camera = worker = logger = None
    try:
        camera, factory, classifier = build_components(args, keys)
        pygame.init()
        surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Gesture car integration")
        font = pygame.font.Font(None, 21)
        clock = pygame.time.Clock()
        controller = GameController()
        path = RESULTS_DIR / f"latency_{datetime.now():%Y%m%d-%H%M%S-%f}.csv" if args.log_latency else None
        logger = LatencyLogger(path)
        if path:
            print(f"Latency CSV: {path}")
        print(f"Running camera={args.camera}, classifier={args.classifier}. Esc quits.")
        if hasattr(controller, "calls"):
            print("Temporary game controller: lane/state testing only; no obstacles or collisions.")
        worker = GestureWorker(camera, factory, classifier)
        worker.start()
        started = time.perf_counter()
        last_action = "none"
        running = True
        while running:
            dt = min(clock.tick(TARGET_FPS)/1000, MAX_DT)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN:
                    _keyboard(controller, event.key)
            keys.update(pygame.key.get_pressed())
            if not running:
                break
            if worker.error is not None:
                pygame.display.set_caption(f"Worker error: {worker.error}")
                raise RuntimeError(f"Gesture worker failed: {worker.error}") from worker.error
            observation = worker.latest()
            if observation is not None:
                controller.set_gesture_info(observation.token, observation.confidence, observation.hand_present)
            for event in worker.poll_events():
                action = gesture_to_action(event.token, controller.state)
                if apply_action(controller, action):
                    t_action = time.perf_counter()
                    logger.log_action(ActionTiming(action, event.token, event.t_onset, event.t_trigger,
                                                   event.t_landmarks, event.t_predict, t_action))
                    last_action = action.value
                    print(f"{event.token} -> {action.value} | state={controller.state.value}")
            controller.update(dt)
            controller.render(surface)
            _status(surface, font, controller, observation, worker, camera, args, last_action)
            pygame.display.flip()
            if args.duration is not None and time.perf_counter()-started >= args.duration:
                running = False
    finally:
        # Close the frame source to wake a blocked camera.read(), then join the worker.
        # Each cleanup is attempted even if another resource fails to close.
        failures = []
        if worker is not None:
            try:
                worker.stop(timeout=0.1)
            except RuntimeError:
                pass  # Closing the camera below wakes a pending read; then join again.
        for cleanup in ([camera.close] if camera is not None else []) + ([worker.stop] if worker is not None else []):
            try:
                cleanup()
            except Exception as exc:
                failures.append(str(exc))
        if logger is not None:
            print(f"Latency summary (ms): {logger.summary()}")
            logger.close()
        pygame.quit()
        if failures:
            print("Cleanup errors: " + "; ".join(failures), file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gesture-controlled car integration")
    parser.add_argument("--classifier", choices=("svm", "mock"), default="svm")
    parser.add_argument("--camera", choices=("webcam", "mock"), default="webcam")
    parser.add_argument("--log-latency", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--duration", type=float, help="Automatically exit after N seconds (smoke tests)")
    args = parser.parse_args(argv)
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    try:
        run(args)
    except (RuntimeError, ValueError, FileNotFoundError, NotImplementedError) as exc:
        parser.exit(1, f"Integration error: {exc}\n")


if __name__ == "__main__":
    main()
