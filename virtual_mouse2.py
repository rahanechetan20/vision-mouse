#!/opt/anaconda3/bin/python3
"""
Virtual Mouse Control System — Two-Handed Segregation Architecture
===================================================================

Right Hand → Cursor movement (EMA-smoothed) + Scroll
Left Hand → Click / Right-Click / Double-Click / Zoom

Dependencies:
    pip install mediapipe opencv-python pyautogui numpy

Model:
    Place 'hand_landmarker.task' in the same directory as this script,
    or update MODEL_PATH below.
"""

from __future__ import annotations

import os
import sys
import math
import time
from typing import Optional

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode,
)

# ─────────────────── Configuration ───────────────────

# PyAutoGUI settings
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

# Model path (resolve relative to this script)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

# Camera
CAM_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Overlay window size (compact preview)
OVERLAY_W = 320
OVERLAY_H = 240
OVERLAY_MARGIN = 10   # pixels from screen edge

# Active Zone (pixels trimmed from each edge of the camera frame)
ACTIVE_ZONE_MARGIN = 100

# Screen resolution
SCREEN_W, SCREEN_H = pyautogui.size()

# EMA smoothing factor (α = 0.3 → heavier smoothing)
EMA_ALPHA = 0.3

# Scroll parameters
SCROLL_SENSITIVITY = 5
SCROLL_DEADZONE    = 3

# Zoom (Pinch) — thumb + index extended, track distance changes
ZOOM_ENGAGE_DIST    = 120
ZOOM_STEP_THRESHOLD = 8

# Thumbs-up strictness
THUMB_UP_MARGIN = 0.05

# Window name
WINDOW_NAME = "Virtual Mouse"

# ─────────────────── Globals (state) ───────────────────

# Smoothed cursor position
smooth_x: float = SCREEN_W / 2
smooth_y: float = SCREEN_H / 2

# Right-hand scroll state
prev_scroll_y: Optional[float] = None
scroll_mode: bool = False

# Left-hand gesture state machine
class GestureState:
    """Tracks left-hand gesture so actions fire only once."""
    IDLE          = "idle"
    LEFT_CLICK    = "left_click"
    RIGHT_CLICK   = "right_click"
    DOUBLE_CLICK  = "double_click"
    ZOOM          = "zoom"

gesture_state = GestureState.IDLE
prev_pinch_dist: Optional[float] = None
zoom_anchor_dist: Optional[float] = None

# Frame-level detection result (written by callback)
latest_result: Optional[HandLandmarkerResult] = None
latest_timestamp_ms: int = 0

# ─────────────────── Utility helpers ───────────────────

def _dist(a, b) -> float:
    """Euclidean distance between two landmarks (uses .x, .y in normalised coords)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _px_dist(a, b, w: int, h: int) -> float:
    """Euclidean distance in pixel space."""
    return math.sqrt(((a.x - b.x) * w) ** 2 + ((a.y - b.y) * h) ** 2)


def _finger_extended(landmarks, tip_id: int, mcp_id: int, wrist_id: int = 0) -> bool:
    """True when the tip is further from the wrist than the MCP knuckle."""
    tip_to_wrist = _dist(landmarks[tip_id], landmarks[wrist_id])
    mcp_to_wrist = _dist(landmarks[mcp_id], landmarks[wrist_id])
    return tip_to_wrist > mcp_to_wrist


def _finger_curled(landmarks, tip_id: int, mcp_id: int, wrist_id: int = 0) -> bool:
    """Opposite of extended – tip is closer to wrist than its MCP."""
    return not _finger_extended(landmarks, tip_id, mcp_id, wrist_id)


# Landmark IDs
WRIST = 0
THUMB_TIP = 4;  THUMB_IP = 3;  THUMB_MCP = 2
INDEX_TIP = 8;  INDEX_MCP = 5
MIDDLE_TIP = 12; MIDDLE_MCP = 9
RING_TIP = 16;  RING_MCP = 13
PINKY_TIP = 20; PINKY_MCP = 17


# ─────────────────── Detection callback ───────────────────

def _on_result(result: HandLandmarkerResult, _image: mp.Image, timestamp_ms: int):
    global latest_result, latest_timestamp_ms
    latest_result = result
    latest_timestamp_ms = timestamp_ms


# ─────────────────── Right-hand logic ───────────────────

def handle_right_hand(landmarks, frame_w: int, frame_h: int, frame):
    """Move cursor or scroll based on right-hand landmarks."""
    global smooth_x, smooth_y, prev_scroll_y, scroll_mode

    # ── Check finger states ──
    index_ext  = _finger_extended(landmarks, INDEX_TIP, INDEX_MCP)
    middle_ext = _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_MCP)
    ring_ext   = _finger_extended(landmarks, RING_TIP, RING_MCP)
    pinky_ext  = _finger_extended(landmarks, PINKY_TIP, PINKY_MCP)

    # ── FREEZE CHECK (open palm = all fingers extended) ──
    # Opening all fingers keeps the index finger stable while freezing the cursor.
    # This lets the user "park" the cursor, then safely remove their hand.
    all_open = index_ext and middle_ext and ring_ext and pinky_ext
    if all_open:
        wx = int(landmarks[WRIST].x * frame_w)
        wy = int(landmarks[WRIST].y * frame_h)
        cv2.putText(frame, "FROZEN", (wx - 30, wy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        cv2.circle(frame, (wx, wy), 12, (0, 140, 255), 3)
        # Reset scroll state too
        scroll_mode = False
        prev_scroll_y = None
        return

    # ── SCROLL MODE: Index + Middle extended BUT ring & pinky curled ──
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        scroll_mode = True
        mid_y = (landmarks[INDEX_TIP].y + landmarks[MIDDLE_TIP].y) / 2 * frame_h

        if prev_scroll_y is not None:
            delta = prev_scroll_y - mid_y
            if abs(delta) > SCROLL_DEADZONE:
                scroll_amount = int(delta / abs(delta)) * max(1, int(abs(delta) * SCROLL_SENSITIVITY / frame_h * 20))
                try:
                    pyautogui.scroll(scroll_amount)
                except pyautogui.FailSafeException:
                    pass

        prev_scroll_y = mid_y

        # Visual: green circles on index + middle tips
        ix = int(landmarks[INDEX_TIP].x * frame_w)
        iy = int(landmarks[INDEX_TIP].y * frame_h)
        mx = int(landmarks[MIDDLE_TIP].x * frame_w)
        my = int(landmarks[MIDDLE_TIP].y * frame_h)
        cv2.circle(frame, (ix, iy), 12, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (mx, my), 12, (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, "SCROLL", (ix - 30, iy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return

    # Reset scroll state when exiting scroll mode
    scroll_mode = False
    prev_scroll_y = None

    # ── CURSOR MODE — track Index Tip ──
    raw_x = landmarks[INDEX_TIP].x * frame_w
    raw_y = landmarks[INDEX_TIP].y * frame_h

    # Active zone boundaries
    az_left   = ACTIVE_ZONE_MARGIN
    az_right  = frame_w - ACTIVE_ZONE_MARGIN
    az_top    = ACTIVE_ZONE_MARGIN
    az_bottom = frame_h - ACTIVE_ZONE_MARGIN

    # Map inside active zone → full screen  (numpy.interp handles clamping)
    screen_x = float(np.interp(raw_x, [az_left, az_right], [0, SCREEN_W]))
    screen_y = float(np.interp(raw_y, [az_top, az_bottom], [0, SCREEN_H]))

    # EMA smoothing
    smooth_x = EMA_ALPHA * screen_x + (1 - EMA_ALPHA) * smooth_x
    smooth_y = EMA_ALPHA * screen_y + (1 - EMA_ALPHA) * smooth_y

    try:
        pyautogui.moveTo(int(smooth_x), int(smooth_y))
    except pyautogui.FailSafeException:
        pass

    # Visual: cyan circle on index tip
    ix = int(landmarks[INDEX_TIP].x * frame_w)
    iy = int(landmarks[INDEX_TIP].y * frame_h)
    cv2.circle(frame, (ix, iy), 14, (255, 255, 0), cv2.FILLED)
    cv2.putText(frame, "CURSOR", (ix - 30, iy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


# ─────────────────── Left-hand logic ───────────────────

def handle_left_hand(landmarks, frame_w: int, frame_h: int, frame):
    """Gesture state machine for click / right-click / double-click / zoom."""
    global gesture_state, prev_pinch_dist, zoom_anchor_dist

    # ── Finger states ──
    index_curled  = _finger_curled(landmarks, INDEX_TIP, INDEX_MCP)
    middle_curled = _finger_curled(landmarks, MIDDLE_TIP, MIDDLE_MCP)
    ring_curled   = _finger_curled(landmarks, RING_TIP, RING_MCP)
    pinky_curled  = _finger_curled(landmarks, PINKY_TIP, PINKY_MCP)

    index_ext     = not index_curled
    middle_ext    = not middle_curled
    thumb_ext     = _finger_extended(landmarks, THUMB_TIP, THUMB_MCP)

    four_curled   = index_curled and middle_curled and ring_curled and pinky_curled

    # ── Robust thumbs-up detection ──
    # The thumb tip must be:
    #   1. Above (lower y) ALL finger MCP joints (the knuckle line)
    #   2. Significantly above the wrist (not just barely above its own joints)
    # This prevents a normal fist from being misidentified as thumbs-up.
    thumb_tip_y = landmarks[THUMB_TIP].y
    knuckle_line_y = min(
        landmarks[INDEX_MCP].y,
        landmarks[MIDDLE_MCP].y,
        landmarks[RING_MCP].y,
        landmarks[PINKY_MCP].y,
    )
    wrist_y = landmarks[WRIST].y
    thumb_up = (
        thumb_tip_y < knuckle_line_y - THUMB_UP_MARGIN
        and thumb_tip_y < landmarks[THUMB_IP].y
        and thumb_tip_y < landmarks[THUMB_MCP].y
        and (wrist_y - thumb_tip_y) > 0.15
    )

    # ── Peace sign (V-shape): Index + Middle extended, thumb/ring/pinky curled ──
    peace_sign = (index_ext and middle_ext and
                  ring_curled and pinky_curled and not thumb_ext)

    # ── Zoom shape: Thumb + Index extended, middle/ring/pinky curled ──
    zoom_shape = (thumb_ext and index_ext and
                  middle_curled and ring_curled and pinky_curled)

    # Pinch distance (pixel space) — used for zoom
    pinch_dist = _px_dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP], frame_w, frame_h)

    # ── Determine current gesture ──
    detected = GestureState.IDLE
    label = ""
    color = (200, 200, 200)

    if zoom_shape:
        # Thumb + Index extended → ZOOM mode (track distance continuously)
        detected = GestureState.ZOOM
        label = "ZOOM"
        color = (255, 0, 255)
    elif peace_sign:
        # Index + Middle up, others down → DOUBLE CLICK
        detected = GestureState.DOUBLE_CLICK
        label = "DOUBLE CLICK"
        color = (0, 165, 255)
    elif four_curled and thumb_up:
        detected = GestureState.RIGHT_CLICK
        label = "RIGHT CLICK"
        color = (0, 0, 255)
    elif four_curled and not thumb_up:
        detected = GestureState.LEFT_CLICK
        label = "LEFT CLICK"
        color = (255, 0, 0)

    # ── Fire action only on state TRANSITION (idle → gesture) ──
    if detected != GestureState.IDLE and gesture_state == GestureState.IDLE:
        if detected == GestureState.LEFT_CLICK:
            try:
                pyautogui.click()
            except pyautogui.FailSafeException:
                pass
        elif detected == GestureState.RIGHT_CLICK:
            try:
                pyautogui.rightClick()
            except pyautogui.FailSafeException:
                pass
        elif detected == GestureState.DOUBLE_CLICK:
            try:
                pyautogui.doubleClick()
            except pyautogui.FailSafeException:
                pass
        elif detected == GestureState.ZOOM:
            # Anchor = the distance when zoom was first entered
            zoom_anchor_dist = pinch_dist
            prev_pinch_dist  = pinch_dist

    # ── Continuous zoom while held ──
    # Compare current distance to the PREVIOUS frame's distance.
    # Spreading fingers apart → zoom in, bringing together → zoom out.
    if detected == GestureState.ZOOM and gesture_state == GestureState.ZOOM:
        if prev_pinch_dist is not None:
            delta = pinch_dist - prev_pinch_dist
            if abs(delta) > ZOOM_STEP_THRESHOLD:
                try:
                    if delta > 0:
                        pyautogui.hotkey("command", "=")
                    else:
                        pyautogui.hotkey("command", "-")
                except pyautogui.FailSafeException:
                    pass
            # Always update prev so small movements accumulate
            prev_pinch_dist = pinch_dist

    # ── Update state ──
    if detected == GestureState.IDLE:
        gesture_state    = GestureState.IDLE
        prev_pinch_dist  = None
        zoom_anchor_dist = None
    else:
        gesture_state = detected

    # ── Visual feedback ──
    tx = int(landmarks[THUMB_TIP].x * frame_w)
    ty = int(landmarks[THUMB_TIP].y * frame_h)
    ix = int(landmarks[INDEX_TIP].x * frame_w)
    iy = int(landmarks[INDEX_TIP].y * frame_h)

    cv2.circle(frame, (tx, ty), 10, color, cv2.FILLED)
    cv2.circle(frame, (ix, iy), 10, color, cv2.FILLED)
    if label:
        cv2.putText(frame, label, (tx - 40, ty - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw distance line for zoom gesture
    if zoom_shape or detected == GestureState.ZOOM:
        cv2.line(frame, (tx, ty), (ix, iy), color, 2)
        direction = ""
        if zoom_anchor_dist is not None:
            diff = pinch_dist - zoom_anchor_dist
            direction = " IN" if diff > 10 else (" OUT" if diff < -10 else "")
        cv2.putText(frame, f"{int(pinch_dist)}px{direction}",
                    ((tx + ix) // 2, (ty + iy) // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw peace sign markers for double-click
    if peace_sign:
        mx = int(landmarks[MIDDLE_TIP].x * frame_w)
        my = int(landmarks[MIDDLE_TIP].y * frame_h)
        cv2.circle(frame, (mx, my), 10, color, cv2.FILLED)


# ─────────────────── Draw helpers ───────────────────

def draw_active_zone(frame):
    """Draw the inner active zone rectangle on the output frame."""
    h, w = frame.shape[:2]
    top_left     = (ACTIVE_ZONE_MARGIN, ACTIVE_ZONE_MARGIN)
    bottom_right = (w - ACTIVE_ZONE_MARGIN, h - ACTIVE_ZONE_MARGIN)
    cv2.rectangle(frame, top_left, bottom_right, (100, 255, 100), 2)
    cv2.putText(frame, "Active Zone", (ACTIVE_ZONE_MARGIN + 5, ACTIVE_ZONE_MARGIN - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)


def draw_hand_connections(frame, landmarks, frame_w, frame_h, color):
    """Draw skeleton connections for a hand."""
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17),             # Palm
    ]
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, color, cv2.FILLED)


# ─────────────────── Overlay window setup ───────────────────

def setup_overlay_window():
    """Create a small, always-on-top OpenCV window positioned at the top-right corner."""
    # Create a resizable window so we can control its size
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Set the window size to the compact overlay dimensions
    cv2.resizeWindow(WINDOW_NAME, OVERLAY_W, OVERLAY_H)

    # Position the window at the top-right corner of the screen
    pos_x = SCREEN_W - OVERLAY_W - OVERLAY_MARGIN
    pos_y = OVERLAY_MARGIN
    cv2.moveWindow(WINDOW_NAME, pos_x, pos_y)

    # Make the window always-on-top
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)


def refresh_overlay_topmost():
    """
    Re-assert always-on-top every few seconds.
    On macOS, the window level can sometimes reset after system events.
    """
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)


# ─────────────────── Main loop ───────────────────

def main():
    global latest_result

    # ── Verify model ──
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        print("       Download from: https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        sys.exit(1)

    # ── Create HandLandmarker ──
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=_on_result,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # ── Open camera ──
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    # ── Set up the always-on-top overlay window ──
    setup_overlay_window()

    print("=" * 60)
    print("  Virtual Mouse — Two-Handed Control  (Overlay Mode)")
    print("=" * 60)
    print("  RIGHT HAND  →  Move cursor / Scroll (Index+Middle up)")
    print("  LEFT  HAND  →  Fist=Click | ThumbsUp=Right-Click")
    print("                 Peace=DblClick | Pinch=Zoom")
    print(f"  Overlay: {OVERLAY_W}×{OVERLAY_H} — always on top (top-right)")
    print("  Press 'q' to quit.")
    print("=" * 60)

    frame_count = 0
    fps_start = time.time()
    fps_display = 0
    topmost_refresh_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # FPS calculation
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Convert to MediaPipe Image (RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Async detection
            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            # ── Process latest result ──
            result = latest_result
            if result and result.hand_landmarks:
                for idx, (hand_lms, handedness) in enumerate(
                    zip(result.hand_landmarks, result.handedness)
                ):
                    # MediaPipe handedness label (from the model's perspective)
                    # The image is already flipped, so "Left" from the model = user's Left hand
                    # and "Right" from the model = user's Right hand
                    # But since we flipped the frame, the labels are swapped:
                    #   model says "Right" → actually user's Left hand (mirrored)
                    #   model says "Left"  → actually user's Right hand (mirrored)
                    label = handedness[0].category_name   # "Left" or "Right"

                    if label == "Left":
                        # Model's Left → User's RIGHT hand (because frame is flipped)
                        draw_hand_connections(frame, hand_lms, w, h, (255, 255, 0))  # Cyan
                        handle_right_hand(hand_lms, w, h, frame)
                    elif label == "Right":
                        # Model's Right → User's LEFT hand (because frame is flipped)
                        draw_hand_connections(frame, hand_lms, w, h, (0, 0, 255))    # Red
                        handle_left_hand(hand_lms, w, h, frame)

            # ── Draw overlays on full-res frame ──
            draw_active_zone(frame)

            # HUD
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Gesture: {gesture_state}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2)

            # Hand count
            n_hands = len(result.hand_landmarks) if result and result.hand_landmarks else 0
            cv2.putText(frame, f"Hands: {n_hands}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)

            # ── Resize frame to compact overlay size and display ──
            overlay_frame = cv2.resize(frame, (OVERLAY_W, OVERLAY_H),
                                       interpolation=cv2.INTER_AREA)

            # Add a thin bright border so the overlay stands out
            cv2.rectangle(overlay_frame, (0, 0),
                          (OVERLAY_W - 1, OVERLAY_H - 1), (0, 255, 200), 2)

            cv2.imshow(WINDOW_NAME, overlay_frame)

            # ── Periodically re-assert always-on-top (every 2 seconds) ──
            # This guards against macOS resetting the window level
            if time.time() - topmost_refresh_time > 2.0:
                refresh_overlay_topmost()
                topmost_refresh_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("\n[INFO] Virtual Mouse stopped.")


if __name__ == "__main__":
    main()
