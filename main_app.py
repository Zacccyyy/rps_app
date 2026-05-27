#!/usr/bin/env python3
"""
main_app.py — RPS Robot standalone fullscreen app.

Three modes accessible from a menu:
  Cheat     — robot always picks the counter-move that beats you.
  Challenge — survival run; robot difficulty ramps with your streak.
  Mirror    — robot hand mirrors your finger curls via BLE.

Camera runs every frame via MediaPipe hand tracking.  All drawing is
done onto a 1280×800 canvas composited from background assets and
live overlays.

Usage:
    python3 main_app.py
Quit:
    press Q
"""

import os
import sys
import time

import cv2
import numpy as np

# ── Shared modules from rps_hand_counter ─────────────────────────────────────
sys.path.insert(0, os.path.expanduser('~/rps_hand_counter'))

from hand_landmarks import (          # MediaPipe wrapper + Kalman wrist filter
    process_hand_frame,
    create_hands_detector,
    create_kalman_wrist_state,
)
from gesture_state import GestureStateTracker   # multi-frame confirmation layer
# gesture_mapper.classify_rps_gesture and front_on_classifier.classify_front_on
# are called internally by process_hand_frame; no direct use needed here.

# ── Mode state modules (in this directory) ────────────────────────────────────
from app_cheat_state     import CheatController
from app_challenge_state import AppChallengeController
from app_mirror_state    import AppMirrorState

# ── Optional BLE bridge (graceful fallback) ───────────────────────────────────
try:
    from ble_bridge import BLEBridge
    ble = BLEBridge()
    ble.start()
    print("[BLE] bridge started")
except Exception as _ble_err:
    ble = None
    print(f"[BLE] unavailable — {_ble_err}")

# =============================================================================
# Constants
# =============================================================================

SCREEN_W, SCREEN_H = 1280, 800
WIN_NAME = "RPS Robot"

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

FONT       = cv2.FONT_HERSHEY_SIMPLEX
WHITE      = (255, 255, 255)
ACCENT_BGR = (210, 160, 60)   # accent blue for mirror bars  (BGR order)
DARK_GREY  = (50,  50,  50)

# Bar layout for Mirror mode
_CURL_LABELS = ['THUMB', 'INDEX', 'MIDDLE', 'RING']
_BAR_X       = 40
_BAR_W       = 300
_BAR_H       = 20
_BAR_Y0      = 520    # top of first bar
_BAR_STEP    = 65     # label-to-label spacing (label height ~18 + bar 20 + gap ~27)

# =============================================================================
# Window setup
# =============================================================================

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# =============================================================================
# Asset loading
# =============================================================================

def _load_asset(filename):
    """Load one PNG from the assets directory, resize to screen dimensions."""
    path = os.path.join(ASSETS_DIR, filename)
    img  = cv2.imread(path)
    if img is None:
        # Graceful fallback: black frame with filename as label
        img = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        cv2.putText(img, f"[missing: {filename}]", (40, SCREEN_H // 2),
                    FONT, 1.0, WHITE, 2, cv2.LINE_AA)
        print(f"[warn] could not load asset: {path}")
    else:
        img = cv2.resize(img, (SCREEN_W, SCREEN_H))
    return img


ASSETS = {
    'title':          _load_asset('TitleScreen.png'),
    'menu_cheat':     _load_asset('RPSCheatSelected.png'),
    'menu_challenge': _load_asset('RPSChallengeSelected.png'),
    'menu_mirror':    _load_asset('RPSMirrorSelected.png'),
    'cheat':          _load_asset('CheatMode.png'),          # has green chroma circles
    'ch_idle':        _load_asset('ChallengeIdle.png'),
    'ch_angry1':      _load_asset('ChallengeAngry1.png'),    # 3+ player wins
    'ch_angry2':      _load_asset('ChallengeAngry2.png'),    # 5+ player wins
    'ch_angry3':      _load_asset('ChallengeAngry3.png'),    # 8+ player wins
    'you_win':        _load_asset('YouWin.png'),
    'you_lose':       _load_asset('YouLose.png'),
    'mirror_bg':      _load_asset('CheatMode.png'),          # reused for Mirror mode
}

# =============================================================================
# Green-circle detection (for Cheat mode chroma-key)
# =============================================================================

def _detect_green_circles(bg_image):
    """
    Find the two solid #00FF00 circles in the Cheat Mode background image.

    Strategy:
      1. Convert to HSV.
      2. Threshold for near-pure green (H≈60°, S/V near max).
      3. Find contours, keep the two largest.
      4. Return list of (cx, cy, radius) tuples in SCREEN_W×SCREEN_H coordinates.

    Returns an empty list when no matching regions exist.
    """
    hsv   = cv2.cvtColor(bg_image, cv2.COLOR_BGR2HSV)
    # Pure #00FF00: OpenCV H=60, S=255, V=255
    lower = np.array([55, 200, 200])
    upper = np.array([65, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)

    # Fill small gaps that can appear at JPEG artefact boundaries
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    circles = []
    for c in contours:
        if cv2.contourArea(c) < 300:   # discard tiny noise blobs
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        circles.append((int(x), int(y), int(r)))

    if circles:
        print(f"[cheat] detected {len(circles)} chroma circle(s): {circles}")
    else:
        print("[cheat] no green circles found in CheatMode.png — camera overlay disabled")
    return circles


# Detect once at startup (background is already screen-sized)
CHEAT_CIRCLES = _detect_green_circles(ASSETS['cheat'])

# =============================================================================
# HandLandmarkDetector — wraps MediaPipe + GestureStateTracker
# =============================================================================

class HandLandmarkDetector:
    """
    Bundles MediaPipe Hands, a Kalman wrist filter, and a multi-frame
    gesture confirmation tracker into a single process() call.

    process(raw_frame) returns a hand_state dict containing:
        raw_gesture        — single-frame MediaPipe classification
        stable_gesture     — majority-voted over a short window
        confirmed_gesture  — held steady for N frames (most reliable)
        robot_ready        — True when it is safe to act on the gesture
        wrist_y            — Kalman-smoothed wrist Y  (0=top, 1=bottom)
        raw_wrist_y        — unsmoothed wrist Y for pump-beat detection
        _landmarks         — raw MediaPipe NormalizedLandmarkList
        _world_landmarks   — rotation-invariant world-space landmarks
        … plus all other fields from process_hand_frame()
    """

    def __init__(self):
        self._hands     = create_hands_detector()
        self._ema       = create_kalman_wrist_state()
        self._tracker   = GestureStateTracker()

    def process(self, raw_frame):
        """
        Process one raw (unflipped) camera frame.

        hand_orientation='Front' activates the hybrid ML + curl-angle classifier
        (front_on_classifier.classify_front_on), which works reliably for both
        palm-facing and side-on hand positions without any settings change.
        """
        _, hand_state, _ = process_hand_frame(
            raw_frame,
            self._hands,
            hand_orientation='Front',
            _ema_state=self._ema,
        )
        tracker_out = self._tracker.update(hand_state.get('raw_gesture', 'Unknown'))
        hand_state.update(tracker_out)
        return hand_state

# =============================================================================
# Game state objects (one each, shared across the lifetime of the app)
# =============================================================================

detector       = HandLandmarkDetector()
cheat_ctrl     = CheatController()
challenge_ctrl = AppChallengeController()
mirror_state   = AppMirrorState(ble_bridge=ble)

# =============================================================================
# App-level state
# =============================================================================

screen   = 'TITLE'   # TITLE | MENU | CHEAT | CHALLENGE | MIRROR | WIN | LOSE
menu_sel = 0         # 0=Cheat, 1=Challenge, 2=Mirror

# BLE dedup: avoid re-sending the same command every frame
_ble_last_cheat_cmd     = None
_ble_last_challenge_cmd = None

# =============================================================================
# Drawing helpers
# =============================================================================

def _text_centred(frame, text, y, scale=1.0, colour=WHITE, thickness=2):
    """Draw text horizontally centred on the canvas at the given Y baseline."""
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (SCREEN_W - tw) // 2
    cv2.putText(frame, text, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def _flash_enter_prompt(frame, now, y=None):
    """
    Draw 'Press ENTER to continue' centred in the bottom quarter.
    Toggles every 500 ms using time.monotonic().
    """
    if int(now * 2) % 2 == 0:
        if y is None:
            y = SCREEN_H * 3 // 4 + 60
        _text_centred(frame, "Press ENTER to continue", y, scale=1.2)


# =============================================================================
# Screen draw functions
# =============================================================================

# ── TITLE ─────────────────────────────────────────────────────────────────────

def draw_title(frame, now):
    frame[:] = ASSETS['title']
    _flash_enter_prompt(frame, now)


# ── MENU ──────────────────────────────────────────────────────────────────────

def draw_menu(frame, sel):
    keys = ('menu_cheat', 'menu_challenge', 'menu_mirror')
    frame[:] = ASSETS[keys[sel]]


# ── CHEAT ─────────────────────────────────────────────────────────────────────

def _composite_camera_circles(bg, raw_frame, circles):
    """
    For each detected green circle: crop the corresponding region from a
    mirror-flipped, screen-scaled camera frame and blend it into the background
    using a circular alpha mask.

    Returns the composited image (same dimensions as bg).
    """
    result = bg.copy()
    if not circles:
        return result

    # Flip + resize once for all circles
    cam = cv2.resize(cv2.flip(raw_frame, 1), (SCREEN_W, SCREEN_H))

    for (cx, cy, r) in circles:
        if r < 5:
            continue
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(SCREEN_W, cx + r)
        y2 = min(SCREEN_H, cy + r)
        rw, rh = x2 - x1, y2 - y1
        if rw <= 0 or rh <= 0:
            continue

        cam_crop = cam[y1:y2, x1:x2].copy()

        # Circular mask in the local (rw × rh) coordinate space
        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.circle(mask, (cx - x1, cy - y1), r, 255, cv2.FILLED)

        inv_mask = cv2.bitwise_not(mask)
        cam_part = cv2.bitwise_and(cam_crop, cam_crop, mask=mask)
        bg_part  = cv2.bitwise_and(result[y1:y2, x1:x2], result[y1:y2, x1:x2], mask=inv_mask)
        result[y1:y2, x1:x2] = cv2.add(cam_part, bg_part)

    return result


def draw_cheat(frame, hand_state, raw_frame):
    global _ble_last_cheat_cmd

    # ── Background: cheat image with camera composited into green circles
    frame[:] = _composite_camera_circles(ASSETS['cheat'], raw_frame, CHEAT_CIRCLES)

    # ── Update controller
    wrist_y = hand_state.get('raw_wrist_y')
    out     = cheat_ctrl.update(wrist_y, hand_state)

    # ── BLE: send once per round when result arrives
    if ble is not None and out.get('state') == 'ROUND_RESULT':
        gest = out.get('computer_gesture', 'Unknown')
        if gest != 'Unknown':
            cmd = f"ROBOT_PLAY_{gest.upper()}"
            if cmd != _ble_last_cheat_cmd:
                try:
                    ble.send_command(cmd)
                    _ble_last_cheat_cmd = cmd
                except Exception:
                    pass
    elif out.get('state') != 'ROUND_RESULT':
        _ble_last_cheat_cmd = None    # reset dedup after round ends

    # ── HUD: Round / gesture summary — top-left, font scale 0.8
    rn  = out.get('session_round', 1)
    lp  = out.get('last_player',  '--')
    lr  = out.get('last_robot',   '--')
    hud = f"Round {rn}  --  YOU: {lp}  ROBOT: {lr}"
    cv2.putText(frame, hud, (20, 44), FONT, 0.8, WHITE, 2, cv2.LINE_AA)

    # ── State text overlay (countdown / SHOOT / result banner)
    state = out.get('state', '')
    main  = out.get('main_text', '')
    sub   = out.get('sub_text',  '')

    if state == 'COUNTDOWN':
        beats = out.get('beat_count', 0)
        label = str(min(beats, 3)) if beats > 0 else "READY"
        _text_centred(frame, label, SCREEN_H // 2, scale=3.5)
    elif main:
        _text_centred(frame, main, SCREEN_H // 2, scale=2.0)

    if sub:
        _text_centred(frame, sub, SCREEN_H // 2 + 60, scale=0.75)

    # Gesture detection status — bottom-left diagnostic
    gest_now = hand_state.get('confirmed_gesture', 'Unknown')
    det_txt  = f"Detected: {gest_now}"
    cv2.putText(frame, det_txt, (20, SCREEN_H - 30), FONT, 0.65, WHITE, 1, cv2.LINE_AA)


# ── CHALLENGE ─────────────────────────────────────────────────────────────────

def draw_challenge(frame, hand_state):
    """
    Draw the Challenge Mode screen.

    Returns:
        'WIN'  — if a game-over triggered and the player won ≥1 round
        'LOSE' — if a game-over triggered and the player won 0 rounds
        None   — normal frame, no transition needed
    """
    global _ble_last_challenge_cmd

    # ── Update controller
    wrist_y = hand_state.get('raw_wrist_y')
    out     = challenge_ctrl.update(wrist_y, hand_state)

    # ── Background based on cumulative session wins
    bg_key = challenge_ctrl.get_background_key()
    frame[:] = ASSETS[bg_key]

    # ── BLE: send robot gesture once per round
    if ble is not None and out.get('state') in ('ROUND_RESULT', 'MATCH_RESULT'):
        gest = out.get('computer_gesture', 'Unknown')
        if gest != 'Unknown':
            cmd = f"ROBOT_PLAY_{gest.upper()}"
            if cmd != _ble_last_challenge_cmd:
                try:
                    ble.send_command(cmd)
                    _ble_last_challenge_cmd = cmd
                except Exception:
                    pass
    elif out.get('state') not in ('ROUND_RESULT', 'MATCH_RESULT'):
        _ble_last_challenge_cmd = None

    # ── Score overlay — bottom-centre, large (scale 1.5)
    p_score = out.get('player_score', 0)   # = current streak
    r_score = out.get('robot_score',  0)   # = high score
    score   = f"YOU  {p_score}  --  ROBOT  {r_score}"
    _text_centred(frame, score, SCREEN_H - 55, scale=1.5)

    # ── Main state text
    state = out.get('state', '')
    main  = out.get('main_text', '')
    sub   = out.get('sub_text',  '')

    if state == 'COUNTDOWN':
        beats = out.get('beat_count', 0)
        label = str(min(beats, 3)) if beats > 0 else "READY"
        _text_centred(frame, label, SCREEN_H // 2 - 40, scale=3.5)
    elif main:
        _text_centred(frame, main, SCREEN_H // 2 - 40, scale=2.2)

    if sub:
        _text_centred(frame, sub, SCREEN_H // 2 + 40, scale=0.85)

    # ── Round counter — top-right
    rn = out.get('round_number', 1)
    rt = out.get('round_text', f"ROUND {rn}")
    cv2.putText(frame, rt, (SCREEN_W - 280, 44), FONT, 0.85, WHITE, 2, cv2.LINE_AA)

    # ── Win/lose transition trigger from AppChallengeController
    trigger = out.get('trigger_game_over')   # 'WIN', 'LOSE', or None
    return trigger   # caller uses this to change screen


# ── MIRROR ────────────────────────────────────────────────────────────────────

def draw_mirror(frame, hand_state):
    frame[:] = ASSETS['mirror_bg']

    # Pass landmarks to the mirror state (sends BLE + updates smoothed curls)
    lm_obj       = hand_state.get('_landmarks')
    world_lm_obj = hand_state.get('_world_landmarks')
    mirror_state.update(lm_obj, world_landmarks=world_lm_obj)

    curls = mirror_state.get_display_curls()   # [thumb, index, middle, ring]

    # ── Finger curl bars — bottom-left region (x=40, y≈500..780) ─────────────
    for i, (label, curl_val) in enumerate(zip(_CURL_LABELS, curls)):
        y_lbl = _BAR_Y0 + i * _BAR_STEP - 18   # label baseline above bar
        y_bar = _BAR_Y0 + i * _BAR_STEP         # bar top edge

        # Label in white, scale 0.7
        cv2.putText(frame, label, (_BAR_X, y_lbl), FONT, 0.7, WHITE, 2, cv2.LINE_AA)

        # Track (dark grey)
        cv2.rectangle(frame,
                      (_BAR_X,           y_bar),
                      (_BAR_X + _BAR_W,  y_bar + _BAR_H),
                      DARK_GREY, cv2.FILLED)

        # Fill (accent blue) proportional to curl 0-100
        fill_w = int(_BAR_W * max(0, min(100, curl_val)) / 100)
        if fill_w > 0:
            cv2.rectangle(frame,
                          (_BAR_X,            y_bar),
                          (_BAR_X + fill_w,   y_bar + _BAR_H),
                          ACCENT_BGR, cv2.FILLED)

        # Percentage value to the right of the bar
        pct_txt = f"{curl_val}%"
        cv2.putText(frame, pct_txt,
                    (_BAR_X + _BAR_W + 8, y_bar + _BAR_H - 2),
                    FONT, 0.6, WHITE, 1, cv2.LINE_AA)

    # Gesture indicator bottom-right
    gest = hand_state.get('confirmed_gesture', 'Unknown')
    cv2.putText(frame, f"Gesture: {gest}",
                (SCREEN_W - 320, SCREEN_H - 30),
                FONT, 0.65, WHITE, 1, cv2.LINE_AA)


# ── WIN / LOSE ────────────────────────────────────────────────────────────────

def draw_result(frame, result_type, now):
    """result_type: 'win' or 'lose'."""
    frame[:] = ASSETS['you_win' if result_type == 'win' else 'you_lose']
    _flash_enter_prompt(frame, now, y=SCREEN_H - 80)


# =============================================================================
# Key handler
# =============================================================================

def handle_keys(key):
    """
    Update the global screen / menu_sel from a waitKey() result.

    Navigation:
        ESC           — back to previous screen
        ENTER (13/10) — confirm / advance
        LEFT  (81/2)  — previous menu item
        RIGHT (83/3)  — next menu item
        Q             — quit (handled in main loop)
    """
    global screen, menu_sel

    if key in (255, 0xFF, -1 & 0xFF):
        return

    esc   = (key == 27)
    enter = (key in (13, 10))
    left  = (key in (81, 2,  ord('a')))
    right = (key in (83, 3,  ord('d')))

    # ── ESC ───────────────────────────────────────────────────────────
    if esc:
        if screen == 'CHEAT':
            cheat_ctrl.reset()
            screen = 'MENU'
        elif screen == 'CHALLENGE':
            challenge_ctrl.reset()
            screen = 'MENU'
        elif screen == 'MIRROR':
            mirror_state.stop()
            screen = 'MENU'
        elif screen in ('WIN', 'LOSE'):
            screen = 'MENU'
        elif screen == 'MENU':
            screen = 'TITLE'

    # ── ENTER ─────────────────────────────────────────────────────────
    elif enter:
        if screen == 'TITLE':
            screen = 'MENU'

        elif screen == 'MENU':
            if menu_sel == 0:
                cheat_ctrl.reset()
                screen = 'CHEAT'
            elif menu_sel == 1:
                challenge_ctrl.reset()
                screen = 'CHALLENGE'
            elif menu_sel == 2:
                mirror_state.start()
                screen = 'MIRROR'

        elif screen in ('WIN', 'LOSE'):
            screen = 'MENU'

    # ── LEFT / RIGHT arrows (menu navigation) ─────────────────────────
    elif left and screen == 'MENU':
        menu_sel = (menu_sel - 1) % 3

    elif right and screen == 'MENU':
        menu_sel = (menu_sel + 1) % 3


# =============================================================================
# Camera
# =============================================================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[error] Could not open camera (index 0)")
    sys.exit(1)

print("[ready] Camera opened — starting main loop")

# =============================================================================
# Main loop
# =============================================================================

while True:
    ret, raw_frame = cap.read()
    if not ret:
        continue

    # Working canvas at screen resolution (filled by each draw function)
    frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    now   = time.monotonic()

    # ── Hand / gesture detection (every frame, all screens) ──────────
    hand_state = detector.process(raw_frame)

    # ── Screen routing ────────────────────────────────────────────────
    if screen == 'TITLE':
        draw_title(frame, now)

    elif screen == 'MENU':
        draw_menu(frame, menu_sel)

    elif screen == 'CHEAT':
        draw_cheat(frame, hand_state, raw_frame)

    elif screen == 'CHALLENGE':
        trigger = draw_challenge(frame, hand_state)
        if trigger:                         # 'WIN' or 'LOSE'
            screen = trigger

    elif screen == 'MIRROR':
        draw_mirror(frame, hand_state)

    elif screen == 'WIN':
        draw_result(frame, 'win', now)

    elif screen == 'LOSE':
        draw_result(frame, 'lose', now)

    # ── Keyboard ──────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    handle_keys(key)

    # ── Display ───────────────────────────────────────────────────────
    cv2.imshow(WIN_NAME, frame)

    if key == ord('q'):
        print("[quit] Q pressed — exiting")
        break

# ── Teardown ──────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
if ble is not None:
    try:
        ble.stop()
    except Exception:
        pass
print("[done]")
