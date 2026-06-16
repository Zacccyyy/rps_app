#!/usr/bin/env python3
"""
main_app.py — RPS Robot standalone fullscreen app.

Three modes accessible from a menu:
  Cheat     — robot always picks the counter-move that beats you.
  Challenge — survival run; robot difficulty ramps with your streak.
  Mirror    — robot hand mirrors your finger curls via BLE.

All visuals are drawn programmatically via ui_renderer.py — no image assets.
Camera feed fills each gameplay screen; animated HUD overlays live on top.

Usage:
    python3 main_app.py
Quit:
    press Q
"""

import math
import os
import sys
import time

import cv2
import numpy as np

# ── Shared modules from rps_hand_counter ────────────────────────────────────
sys.path.insert(0, os.path.expanduser('~/rps_hand_counter'))

from hand_landmarks import (
    process_hand_frame,
    create_hands_detector,
    create_kalman_wrist_state,
)
from gesture_state import GestureStateTracker

# ── Mode state modules ───────────────────────────────────────────────────────
from app_cheat_state     import CheatController
from app_challenge_state import AppChallengeController
from app_mirror_state    import AppMirrorState

# ── Visual design system ────────────────────────────────────────────────────
import ui_renderer as UI

# ── Sound engine ────────────────────────────────────────────────────────────
import sound_engine as SFX
SFX.init()

# ── Optional BLE bridge ──────────────────────────────────────────────────────
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

RESULT_MIN_DISPLAY    = 4.0        # seconds the result screen must stay visible
result_show_time      = 0.0
result_player_gesture = 'Unknown'
result_robot_gesture  = 'Unknown'

# BLE dedup: avoid re-sending the same command every frame
_ble_last_cheat_cmd     = None
_ble_last_challenge_cmd = None

# =============================================================================
# Window setup
# =============================================================================

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# =============================================================================
# Hand landmark detector
# =============================================================================

class HandLandmarkDetector:
    """Bundles MediaPipe Hands + Kalman wrist filter + multi-frame tracker."""

    def __init__(self):
        self._hands   = create_hands_detector()
        self._ema     = create_kalman_wrist_state()
        self._tracker = GestureStateTracker()

    def process(self, raw_frame):
        _, hand_state, _ = process_hand_frame(
            raw_frame,
            self._hands,
            target_hand='Auto',
            display_mode='Game',
            hand_orientation='Side',
            _ema_state=self._ema,
        )
        tracker_out = self._tracker.update(hand_state.get('raw_gesture', 'Unknown'))
        hand_state.update(tracker_out)
        return hand_state

# =============================================================================
# Game state objects
# =============================================================================

detector       = HandLandmarkDetector()
cheat_ctrl     = CheatController()
challenge_ctrl = AppChallengeController()
mirror_state   = AppMirrorState(ble_bridge=ble)

# =============================================================================
# App state
# =============================================================================

screen   = 'TITLE'   # TITLE | MENU | CHEAT | CHALLENGE | MIRROR | WIN | LOSE
menu_sel = 0         # 0=Cheat 1=Challenge 2=Mirror

# ── Sound state tracking — detect transitions so we fire each sound once ─────
_prev_cheat_state      = 'WAITING_FOR_ROCK'
_prev_challenge_state  = 'WAITING_FOR_ROCK'
_prev_beat_ch          = 0
_prev_beat_chal        = 0


def _reset_cheat_sound_state():
    global _prev_cheat_state, _prev_beat_ch
    _prev_cheat_state = 'WAITING_FOR_ROCK'
    _prev_beat_ch     = 0


def _reset_challenge_sound_state():
    global _prev_challenge_state, _prev_beat_chal
    _prev_challenge_state = 'WAITING_FOR_ROCK'
    _prev_beat_chal       = 0


def _enter_result_screen(screen_name, player_gest='Unknown', robot_gest='Unknown'):
    global screen, result_show_time, result_player_gesture, result_robot_gesture
    screen                = screen_name
    result_show_time      = time.monotonic()
    result_player_gesture = player_gest
    result_robot_gesture  = robot_gest
    SFX.play('game_win' if screen_name == 'WIN' else 'game_lose')

# =============================================================================
# Mirror-mode curl bars (drawn inline, styled with neon palette)
# =============================================================================

_CURL_LABELS = ['THUMB', 'INDEX', 'MIDDLE', 'RING']
_BAR_X  = 30
_BAR_W  = 260
_BAR_H  = 18
_BAR_Y0 = 90     # y of first bar top (clears top HUD bar)
_BAR_STEP = 52


def _draw_mirror_bars(frame, curls):
    """Four neon curl bars in the top-left corner for Mirror mode."""
    col   = UI.P['cyan']
    dark  = UI.P['dark_grey']
    panel = UI.P['panel']

    for i, (label, val) in enumerate(zip(_CURL_LABELS, curls)):
        y_lbl = _BAR_Y0 + i * _BAR_STEP
        y_bar = y_lbl + 20

        # Label
        cv2.putText(frame, label, (_BAR_X, y_lbl),
                    UI.FS, 0.60, col, 1, cv2.LINE_AA)

        # Track — dark fill + border
        cv2.rectangle(frame, (_BAR_X, y_bar), (_BAR_X + _BAR_W, y_bar + _BAR_H),
                      panel, cv2.FILLED)
        cv2.rectangle(frame, (_BAR_X, y_bar), (_BAR_X + _BAR_W, y_bar + _BAR_H),
                      dark, 1)

        # Fill proportional to 0-100 curl value
        fill_w = int(_BAR_W * max(0, min(100, val)) / 100)
        if fill_w > 0:
            cv2.rectangle(frame, (_BAR_X, y_bar), (_BAR_X + fill_w, y_bar + _BAR_H),
                          col, cv2.FILLED)

        # Percentage label
        pct = f"{val}%"
        cv2.putText(frame, pct, (_BAR_X + _BAR_W + 8, y_bar + _BAR_H - 2),
                    UI.FS, 0.55, col, 1, cv2.LINE_AA)

# =============================================================================
# Screen draw functions
# =============================================================================

# ── TITLE ────────────────────────────────────────────────────────────────────

def draw_title(frame, now):
    UI.draw_title_screen(frame, now)


# ── MENU ─────────────────────────────────────────────────────────────────────

def draw_menu(frame, sel, now):
    UI.draw_menu_screen(frame, sel, now)


# ── CHEAT ────────────────────────────────────────────────────────────────────

def draw_cheat(frame, hand_state, raw_frame, now):
    global _ble_last_cheat_cmd, _prev_cheat_state, _prev_beat_ch

    # Full-screen camera background (slightly darkened)
    UI.draw_camera_bg(frame, raw_frame, darken=0.45)

    # Update controller
    wrist_y = hand_state.get('raw_wrist_y')
    out     = cheat_ctrl.update(wrist_y, hand_state)

    # ── Sound triggers (fire once per state transition / beat) ─────────
    state = out.get('state', '')
    beats = out.get('beat_count', 0)
    if state != _prev_cheat_state:
        if state == 'COUNTDOWN':
            SFX.play('rock_ready')
        elif state == 'SHOOT_WINDOW':
            SFX.play('shoot')
    if state == 'COUNTDOWN' and beats != _prev_beat_ch and 1 <= beats <= 3:
        SFX.play('beat')
    _prev_cheat_state = state
    _prev_beat_ch     = beats

    # BLE — send once per round when result first arrives
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
        _ble_last_cheat_cmd = None

    # Top HUD
    rn      = out.get('session_round', 1)
    p_score = out.get('player_score',  0)
    r_score = out.get('robot_score',   0)
    UI.draw_top_hud(frame, 'CHEAT', f'ROUND {rn}',
                    f'YOU {p_score}   ROBOT {r_score}')

    # State text / countdown
    state = out.get('state', '')
    main  = out.get('main_text', '')
    sub   = out.get('sub_text',  '')

    if state == 'COUNTDOWN':
        beats = out.get('beat_count', 0)
        UI.draw_countdown(frame, str(min(beats, 3)) if beats > 0 else 'READY')
    elif state == 'ROUND_RESULT' and main:
        UI.draw_state_pill(frame, main, sub, colour_key='magenta')
    elif main and state != 'ROUND_INTRO':
        UI.draw_state_pill(frame, main, sub, colour_key='cyan')

    # Robot face panel — shows what robot is playing, smug on result
    robot_gest = out.get('computer_gesture', 'Unknown')
    mood       = 'smug' if state == 'ROUND_RESULT' else 'neutral'
    label      = f'ROBOT: {robot_gest}' if robot_gest not in ('Unknown', None) else 'ROBOT'
    UI.draw_robot_panel(frame, mood=mood, now=now, gesture_label=label)

    # Gesture indicator (bottom-left, small)
    UI.draw_gesture_indicator(frame, hand_state.get('confirmed_gesture', 'Unknown'))


# ── CHALLENGE ────────────────────────────────────────────────────────────────

def _session_wins_to_mood(wins):
    if wins >= 8:   return 'angry3'
    if wins >= 5:   return 'angry2'
    if wins >= 3:   return 'angry1'
    return 'neutral'


def draw_challenge(frame, hand_state, raw_frame, now):
    """Returns 'WIN', 'LOSE', or None for the game-over transition."""
    global _ble_last_challenge_cmd, _prev_challenge_state, _prev_beat_chal

    # Update controller
    wrist_y = hand_state.get('raw_wrist_y')
    out     = challenge_ctrl.update(wrist_y, hand_state)
    wins    = out.get('session_wins', 0)

    # ── Sound triggers ──────────────────────────────────────────────────
    state = out.get('state', '')
    beats = out.get('beat_count', 0)
    if state != _prev_challenge_state:
        if state == 'COUNTDOWN':
            SFX.play('rock_ready')
        elif state == 'SHOOT_WINDOW':
            SFX.play('shoot')
        elif state == 'ROUND_RESULT':
            result = getattr(challenge_ctrl, 'last_round_result', None)
            SFX.play('round_win' if result == 'player_win' else 'round_lose')
    if state == 'COUNTDOWN' and beats != _prev_beat_chal and 1 <= beats <= 3:
        SFX.play('beat')
    _prev_challenge_state = state
    _prev_beat_chal       = beats

    # Camera background — darken more as robot gets angrier
    darken = 0.35 + 0.15 * (wins / 10)
    UI.draw_camera_bg(frame, raw_frame, darken=min(darken, 0.60))

    # Red vignette that grows with anger
    vig = UI.challenge_vignette_intensity(wins)
    if vig > 0:
        pulse = 1.0 if wins < 8 else (0.75 + 0.25 * math.sin(now * 6 * math.pi))
        UI.vignette(frame, UI.P['red'], vig * pulse)

    # BLE
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

    # Top HUD
    p_score = out.get('player_score', 0)   # current streak
    r_score = out.get('robot_score',  0)   # high score
    rn      = out.get('round_number', 1)
    UI.draw_top_hud(frame, 'CHALLENGE', f'ROUND {rn}',
                    f'STREAK {p_score}   BEST {r_score}')

    # Robot face — grows more aggressively with wins
    mood = _session_wins_to_mood(wins)
    UI.draw_challenge_robot(frame, mood, now, wins)

    # State text / countdown
    state = out.get('state', '')
    main  = out.get('main_text', '')
    sub   = out.get('sub_text',  '')

    if state == 'COUNTDOWN':
        beats = out.get('beat_count', 0)
        UI.draw_countdown(frame, str(min(beats, 3)) if beats > 0 else 'READY')
    elif state == 'ROUND_RESULT' and main:
        UI.draw_state_pill(frame, main, sub, colour_key='magenta')
    elif main and state != 'ROUND_INTRO':
        UI.draw_state_pill(frame, main, sub, colour_key='cyan')

    # Gesture indicator
    UI.draw_gesture_indicator(frame, hand_state.get('confirmed_gesture', 'Unknown'))

    return out.get('trigger_game_over')   # 'WIN', 'LOSE', or None


# ── MIRROR ───────────────────────────────────────────────────────────────────

def draw_mirror(frame, hand_state, raw_frame, now):
    UI.draw_camera_bg(frame, raw_frame, darken=0.50)

    # Update mirror state
    mirror_state.update(
        hand_state.get('_landmarks'),
        world_landmarks=hand_state.get('_world_landmarks'),
    )
    curls = mirror_state.get_display_curls()

    # Top HUD
    UI.draw_top_hud(frame, 'MIRROR', 'CURL YOUR FINGERS', '')

    # Curl bars (top-left, below HUD)
    _draw_mirror_bars(frame, curls)

    # Robot face — neutral, just watching
    UI.draw_robot_panel(frame, mood='neutral', now=now, gesture_label='MIRROR')

    # Gesture indicator
    UI.draw_gesture_indicator(frame, hand_state.get('confirmed_gesture', 'Unknown'))


# ── WIN / LOSE ───────────────────────────────────────────────────────────────

def draw_result(frame, result_type, now):
    UI.draw_win_lose_screen(
        frame, result_type,
        result_player_gesture, result_robot_gesture,
        result_show_time, now, RESULT_MIN_DISPLAY,
    )

# =============================================================================
# Key handler
# =============================================================================

def handle_keys(key, now):
    global screen, menu_sel

    if key in (255, 0xFF, -1 & 0xFF):
        return

    esc   = (key == 27)
    enter = (key in (13, 10))
    left  = (key in (81, 2,  ord('a')))
    right = (key in (83, 3,  ord('d')))

    if esc:
        if screen == 'CHEAT':
            cheat_ctrl.reset()
            _reset_cheat_sound_state()
            screen = 'MENU'
            SFX.play('menu_tick')
        elif screen == 'CHALLENGE':
            challenge_ctrl.reset()
            _reset_challenge_sound_state()
            screen = 'MENU'
            SFX.play('menu_tick')
        elif screen == 'MIRROR':
            mirror_state.stop()
            screen = 'MENU'
            SFX.play('menu_tick')
        elif screen in ('WIN', 'LOSE'):
            if now - result_show_time > RESULT_MIN_DISPLAY:
                screen = 'MENU'
        elif screen == 'MENU':
            screen = 'TITLE'
            SFX.play('menu_tick')

    elif enter:
        if screen == 'TITLE':
            screen = 'MENU'
            SFX.play('menu_tick')
        elif screen == 'MENU':
            SFX.play('menu_tick')
            if menu_sel == 0:
                cheat_ctrl.reset()
                _reset_cheat_sound_state()
                screen = 'CHEAT'
            elif menu_sel == 1:
                challenge_ctrl.reset()
                _reset_challenge_sound_state()
                screen = 'CHALLENGE'
            elif menu_sel == 2:
                mirror_state.start()
                screen = 'MIRROR'
        elif screen in ('WIN', 'LOSE'):
            if now - result_show_time > RESULT_MIN_DISPLAY:
                screen = 'MENU'

    elif left and screen == 'MENU':
        menu_sel = (menu_sel - 1) % 3
        SFX.play('menu_tick')

    elif right and screen == 'MENU':
        menu_sel = (menu_sel + 1) % 3
        SFX.play('menu_tick')

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

    frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    now   = time.monotonic()

    hand_state = detector.process(raw_frame)

    if screen == 'TITLE':
        draw_title(frame, now)

    elif screen == 'MENU':
        draw_menu(frame, menu_sel, now)

    elif screen == 'CHEAT':
        draw_cheat(frame, hand_state, raw_frame, now)

    elif screen == 'CHALLENGE':
        trigger = draw_challenge(frame, hand_state, raw_frame, now)
        if trigger:
            p_gest = challenge_ctrl._game_over_player_gesture or 'Unknown'
            r_gest = challenge_ctrl._game_over_robot_gesture  or 'Unknown'
            _enter_result_screen(trigger, p_gest, r_gest)

    elif screen == 'MIRROR':
        draw_mirror(frame, hand_state, raw_frame, now)

    elif screen == 'WIN':
        draw_result(frame, 'win', now)

    elif screen == 'LOSE':
        draw_result(frame, 'lose', now)

    key = cv2.waitKey(1) & 0xFF
    handle_keys(key, now)
    cv2.imshow(WIN_NAME, frame)

    if key == ord('q'):
        print("[quit] Q pressed — exiting")
        break

# =============================================================================
# Teardown
# =============================================================================

cap.release()
cv2.destroyAllWindows()
if ble is not None:
    try:
        ble.stop()
    except Exception:
        pass
print("[done]")
