"""
app_cheat_state.py
==================
Cheat Mode state for the standalone RPS App.

Wraps RPSGameController from ~/rps_hand_counter/rps_game_state.py and adds
a per-session round counter and gesture-abbreviation tracking so the renderer
can display:  "Round N  --  YOU: X  ROBOT: Y"

The robot ALWAYS beats the player — it detects the player's gesture via
MediaPipe then selects the winning counter-move.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser('~/rps_hand_counter'))

from rps_game_state import RPSGameController, WIN_MAP, VALID_GESTURES  # noqa: F401 (re-export)

# ── Tuning constants ──────────────────────────────────────────────────────────
# How many consecutive frames of raw OR stable "Rock" the controller needs
# before it accepts the fist and starts the countdown.  The default path uses
# GestureStateTracker.confirmed_gesture which takes ~10 frames (7-frame majority
# window + 3-frame stable streak).  6 is faster while still preventing spurious
# single-frame Rock blips from accidentally triggering the countdown.
ROCK_HOLD_FRAMES_REQUIRED = 6

# Minimum normalised wrist-Y drop to register one pump beat.
# Original rps_game_state default is 0.045; 0.04 is ~11 % more sensitive.
PUMP_MIN_DELTA_Y = 0.04

# Minimum time (seconds) between two registered pump beats.
# At 30 fps, 0.10 s ≈ 3 frames.  Original default is 0.18 s (~5-6 frames).
PUMP_DEBOUNCE_SECS = 0.10

# Compact display abbreviations
_ABBREV = {"Rock": "R", "Paper": "P", "Scissors": "S", "Unknown": "--"}


class CheatController(RPSGameController):
    """
    Extends RPSGameController with a per-session round counter and
    per-round gesture abbreviations for the top-left HUD overlay.

    Extra keys injected into the update() output dict:
        session_round  — 1-indexed round counter (increments after each throw)
        player_score   — always 0 (player never wins in cheat mode)
        robot_score    — cumulative robot wins this session
        last_player    — single-char abbreviation of the player's last gesture
        last_robot     — single-char abbreviation of the robot's last gesture
    """

    def __init__(self, robot_output=None, **kwargs):
        # Override pump thresholds with the more-sensitive cheat-mode values
        # unless the caller has explicitly passed their own values.
        kwargs.setdefault('down_threshold', PUMP_MIN_DELTA_Y)
        kwargs.setdefault('beat_cooldown',  PUMP_DEBOUNCE_SECS)
        super().__init__(robot_output=robot_output, **kwargs)
        self._init_session_state()

    # ------------------------------------------------------------------
    # Session state helpers
    # ------------------------------------------------------------------

    def _init_session_state(self):
        self.session_round    = 1
        self.player_score     = 0         # always 0 — cheat AI never loses
        self.robot_score      = 0
        self.last_player      = "--"
        self.last_robot       = "--"
        self._counted_round   = False     # guard: count each ROUND_RESULT only once
        self._prev_state_ch   = "WAITING_FOR_ROCK"
        self._rock_hold_count = 0         # consecutive frames raw/stable Rock visible

    def reset(self):
        """Full reset including session counters — call when re-entering cheat mode."""
        super().reset()
        self._init_session_state()

    # ------------------------------------------------------------------
    # Override update to inject session data
    # ------------------------------------------------------------------

    def update(self, wrist_y, tracker_state, now=None):
        # ── Fix 1: fast Rock-hold counter bypasses confirmed_gesture lag ──
        #
        # GestureStateTracker.confirmed_gesture requires a 7-frame majority
        # window PLUS a 3-frame stable streak ≈ 10 frames before WAITING_FOR_ROCK
        # will transition.  Instead, count consecutive frames where the raw OR
        # stable gesture is already Rock.  Once the count reaches
        # ROCK_HOLD_FRAMES_REQUIRED we inject "Rock" as confirmed_gesture —
        # but ONLY during WAITING_FOR_ROCK so SHOOT_WINDOW detection is unaffected.
        raw_g    = tracker_state.get('raw_gesture',    'Unknown')
        stable_g = tracker_state.get('stable_gesture', 'Unknown')

        if raw_g == 'Rock' or stable_g == 'Rock':
            self._rock_hold_count += 1
        else:
            # Decay rather than hard-zero so a brief 1-frame flicker doesn't
            # fully erase progress.
            self._rock_hold_count = max(0, self._rock_hold_count - 2)

        # Inject fast confirm only while waiting for the starting fist
        if (self.state == 'WAITING_FOR_ROCK'
                and self._rock_hold_count >= ROCK_HOLD_FRAMES_REQUIRED):
            tracker_state = dict(tracker_state)          # shallow copy, don't mutate caller's dict
            tracker_state['confirmed_gesture'] = 'Rock'

        prev_state = self.state
        out = super().update(wrist_y, tracker_state, now=now)

        # ── Entering ROUND_RESULT for the first time this round ───────
        if self.state == "ROUND_RESULT" and not self._counted_round:
            self._counted_round = True
            self.last_player   = _ABBREV.get(self.player_gesture,   "--")
            self.last_robot    = _ABBREV.get(self.computer_gesture, "--")
            # Robot always wins — every completed round is a robot win
            self.robot_score  += 1

        # ── Leaving ROUND_RESULT → increment round counter ───────────
        if prev_state == "ROUND_RESULT" and self.state != "ROUND_RESULT":
            self.session_round += 1
            self._counted_round = False

        # Inject extras into the output dict
        out["session_round"] = self.session_round
        out["player_score"]  = self.player_score
        out["robot_score"]   = self.robot_score
        out["last_player"]   = self.last_player
        out["last_robot"]    = self.last_robot
        return out
