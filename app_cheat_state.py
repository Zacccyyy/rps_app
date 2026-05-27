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

    def reset(self):
        """Full reset including session counters — call when re-entering cheat mode."""
        super().reset()
        self._init_session_state()

    # ------------------------------------------------------------------
    # Override update to inject session data
    # ------------------------------------------------------------------

    def update(self, wrist_y, tracker_state, now=None):
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
