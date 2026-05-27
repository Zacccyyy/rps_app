"""
app_challenge_state.py
======================
Challenge Mode state for the standalone RPS App.

Wraps ChallengeController from ~/rps_hand_counter/challenge_mode_state.py and
adds:
  - session_wins  : cumulative player-win count since this visit to the mode
                    (resets on reset(), persists across runs within a session)
  - trigger_game_over: injected into the update() output dict the frame the
                    game-over timer expires, so the main app can transition to
                    the WIN or LOSE screen.
  - get_background_key(): maps session_wins to the correct ASSETS key.

Background mapping:
    0-2 wins  → ch_idle
    3-4 wins  → ch_angry1
    5-7 wins  → ch_angry2
    8+  wins  → ch_angry3
"""

import os
import sys

sys.path.insert(0, os.path.expanduser('~/rps_hand_counter'))

from challenge_mode_state import ChallengeController, VALID_GESTURES, BEATS, compare_rps  # noqa: F401
from challenge_ai import ChallengeAI  # noqa: F401 (available for callers)


class AppChallengeController(ChallengeController):
    """
    Challenge Mode controller with session-level win tracking and
    WIN/LOSE screen trigger for the standalone app.
    """

    def __init__(self):
        # Disable stats logging (no Excel file dependency)
        super().__init__(stats_logger=None)
        self._init_session()

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------

    def _init_session(self):
        self.session_wins              = 0
        self._game_over_streak         = None   # streak value captured when MATCH_RESULT starts
        self._game_over_player_gesture = None   # player's final throw (captured before reset)
        self._game_over_robot_gesture  = None   # robot's final throw (captured before reset)

    def reset(self):
        """Full reset — call when re-entering challenge mode from the menu."""
        super().reset()
        self._init_session()

    # ------------------------------------------------------------------
    # Background key helper
    # ------------------------------------------------------------------

    def get_background_key(self):
        """Return the ASSETS dict key for the background based on session_wins."""
        if self.session_wins >= 8:
            return 'ch_angry3'
        elif self.session_wins >= 5:
            return 'ch_angry2'
        elif self.session_wins >= 3:
            return 'ch_angry1'
        return 'ch_idle'

    # ------------------------------------------------------------------
    # Override update to track wins and game-over transitions
    # ------------------------------------------------------------------

    def update(self, wrist_y, tracker_state, now=None):
        prev_state = self.state
        out = super().update(wrist_y, tracker_state, now=now)

        # ── Count a player win when we first land in ROUND_RESULT ─────
        # last_round_result is set by _resolve_round() before state changes
        if (prev_state == "SHOOT_WINDOW"
                and self.state == "ROUND_RESULT"
                and self.last_round_result == "player_win"):
            self.session_wins += 1

        # ── Entering MATCH_RESULT → save streak + final gestures ─────
        # self.streak / player_gesture / computer_gesture are intact at this
        # point; reset_run() hasn't been called yet.
        if prev_state != "MATCH_RESULT" and self.state == "MATCH_RESULT":
            self._game_over_streak         = self.streak
            self._game_over_player_gesture = self.player_gesture
            self._game_over_robot_gesture  = self.computer_gesture

        # ── Leaving MATCH_RESULT (reset_run called) → fire trigger ────
        trigger = None
        if prev_state == "MATCH_RESULT" and self.state != "MATCH_RESULT":
            if self._game_over_streak is not None:
                trigger = "WIN" if self._game_over_streak > 0 else "LOSE"
                self._game_over_streak = None

        out["session_wins"]      = self.session_wins
        out["trigger_game_over"] = trigger   # None or 'WIN' or 'LOSE'
        return out
