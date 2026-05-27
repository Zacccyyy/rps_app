"""
app_mirror_state.py
===================
Mirror Mode state for the standalone RPS App.

Wraps MirrorModeState from ~/rps_hand_counter/mirror_mode_state.py and
re-exports extract_finger_curls so the main app can access curl values
for the finger-bar UI overlay.

The finger bars display: THUMB / INDEX / MIDDLE / RING, each 0-100.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser('~/rps_hand_counter'))

from mirror_mode_state import MirrorModeState, extract_finger_curls  # noqa: F401 (re-export)


class AppMirrorState(MirrorModeState):
    """
    Mirror Mode state with a safe get_display_curls() accessor.

    The base class sends BLE commands; this subclass adds a cached curl
    readout so the renderer can always read the latest values even when
    no landmarks are detected this frame.
    """

    def __init__(self, ble_bridge=None):
        super().__init__(ble_bridge=ble_bridge)
        self._display_curls = [0, 0, 0, 0]

    def get_display_curls(self):
        """
        Return latest smoothed [thumb, index, middle, ring] curl values (0-100).
        Returns [0,0,0,0] when no hand has been seen yet.
        """
        return list(self._display_curls)

    def update(self, landmarks, world_landmarks=None):
        """
        Call every frame with MediaPipe NormalizedLandmarkList (or None).
        Updates the cached display curls and, if BLE is connected, sends
        the mirror command at 10 fps.
        Returns [t, i, m, r] or None when landmarks are absent.
        """
        if landmarks is None:
            return None

        result = super().update(landmarks, world_landmarks=world_landmarks)
        if result is not None:
            self._display_curls = list(result)
        return result
