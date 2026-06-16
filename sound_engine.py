"""
sound_engine.py — Synthesised game audio for RPS Robot.

All sounds are generated programmatically from sine/square waves using numpy
and pygame.mixer.  No audio files needed.

Usage:
    import sound_engine as SFX
    SFX.init()          # call once at startup
    SFX.play('beat')    # fire a named sound
"""

import math
import numpy as np

_READY  = False
_SOUNDS = {}

SR = 44100   # sample rate (Hz)


# ── Wave generators ──────────────────────────────────────────────────────────

def _sine(freq, dur, vol=0.45, attack=0.005, release=0.0):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    w = np.sin(2 * math.pi * freq * t) * vol
    a = int(SR * attack)
    if 0 < a < n:
        w[:a] *= np.linspace(0, 1, a)
    r = int(SR * release)
    if 0 < r <= n:
        w[n-r:] *= np.linspace(1, 0, r)
    return w


def _square(freq, dur, vol=0.25, attack=0.004, release=0.0):
    """Buzzy square wave for failure / angry sounds."""
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    w = (np.sin(2 * math.pi * freq * t) > 0).astype(np.float32) * 2 - 1
    w *= vol
    a = int(SR * attack)
    if 0 < a < n:
        w[:a] *= np.linspace(0, 1, a)
    r = int(SR * release)
    if 0 < r <= n:
        w[n-r:] *= np.linspace(1, 0, r)
    return w


def _concat(*waves):
    return np.concatenate(waves)


def _gap(dur):
    return np.zeros(int(SR * dur))


def _make(wave):
    import pygame
    s16 = np.clip(wave * 32767, -32767, 32767).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([s16, s16]))
    return pygame.sndarray.make_sound(stereo)


# ── Public API ───────────────────────────────────────────────────────────────

def init():
    """
    Initialise pygame.mixer and pre-build all game sounds.
    Safe to call even if pygame is not installed — degrades gracefully.
    """
    global _READY

    try:
        import pygame
        pygame.mixer.pre_init(SR, -16, 2, 256)
        pygame.mixer.init()

        # ── Navigation tick: short 8-bit chirp ────────────────────────
        _SOUNDS['menu_tick'] = _make(
            _sine(880, 0.035, vol=0.20, release=0.02)
        )

        # ── Rock confirmed (fist held) ─────────────────────────────────
        # Low soft thud — reassuring feedback that the round is starting
        _SOUNDS['rock_ready'] = _make(
            _sine(160, 0.10, vol=0.35, attack=0.003, release=0.06)
        )

        # ── Countdown beats (3 / 2 / 1): identical beep, played once per beat
        _SOUNDS['beat'] = _make(
            _sine(880, 0.090, vol=0.40, attack=0.003, release=0.04)
        )

        # ── SHOOT! — two-note stinger: punchy then bright ─────────────
        _SOUNDS['shoot'] = _make(_concat(
            _sine(1046.5, 0.055, vol=0.50, attack=0.002),
            _sine(1318.5, 0.130, vol=0.55, attack=0.003, release=0.08),
        ))

        # ── Round won (challenge mode) ─────────────────────────────────
        _SOUNDS['round_win'] = _make(_concat(
            _sine(523.25, 0.065, vol=0.38, release=0.02),
            _gap(0.018),
            _sine(659.25, 0.110, vol=0.42, release=0.06),
        ))

        # ── Round lost / robot scores (challenge mode) ─────────────────
        _SOUNDS['round_lose'] = _make(_concat(
            _square(440, 0.055, vol=0.28),
            _gap(0.018),
            _square(330, 0.110, vol=0.22, release=0.07),
        ))

        # ── Game WIN: ascending C-E-G-C fanfare ───────────────────────
        _SOUNDS['game_win'] = _make(_concat(
            _sine(523.25, 0.095, vol=0.46, release=0.02),
            _gap(0.018),
            _sine(659.25, 0.095, vol=0.46, release=0.02),
            _gap(0.018),
            _sine(783.99, 0.095, vol=0.46, release=0.02),
            _gap(0.018),
            _sine(1046.5, 0.260, vol=0.55, release=0.18),
        ))

        # ── Game LOSE: descending square-wave wail ────────────────────
        _SOUNDS['game_lose'] = _make(_concat(
            _square(440, 0.080, vol=0.32),
            _gap(0.018),
            _square(370, 0.080, vol=0.30),
            _gap(0.018),
            _square(311, 0.200, vol=0.26, release=0.14),
        ))

        _READY = True
        print("[audio] Sound engine ready — 7 sounds loaded")

    except Exception as e:
        _READY = False
        print(f"[audio] unavailable — {e}")


def play(name):
    """Fire a named sound, silently ignoring if audio is unavailable."""
    if _READY and name in _SOUNDS:
        _SOUNDS[name].play()
