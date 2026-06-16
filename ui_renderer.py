"""
ui_renderer.py — Programmatic visual design for RPS Robot.

Every screen is drawn entirely from OpenCV primitives — no image files needed.
Tune the PALETTE dict and the module constants at the top to retheme anything.
All draw_* functions write into the caller's frame (1280×800 numpy array).
Animated functions require `now = time.monotonic()` to be passed in.
"""

import math
import cv2
import numpy as np

# =============================================================================
# PALETTE  (all colours are BGR — OpenCV order)
# =============================================================================
# fmt: off
P = {
    'bg':          ( 15,  10,  10),   # #0a0a0f — near-black with purple tint
    'panel':       ( 26,  18,  18),   # #12101a — dark panel background
    'panel_sel':   ( 48,  34,  36),   # #24222f — selected / active panel
    'border':      ( 65,  55,  72),   # #484138 — subtle separator
    'cyan':        (255, 245,   0),   # #00f5ff — primary neon accent
    'magenta':     (170,   0, 255),   # #ff00aa — secondary neon accent
    'green':       ( 40, 210,  60),   # #3cd228 — positive / WIN
    'yellow':      (  0, 210, 255),   # #ffd200 — caution / angry-1
    'orange':      (  0, 140, 255),   # #ff8c00 — angry-2
    'red':         ( 40,  40, 210),   # #d22828 — angry-3 / LOSE
    'white':       (255, 255, 255),
    'light_grey':  (180, 180, 185),
    'grey':        (110, 110, 115),
    'dark_grey':   ( 55,  55,  60),
    'black':       (  0,   0,   0),
}
# fmt: on

# Expose colour palette as PALETTE for external callers
PALETTE = P

# Screen dimensions (must match main_app.py SCREEN_W / SCREEN_H)
W = 1280
H = 800

# Font shortcuts
FD = cv2.FONT_HERSHEY_DUPLEX
FS = cv2.FONT_HERSHEY_SIMPLEX

# Mood → neon colour mapping (used by draw_robot_face and callers)
MOOD_COLOUR = {
    'neutral': P['cyan'],
    'smug':    P['cyan'],
    'happy':   P['green'],
    'angry1':  P['yellow'],
    'angry2':  P['orange'],
    'angry3':  P['red'],
    'sad':     (160,  60, 200),   # muted purple
}

# =============================================================================
# Low-level helpers
# =============================================================================

def _blend(frame, x1, y1, x2, y2, colour, alpha):
    """Semi-transparent filled rectangle (alpha 0=invisible, 1=opaque)."""
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), colour, cv2.FILLED)
    cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0, frame)


def _tw(text, font, scale, thick):
    """Return pixel width of rendered text."""
    return cv2.getTextSize(text, font, scale, thick)[0][0]


def _put(frame, text, x, y, font, scale, colour, thick=2):
    cv2.putText(frame, text, (x, y), font, scale, colour, thick, cv2.LINE_AA)


def _put_c(frame, text, y, font, scale, colour, thick=2):
    """Horizontally centred put."""
    x = (W - _tw(text, font, scale, thick)) // 2
    _put(frame, text, x, y, font, scale, colour, thick)


def neon_text(frame, text, x, y, font, scale, colour, thick=2):
    """
    Text with a dark halo so it pops on any background.
    Draws 8 offset copies in near-black first, then the bright text on top.
    """
    halo = (8, 5, 12)
    for dx, dy in ((-2,-2),(2,-2),(-2,2),(2,2),(-3,0),(3,0),(0,-3),(0,3)):
        _put(frame, text, x+dx, y+dy, font, scale, halo, thick+2)
    _put(frame, text, x, y, font, scale, colour, thick)


def neon_text_c(frame, text, y, font, scale, colour, thick=2):
    x = (W - _tw(text, font, scale, thick)) // 2
    neon_text(frame, text, x, y, font, scale, colour, thick)


# =============================================================================
# Background patterns
# =============================================================================

def dot_grid(frame, spacing=64, colour=(28, 20, 35)):
    """Subtle dot grid — used on title and menu screens."""
    for y in range(spacing // 2, H, spacing):
        for x in range(spacing // 2, W, spacing):
            cv2.circle(frame, (x, y), 1, colour, cv2.FILLED)


def hex_grid(frame, colour=(22, 16, 28), size=40):
    """Subtle hexagonal grid (drawn as diagonal lines for performance)."""
    for y in range(-size, H + size, size):
        for x in range(-size, W + size, size * 2):
            offset = (y // size % 2) * size
            pts = np.array([
                [x + offset,        y],
                [x + offset + size, y + size // 2],
                [x + offset + size, y + size * 3 // 2],
                [x + offset,        y + size * 2],
            ], dtype=np.int32)
            cv2.polylines(frame, [pts], False, colour, 1)


def vignette(frame, colour, intensity):
    """
    Fast radial vignette using numpy broadcasting.
    colour: BGR tuple   intensity: 0.0–1.0
    """
    if intensity <= 0:
        return
    cy, cx = H / 2.0, W / 2.0
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    mask = np.clip((dist - 0.35) / 0.65, 0.0, 1.0) * intensity
    mask3 = mask[:, :, np.newaxis]
    col_arr = np.array(colour, dtype=np.float32)
    frame[:] = np.clip(
        frame.astype(np.float32) * (1.0 - mask3) + col_arr * mask3,
        0, 255
    ).astype(np.uint8)


# =============================================================================
# Rounded rectangle
# =============================================================================

def rounded_rect(frame, x1, y1, x2, y2, r, colour, thick=-1):
    """Filled (thick==-1) or outlined rounded rectangle."""
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    if thick == -1:
        cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), colour, -1)
        cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), colour, -1)
        for cx, cy in ((x1+r, y1+r),(x2-r, y1+r),(x1+r, y2-r),(x2-r, y2-r)):
            cv2.circle(frame, (cx, cy), r, colour, -1)
    else:
        cv2.line(frame,  (x1+r,  y1),  (x2-r,  y1),  colour, thick)
        cv2.line(frame,  (x1+r,  y2),  (x2-r,  y2),  colour, thick)
        cv2.line(frame,  (x1,  y1+r),  (x1,  y2-r),  colour, thick)
        cv2.line(frame,  (x2,  y1+r),  (x2,  y2-r),  colour, thick)
        cv2.ellipse(frame, (x1+r, y1+r), (r,r), 180, 0, 90, colour, thick)
        cv2.ellipse(frame, (x2-r, y1+r), (r,r), 270, 0, 90, colour, thick)
        cv2.ellipse(frame, (x1+r, y2-r), (r,r),  90, 0, 90, colour, thick)
        cv2.ellipse(frame, (x2-r, y2-r), (r,r),   0, 0, 90, colour, thick)


# =============================================================================
# Robot face
# =============================================================================

def draw_robot_face(frame, cx, cy, size=160, mood='neutral', now=0.0):
    """
    Geometric robot face centred at (cx, cy).

    Parameters
    ----------
    size  : overall scale (head height ≈ size px)
    mood  : 'neutral' | 'smug' | 'happy' | 'angry1' | 'angry2' | 'angry3' | 'sad'
    now   : time.monotonic() for animations
    """
    col = list(MOOD_COLOUR.get(mood, P['cyan']))

    # angry3 pulses in intensity
    if mood == 'angry3':
        pulse = 0.65 + 0.35 * math.sin(now * 6 * math.pi)
        col = [int(c * pulse) for c in col]
    col = tuple(col)

    hw = int(size * 0.62)    # head half-width
    hh = int(size * 0.50)    # head half-height

    # ── Head body ──────────────────────────────────────────────────────
    x1, y1 = cx - hw, cy - hh
    x2, y2 = cx + hw, cy + hh

    # Dark fill
    _blend(frame, x1, y1, x2, y2, P['panel'], 0.88)

    # Neon border (thicker when angrier)
    bthick = 2 + int(mood in ('angry2', 'angry3'))
    rounded_rect(frame, x1, y1, x2, y2, 10, col, bthick)

    # Accent line across forehead
    fy = y1 + size // 7
    cv2.line(frame, (x1 + 8, fy), (x2 - 8, fy), col, 1)

    # ── Ear panels ─────────────────────────────────────────────────────
    ear_w = max(6, size // 11)
    ear_h = size // 3
    ey1, ey2 = cy - ear_h // 2, cy + ear_h // 2

    for ex1, ex2 in ((x1 - ear_w, x1), (x2, x2 + ear_w)):
        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), col, 2)
        cv2.line(frame, (ex1 + 2, cy), (ex2 - 2, cy), col, 1)

    # ── Antenna ────────────────────────────────────────────────────────
    ant_top = y1 - size // 4
    blink   = math.sin(now * 4 * math.pi) > 0   # 2 Hz blink
    cv2.line(frame, (cx, y1), (cx, ant_top), col, 2)
    tip_col = P['white'] if blink else col
    cv2.circle(frame, (cx, ant_top), max(4, size // 16), tip_col, -1)
    cv2.circle(frame, (cx, ant_top), max(6, size // 11), col, 2)

    # ── Eyes ───────────────────────────────────────────────────────────
    eye_y   = cy - hh // 3
    eye_r   = max(9, size // 9)
    eye_gap = hw // 2
    lx, rx  = cx - eye_gap, cx + eye_gap

    for ex in (lx, rx):
        cv2.circle(frame, (ex, eye_y), eye_r + 6, col, 1)   # outer glow ring
        cv2.circle(frame, (ex, eye_y), eye_r,     col, -1)  # filled eye

        if mood in ('happy', 'smug'):
            # Squint: dark bar over bottom third of eye
            bar_y = eye_y + eye_r // 3
            cv2.rectangle(frame, (ex - eye_r, bar_y), (ex + eye_r, eye_y + eye_r + 4),
                          P['panel'], -1)
            cv2.line(frame, (ex - eye_r, bar_y), (ex + eye_r, bar_y), col, 2)
        else:
            cv2.circle(frame, (ex, eye_y), max(3, eye_r // 3), P['white'], -1)

    # ── Eyebrows ───────────────────────────────────────────────────────
    brow_y  = eye_y - eye_r - 7
    bw      = eye_r + 5
    bt      = 3

    if mood in ('angry1', 'angry2', 'angry3'):
        drop = max(4, size // 10)
        cv2.line(frame, (lx - bw, brow_y - drop), (lx + bw, brow_y + drop), col, bt)
        cv2.line(frame, (rx - bw, brow_y + drop), (rx + bw, brow_y - drop), col, bt)
    elif mood == 'happy':
        for ex in (lx, rx):
            pts = np.array([[ex - bw, brow_y + 4], [ex, brow_y - 5], [ex + bw, brow_y + 4]],
                           dtype=np.int32)
            cv2.polylines(frame, [pts], False, col, bt, cv2.LINE_AA)
    else:
        for ex in (lx, rx):
            cv2.line(frame, (ex - bw, brow_y), (ex + bw, brow_y), col, bt)

    # ── Mouth arc ──────────────────────────────────────────────────────
    mouth_cy = cy + hh // 3
    m_half   = hw // 2
    amp      = max(4, hh // 8)
    n        = 20

    pts = []
    for i in range(n + 1):
        t  = math.pi * i / n          # 0..π
        px = cx - m_half + int(2 * m_half * i / n)
        if mood in ('happy', 'smug'):
            py = mouth_cy + int(amp * math.sin(t))    # ∪ smile
        elif mood in ('angry1', 'angry2', 'angry3', 'sad'):
            py = mouth_cy - int(amp * math.sin(t))    # ∩ frown
        else:
            py = mouth_cy                              # flat
        pts.append([px, py])

    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, col, 3, cv2.LINE_AA)

    # Chin detail dots
    chin_y = y2 - max(4, size // 12)
    for dx in (-hw // 3, 0, hw // 3):
        cv2.line(frame, (cx + dx, chin_y - 3), (cx + dx, chin_y + 3), col, 2)


# =============================================================================
# HUD bars & pill badge
# =============================================================================

def hud_bar(frame, y1, y2, alpha=0.82, accent_line='bottom'):
    """Full-width semi-transparent HUD bar with a neon accent edge."""
    _blend(frame, 0, y1, W, y2, P['panel'], alpha)
    line_y = y2 if accent_line == 'bottom' else y1
    cv2.line(frame, (0, line_y), (W, line_y), P['cyan'], 1)


def pill(frame, cx, cy, text, colour, font=FD, scale=0.95, thick=2,
         pad_x=28, pad_y=14):
    """Pill-shaped badge centred at (cx, cy)."""
    tw = _tw(text, font, scale, thick)
    (_, th), _ = cv2.getTextSize(text, font, scale, thick)
    rx, ry = tw // 2 + pad_x, th // 2 + pad_y
    _blend(frame, cx - rx, cy - ry, cx + rx, cy + ry, P['panel'], 0.88)
    rounded_rect(frame, cx - rx, cy - ry, cx + rx, cy + ry, ry // 2 + 2, colour, 2)
    _put(frame, text, cx - tw // 2, cy + th // 2, font, scale, colour, thick)


# =============================================================================
# Title screen
# =============================================================================

def draw_title_screen(frame, now):
    """Animated dark-neon title screen drawn entirely in code."""
    frame[:] = P['bg']
    dot_grid(frame)

    # Pulsing glow ring behind the robot face
    ring_cx, ring_cy = W // 2, 270
    pulse    = math.sin(now * 2 * math.pi)       # 1 Hz
    ring_r   = 155 + int(10 * pulse)

    for offset, am in ((22, 0.10), (12, 0.20), (4, 0.45)):
        ov = frame.copy()
        cv2.circle(ov, (ring_cx, ring_cy), ring_r + offset, P['cyan'], 2)
        cv2.addWeighted(ov, am, frame, 1 - am, 0, frame)

    # Vertical accent lines flanking the ring
    line_x1 = ring_cx - ring_r - 40
    line_x2 = ring_cx + ring_r + 40
    for lx in (line_x1, line_x2):
        cv2.line(frame, (lx, ring_cy - ring_r - 15), (lx, ring_cy + ring_r + 15),
                 P['border'], 1)

    # Robot face
    draw_robot_face(frame, ring_cx, ring_cy, size=175, mood='smug', now=now)

    # Title — "RPS" large cyan
    title_y = 495
    neon_text_c(frame, "R P S", title_y, FD, 2.8, P['cyan'], 5)

    # "ROBOT" magenta, slightly smaller, tight to "RPS"
    neon_text_c(frame, "ROBOT", title_y + 82, FD, 2.6, P['magenta'], 5)

    # Horizontal rule
    rule_y = title_y + 105
    rule_w = 300
    cv2.line(frame, (W//2 - rule_w, rule_y), (W//2 + rule_w, rule_y), P['border'], 1)
    # Small diamond on rule
    d = 4
    cv2.fillPoly(frame,
                 [np.array([[W//2 - d, rule_y], [W//2, rule_y - d],
                             [W//2 + d, rule_y], [W//2, rule_y + d]], dtype=np.int32)],
                 P['cyan'])

    # "Press ENTER" — flashing at 1 Hz
    if int(now * 2) % 2 == 0:
        neon_text_c(frame, "PRESS  ENTER  TO  CONTINUE",
                    title_y + 140, FS, 0.80, P['white'], 2)

    # Version / branding hint (very subtle)
    _put(frame, "RPS ROBOT", W - 130, H - 14, FS, 0.45, P['dark_grey'], 1)


# =============================================================================
# Menu screen
# =============================================================================

_CARDS = [
    ('CHEAT',     'Robot always counters',   P['cyan'],    'smug'),
    ('CHALLENGE', 'Build your streak',       P['magenta'], 'angry2'),
    ('MIRROR',    'Mirror your fingers',     P['green'],   'neutral'),
]

def draw_menu_screen(frame, sel, now):
    """Three card selection menu with glowing selection highlight."""
    frame[:] = P['bg']
    dot_grid(frame)

    # Header
    neon_text_c(frame, "SELECT MODE", 72, FD, 1.5, P['white'], 2)
    # Subtitle
    _put_c(frame, "Arrow keys  |  Enter to launch  |  ESC back", 100, FS, 0.58, P['grey'], 1)

    # Horizontal rule
    cv2.line(frame, (80, 115), (W - 80, 115), P['border'], 1)

    card_w    = 280
    card_h    = 390
    gap       = 60
    total_w   = 3 * card_w + 2 * gap
    x0        = (W - total_w) // 2
    y0        = 148

    for i, (name, desc, col, face_mood) in enumerate(_CARDS):
        cx1 = x0 + i * (card_w + gap)
        cy1 = y0
        cx2 = cx1 + card_w
        cy2 = cy1 + card_h
        ccx = (cx1 + cx2) // 2
        ccy = (cy1 + cy2) // 2

        is_sel = (i == sel)

        # Panel fill
        fill = P['panel_sel'] if is_sel else P['panel']
        _blend(frame, cx1, cy1, cx2, cy2, fill, 0.92)

        # Selected: animated inner glow
        if is_sel:
            pulse = 0.5 + 0.5 * math.sin(now * 4 * math.pi)
            ov = frame.copy()
            cv2.rectangle(ov, (cx1, cy1), (cx2, cy2), col, -1)
            cv2.addWeighted(ov, 0.06 + 0.03 * pulse, frame, 1 - 0.06 - 0.03 * pulse, 0, frame)

        # Border
        b_col   = col if is_sel else P['border']
        b_thick = 3   if is_sel else 1
        rounded_rect(frame, cx1, cy1, cx2, cy2, 12, b_col, b_thick)

        # Corner accent dots on selected card
        if is_sel:
            for dx, dy in ((cx1+4, cy1+4),(cx2-4, cy1+4),(cx1+4, cy2-4),(cx2-4, cy2-4)):
                cv2.circle(frame, (dx, dy), 3, col, -1)

        # Robot face inside card
        face_y = cy1 + 130
        draw_robot_face(frame, ccx, face_y, size=105,
                        mood=face_mood if not is_sel else face_mood,
                        now=now)

        # Mode name
        tw = _tw(name, FD, 0.95, 2)
        neon_text(frame, name, ccx - tw // 2, cy1 + 265, FD, 0.95,
                  col if is_sel else P['light_grey'], 2)

        # Description
        tw2 = _tw(desc, FS, 0.55, 1)
        _put(frame, desc, ccx - tw2 // 2, cy1 + 298, FS, 0.55, P['grey'], 1)

        # Selection indicator bar at bottom of card
        if is_sel:
            bar_x1, bar_x2 = cx1 + 20, cx2 - 20
            cv2.line(frame, (bar_x1, cy2 - 18), (bar_x2, cy2 - 18), col, 3)
            _put_c(frame, "▶  SELECTED  ◀", cy2 - 5, FS, 0.50, col, 1)

    # Bottom nav hint
    _put_c(frame, "←  →   navigate      ENTER   launch      ESC   title",
           H - 20, FS, 0.52, P['dark_grey'], 1)


# =============================================================================
# Gameplay overlay helpers
# =============================================================================

def draw_camera_bg(frame, raw_frame, darken=0.45):
    """Fill frame with mirrored, slightly darkened camera feed."""
    cam = cv2.resize(cv2.flip(raw_frame, 1), (W, H))
    frame[:] = (cam.astype(np.float32) * (1.0 - darken)).astype(np.uint8)


def draw_top_hud(frame, mode_label, round_str, score_str):
    """
    Full-width HUD bar at top (y=0..58).
    mode_label on left  |  round_str centred  |  score_str on right.
    """
    hud_bar(frame, 0, 58, alpha=0.84, accent_line='bottom')
    _put(frame, mode_label, 16, 38, FD, 0.85, P['cyan'], 2)
    _put_c(frame, round_str, 38, FS, 0.75, P['white'], 2)
    tw = _tw(score_str, FS, 0.75, 2)
    _put(frame, score_str, W - tw - 16, 38, FS, 0.75, P['white'], 2)


def draw_bottom_hud(frame, left_str, right_str, alpha=0.82):
    """Full-width HUD bar at bottom (y=748..800)."""
    hud_bar(frame, 748, H, alpha=alpha, accent_line='top')
    _put(frame, left_str,  40, 782, cv2.FONT_HERSHEY_DUPLEX, 0.82, P['white'], 2)
    tw = _tw(right_str, cv2.FONT_HERSHEY_DUPLEX, 0.82, 2)
    _put(frame, right_str, W - tw - 40, 782, cv2.FONT_HERSHEY_DUPLEX, 0.82, P['white'], 2)


def draw_state_pill(frame, main_text, sub_text='', colour_key='cyan'):
    """
    Large centred pill badge for game state text (MAKE A FIST / SHOOT! / etc.).
    Drawn in vertical centre of frame.
    """
    col  = PALETTE.get(colour_key, P['cyan'])
    cy   = H // 2

    if main_text:
        pill(frame, W // 2, cy - 10, main_text,
             col, font=FD, scale=1.55, thick=2, pad_x=44, pad_y=18)
    if sub_text:
        _put_c(frame, sub_text, cy + 68, FS, 0.68, P['light_grey'], 1)


def draw_countdown(frame, label):
    """
    Large glowing countdown digit / 'READY' centred on frame.
    Draws a glow halo then the bright character on top.
    """
    is_num  = label.lstrip('-').isdigit()
    scale   = 5.5 if is_num else 3.2
    thick   = 7   if is_num else 4
    col     = P['cyan']

    tw = _tw(label, FD, scale, thick)
    (_, th), _ = cv2.getTextSize(label, FD, scale, thick)
    x = (W - tw) // 2
    y = H // 2 + th // 2

    # Multi-layer glow
    for off, am in ((12, 0.08), (7, 0.18), (3, 0.50)):
        dark = tuple(int(c * 0.25) for c in col)
        ov = frame.copy()
        cv2.putText(ov, label, (x, y), FD, scale, dark, thick + off * 2, cv2.LINE_AA)
        cv2.addWeighted(ov, am, frame, 1 - am, 0, frame)

    _put(frame, label, x, y, FD, scale, col, thick)


def draw_robot_panel(frame, mood='neutral', now=0.0, gesture_label=''):
    """
    Robot face in a dark panel — bottom-right corner of frame.
    gesture_label: e.g. "ROBOT: Paper" shown under the face.
    """
    face_size = 135
    pw        = int(face_size * 1.65)
    ph        = int(face_size * 1.90)
    px1       = W - pw - 18
    py1       = H - ph - 18
    px2       = W - 18
    py2       = H - 18

    col = MOOD_COLOUR.get(mood, P['cyan'])
    _blend(frame, px1, py1, px2, py2, P['panel'], 0.90)
    rounded_rect(frame, px1, py1, px2, py2, 10, col, 2)

    face_cx = (px1 + px2) // 2
    face_cy = py1 + ph // 2 - 12
    draw_robot_face(frame, face_cx, face_cy, size=face_size, mood=mood, now=now)

    if gesture_label:
        tw = _tw(gesture_label, FS, 0.52, 1)
        _put(frame, gesture_label, face_cx - tw // 2, py2 - 8,
             FS, 0.52, col, 1)


def draw_gesture_indicator(frame, gesture_text):
    """Bottom-left diagnostic — small gesture label."""
    _put(frame, f"YOU: {gesture_text}", 16, H - 20, FS, 0.62, P['grey'], 1)


# =============================================================================
# Challenge mode — angry vignette + large robot
# =============================================================================

def draw_challenge_robot(frame, mood, now, session_wins):
    """
    Draws the robot face for challenge mode.
    At anger level 3 (8+ wins) the face is large and centred on the right half;
    otherwise it lives in the standard bottom-right panel.
    """
    if mood == 'angry3' and session_wins >= 8:
        # Large face on the right half of the screen
        face_cx = W * 3 // 4
        face_cy = H // 2
        face_sz = 200
        col = MOOD_COLOUR['angry3']
        # Dark backing panel
        _blend(frame, W // 2 + 20, 80, W - 20, H - 80, P['panel'], 0.75)
        rounded_rect(frame, W // 2 + 20, 80, W - 20, H - 80, 14, col, 2)
        draw_robot_face(frame, face_cx, face_cy, size=face_sz, mood=mood, now=now)
    elif mood == 'angry2':
        # Medium panel, slightly enlarged
        draw_robot_panel(frame, mood=mood, now=now, gesture_label='ROBOT')
    else:
        draw_robot_panel(frame, mood=mood, now=now, gesture_label='ROBOT')


def challenge_vignette_intensity(session_wins):
    """Map session_wins to a red vignette intensity."""
    if session_wins >= 8:
        return 0.50
    if session_wins >= 5:
        return 0.28
    if session_wins >= 3:
        return 0.12
    return 0.0


# =============================================================================
# Win / Lose screen
# =============================================================================

def draw_win_lose_screen(frame, result_type, player_gest, robot_gest,
                         result_show_time, now, result_min_display=4.0):
    """
    Full programmatic WIN / LOSE screen.

    Parameters
    ----------
    result_type       : 'win' or 'lose'
    player_gest       : gesture string ('Rock', 'Paper', 'Scissors', 'Unknown')
    robot_gest        : gesture string
    result_show_time  : time.monotonic() when screen was entered
    now               : current time.monotonic()
    result_min_display: seconds before ENTER is accepted
    """
    is_win   = result_type == 'win'
    word     = 'WIN' if is_win else 'LOSE'
    hl_col   = P['green'] if is_win else P['red']
    bg_tint  = (0, 8, 4) if is_win else (5, 0, 8)
    robot_mood = 'sad' if is_win else 'smug'

    # Background
    frame[:] = bg_tint
    dot_grid(frame, colour=(20, 18, 26) if is_win else (28, 16, 20))

    # Animated ring
    ring_cx, ring_cy = W // 2, H // 2
    pulse  = math.sin(now * 3 * math.pi)
    ring_r = 195 + int(12 * pulse)
    for off, am in ((24, 0.10), (12, 0.22), (2, 0.75)):
        ov = frame.copy()
        cv2.circle(ov, (ring_cx, ring_cy), ring_r + off, hl_col, 2)
        cv2.addWeighted(ov, am, frame, 1 - am, 0, frame)

    # Large result word at top
    neon_text_c(frame, word, 130, FD, 4.8, hl_col, 7)

    # Robot face centred
    draw_robot_face(frame, W // 2, H // 2 - 20, size=185, mood=robot_mood, now=now)

    # Info panel — bottom quarter
    panel_y1 = H - 230
    _blend(frame, 90, panel_y1, W - 90, H - 18, P['panel'], 0.90)
    rounded_rect(frame, 90, panel_y1, W - 90, H - 18, 14, hl_col, 2)

    _put_c(frame, f"YOU  threw:   {player_gest}",  panel_y1 + 55,  FD, 1.35, P['white'],      3)
    _put_c(frame, f"ROBOT  threw:  {robot_gest}",  panel_y1 + 115, FD, 1.35, P['light_grey'], 3)

    # Lock-out: progress bar until dismissable
    elapsed    = now - result_show_time
    bar_x1     = 160
    bar_x2     = W - 160
    bar_y      = panel_y1 + 165
    bar_h_val  = 8

    # Track
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h_val),
                  P['dark_grey'], -1)
    # Fill
    fill_w = int((bar_x2 - bar_x1) * min(elapsed / result_min_display, 1.0))
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + fill_w, bar_y + bar_h_val),
                      hl_col, -1)

    if elapsed >= result_min_display:
        if int(now * 2) % 2 == 0:
            _put_c(frame, "PRESS  ENTER  TO  CONTINUE",
                   panel_y1 + 200, FS, 0.78, (255, 220, 50), 2)
