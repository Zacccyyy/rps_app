"""
ui_renderer.py — Programmatic visual design for RPS Robot.

Every screen is drawn from OpenCV primitives + Pillow TTF text.
Tune the PALETTE dict and module constants at the top to retheme.
All draw_* functions write into the caller's frame (1280×800 numpy array).
Animated functions require  now = time.monotonic()  to be passed in.

Requires: opencv-python, numpy, Pillow
Optional: fonts/Orbitron-Bold.ttf, fonts/ShareTechMono-Regular.ttf
          (falls back to cv2 fonts if not present)
"""

import math
import os

import cv2
import numpy as np

# ── Pillow setup (graceful fallback if not installed) ────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter as _IFilter

    _FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')

    def _load_ttf(filename, size, bold_weight=None):
        path = os.path.join(_FONTS_DIR, filename)
        try:
            f = ImageFont.truetype(path, size)
            if bold_weight is not None:
                try:
                    f.set_variation_by_axes([bold_weight])
                except Exception:
                    pass
            return f
        except Exception:
            return ImageFont.load_default()

    # Font sizes — tune here to adjust all text at once
    _F = {
        'huge':    _load_ttf('Orbitron-Bold.ttf', 96,  700),
        'title':   _load_ttf('Orbitron-Bold.ttf', 68,  700),
        'heading': _load_ttf('Orbitron-Bold.ttf', 38,  700),
        'card':    _load_ttf('Orbitron-Bold.ttf', 26,  700),
        'ui':      _load_ttf('Orbitron-Bold.ttf', 22,  700),
        'mono_lg': _load_ttf('ShareTechMono-Regular.ttf', 22),
        'mono_sm': _load_ttf('ShareTechMono-Regular.ttf', 16),
    }

    _PIL_OK = True
    print("[ui] Pillow + TTF fonts loaded")

except ImportError:
    _PIL_OK = False
    print("[ui] Pillow unavailable — falling back to cv2 fonts")


# =============================================================================
# PALETTE  (all colours are BGR — OpenCV order)
# =============================================================================
# fmt: off
P = {
    'bg':          ( 15,  10,  10),
    'panel':       ( 26,  18,  18),
    'panel_sel':   ( 48,  34,  36),
    'border':      ( 65,  55,  72),
    'cyan':        (255, 245,   0),   # #00f5ff
    'magenta':     (170,   0, 255),   # #ff00aa
    'green':       ( 40, 210,  60),   # #3cd228
    'yellow':      (  0, 210, 255),   # #ffd200
    'orange':      (  0, 140, 255),   # #ff8c00
    'red':         ( 40,  40, 210),   # #d22828
    'white':       (255, 255, 255),
    'light_grey':  (180, 180, 185),
    'grey':        (110, 110, 115),
    'dark_grey':   ( 55,  55,  60),
    'black':       (  0,   0,   0),
}
# fmt: on

PALETTE = P

W = 1280
H = 800

FD = cv2.FONT_HERSHEY_DUPLEX    # fallback fonts (used if PIL unavailable)
FS = cv2.FONT_HERSHEY_SIMPLEX

MOOD_COLOUR = {
    'neutral': P['cyan'],
    'smug':    P['cyan'],
    'happy':   P['green'],
    'angry1':  P['yellow'],
    'angry2':  P['orange'],
    'angry3':  P['red'],
    'sad':     (160, 60, 200),
}


# =============================================================================
# OpenCV helpers
# =============================================================================

def _blend(frame, x1, y1, x2, y2, colour, alpha):
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), colour, cv2.FILLED)
    cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0, frame)


def _cv_put(frame, text, x, y, font, scale, colour, thick=2):
    cv2.putText(frame, text, (x, y), font, scale, colour, thick, cv2.LINE_AA)


def _cv_put_c(frame, text, y, font, scale, colour, thick=2):
    tw = cv2.getTextSize(text, font, scale, thick)[0][0]
    _cv_put(frame, text, (W - tw) // 2, y, font, scale, colour, thick)


# =============================================================================
# Pillow text helpers
# =============================================================================

def _bgr2rgb(c):
    return (c[2], c[1], c[0])


def _pil_size(text, font):
    """(width, height) of text in the given Pillow font."""
    d = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    bb = d.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def _pil_composite(frame, chip_rgba, dst_x, dst_y):
    """Alpha-composite a Pillow RGBA chip (as numpy RGBA array) onto frame."""
    h, w = frame.shape[:2]
    ch, cw = chip_rgba.shape[:2]

    src_x = max(0, -dst_x);  dst_x = max(0, dst_x)
    src_y = max(0, -dst_y);  dst_y = max(0, dst_y)
    end_x = min(w, dst_x + cw - src_x)
    end_y = min(h, dst_y + ch - src_y)
    sw, sh = end_x - dst_x, end_y - dst_y
    if sw <= 0 or sh <= 0:
        return

    alpha = chip_rgba[src_y:src_y+sh, src_x:src_x+sw, 3:4].astype(np.float32) / 255.0
    text_bgr = chip_rgba[src_y:src_y+sh, src_x:src_x+sw, :3][:, :, ::-1]
    fg = frame[dst_y:end_y, dst_x:end_x]
    frame[dst_y:end_y, dst_x:end_x] = (fg * (1 - alpha) + text_bgr * alpha).astype(np.uint8)


def _pil_put(frame, text, x, y, font, colour,
             glow=False, glow_r=14, glow_str=0.65, anchor='lt'):
    """
    Draw TTF text onto an OpenCV BGR frame.

    anchor: 'lt'=left-top  'mm'=centre-centre  'lm'=left-middle
            'rm'=right-middle  'rt'=right-top
    glow  : when True, adds a Gaussian-blur glow halo before the sharp text.
    """
    if not _PIL_OK:
        # fallback
        _cv_put(frame, text, x, y, FD, 0.8, colour, 2)
        return

    tw, th = _pil_size(text, font)
    if tw == 0 or th == 0:
        return

    # Resolve anchor to top-left draw position
    if   anchor == 'mm': x, y = x - tw // 2, y - th // 2
    elif anchor == 'lm': y = y - th // 2
    elif anchor == 'rm': x, y = x - tw, y - th // 2
    elif anchor == 'rt': x = x - tw

    pil_col = _bgr2rgb(colour) + (255,)

    if glow:
        pad = glow_r * 2
        chip_w, chip_h = tw + pad * 2, th + pad * 2
        glow_img = Image.new('RGB', (chip_w, chip_h), (0, 0, 0))
        ImageDraw.Draw(glow_img).text((pad, pad), text, font=font,
                                      fill=_bgr2rgb(colour))
        glow_np = np.array(
            glow_img.filter(_IFilter.GaussianBlur(radius=glow_r))
        )[:, :, ::-1]   # RGB→BGR

        gx, gy = x - pad, y - pad
        dx, dy = max(0, gx), max(0, gy)
        sx, sy = max(0, -gx), max(0, -gy)
        ex = min(W, dx + chip_w - sx)
        ey = min(H, dy + chip_h - sy)
        sw2, sh2 = ex - dx, ey - dy
        if sw2 > 0 and sh2 > 0:
            g = (glow_np[sy:sy+sh2, sx:sx+sw2] * glow_str).astype(np.uint8)
            frame[dy:ey, dx:ex] = cv2.add(frame[dy:ey, dx:ex], g)

    # Sharp text
    pad2 = 2
    chip = Image.new('RGBA', (tw + pad2*2, th + pad2*2), (0, 0, 0, 0))
    ImageDraw.Draw(chip).text((pad2, pad2), text, font=font, fill=pil_col)
    _pil_composite(frame, np.array(chip), x - pad2, y - pad2)


def _pil_put_c(frame, text, y, font, colour,
               glow=False, glow_r=14, glow_str=0.65, anchor_y='t'):
    """Horizontally centred TTF text.  anchor_y='t' top, 'm' middle."""
    tw, th = _pil_size(text, font)
    ty = y if anchor_y == 't' else y - th // 2
    _pil_put(frame, text, (W - tw) // 2, ty, font, colour,
             glow=glow, glow_r=glow_r, glow_str=glow_str)


def _pil_tw(text, font):
    """Pixel width of text in font."""
    return _pil_size(text, font)[0]


# =============================================================================
# Background patterns
# =============================================================================

def dot_grid(frame, spacing=64, colour=(28, 20, 35)):
    for y in range(spacing // 2, H, spacing):
        for x in range(spacing // 2, W, spacing):
            cv2.circle(frame, (x, y), 1, colour, cv2.FILLED)


def vignette(frame, colour, intensity):
    """Fast numpy radial vignette."""
    if intensity <= 0:
        return
    cy, cx = H / 2.0, W / 2.0
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    mask = np.clip((dist - 0.35) / 0.65, 0.0, 1.0) * intensity
    m3   = mask[:, :, np.newaxis]
    c    = np.array(colour, dtype=np.float32)
    frame[:] = np.clip(frame.astype(np.float32) * (1 - m3) + c * m3,
                       0, 255).astype(np.uint8)


# =============================================================================
# Rounded rectangle
# =============================================================================

def rounded_rect(frame, x1, y1, x2, y2, r, colour, thick=-1):
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    if thick == -1:
        cv2.rectangle(frame, (x1+r, y1), (x2-r, y2), colour, -1)
        cv2.rectangle(frame, (x1, y1+r), (x2, y2-r), colour, -1)
        for cx, cy in ((x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)):
            cv2.circle(frame, (cx, cy), r, colour, -1)
    else:
        cv2.line(frame, (x1+r, y1), (x2-r, y1), colour, thick)
        cv2.line(frame, (x1+r, y2), (x2-r, y2), colour, thick)
        cv2.line(frame, (x1, y1+r), (x1, y2-r), colour, thick)
        cv2.line(frame, (x2, y1+r), (x2, y2-r), colour, thick)
        cv2.ellipse(frame, (x1+r, y1+r), (r,r), 180, 0, 90, colour, thick)
        cv2.ellipse(frame, (x2-r, y1+r), (r,r), 270, 0, 90, colour, thick)
        cv2.ellipse(frame, (x1+r, y2-r), (r,r),  90, 0, 90, colour, thick)
        cv2.ellipse(frame, (x2-r, y2-r), (r,r),   0, 0, 90, colour, thick)


# =============================================================================
# Glow helpers
# =============================================================================

def _circle_glow(frame, cx, cy, r, colour, glow_r=20):
    """
    Draw a glowing filled circle: blurred halo first, sharp circle on top.
    Only blurs the small region around the circle — fast at 30 fps.
    """
    pad = glow_r * 2
    x1, y1 = max(0, cx - r - pad), max(0, cy - r - pad)
    x2, y2 = min(W, cx + r + pad), min(H, cy + r + pad)
    if x2 <= x1 or y2 <= y1:
        return
    chip = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
    cv2.circle(chip, (cx-x1, cy-y1), r, colour, -1)
    ks = glow_r * 2 + 1
    blurred = cv2.GaussianBlur(chip, (ks, ks), glow_r // 2)
    frame[y1:y2, x1:x2] = cv2.add(frame[y1:y2, x1:x2], blurred)
    cv2.circle(frame, (cx, cy), r, colour, -1)


def _ring_glow(frame, cx, cy, r, colour, thick=2, glow_r=22):
    """Glowing ring (outline only)."""
    pad = glow_r * 2
    x1, y1 = max(0, cx - r - pad), max(0, cy - r - pad)
    x2, y2 = min(W, cx + r + pad), min(H, cy + r + pad)
    if x2 <= x1 or y2 <= y1:
        return
    chip = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
    cv2.circle(chip, (cx-x1, cy-y1), r, colour, thick + 2)
    ks = glow_r * 2 + 1
    blurred = cv2.GaussianBlur(chip, (ks, ks), glow_r // 2)
    frame[y1:y2, x1:x2] = cv2.add(frame[y1:y2, x1:x2], blurred)
    cv2.circle(frame, (cx, cy), r, colour, thick)


# =============================================================================
# Robot face
# =============================================================================

def draw_robot_face(frame, cx, cy, size=160, mood='neutral', now=0.0):
    """
    Geometric robot face centred at (cx, cy).
    mood  : 'neutral' | 'smug' | 'happy' | 'angry1' | 'angry2' | 'angry3' | 'sad'
    now   : time.monotonic() for blink / pulse animations
    """
    col = list(MOOD_COLOUR.get(mood, P['cyan']))
    if mood == 'angry3':
        pulse = 0.65 + 0.35 * math.sin(now * 6 * math.pi)
        col = [int(c * pulse) for c in col]
    col = tuple(col)

    hw = int(size * 0.62)
    hh = int(size * 0.50)
    x1, y1, x2, y2 = cx - hw, cy - hh, cx + hw, cy + hh

    # Head fill + border
    _blend(frame, x1, y1, x2, y2, P['panel'], 0.88)
    bthick = 2 + int(mood in ('angry2', 'angry3'))
    rounded_rect(frame, x1, y1, x2, y2, 10, col, bthick)

    # Forehead accent line
    cv2.line(frame, (x1 + 8, y1 + size//7), (x2 - 8, y1 + size//7), col, 1)

    # Ear panels
    ear_w = max(6, size // 11)
    ear_h = size // 3
    ey1, ey2 = cy - ear_h//2, cy + ear_h//2
    for ex1, ex2 in ((x1 - ear_w, x1), (x2, x2 + ear_w)):
        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), col, 2)
        cv2.line(frame, (ex1+2, cy), (ex2-2, cy), col, 1)

    # Antenna with blink glow
    ant_top = y1 - size // 4
    blink   = math.sin(now * 4 * math.pi) > 0
    cv2.line(frame, (cx, y1), (cx, ant_top), col, 2)
    tip_col  = P['white'] if blink else col
    tip_r    = max(4, size // 16)
    _circle_glow(frame, cx, ant_top, tip_r, tip_col, glow_r=tip_r * 3)
    cv2.circle(frame, (cx, ant_top), max(6, size // 11), col, 2)

    # Eye positions
    eye_y  = cy - hh // 3
    eye_r  = max(9, size // 9)
    gap    = hw // 2
    lx, rx = cx - gap, cx + gap

    # Eye glow — single blur pass covering both eyes
    e_pad = eye_r * 3 + 10
    ex1_g = max(0, lx - e_pad);  ey1_g = max(0, eye_y - e_pad)
    ex2_g = min(W, rx + e_pad);  ey2_g = min(H, eye_y + e_pad)
    if ex2_g > ex1_g and ey2_g > ey1_g:
        gchip = np.zeros((ey2_g - ey1_g, ex2_g - ex1_g, 3), dtype=np.uint8)
        for ex in (lx, rx):
            cv2.circle(gchip, (ex - ex1_g, eye_y - ey1_g), eye_r + 6, col, -1)
        ks = eye_r * 2 + 5
        ks = ks if ks % 2 == 1 else ks + 1
        blurred = cv2.GaussianBlur(gchip, (ks, ks), eye_r // 2 + 1)
        frame[ey1_g:ey2_g, ex1_g:ex2_g] = cv2.add(
            frame[ey1_g:ey2_g, ex1_g:ex2_g], blurred
        )

    # Sharp eyes
    for ex in (lx, rx):
        cv2.circle(frame, (ex, eye_y), eye_r + 6, col, 1)
        cv2.circle(frame, (ex, eye_y), eye_r, col, -1)
        if mood in ('happy', 'smug'):
            bar_y = eye_y + eye_r // 3
            cv2.rectangle(frame, (ex - eye_r, bar_y), (ex + eye_r, eye_y + eye_r + 4),
                          P['panel'], -1)
            cv2.line(frame, (ex - eye_r, bar_y), (ex + eye_r, bar_y), col, 2)
        else:
            cv2.circle(frame, (ex, eye_y), max(3, eye_r // 3), P['white'], -1)

    # Eyebrows
    brow_y = eye_y - eye_r - 7
    bw, bt = eye_r + 5, 3
    if mood in ('angry1', 'angry2', 'angry3'):
        drop = max(4, size // 10)
        cv2.line(frame, (lx - bw, brow_y - drop), (lx + bw, brow_y + drop), col, bt)
        cv2.line(frame, (rx - bw, brow_y + drop), (rx + bw, brow_y - drop), col, bt)
    elif mood == 'happy':
        for ex in (lx, rx):
            pts = np.array([[ex-bw, brow_y+4],[ex, brow_y-5],[ex+bw, brow_y+4]], dtype=np.int32)
            cv2.polylines(frame, [pts], False, col, bt, cv2.LINE_AA)
    else:
        cv2.line(frame, (lx - bw, brow_y), (lx + bw, brow_y), col, bt)
        cv2.line(frame, (rx - bw, brow_y), (rx + bw, brow_y), col, bt)

    # Mouth arc
    mouth_cy = cy + hh // 3
    m_half   = hw // 2
    amp      = max(4, hh // 8)
    pts      = []
    for i in range(21):
        t  = math.pi * i / 20
        px = cx - m_half + int(2 * m_half * i / 20)
        if mood in ('happy', 'smug'):
            py = mouth_cy + int(amp * math.sin(t))
        elif mood in ('angry1', 'angry2', 'angry3', 'sad'):
            py = mouth_cy - int(amp * math.sin(t))
        else:
            py = mouth_cy
        pts.append([px, py])
    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, col, 3, cv2.LINE_AA)

    # Chin detail
    chin_y = y2 - max(4, size // 12)
    for dx in (-hw//3, 0, hw//3):
        cv2.line(frame, (cx+dx, chin_y-3), (cx+dx, chin_y+3), col, 2)


# =============================================================================
# HUD bars and pill badge
# =============================================================================

def hud_bar(frame, y1, y2, alpha=0.82, accent_line='bottom'):
    _blend(frame, 0, y1, W, y2, P['panel'], alpha)
    ly = y2 if accent_line == 'bottom' else y1
    cv2.line(frame, (0, ly), (W, ly), P['cyan'], 1)


def pill(frame, cx, cy, text, colour, font_key='ui', pad_x=32, pad_y=16):
    """Pill badge centred at (cx, cy) with TTF text."""
    if not _PIL_OK:
        tw = cv2.getTextSize(text, FD, 0.95, 2)[0][0]
        th = 28
    else:
        tw, th = _pil_size(text, _F[font_key])
    rx, ry = tw // 2 + pad_x, th // 2 + pad_y
    _blend(frame, cx - rx, cy - ry, cx + rx, cy + ry, P['panel'], 0.88)
    rounded_rect(frame, cx - rx, cy - ry, cx + rx, cy + ry, ry // 2 + 2, colour, 2)
    if _PIL_OK:
        _pil_put(frame, text, cx - tw // 2, cy - th // 2, _F[font_key], colour)
    else:
        _cv_put(frame, text, cx - tw // 2, cy + th // 2, FD, 0.95, colour, 2)


# =============================================================================
# Title screen
# =============================================================================

def draw_title_screen(frame, now):
    frame[:] = P['bg']
    dot_grid(frame)

    ring_cx, ring_cy = W // 2, 270
    pulse   = math.sin(now * 2 * math.pi)
    ring_r  = 158 + int(10 * pulse)

    # Glowing ring
    _ring_glow(frame, ring_cx, ring_cy, ring_r, P['cyan'], thick=2, glow_r=26)

    # Vertical accent lines
    for lx in (ring_cx - ring_r - 38, ring_cx + ring_r + 38):
        cv2.line(frame, (lx, ring_cy - ring_r - 18), (lx, ring_cy + ring_r + 18),
                 P['border'], 1)

    # Robot face
    draw_robot_face(frame, ring_cx, ring_cy, size=175, mood='smug', now=now)

    # "R P S" — large, glowing cyan
    _pil_put_c(frame, "R P S", 496, _F['title'], P['cyan'], glow=True, glow_r=20, glow_str=0.70)

    # "ROBOT" — magenta, slightly smaller
    _pil_put_c(frame, "ROBOT", 576, _F['heading'], P['magenta'], glow=True, glow_r=16, glow_str=0.65)

    # Horizontal rule with diamond
    rule_y = 605
    cv2.line(frame, (W//2 - 290, rule_y), (W//2 + 290, rule_y), P['border'], 1)
    d = 4
    cv2.fillPoly(frame, [np.array([
        [W//2 - d, rule_y], [W//2, rule_y - d],
        [W//2 + d, rule_y], [W//2, rule_y + d]], dtype=np.int32)], P['cyan'])

    # Flashing ENTER prompt
    if int(now * 2) % 2 == 0:
        _pil_put_c(frame, "PRESS  ENTER  TO  CONTINUE",
                   625, _F['mono_lg'], P['white'])

    _pil_put(frame, "RPS ROBOT", W - 140, H - 22, _F['mono_sm'], P['dark_grey'])


# =============================================================================
# Menu screen
# =============================================================================

_CARDS = [
    ('CHEAT',     'Robot always counters',  P['cyan'],    'smug'),
    ('CHALLENGE', 'Build your streak',      P['magenta'], 'angry2'),
    ('MIRROR',    'Mirror your fingers',    P['green'],   'neutral'),
]

def draw_menu_screen(frame, sel, now):
    frame[:] = P['bg']
    dot_grid(frame)

    _pil_put_c(frame, "SELECT MODE", 28, _F['heading'], P['white'], glow=True, glow_r=10, glow_str=0.45)
    _pil_put_c(frame, "Arrow keys  |  Enter to launch  |  ESC back",
               76, _F['mono_sm'], P['grey'])
    cv2.line(frame, (80, 96), (W - 80, 96), P['border'], 1)

    card_w, card_h = 280, 390
    gap            = 60
    x0 = (W - (3 * card_w + 2 * gap)) // 2
    y0 = 115

    for i, (name, desc, col, face_mood) in enumerate(_CARDS):
        cx1, cy1 = x0 + i * (card_w + gap), y0
        cx2, cy2 = cx1 + card_w, cy1 + card_h
        ccx = (cx1 + cx2) // 2
        is_sel = (i == sel)

        _blend(frame, cx1, cy1, cx2, cy2, P['panel_sel'] if is_sel else P['panel'], 0.92)

        if is_sel:
            pulse = 0.5 + 0.5 * math.sin(now * 4 * math.pi)
            ov = frame.copy()
            cv2.rectangle(ov, (cx1, cy1), (cx2, cy2), col, -1)
            cv2.addWeighted(ov, 0.06 + 0.03 * pulse, frame, 1 - 0.06 - 0.03 * pulse, 0, frame)

        b_col   = col if is_sel else P['border']
        b_thick = 3   if is_sel else 1
        rounded_rect(frame, cx1, cy1, cx2, cy2, 12, b_col, b_thick)

        if is_sel:
            for dxy in ((cx1+4,cy1+4),(cx2-4,cy1+4),(cx1+4,cy2-4),(cx2-4,cy2-4)):
                cv2.circle(frame, dxy, 3, col, -1)

        draw_robot_face(frame, ccx, cy1 + 130, size=105, mood=face_mood, now=now)

        tw = _pil_tw(name, _F['card'])
        _pil_put(frame, name, ccx - tw//2, cy1 + 262,
                 _F['card'], col if is_sel else P['light_grey'],
                 glow=is_sel, glow_r=8, glow_str=0.50)

        tw2 = _pil_tw(desc, _F['mono_sm'])
        _pil_put(frame, desc, ccx - tw2//2, cy1 + 300, _F['mono_sm'], P['grey'])

        if is_sel:
            cv2.line(frame, (cx1+20, cy2-18), (cx2-20, cy2-18), col, 3)
            tw3 = _pil_tw("SELECTED", _F['mono_sm'])
            _pil_put(frame, "SELECTED", ccx - tw3//2, cy2 - 15, _F['mono_sm'], col)

    _pil_put_c(frame, "left / right   navigate      enter   launch      esc   title",
               H - 16, _F['mono_sm'], P['dark_grey'])


# =============================================================================
# Gameplay overlay helpers
# =============================================================================

def draw_camera_bg(frame, raw_frame, darken=0.45):
    cam = cv2.resize(cv2.flip(raw_frame, 1), (W, H))
    frame[:] = (cam.astype(np.float32) * (1.0 - darken)).astype(np.uint8)


def draw_top_hud(frame, mode_label, round_str, score_str):
    hud_bar(frame, 0, 58, alpha=0.84, accent_line='bottom')
    _pil_put(frame, mode_label, 14, 8, _F['ui'], P['cyan'], glow=True, glow_r=8, glow_str=0.50)
    _pil_put_c(frame, round_str, 10, _F['mono_lg'], P['white'], anchor_y='t')
    if score_str:
        tw = _pil_tw(score_str, _F['mono_lg'])
        _pil_put(frame, score_str, W - tw - 14, 10, _F['mono_lg'], P['white'])


def draw_bottom_hud(frame, left_str, right_str, alpha=0.82):
    hud_bar(frame, 748, H, alpha=alpha, accent_line='top')
    _pil_put(frame, left_str,  36, 756, _F['mono_lg'], P['white'])
    tw = _pil_tw(right_str, _F['mono_lg'])
    _pil_put(frame, right_str, W - tw - 36, 756, _F['mono_lg'], P['white'])


def draw_state_pill(frame, main_text, sub_text='', colour_key='cyan'):
    """Large centred pill badge for game state text."""
    col = PALETTE.get(colour_key, P['cyan'])
    pill(frame, W // 2, H // 2 - 10, main_text, col, font_key='ui', pad_x=42, pad_y=18)
    if sub_text:
        _pil_put_c(frame, sub_text, H // 2 + 52, _F['mono_sm'], P['light_grey'], anchor_y='t')


def draw_countdown(frame, label):
    """Large glowing countdown digit / 'READY' centred on frame."""
    col = P['cyan']
    if _PIL_OK:
        font = _F['huge']
        tw, th = _pil_size(label, font)
        x = (W - tw) // 2
        y = H // 2 - th // 2
        _pil_put(frame, label, x, y, font, col, glow=True, glow_r=28, glow_str=0.80)
    else:
        scale = 5.0 if len(label) == 1 else 3.0
        tw = cv2.getTextSize(label, FD, scale, 6)[0][0]
        _cv_put(frame, label, (W - tw) // 2, H // 2 + 40, FD, scale, col, 6)


def draw_robot_panel(frame, mood='neutral', now=0.0, gesture_label=''):
    face_size = 135
    pw = int(face_size * 1.65)
    ph = int(face_size * 1.90)
    px1, py1 = W - pw - 18, H - ph - 18
    px2, py2 = W - 18, H - 18

    col = MOOD_COLOUR.get(mood, P['cyan'])
    _blend(frame, px1, py1, px2, py2, P['panel'], 0.90)
    rounded_rect(frame, px1, py1, px2, py2, 10, col, 2)

    face_cx = (px1 + px2) // 2
    face_cy = py1 + ph // 2 - 14
    draw_robot_face(frame, face_cx, face_cy, size=face_size, mood=mood, now=now)

    if gesture_label:
        tw = _pil_tw(gesture_label, _F['mono_sm'])
        _pil_put(frame, gesture_label, face_cx - tw // 2, py2 - 20, _F['mono_sm'], col)


def draw_gesture_indicator(frame, gesture_text):
    """Small bottom-left diagnostic label."""
    _pil_put(frame, f"YOU: {gesture_text}", 14, H - 22, _F['mono_sm'], P['grey'])


# =============================================================================
# Challenge mode — anger vignette + large robot
# =============================================================================

def draw_challenge_robot(frame, mood, now, session_wins):
    if mood == 'angry3' and session_wins >= 8:
        col = MOOD_COLOUR['angry3']
        _blend(frame, W//2 + 20, 80, W - 20, H - 80, P['panel'], 0.75)
        rounded_rect(frame, W//2 + 20, 80, W - 20, H - 80, 14, col, 2)
        draw_robot_face(frame, W * 3 // 4, H // 2, size=200, mood=mood, now=now)
    else:
        draw_robot_panel(frame, mood=mood, now=now, gesture_label='ROBOT')


def challenge_vignette_intensity(session_wins):
    if session_wins >= 8:  return 0.50
    if session_wins >= 5:  return 0.28
    if session_wins >= 3:  return 0.12
    return 0.0


# =============================================================================
# Win / Lose screen
# =============================================================================

def draw_win_lose_screen(frame, result_type, player_gest, robot_gest,
                         result_show_time, now, result_min_display=4.0):
    is_win   = result_type == 'win'
    word     = 'WIN' if is_win else 'LOSE'
    hl_col   = P['green'] if is_win else P['red']
    bg_tint  = (0, 8, 4) if is_win else (5, 0, 8)
    r_mood   = 'sad' if is_win else 'smug'

    frame[:] = bg_tint
    dot_grid(frame, colour=(20, 18, 26) if is_win else (28, 16, 20))

    # Animated glowing ring
    ring_cx, ring_cy = W // 2, H // 2
    ring_r = 195 + int(12 * math.sin(now * 3 * math.pi))
    _ring_glow(frame, ring_cx, ring_cy, ring_r, hl_col, thick=2, glow_r=28)

    # WIN / LOSE word — huge, glowing
    _pil_put_c(frame, word, 28, _F['huge'], hl_col, glow=True, glow_r=30, glow_str=0.80)

    # Robot face
    draw_robot_face(frame, W // 2, H // 2 - 20, size=185, mood=r_mood, now=now)

    # Info panel — bottom quarter
    panel_y1 = H - 230
    _blend(frame, 90, panel_y1, W - 90, H - 18, P['panel'], 0.90)
    rounded_rect(frame, 90, panel_y1, W - 90, H - 18, 14, hl_col, 2)

    _pil_put_c(frame, f"YOU  threw:   {player_gest}",
               panel_y1 + 22, _F['card'], P['white'], anchor_y='t')
    _pil_put_c(frame, f"ROBOT  threw:  {robot_gest}",
               panel_y1 + 80, _F['card'], P['light_grey'], anchor_y='t')

    # Lock-out progress bar
    elapsed   = now - result_show_time
    bx1, bx2  = 160, W - 160
    by        = panel_y1 + 155
    cv2.rectangle(frame, (bx1, by), (bx2, by + 8), P['dark_grey'], -1)
    fill_w = int((bx2 - bx1) * min(elapsed / result_min_display, 1.0))
    if fill_w > 0:
        cv2.rectangle(frame, (bx1, by), (bx1 + fill_w, by + 8), hl_col, -1)

    if elapsed >= result_min_display and int(now * 2) % 2 == 0:
        _pil_put_c(frame, "PRESS  ENTER  TO  CONTINUE",
                   panel_y1 + 178, _F['mono_lg'], (255, 220, 50), anchor_y='t')
