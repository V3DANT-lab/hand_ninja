#!/usr/bin/env python3
"""
HAND NINJA  -  slice flying fruit with your real hand.
=====================================================

Your index fingertip (tracked by MediaPipe) is the blade.  Swipe through fruit,
dodge the bombs, chain combos.

What's new in this version
--------------------------
* SMOOTHER BLADE: the blade rides a critically-damped spring (frame-rate
  independent) fed by the One-Euro fingertip filter plus a small velocity
  look-ahead.  Camera-rate stair-stepping and chase jitter are gone - slow
  moves are steady, fast swipes stay tight.  If tracking drops out mid-swipe
  the blade glides on for ~150 ms and finishes the cut instead of freezing.
* EASIER HAND DETECTION: tracks up to two hands with smooth hand-over (the
  fingertip closest to the last one wins), lower detection/tracking
  thresholds, an automatic contrast boost for dim rooms, and an adaptive
  inference rate that keeps the tracking thread from falling behind the
  camera (latency stays low even on slow machines).
* FIXED: "module 'mediapipe' has no attribute 'solutions'".  MediaPipe 1.x removed
  the old `mp.solutions` API.  This version uses the new MediaPipe *Tasks*
  HandLandmarker, downloads its model file automatically on first run, and still
  works with old MediaPipe (0.10.x) and even without MediaPipe (OpenCV fallback).
* Camera AUTO-DETECTION (scans /dev/video* or indices, skips dead / metadata /
  black devices, auto-reconnects if the camera drops).  Override: --camera N
* Camera + hand tracking run in a background thread -> the game stays at 60 FPS.
* Butter-smooth blade: One-Euro fingertip filter + critically-damped spring +
  velocity look-ahead (steady when still, instant when you swipe fast).
* Photo-style procedural fruit: lit/specular-shaded apple, orange, lemon, kiwi,
  watermelon with real cut cross-sections (seeds, segments, core, rind).
* Wood-plank dojo background, juice splats, drop shadows, floating score pop-ups,
  start menu you slice to begin, persistent best score.
* Mouse also works as a blade (hold left button) - handy without a camera.

Controls
--------
    INDEX FINGER   blade            (or hold LEFT MOUSE BUTTON)
    SPACE / slice the fruit        start / restart
    C   cycle camera view: off -> picture-in-picture -> full background
    M   mute        F  fullscreen       P  pause       D  debug/FPS
    ESC quit

Setup
-----
    pip install mediapipe opencv-python pygame numpy
    python hand_ninja.py

    --list-cameras     show the cameras that were found
    --camera 2         force a specific camera index
    --mouse            ignore the camera, play with the mouse
    --model PATH       use a local hand_landmarker.task file
"""

import os
import sys
import math
import time
import json
import random
import argparse
import threading
import urllib.request
from collections import deque, namedtuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np
import pygame

try:
    import mediapipe as mp
except Exception as _e:  # pragma: no cover
    mp = None
    MP_IMPORT_ERROR = _e
else:
    MP_IMPORT_ERROR = None

try:
    cv2.setLogLevel(0)
except Exception:
    pass


# =============================================================== constants ==

WIDTH, HEIGHT = 1024, 576
FPS = 60

# Camera / tracking
CAM_REQUEST = (1280, 720)       # asked from the camera; auto-falls back
DETECT_SIZE = (640, 360)        # frame size handed to the hand model
MARGIN_X, MARGIN_Y = 0.12, 0.10 # unused border of the camera frame (easier to reach edges)

# --- blade motion (all frame-rate independent, speeds in px/second) ---
BLADE_OMEGA = 58.0              # critically-damped spring stiffness (higher = tighter)
MAX_BLADE_SPEED = 6000.0        # px/s safety clamp on the spring
PREDICT_S = 0.050               # seconds of velocity look-ahead (hides camera latency)
PREDICT_FULL_SPEED = 900.0      # px/s at which look-ahead reaches full strength
MAX_TRACK_SPEED = 4000.0        # px/s clamp on the fingertip velocity estimate
HAND_GRACE_FRAMES = 9           # frames a dropout is bridged by gliding on
GLIDE_DECAY = 0.80              # per-frame velocity decay while gliding
MIN_SLICE_SPEED = 200.0         # px/s needed to cut (stops accidental touches)
SWISH_SPEED = 1300.0            # px/s before the swish sound plays
TELEPORT_DIST = 420             # target jumps farther than this = new hand, no slice
TRAIL_LIFE = 0.26               # seconds the blade trail stays visible

# --- fingertip filter ---
ONE_EURO_CUTOFF = 2.2           # lower = smoother, higher = tighter
ONE_EURO_BETA = 0.022           # how quickly smoothing opens up on fast swipes

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_NAME = "hand_landmarker.task"

# Physics / spawning
GRAVITY = 0.36
FRUIT_RADIUS = 44
SPAWN_Y = HEIGHT + 60
SPAWN_VY_MIN, SPAWN_VY_VAR = 15.5, 4.0
SPAWN_VX_RANGE = 5.5
DESPAWN_Y = HEIGHT + 110
HALF_GRAVITY_FACTOR = 0.85
HALF_SPEED = 3.6
SPAWN_INTERVAL_MIN, SPAWN_INTERVAL_MAX = 42, 68
SPAWN_COUNT_MIN, SPAWN_COUNT_MAX = 1, 3
DIFFICULTY_STEP = 45
MIN_INTERVAL = 20
BOMB_CHANCE = 0.12

COMBO_WINDOW_MS = 550
COMBO_BONUS_BASE = 5
MAX_LIVES = 3

# Particles / FX
JUICE_MIN, JUICE_MAX = 18, 30
PARTICLE_GRAVITY = 0.42
SHAKE_FRAMES, FLASH_FRAMES = 22, 16

HUD_COLOR = (248, 244, 236)
COMBO_COLOR = (255, 214, 90)
FRUIT_TYPES = ["apple", "orange", "lemon", "kiwi", "watermelon"]

JUICE_COLORS = {
    "apple": (250, 238, 180), "orange": (255, 150, 30), "lemon": (255, 236, 90),
    "kiwi": (140, 205, 40), "watermelon": (235, 45, 70), "bomb": (255, 160, 40),
}

SS = 3  # supersampling factor for procedural art
BEST_FILE = os.path.join(os.path.expanduser("~"), ".hand_ninja_best")

HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
              (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


# ================================================================== audio ==

SR = 22050


def _to_sound(wave):
    if not pygame.mixer.get_init():
        return None
    wave = np.clip(wave, -1.0, 1.0)
    st = np.ascontiguousarray((np.column_stack([wave, wave]) * 32767).astype(np.int16))
    try:
        return pygame.sndarray.make_sound(st)
    except Exception:
        return None


def _env(n, attack=0.005, release=0.03):
    e = np.ones(n, np.float32)
    a, r = int(SR * attack), int(SR * release)
    if a > 0:
        e[:a] = np.linspace(0, 1, a)
    if 0 < r < n:
        e[-r:] = np.linspace(1, 0, r)
    return e


def _sweep(f0, f1, dur, vol, decay=6.0):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, False)
    f = np.linspace(f0, f1, n)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return vol * np.exp(-t * decay) * np.sin(ph) * _env(n)


def _noise_burst(dur, vol, a0=0.55, a1=0.08, decay=5.0):
    n = int(SR * dur)
    x = np.random.standard_normal(n).astype(np.float32)
    a = np.linspace(a0, a1, n)
    y = np.empty(n, np.float32)
    acc = 0.0
    for i in range(n):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    y /= (np.abs(y).max() + 1e-6)
    t = np.linspace(0, dur, n, False)
    return vol * y * np.exp(-t * decay) * _env(n)


def build_sounds():
    s = {}
    try:
        s["swish"] = _to_sound(_noise_burst(0.16, 0.35, 0.6, 0.12, 9.0))
        s["slice"] = _to_sound(_mix(_noise_burst(0.14, 0.45, 0.7, 0.15, 12.0),
                                     _sweep(750, 230, 0.14, 0.30, 14.0)))
        s["miss"] = _to_sound(_sweep(330, 140, 0.22, 0.35, 7.0))
        s["combo"] = _to_sound(np.concatenate([_sweep(660, 660, 0.07, 0.25, 8), _sweep(880, 880, 0.07, 0.25, 8),
                                               _sweep(1320, 1320, 0.12, 0.25, 7)]))
        n = int(SR * 0.7)
        t = np.linspace(0, 0.7, n, False)
        f = 70 + 90 * np.exp(-t * 9)
        boom = 0.7 * np.exp(-t * 3.2) * np.sin(2 * np.pi * np.cumsum(f) / SR)
        boom += 0.45 * np.exp(-t * 4.0) * (np.random.rand(n) * 2 - 1)
        s["bomb"] = _to_sound(boom)
        s["start"] = _to_sound(np.concatenate([_sweep(440, 440, 0.06, 0.3, 6), _sweep(660, 660, 0.06, 0.3, 6),
                                               _sweep(990, 990, 0.14, 0.3, 5)]))
    except Exception as e:
        print("[audio] disabled:", e)
    return s


def _mix(a, b):
    n = max(len(a), len(b))
    out = np.zeros(n, np.float32)
    out[:len(a)] += a
    out[:len(b)] += b
    return out


# ============================================================= math helpers ==

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mix(c0, c1, t):
    c0 = np.asarray(c0, np.float32)
    c1 = np.asarray(c1, np.float32)
    return c0 + (c1 - c0) * np.asarray(t, np.float32)[..., None]


def fbm(h, w, cy, cx, rng, octaves=4, gain=0.5):
    """Fractal value noise in [0,1], (cy, cx) = lattice cells in the first octave."""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for _ in range(octaves):
        gy, gx = int(min(max(cy, 1), h)), int(min(max(cx, 1), w))
        g = rng.random((max(2, gy + 1), max(2, gx + 1))).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= gain
        cy *= 2
        cx *= 2
    out /= tot
    lo, hi = float(out.min()), float(out.max())
    return np.clip((out - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def polar_noise(rr, th, rng, cells_r=6, cells_t=90, octaves=3):
    nt = fbm(256, 256, cells_r, cells_t, rng, octaves)
    ti = np.clip(((th + np.pi) / (2 * np.pi) * 255), 0, 255).astype(np.int32)
    ri = np.clip(rr * 255, 0, 255).astype(np.int32)
    return nt[ri, ti]


# ========================================================== procedural fruit ==

PAD = 0.40


def canvas_size(radius, pad=PAD):
    return int(round(radius * (1 + pad) * 2))


def coords(radius, pad=PAD):
    size = canvas_size(radius, pad)
    n = size * SS
    r = radius * SS
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    return n, size, r, (xs - n / 2 + 0.5) / r, (ys - n / 2 + 0.5) / r


SHAPES = {
    "apple": dict(axis="y", L=0.97, jitter=0.006),
    "orange": dict(axis="y", L=0.98, jitter=0.012),
    "lemon": dict(axis="x", L=1.10, jitter=0.010),
    "kiwi": dict(axis="x", L=1.00, jitter=0.020),
    "watermelon": dict(axis="x", L=1.06, jitter=0.008),
    "bomb": dict(axis="y", L=1.0, jitter=0.0),
}


def profile(kind, t):
    t = np.clip(np.asarray(t, np.float32), -1.0, 1.0)
    base = np.sqrt(np.clip(1 - t * t, 0, 1))
    if kind == "apple":
        return base * (1.03 - 0.11 * t - 0.05 * t * t)
    if kind == "orange":
        return base
    if kind == "lemon":
        p = 0.80 * np.sqrt(np.clip(1 - np.abs(t) ** 2.4, 0, 1))
        return p + 0.085 * np.exp(-((1 - np.abs(t)) / 0.045) ** 2)
    if kind == "kiwi":
        return 0.84 * base
    if kind == "watermelon":
        return 0.86 * base
    return base


def _teardrop(surf, x, y, length, width, ang, color, shine=True):
    pts = []
    for i in range(28):
        t = 2 * math.pi * i / 28
        px = length * 0.5 * math.cos(t)
        py = width * 0.5 * math.sin(t) * (1.0 - 0.45 * math.cos(t))
        pts.append((x + px * math.cos(ang) - py * math.sin(ang),
                    y + px * math.sin(ang) + py * math.cos(ang)))
    pygame.draw.polygon(surf, color, pts)
    if shine:
        hl = tuple(min(255, int(c + 70)) for c in color)
        pygame.draw.circle(surf, hl, (int(x - 0.12 * length * math.cos(ang) - 0.12 * width),
                                      int(y - 0.12 * length * math.sin(ang) - 0.14 * width)),
                           max(1, int(width * 0.13)))


def _stem(surf, x, y, R, length=0.30, thick=0.065, bend=0.10, color=(98, 64, 34)):
    left, right = [], []
    for i in range(14):
        s = i / 13
        cx = x + bend * R * s * s
        cy = y - length * R * s
        w = thick * R * (1.0 - 0.30 * s)
        left.append((cx - w, cy))
        right.append((cx + w, cy))
    pygame.draw.polygon(surf, color, left + right[::-1])
    hl = tuple(min(255, c + 45) for c in color)
    pygame.draw.lines(surf, hl, False, [(p[0] + thick * R * 0.5, p[1]) for p in left], max(1, int(R * 0.025)))
    pygame.draw.ellipse(surf, tuple(int(c * 0.6) for c in color),
                        (x - thick * R * 1.3, y - thick * R * 0.6, thick * R * 2.6, thick * R * 1.2))


def _leaf(surf, x, y, R, ang_deg, size=0.50, base=(66, 152, 52)):
    ang = math.radians(ang_deg)
    d = (math.cos(ang), math.sin(ang))
    nrm = (-d[1], d[0])
    up, lo, mid = [], [], []
    for i in range(16):
        s = i / 15
        w = size * R * 0.27 * (math.sin(math.pi * s) ** 0.85)
        curve = 0.10 * size * R * math.sin(math.pi * s)
        px = x + d[0] * s * size * R + nrm[0] * curve
        py = y + d[1] * s * size * R + nrm[1] * curve
        up.append((px + nrm[0] * w, py + nrm[1] * w))
        lo.append((px - nrm[0] * w, py - nrm[1] * w))
        mid.append((px, py))
    light = tuple(min(255, int(c * 1.22)) for c in base)
    dark = tuple(int(c * 0.72) for c in base)
    pygame.draw.polygon(surf, light, up + mid[::-1])
    pygame.draw.polygon(surf, dark, mid + lo[::-1])
    pygame.draw.lines(surf, tuple(int(c * 0.55) for c in base), False, mid, max(1, int(R * 0.022)))


def _to_surface(col, alpha, n):
    rgba = np.dstack([np.clip(col, 0, 1) * 255, alpha.astype(np.float32) * 255]).astype(np.uint8)
    return pygame.image.frombuffer(np.ascontiguousarray(rgba).tobytes(), (n, n), "RGBA").copy()


def _downscale(surf, size):
    return pygame.transform.smoothscale(surf, (size, size))


def _shade(col, nx, ny, nz, ks, shin, sss=0.0, metal=0.0):
    ln = np.array([-0.50, -0.62, 0.60], np.float32)
    ln /= np.linalg.norm(ln)
    hv = ln + np.array([0, 0, 1], np.float32)
    hv /= np.linalg.norm(hv)
    diff = np.clip(nx * ln[0] + ny * ln[1] + nz * ln[2], 0, 1)
    amb = 0.33 + 0.14 * (-ny)
    sp = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1)
    spec = ks * np.power(sp, shin) + 0.10 * ks * np.power(sp, max(2.0, shin / 6))
    fres = np.power(np.clip(1 - nz, 0, 1), 3)
    lit = col * (amb + 0.95 * diff)[..., None]
    lit += col * (0.10 * np.clip(ny, 0, 1))[..., None]                       # bounce from the table
    if sss > 0:
        lit += col * (sss * np.exp(-((diff - 0.12) / 0.20) ** 2))[..., None]  # soft subsurface glow
    lit *= (1 - 0.40 * np.power(np.clip(1 - nz, 0, 1), 2.2))[..., None]    # edge darkening
    lit += spec[..., None] * np.array([1.0, 0.98, 0.94], np.float32)
    lit += (0.10 + 0.25 * metal) * fres[..., None] * np.array([0.55, 0.66, 0.88], np.float32)
    if metal:
        sky = np.clip(-ny, 0, 1) * 0.22 * metal
        lit += sky[..., None] * np.array([0.6, 0.7, 0.9], np.float32)
    return lit


def render_fruit(kind, radius, seed=3):
    """Photo-ish whole fruit sprite (RGBA pygame Surface)."""
    rng = np.random.default_rng(seed + hash(kind) % 997)
    n, size, R, u, v = coords(radius)
    sp = SHAPES[kind]
    L = sp["L"]
    if sp["axis"] == "x":
        t, s = u / L, v
    else:
        t, s = v / L, u
    p = profile(kind, t)
    e = 0.01
    dp = (profile(kind, t + e) - profile(kind, t - e)) / (2 * e) / L
    jit = (fbm(n, n, 6, 6, rng, 3) - 0.5) * sp["jitter"] * 2
    inside = (np.abs(t) <= 1.0) & (np.abs(s) <= p + jit * (p > 0.05))
    z = np.sqrt(np.clip(p * p - s * s, 1e-4, None))
    k = np.clip(-p * dp, -4, 4)
    if sp["axis"] == "x":
        nu, nv = k, s.copy()
    else:
        nu, nv = s.copy(), k
    nz = z.copy()

    # ----- per-fruit surface texture -----
    ks, shin, sss, bump_k = 0.5, 40, 0.0, 0.0
    hf = None
    if kind == "apple":
        streak = fbm(n, n, 3, 52, rng, 4)
        blotch = fbm(n, n, 4, 4, rng, 3)
        col = mix((0.50, 0.03, 0.06), (0.90, 0.13, 0.15), streak)
        yel = np.clip((blotch - 0.60) * 4.5, 0, 1) * (0.45 + 0.55 * np.clip(v + 0.2, 0, 1))
        col = mix(col, (0.90, 0.74, 0.22), yel * 0.55)
        dots = (rng.random((n, n)) > 0.9985).astype(np.float32)
        dots = cv2.dilate(dots, np.ones((3, 3), np.uint8))
        col = col + dots[..., None] * np.array([0.35, 0.30, 0.15], np.float32) * (0.4 + 0.6 * nz[..., None])
        dim = np.exp(-(((u) / 0.20) ** 2 + ((v + 0.90) / 0.075) ** 2))
        col = mix(col, (0.30, 0.04, 0.04), 0.75 * dim)
        ks, shin, bump_k, sss = 0.95, 60, 0.12, 0.04
        hf = fbm(n, n, n / 4, n / 4, rng, 2)
    elif kind == "orange":
        n1 = fbm(n, n, 8, 8, rng, 4)
        col = mix((0.90, 0.40, 0.04), (1.0, 0.62, 0.10), n1)
        hf = fbm(n, n, n / 3, n / 3, rng, 2)
        col = col * (0.90 + 0.18 * hf[..., None])
        ks, shin, bump_k, sss = 0.42, 30, 1.25, 0.10
    elif kind == "lemon":
        n1 = fbm(n, n, 7, 7, rng, 4)
        col = mix((0.97, 0.80, 0.10), (1.0, 0.93, 0.30), n1)
        col = mix(col, (0.72, 0.80, 0.18), np.clip((fbm(n, n, 3, 3, rng, 2) - 0.7) * 3, 0, 1) * 0.35)
        hf = fbm(n, n, n / 3, n / 3, rng, 2)
        col = col * (0.92 + 0.14 * hf[..., None])
        ks, shin, bump_k, sss = 0.45, 32, 0.95, 0.08
    elif kind == "kiwi":
        hairs = fbm(n, n, n / 2, n / 2, rng, 2)
        base = fbm(n, n, 5, 5, rng, 3)
        col = mix((0.38, 0.26, 0.15), (0.58, 0.42, 0.26), base)
        col = col * (0.72 + 0.55 * hairs[..., None])
        tips = (rng.random((n, n)) > 0.985).astype(np.float32)
        col = col + tips[..., None] * 0.18
        hf = hairs
        ks, shin, bump_k, sss = 0.05, 8, 0.45, 0.05
    elif kind == "watermelon":
        phi = np.arctan2(z, s)
        warp = fbm(n, n, 5, 5, rng, 3)
        stripes = 0.5 + 0.5 * np.sin(phi * 7.0 + 2.2 * warp + 0.9 * np.sin(u * 2.4))
        stripes = smoothstep(0.35, 0.65, stripes)
        col = mix((0.07, 0.30, 0.10), (0.34, 0.60, 0.26), stripes)
        col = col * (0.88 + 0.22 * fbm(n, n, n / 6, n / 6, rng, 2)[..., None])
        ks, shin, bump_k, sss = 0.55, 42, 0.10, 0.04
        hf = fbm(n, n, n / 5, n / 5, rng, 2)

    if hf is not None and bump_k > 0:
        gy, gx = np.gradient(hf)
        nu = nu - bump_k * gx * 3.0
        nv = nv - bump_k * gy * 3.0
    norm = np.sqrt(nu * nu + nv * nv + nz * nz) + 1e-6
    nu, nv, nz = nu / norm, nv / norm, nz / norm
    lit = _shade(col.astype(np.float32), nu, nv, nz, ks, shin, sss)
    surf = _to_surface(lit, inside, n)

    # ----- decorations drawn on the supersampled surface -----
    cx = cy = n / 2
    if kind == "apple":
        _leaf(surf, cx + 0.04 * R, cy - 0.86 * R, R, -38, 0.56)
        _stem(surf, cx - 0.01 * R, cy - 0.84 * R, R, 0.34, 0.06, 0.12)
    elif kind == "orange":
        pygame.draw.ellipse(surf, (70, 100, 35), (cx - 0.11 * R, cy - 0.98 * R, 0.22 * R, 0.09 * R))
        pygame.draw.ellipse(surf, (140, 160, 70), (cx - 0.06 * R, cy - 0.975 * R, 0.10 * R, 0.04 * R))
        _leaf(surf, cx + 0.02 * R, cy - 0.95 * R, R, -28, 0.50)
    elif kind == "watermelon":
        _stem(surf, cx + 0.02 * R, cy - 0.83 * R, R, 0.27, 0.05, 0.16, (88, 104, 42))
    return _downscale(surf, size)


def render_bomb(radius, seed=5):
    """Glossy black bomb with a metal cap and fuse.  Returns (surface, spark_offset)."""
    pad = 0.75
    n, size, R, u, v = coords(radius, pad)
    rr2 = u * u + v * v
    inside = rr2 <= 1.0
    nz = np.sqrt(np.clip(1 - rr2, 1e-4, None))
    rng = np.random.default_rng(seed)
    col = np.dstack([np.full((n, n), 0.055)] * 3).astype(np.float32)
    col += 0.02 * fbm(n, n, 6, 6, rng, 3)[..., None]
    lit = _shade(col, u, v, nz, 1.0, 46, 0.0, metal=1.0)
    surf = _to_surface(lit, inside, n)
    cx = cy = n / 2
    # cap
    cap_w, cap_h = 0.42 * R, 0.20 * R
    top = cy - 0.93 * R
    for i, c in enumerate([(60, 60, 66), (112, 112, 120), (170, 170, 178), (112, 112, 120), (52, 52, 58)]):
        w = cap_w * 2 / 5
        pygame.draw.rect(surf, c, (cx - cap_w + i * w, top - cap_h * 0.7, w + 1, cap_h))
    pygame.draw.ellipse(surf, (40, 40, 46), (cx - cap_w, top - cap_h * 0.7 - 0.05 * R, cap_w * 2, 0.12 * R))
    # fuse (quadratic curve)
    p0 = (cx, top - cap_h * 0.7)
    p1 = (cx + 0.05 * R, top - 0.75 * R)
    p2 = (cx + 0.50 * R, top - 0.58 * R)
    pts = []
    for i in range(21):
        s = i / 20
        pts.append(((1 - s) ** 2 * p0[0] + 2 * (1 - s) * s * p1[0] + s * s * p2[0],
                    (1 - s) ** 2 * p0[1] + 2 * (1 - s) * s * p1[1] + s * s * p2[1]))
    pygame.draw.lines(surf, (118, 90, 52), False, pts, max(2, int(0.085 * R)))
    pygame.draw.lines(surf, (190, 160, 100), False, [(x - 0.02 * R, y - 0.02 * R) for x, y in pts], max(1, int(0.03 * R)))
    # skull-ish warning mark: white "X" eyes + stripe so bombs read instantly
    ew = 0.11 * R
    for ex in (-0.26 * R, 0.26 * R):
        a = (cx + ex, cy + 0.02 * R)
        pygame.draw.line(surf, (225, 225, 230), (a[0] - ew, a[1] - ew), (a[0] + ew, a[1] + ew), max(2, int(0.06 * R)))
        pygame.draw.line(surf, (225, 225, 230), (a[0] - ew, a[1] + ew), (a[0] + ew, a[1] - ew), max(2, int(0.06 * R)))
    pygame.draw.line(surf, (200, 40, 40), (cx - 0.34 * R, cy + 0.42 * R), (cx + 0.34 * R, cy + 0.42 * R), max(2, int(0.07 * R)))
    tip = ((p2[0] - n / 2) / SS, (p2[1] - n / 2) / SS)
    return _downscale(surf, size), tip


def render_cut(kind, radius, seed=11):
    """Cross-section face of a fruit (what you see on the sliced halves)."""
    rng = np.random.default_rng(seed + hash(kind) % 991)
    n, size, R, u, v = coords(radius)
    sp = SHAPES[kind]
    L = sp["L"]
    pm = float(profile(kind, np.array(0.0)))
    if sp["axis"] == "x":
        axh, ayh = L, pm
        t, s = u / L, v
    else:
        axh, ayh = pm, L
        t, s = v / L, u
    pr = profile(kind, t)
    inside = (np.abs(t) <= 1.0) & (np.abs(s) <= pr)
    ex, ey = u / axh, v / ayh
    rr = np.sqrt(ex * ex + ey * ey)
    th = np.arctan2(ey, ex)
    seeds = []

    def pt(r, a):
        return (n / 2 + r * math.cos(a) * axh * R, n / 2 + r * math.sin(a) * ayh * R)

    fib = polar_noise(rr, th, rng, 6, 100, 3)
    wet = fbm(n, n, n / 7, n / 7, rng, 2)

    if kind == "apple":
        col = mix((0.97, 0.95, 0.78), (0.93, 0.85, 0.50), smoothstep(0.55, 0.15, rr))
        col = col * (0.93 + 0.12 * fib[..., None])
        col = mix(col, (0.92, 0.88, 0.62), (rr > 0.90) * 0.6)
        core_r = 0.25 + 0.045 * np.cos(5 * th + math.pi)
        col = mix(col, (0.86, 0.80, 0.52), (rr < core_r) * 1.0)
        col = mix(col, (0.66, 0.56, 0.32), np.exp(-((rr - core_r) / 0.012) ** 2) * 0.8)
        col = mix(col, (0.74, 0.08, 0.10), (rr > 0.955) * 1.0)
        for k in range(5):
            a = -math.pi / 2 + k * 2 * math.pi / 5
            seeds.append(("pocket", pt(0.145, a), a))
            seeds.append(("seed", pt(0.145, a), a, (70, 38, 20), 0.15, 0.075))
    elif kind in ("orange", "lemon"):
        nseg = 10 if kind == "orange" else 9
        if kind == "orange":
            zest, pith, f0, f1 = (0.98, 0.50, 0.07), (1.0, 0.95, 0.80), (0.99, 0.46, 0.04), (1.0, 0.68, 0.16)
        else:
            zest, pith, f0, f1 = (0.98, 0.84, 0.12), (1.0, 0.98, 0.84), (0.97, 0.88, 0.30), (1.0, 0.97, 0.58)
        seg = 2 * math.pi / nseg
        a = np.mod(th + math.pi / 2, seg)
        dm = np.minimum(a, seg - a) * rr
        ves = fbm(n, n, n / 9, n / 9, rng, 2)
        base = mix(f0, f1, 0.25 + 0.75 * ves)
        base = base * (0.90 + 0.18 * fib[..., None])
        base = mix(base, f1, smoothstep(0.88, 0.45, rr) * 0.25 * ves)
        mem = np.clip(1 - dm / 0.022, 0, 1) * (rr > 0.07)
        col = mix(base, pith, np.clip(mem * 0.85, 0, 1))
        col = mix(col, pith, (rr < 0.085) * 1.0)
        col = mix(col, pith, np.exp(-((rr - 0.875) / 0.014) ** 2) * 0.9)
        col = mix(col, pith, (rr > 0.885) * (rr <= 0.945) * 1.0)
        col = col * np.where((rr > 0.885) & (rr <= 0.945), 0.97 + 0.03 * fib, 1.0)[..., None]
        zc = mix(zest, np.asarray(zest) * 0.9, wet) * (0.92 + 0.12 * wet[..., None])
        col = np.where((rr > 0.945)[..., None], zc, col)
        if kind == "lemon":
            for aa in (0.9, 3.1, 5.2):
                seeds.append(("seed", pt(0.34, aa), aa + 1.2, (236, 224, 168), 0.13, 0.065))
    elif kind == "kiwi":
        flesh = mix((0.40, 0.68, 0.10), (0.56, 0.80, 0.20), smoothstep(0.95, 0.45, rr))
        flesh = mix(flesh, (0.82, 0.93, 0.52), smoothstep(0.50, 0.22, rr))
        flesh = flesh * (0.90 + 0.22 * fib[..., None])
        col = mix(flesh, (0.96, 0.98, 0.84), smoothstep(0.26, 0.20, rr) * 0.95)
        lines = np.clip(polar_noise(rr, th, rng, 3, 60, 2) - 0.5, 0, 1)
        col = col + (lines * 0.35 * smoothstep(0.30, 0.55, rr) * (rr < 0.92))[..., None]
        col = mix(col, (0.76, 0.86, 0.46), (rr > 0.935) * (rr <= 0.965) * 1.0)
        col = mix(col, (0.46, 0.33, 0.20) * (0.8 + 0.4 * fib[..., None]), (rr > 0.965) * 1.0)
        cnt = 34
        for k in range(cnt):
            aa = k * 2 * math.pi / cnt + rng.uniform(-0.06, 0.06)
            rg = 0.37 + rng.uniform(-0.035, 0.035)
            seeds.append(("seed", pt(rg, aa), aa, (18, 15, 14), 0.075, 0.036))
    elif kind == "watermelon":
        flesh = mix((0.93, 0.10, 0.20), (0.99, 0.50, 0.52), smoothstep(0.30, 0.82, rr))
        flesh = flesh * (0.92 + 0.14 * fib[..., None])
        sparkle = (rng.random((n, n)) > 0.996).astype(np.float32)
        sparkle = cv2.dilate(sparkle, np.ones((3, 3), np.uint8))
        flesh = flesh + sparkle[..., None] * 0.22
        col = mix(flesh, (0.93, 0.97, 0.88), (rr > 0.83) * 1.0)
        col = mix(col, (0.56, 0.78, 0.36), (rr > 0.895) * 1.0)
        col = mix(col, (0.13, 0.42, 0.17), (rr > 0.945) * 1.0)
        placed = []
        tries = 0
        while len(placed) < 15 and tries < 400:
            tries += 1
            rg = rng.uniform(0.18, 0.70)
            aa = rng.uniform(0, 2 * math.pi)
            p_ = pt(rg, aa)
            if all((p_[0] - q[0]) ** 2 + (p_[1] - q[1]) ** 2 > (0.24 * R) ** 2 for q in placed):
                placed.append(p_)
                seeds.append(("seed", p_, aa + rng.uniform(-0.8, 0.8) + math.pi / 2, (14, 12, 12), 0.15, 0.075))
    else:
        col = np.zeros((n, n, 3), np.float32)

    # wet highlight on flesh
    hl = np.exp(-(((u + 0.30) / 0.46) ** 2 + ((v + 0.34) / 0.40) ** 2)) * 0.20
    col = col + (hl * (rr < 0.94))[..., None]
    col = col * (1 - 0.18 * smoothstep(0.90, 1.0, rr))[..., None]
    surf = _to_surface(col.astype(np.float32), inside, n)
    for it in seeds:
        if it[0] == "pocket":
            pygame.draw.circle(surf, (205, 190, 125), (int(it[1][0]), int(it[1][1])), int(0.075 * R))
        else:
            _, (x, y), ang, c, ln, wd = it
            _teardrop(surf, x, y, ln * R, wd * R, ang, c)
    return _downscale(surf, size)


class Assets:
    def __init__(self, radius):
        self.radius = radius
        self.whole, self.cut = {}, {}
        for k in FRUIT_TYPES:
            self.whole[k] = render_fruit(k, radius)
            self.cut[k] = render_cut(k, radius)
        self.whole["bomb"], self.bomb_tip = render_bomb(radius)
        self.shadow = make_shadow(radius)


def make_shadow(radius):
    size = int(radius * 3)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.sqrt((xs - size / 2) ** 2 + (ys - size / 2) ** 2) / (radius * 1.05)
    a = np.clip(1 - d, 0, 1) ** 1.6 * 110
    rgba = np.dstack([np.zeros((size, size, 3), np.uint8), a.astype(np.uint8)])
    return pygame.image.frombuffer(np.ascontiguousarray(rgba).tobytes(), (size, size), "RGBA").copy()


def split_surface(surf, nx, ny):
    """Cut a sprite along a line through its centre (normal = (nx, ny)).
    Returns [(half_surface, (offset_x, offset_y)), ...] for the +normal / -normal sides."""
    w, h = surf.get_size()
    xs = np.arange(w)[:, None] - (w - 1) / 2.0
    ys = np.arange(h)[None, :] - (h - 1) / 2.0
    side = xs * nx + ys * ny
    out = []
    for sign in (1, -1):
        s = surf.copy()
        a = pygame.surfarray.pixels_alpha(s)
        a[:] = np.where(side * sign >= 1.0, a, 0)
        del a
        rect = s.get_bounding_rect(min_alpha=1)
        if rect.w < 2 or rect.h < 2:
            rect = pygame.Rect(0, 0, 2, 2)
        sub = s.subsurface(rect).copy()
        lip = (np.abs(side[rect.x:rect.x + rect.w, rect.y:rect.y + rect.h]) < 2.2)
        px = pygame.surfarray.pixels3d(sub)
        px[lip] = (px[lip] * 0.55 + 255 * 0.45).astype(np.uint8)
        del px
        out.append((sub, (rect.centerx - w / 2.0, rect.centery - h / 2.0)))
    return out


# ============================================================ background etc ==

def make_background(w, h):
    rng = np.random.default_rng(99)
    img = np.zeros((h, w, 3), np.float32)
    plank = 128
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    grain = fbm(h, w, 5, w // 3, rng, 4)
    rings = 0.5 + 0.5 * np.sin(xs * 0.07 + 9.0 * grain + ys * 0.004)
    for i in range(0, w // plank + 2):
        x0, x1 = i * plank, min(w, (i + 1) * plank)
        if x0 >= w:
            break
        tone = 0.72 + 0.50 * rng.random()
        base = np.array([0.30, 0.17, 0.09], np.float32) * tone
        seg = base[None, None, :] * (0.78 + 0.34 * rings[:, x0:x1, None]) * (0.88 + 0.24 * grain[:, x0:x1, None])
        seg[:, :3, :] *= 0.35
        seg[:, 3:5, :] *= 1.18
        img[:, x0:x1, :] = seg
    cxn, cyn = w / 2, h * 0.30
    d = np.sqrt(((xs - cxn) / (w * 0.62)) ** 2 + ((ys - cyn) / (h * 0.95)) ** 2)
    light = np.clip(1.15 - d, 0.0, 1.0)
    img *= (0.22 + 0.95 * light ** 1.4)[..., None]
    img += (np.exp(-(((xs - cxn) / (w * 0.30)) ** 2 + ((ys - cyn) / (h * 0.35)) ** 2)) * 0.07)[..., None] \
        * np.array([1.0, 0.7, 0.35], np.float32)
    arr = np.clip(img * 255, 0, 255).astype(np.uint8)
    return pygame.surfarray.make_surface(arr.swapaxes(0, 1)).convert()


def make_heart(size, color):
    big = size * 4
    surf = pygame.Surface((big, big), pygame.SRCALPHA)
    pts = []
    for i in range(80):
        t = 2 * math.pi * i / 80
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((big / 2 + x * big / 36, big / 2 - y * big / 36 - big * 0.02))
    pygame.draw.polygon(surf, color, pts)
    hl = tuple(min(255, c + 70) for c in color[:3])
    pygame.draw.ellipse(surf, hl, (big * 0.22, big * 0.20, big * 0.18, big * 0.12))
    return pygame.transform.smoothscale(surf, (size, size))


def make_splat(color, rng):
    s = 120
    surf = pygame.Surface((s, s), pygame.SRCALPHA)
    c = (*color, 150)
    pygame.draw.circle(surf, c, (s // 2, s // 2), rng.randint(14, 22))
    for _ in range(rng.randint(8, 13)):
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(14, 44)
        r = rng.randint(3, 10)
        x, y = s / 2 + math.cos(a) * d, s / 2 + math.sin(a) * d
        pygame.draw.line(surf, c, (s / 2 + math.cos(a) * 10, s / 2 + math.sin(a) * 10), (x, y), max(2, r // 2))
        pygame.draw.circle(surf, c, (int(x), int(y)), r)
    return surf


# =============================================================== camera layer ==

def candidate_cameras():
    idx = []
    if sys.platform.startswith("linux"):
        try:
            for name in os.listdir("/dev"):
                if name.startswith("video") and name[5:].isdigit():
                    idx.append(int(name[5:]))
        except Exception:
            pass
    if not idx:
        idx = list(range(0, 6))
    return sorted(set(idx))


def _backends():
    if sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_camera(index):
    """Open camera `index`; returns (cap, w, h) only if it really delivers frames."""
    for be in _backends():
        cap = None
        try:
            cap = cv2.VideoCapture(index, be)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_REQUEST[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_REQUEST[1])
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            good = 0
            frame = None
            for _ in range(25):                      # let auto-exposure warm up
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    good += 1
                    if float(frame.mean()) > 2.0 and good >= 2:
                        return cap, frame.shape[1], frame.shape[0]
                else:
                    time.sleep(0.02)
            if good >= 2:                            # works but very dark (covered / dim room)
                return cap, frame.shape[1], frame.shape[0]
            cap.release()
        except Exception:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
    return None, 0, 0


def find_camera(preferred=None, log=print):
    order = ([preferred] if preferred is not None else []) + [i for i in candidate_cameras() if i != preferred]
    for i in order:
        cap, w, h = open_camera(i)
        if cap is not None:
            log(f"[camera] using camera {i} ({w}x{h})")
            return cap, i, w, h
    return None, None, 0, 0


# ============================================================= hand trackers ==

def _download_model(dest, log=print):
    tmp = dest + ".part"
    log("[model] downloading hand model (~7.5 MB, one time only)...")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "hand-ninja"})
    with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r[model] {done * 100 // total}%", end="", flush=True)
    print()
    os.replace(tmp, dest)
    log("[model] saved to " + dest)


def ensure_model(path_hint=None, log=print):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [p for p in [path_hint, os.path.join(here, MODEL_NAME), os.path.join(os.getcwd(), MODEL_NAME),
                         os.path.join(os.path.expanduser("~"), ".cache", "hand_ninja", MODEL_NAME)] if p]
    for p in cands:
        if os.path.isfile(p) and os.path.getsize(p) > 1_000_000:
            return p
    for dest in (os.path.join(here, MODEL_NAME), os.path.join(os.path.expanduser("~"), ".cache", "hand_ninja", MODEL_NAME)):
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            _download_model(dest, log)
            return dest
        except Exception as e:
            log(f"[model] download failed ({e})")
    return None


class TasksTracker:
    """MediaPipe >= 0.10 Tasks API (works on MediaPipe 1.x where mp.solutions is gone)."""
    name = "MediaPipe HandLandmarker"

    def __init__(self, model_path):
        from mediapipe.tasks.python import vision, BaseOptions
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,                     # 2nd hand catches the 1st one's dropouts
            min_hand_detection_confidence=0.50,
            min_hand_presence_confidence=0.50,
            min_tracking_confidence=0.40,    # low -> keeps lock through hard poses
        )
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.last_ts = 0

    def detect(self, rgb, ts_ms):
        """Returns [(tip, landmarks), ...] for every hand found, or None."""
        ts_ms = max(int(ts_ms), self.last_ts + 1)
        self.last_ts = ts_ms
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.lm.detect_for_video(img, ts_ms)
        if not res.hand_landmarks:
            return None
        out = []
        for hl in res.hand_landmarks:
            pts = [(p.x, p.y) for p in hl]
            out.append((pts[8], pts))       # landmark 8 = index fingertip
        return out

    def close(self):
        try:
            self.lm.close()
        except Exception:
            pass


class LegacyTracker:
    """Old MediaPipe (<= 0.10.14) mp.solutions.hands."""
    name = "MediaPipe Hands (legacy)"

    def __init__(self):
        self.h = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0,
                                          min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect(self, rgb, ts_ms):
        """Returns [(tip, landmarks), ...] for every hand found, or None."""
        r = self.h.process(rgb)
        if not r.multi_hand_landmarks:
            return None
        out = []
        for hl in r.multi_hand_landmarks:
            pts = [(p.x, p.y) for p in hl.landmark]
            out.append((pts[8], pts))
        return out

    def close(self):
        try:
            self.h.close()
        except Exception:
            pass


class SkinTracker:
    """Last-resort OpenCV tracker: moving skin-coloured blob, topmost point = fingertip."""
    name = "OpenCV skin fallback (low accuracy)"

    def __init__(self):
        self.prev = None

    def detect(self, rgb, ts_ms):
        small = cv2.GaussianBlur(cv2.resize(rgb, (320, 180)), (5, 5), 0)
        skin = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb), (0, 135, 85), (255, 180, 135))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        if self.prev is None:
            self.prev = gray.astype(np.float32)
            return None
        motion = cv2.threshold(cv2.absdiff(gray, cv2.convertScaleAbs(self.prev)), 16, 255, cv2.THRESH_BINARY)[1]
        cv2.accumulateWeighted(gray, self.prev, 0.25)
        mask = cv2.bitwise_and(skin, cv2.dilate(motion, None, iterations=6))
        mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), None, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 250:
            return None
        x, y = min(c[:, 0, :].tolist(), key=lambda p: p[1])
        return [((x / 320.0, y / 180.0), None)]

    def close(self):
        pass


def make_tracker(model_hint, log=print):
    if mp is None:
        log(f"[tracker] mediapipe import failed ({MP_IMPORT_ERROR}) -> OpenCV fallback")
        return SkinTracker()
    if hasattr(mp, "tasks"):
        path = ensure_model(model_hint, log)
        if path:
            try:
                t = TasksTracker(path)
                log("[tracker] MediaPipe HandLandmarker ready")
                return t
            except Exception as e:
                log(f"[tracker] Tasks API failed: {e}")
        else:
            log("[tracker] no model file.  Download it manually from:\n          " + MODEL_URL +
                "\n          and put it next to hand_ninja.py (or use --model PATH)")
    if hasattr(mp, "solutions"):
        try:
            t = LegacyTracker()
            log("[tracker] MediaPipe legacy Hands ready")
            return t
        except Exception as e:
            log(f"[tracker] legacy Hands failed: {e}")
    log("[tracker] using OpenCV skin fallback - accuracy is much lower!")
    return SkinTracker()


Snapshot = namedtuple("Snapshot", "seq tip lms frame fps status cam_ok tracker")
EMPTY_SNAP = Snapshot(0, None, None, None, 0.0, "Starting camera...", False, "")


class VisionThread(threading.Thread):
    def __init__(self, preferred_cam=None, model_hint=None):
        super().__init__(daemon=True)
        self.preferred = preferred_cam
        self.model_hint = model_hint
        self.stop_flag = False
        self.snap = EMPTY_SNAP
        self.seq = 0
        self.status = "Starting camera..."
        self.cam_index = None
        try:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        except Exception:
            self.clahe = None
        self.detect_every = 1      # run inference every Nth camera frame (adapts)
        self.frame_i = 0
        self.last_tip = None       # for continuity when 2 hands are visible

    def log(self, msg):
        print(msg, flush=True)

    def _publish(self, tip, lms, frame, fps, cam_ok, tracker):
        self.seq += 1
        self.snap = Snapshot(self.seq, tip, lms, frame, fps, self.status, cam_ok, tracker)

    def _pick_hand(self, hands):
        """Keep the hand whose fingertip is closest to the last tracked one, so a
        second visible hand catches the first one's dropouts instead of the blade
        dying (and the blade never flickers between two hands)."""
        if not hands:
            return None, None
        if self.last_tip is None or len(hands) == 1:
            return hands[0]
        lt = self.last_tip
        best, bd = hands[0], 1e9
        for h in hands:
            d = (h[0][0] - lt[0]) ** 2 + (h[0][1] - lt[1]) ** 2
            if d < bd:
                best, bd = h, d
        if bd ** 0.5 > 0.35:            # nothing nearby -> treat as a fresh hand
            return hands[0]
        return best

    def run(self):
        self.status = "Loading hand tracker..."
        self._publish(None, None, None, 0.0, False, "")
        tracker = make_tracker(self.model_hint, self.log)
        tname = tracker.name
        cap = None
        fails = 0
        fps_t, fps_n, fps = time.time(), 0, 0.0
        t0 = time.monotonic()
        tip, lms = None, None
        while not self.stop_flag:
            if cap is None:
                tip, lms, self.last_tip = None, None, None
                self.status = "Scanning for cameras..."
                self._publish(None, None, None, 0.0, False, tname)
                cap, self.cam_index, _, _ = find_camera(self.preferred, self.log)
                if cap is None:
                    self.status = "No camera found - retrying (or use the mouse)"
                    self._publish(None, None, None, 0.0, False, tname)
                    self.log("[camera] no working camera found; will retry. (Linux: check `ls /dev/video*` "
                             "and that your user is in the 'video' group.)")
                    for _ in range(30):
                        if self.stop_flag:
                            break
                        time.sleep(0.1)
                    continue
                self.preferred = self.cam_index
                self.status = f"Camera {self.cam_index} OK"
                fails = 0
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                fails += 1
                if fails > 40:
                    self.log("[camera] lost connection - rescanning")
                    cap.release()
                    cap = None
                time.sleep(0.01)
                continue
            fails = 0
            frame = cv2.flip(frame, 1)
            small = cv2.resize(frame, DETECT_SIZE, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # dim room? boost contrast on a copy so the hand model still finds the hand
            det_rgb = rgb
            if self.clahe is not None and float(rgb.mean()) < 58.0:
                try:
                    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                    lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
                    det_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                except Exception:
                    det_rgb = rgb

            # Inference runs every Nth frame; N adapts to how slow the model is so
            # this thread never falls behind the camera (keeps latency low).
            self.frame_i += 1
            if self.frame_i % self.detect_every == 0:
                t_d = time.perf_counter()
                try:
                    hands = tracker.detect(det_rgb, (time.monotonic() - t0) * 1000.0)
                except Exception as e:
                    self.log(f"[tracker] error: {e}")
                    hands = None
                det_s = time.perf_counter() - t_d
                if det_s > 0.055 and self.detect_every < 3:
                    self.detect_every += 1
                elif det_s < 0.024 and self.detect_every > 1:
                    self.detect_every -= 1
                if hands:
                    tip, lms = self._pick_hand(hands)
                    self.last_tip = tip
                else:
                    tip, lms, self.last_tip = None, None, None
                self.seq += 1          # only fresh detections bump the sequence
                fps_n += 1
                if time.time() - fps_t >= 1.0:
                    fps = fps_n / (time.time() - fps_t)
                    fps_t, fps_n = time.time(), 0
            disp = cv2.resize(rgb, (512, 288), interpolation=cv2.INTER_AREA)
            self.status = f"Camera {self.cam_index} OK"
            self.snap = Snapshot(self.seq, tip, lms, disp, fps, self.status, True, tname)
        if cap is not None:
            cap.release()
        tracker.close()


class OneEuro:
    """One-Euro filter: heavy smoothing when slow, almost none when fast."""

    def __init__(self, mincutoff=ONE_EURO_CUTOFF, beta=ONE_EURO_BETA, dcutoff=1.0):
        self.mc, self.beta, self.dc = mincutoff, beta, dcutoff
        self.x = None
        self.dx = 0.0
        self.t = None

    @staticmethod
    def _a(cut, dt):
        tau = 1.0 / (2 * math.pi * cut)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(1e-3, t - self.t)
        dx = (x - self.x) / dt
        self.dx += self._a(self.dc, dt) * (dx - self.dx)
        cutoff = self.mc + self.beta * abs(self.dx)
        self.x += self._a(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


# ============================================================== geometry =====

def segment_circle_intersect(p1, p2, center, radius):
    px, py = float(p1[0]), float(p1[1])
    qx, qy = float(p2[0]), float(p2[1])
    cx, cy = float(center[0]), float(center[1])
    dx, dy = qx - px, qy - py
    fx, fy = px - cx, py - cy
    a = dx * dx + dy * dy
    if a < 1e-9:
        return fx * fx + fy * fy <= radius * radius
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1, t2 = (-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0) or (t1 < 0.0 and t2 > 1.0)


def _chaikin(pts):
    """Chaikin corner-cutting: replaces every corner with two points -> smooth curve.
    pts are (x, y, t) tuples; t is interpolated along with the position."""
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for i in range(len(pts) - 1):
        p, q = pts[i], pts[i + 1]
        out.append((p[0] * 0.75 + q[0] * 0.25, p[1] * 0.75 + q[1] * 0.25, p[2] * 0.75 + q[2] * 0.25))
        out.append((p[0] * 0.25 + q[0] * 0.75, p[1] * 0.25 + q[1] * 0.75, p[2] * 0.25 + q[2] * 0.75))
    out.append(pts[-1])
    return out


# ================================================================== the game ==

class Game:
    def __init__(self, args):
        self.args = args
        pygame.init()
        try:
            pygame.mixer.init(frequency=SR, size=-16, channels=2, buffer=256)
        except Exception:
            pass
        flags = pygame.SCALED | pygame.RESIZABLE
        if args.fullscreen:
            flags |= pygame.FULLSCREEN
        try:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
        except Exception:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hand Ninja")
        self.clock = pygame.time.Clock()
        names = "poppins,montserrat,segoeui,arial,dejavusans,liberationsans,helvetica,sans"
        self.f_huge = pygame.font.SysFont(names, 78, bold=True)
        self.f_big = pygame.font.SysFont(names, 34, bold=True)
        self.f_mid = pygame.font.SysFont(names, 24, bold=True)
        self.f_small = pygame.font.SysFont(names, 17, bold=True)
        self.f_tiny = pygame.font.SysFont(names, 13)
        self.splash("Preparing the dojo...")

        # camera / tracking starts in the background immediately
        self.vision = None
        self.mouse_mode = bool(args.mouse or args.no_camera)
        if not args.no_camera and not args.mouse:
            self.vision = VisionThread(args.camera, args.model)
            self.vision.start()

        # art
        self.splash("Slicing up some fruit textures...")
        t0 = time.time()
        self.assets = Assets(FRUIT_RADIUS)
        self.big_assets = {k: (render_fruit(k, 84), render_cut(k, 84)) for k in ("watermelon", "apple")}
        print(f"[art] procedural fruit rendered in {time.time() - t0:.1f}s")
        self.bg = make_background(WIDTH, HEIGHT)
        self.hearts = (make_heart(30, (232, 56, 78)), make_heart(30, (70, 36, 44)))
        self.glow = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA).convert_alpha()
        self.psurf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA).convert_alpha()
        self.vignette = self._vignette()
        self.sfx = build_sounds()
        self.muted = False

        self.best = self._load_best()
        self.cam_mode = 1          # 0 off, 1 PiP, 2 background
        self.debug = False
        self.paused = False
        self.rng = random.Random()

        # input state (the blade rides a critically-damped spring - see poll_input)
        self.filt_x, self.filt_y = OneEuro(), OneEuro()
        self.last_seq = -1
        self.target = None
        self.target_t = 0.0
        self.tvel = (0.0, 0.0)
        self.blade = None
        self.blade_vel = (0.0, 0.0)
        self.prev_blade = None
        self.no_hand = 999
        self.snap = EMPTY_SNAP
        self.last_swish = 0.0
        self.last_poll_t = time.time()
        self.frame_dt = 1.0 / FPS
        self.cam_surface = None

        self.state = "menu"
        self.t = 0.0
        self.frame_no = 0
        self.fruits, self.halves, self.particles, self.splats, self.popups = [], [], [], [], []
        self.trail = deque()
        self.shake = self.flash = 0
        self.reset_round()
        self.menu_target = None
        self.make_menu_target()

        # autoplay (selftest / attract mode)
        self.autoplay = args.autoplay or args.selftest > 0
        self.ap_swipe = 0
        self.ap_from = self.ap_to = (0, 0)

    # ---------------------------------------------------------------- utils --
    def splash(self, text):
        self.screen.fill((20, 12, 8))
        s = self.f_big.render(text, True, (240, 220, 190))
        self.screen.blit(s, s.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
        pygame.display.flip()
        pygame.event.pump()

    def _vignette(self):
        ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        d = np.sqrt(((xs - WIDTH / 2) / (WIDTH * 0.62)) ** 2 + ((ys - HEIGHT / 2) / (HEIGHT * 0.66)) ** 2)
        a = (np.clip(d - 0.55, 0, 1) ** 1.6 * 190).astype(np.uint8)
        rgba = np.dstack([np.zeros((HEIGHT, WIDTH, 3), np.uint8), a])
        return pygame.image.frombuffer(np.ascontiguousarray(rgba).tobytes(), (WIDTH, HEIGHT), "RGBA").copy().convert_alpha()

    def _load_best(self):
        try:
            return int(open(BEST_FILE).read().strip())
        except Exception:
            return 0

    def _save_best(self):
        try:
            with open(BEST_FILE, "w") as f:
                f.write(str(self.best))
        except Exception:
            pass

    def play(self, name):
        if self.muted:
            return
        s = self.sfx.get(name)
        if s:
            s.play()

    def text(self, font, txt, pos, color=HUD_COLOR, anchor="topleft", shadow=True, alpha=255, surf=None):
        surf = surf or self.screen
        img = font.render(txt, True, color)
        if shadow:
            sh = font.render(txt, True, (0, 0, 0))
            sh.set_alpha(min(200, alpha))
            r = sh.get_rect(**{anchor: (pos[0] + 2, pos[1] + 2)})
            surf.blit(sh, r)
        if alpha < 255:
            img.set_alpha(alpha)
        surf.blit(img, img.get_rect(**{anchor: pos}))

    # --------------------------------------------------------------- state ---
    def reset_round(self):
        self.score = 0
        self.lives = MAX_LIVES
        self.fruits.clear()
        self.combo = 0
        self.combo_pulse = 1.0
        self.last_slice_ms = 0.0
        self.spawn_timer = 0
        self.next_spawn = self.rng.randint(SPAWN_INTERVAL_MIN, SPAWN_INTERVAL_MAX)
        self.shake = self.flash = 0

    def make_menu_target(self, kind=None):
        kind = kind or ("watermelon" if self.state == "menu" else "apple")
        whole, cut = self.big_assets[kind]
        self.menu_target = dict(kind=kind, whole=whole, cut=cut, x=WIDTH / 2, y=HEIGHT * 0.66,
                                r=76, angle=0.0, hit=False)

    def start_game(self):
        self.reset_round()
        self.state = "playing"
        self.play("start")

    def end_game(self):
        self.state = "gameover"
        if self.score > self.best:
            self.best = self.score
            self._save_best()
        self.make_menu_target("apple")

    # --------------------------------------------------------------- input ---
    def _predicted_target(self, now):
        """Dead-reckoned blade target: filtered fingertip + velocity look-ahead.
        Extrapolates between camera updates (30 Hz) so the spring always chases a
        continuously moving target -> no stair-steps at the camera rate, and the
        look-ahead hides the remaining camera/filter/spring latency."""
        if self.target is None:
            return None
        vx, vy = self.tvel
        speed = math.hypot(vx, vy)
        ahead = PREDICT_S * min(1.0, speed / PREDICT_FULL_SPEED)
        since = min(0.15, max(0.0, now - self.target_t))   # time since last camera update
        x = self.target[0] + vx * (since + ahead)
        y = self.target[1] + vy * (since + ahead)
        return (min(max(x, 0.0), float(WIDTH)), min(max(y, 0.0), float(HEIGHT)))

    def _reset_blade_at(self, p):
        """Snap the blade to a point with zero speed (new hand / teleport)."""
        self.blade = [float(p[0]), float(p[1])]
        self.blade_vel = (0.0, 0.0)
        self.prev_blade = None
        if self.trail and math.hypot(p[0] - self.trail[-1][0], p[1] - self.trail[-1][1]) > TELEPORT_DIST * 0.5:
            self.trail.clear()

    def _spring_to(self, tgt, dt):
        """Critically-damped spring: buttery motion, no overshoot, and identical
        feel at any frame rate (sub-stepped for stability)."""
        steps = max(1, min(8, int(math.ceil(dt / 0.008))))
        h = dt / steps
        w2 = BLADE_OMEGA * BLADE_OMEGA
        vx, vy = self.blade_vel
        for _ in range(steps):
            vx += (w2 * (tgt[0] - self.blade[0]) - 2.0 * BLADE_OMEGA * vx) * h
            vy += (w2 * (tgt[1] - self.blade[1]) - 2.0 * BLADE_OMEGA * vy) * h
            sp = math.hypot(vx, vy)
            if sp > MAX_BLADE_SPEED:
                s = MAX_BLADE_SPEED / sp
                vx, vy = vx * s, vy * s
            self.blade[0] += vx * h
            self.blade[1] += vy * h
        self.blade_vel = (vx, vy)

    def poll_input(self):
        snap = self.vision.snap if self.vision else EMPTY_SNAP
        self.snap = snap
        now = time.time()
        dt = max(1e-3, min(1.0 / 30.0, now - self.last_poll_t))
        self.last_poll_t = now

        # ---- hand target: OneEuro-filter on fresh camera frames -------------
        if snap.cam_ok and snap.tip is not None:
            if snap.seq != self.last_seq:
                self.last_seq = snap.seq
                self.no_hand = 0
                nx = min(1.0, max(0.0, (snap.tip[0] - MARGIN_X) / (1 - 2 * MARGIN_X)))
                ny = min(1.0, max(0.0, (snap.tip[1] - MARGIN_Y) / (1 - 2 * MARGIN_Y)))
                new = (self.filt_x(nx * WIDTH, now), self.filt_y(ny * HEIGHT, now))
                if self.target is not None:
                    jump = math.hypot(new[0] - self.target[0], new[1] - self.target[1])
                    if jump > TELEPORT_DIST:        # tracking glitch / other hand
                        self._reset_blade_at(new)   # snap over, don't slice across
                        self.tvel = (0.0, 0.0)      # a jump is not motion - don't predict past it
                    else:
                        dts = max(1e-3, now - self.target_t)
                        tvx = (new[0] - self.target[0]) / dts
                        tvy = (new[1] - self.target[1]) / dts
                        tsp = math.hypot(tvx, tvy)
                        if tsp > MAX_TRACK_SPEED:   # measurement spike - clamp it
                            k = MAX_TRACK_SPEED / tsp
                            tvx, tvy = tvx * k, tvy * k
                        self.tvel = (tvx, tvy)
                self.target, self.target_t = new, now
            tgt = self._predicted_target(now)
        else:
            self.no_hand += 1
            if self.target is None or self.no_hand > HAND_GRACE_FRAMES:
                if self.target is not None:
                    self.filt_x, self.filt_y = OneEuro(), OneEuro()
                    self.target, self.tvel = None, (0.0, 0.0)
                tgt = None
            else:
                # brief dropout: glide on momentum so a swipe in progress
                # still finishes instead of the blade freezing mid-air
                decay = GLIDE_DECAY ** self.no_hand
                self.target = (self.target[0] + self.tvel[0] * dt * decay,
                               self.target[1] + self.tvel[1] * dt * decay)
                self.target_t = now
                tgt = self._predicted_target(now)

        mouse_down = pygame.mouse.get_pressed()[0]
        if mouse_down or (self.mouse_mode and pygame.mouse.get_focused()):
            tgt = pygame.mouse.get_pos()
        if self.autoplay:
            tgt = self.autoplay_target()
        if tgt is None:
            self.blade = None
            self.prev_blade = None
            self.blade_vel = (0.0, 0.0)
            return
        if self.blade is None:
            self._reset_blade_at(tgt)
            return
        self.prev_blade = (self.blade[0], self.blade[1])
        self._spring_to(tgt, dt)

    def autoplay_target(self):
        if self.state != "playing":
            tgt = self.menu_target
            if self.ap_swipe <= 0 and tgt and not tgt["hit"] and self.frame_no % 50 == 20:
                self.ap_from, self.ap_to, self.ap_swipe = (tgt["x"] - 130, tgt["y"] - 70), (tgt["x"] + 130, tgt["y"] + 70), 8
        elif self.ap_swipe <= 0:
            best = None
            for f in self.fruits:
                if f["type"] != "bomb" and not f["sliced"] and 80 < f["y"] < HEIGHT - 60 and f["vy"] < 6:
                    if best is None or f["y"] < best["y"]:
                        best = f
            if best and self.rng.random() < 0.55:
                ang = self.rng.uniform(-0.8, 0.8)
                dx, dy = math.cos(ang) * 120, math.sin(ang) * 80
                self.ap_from = (best["x"] - dx, best["y"] - dy)
                self.ap_to = (best["x"] + dx, best["y"] + dy)
                self.ap_swipe = 8
        if self.ap_swipe > 0:
            k = 1.0 - self.ap_swipe / 8.0
            self.ap_swipe -= 1
            return (self.ap_from[0] + (self.ap_to[0] - self.ap_from[0]) * k,
                    self.ap_from[1] + (self.ap_to[1] - self.ap_from[1]) * k)
        return None

    # ---------------------------------------------------------------- logic --
    def make_fruit(self):
        kind = "bomb" if self.rng.random() < BOMB_CHANCE else self.rng.choice(FRUIT_TYPES)
        x = self.rng.uniform(90, WIDTH - 90)
        vy = -(SPAWN_VY_MIN + self.rng.random() * SPAWN_VY_VAR)
        vx = self.rng.uniform(-SPAWN_VX_RANGE, SPAWN_VX_RANGE)
        if x < WIDTH * 0.3:
            vx = abs(vx)
        elif x > WIDTH * 0.7:
            vx = -abs(vx)
        return dict(x=x, y=float(SPAWN_Y), vx=vx, vy=vy, radius=FRUIT_RADIUS, type=kind, sliced=False,
                    angle=self.rng.uniform(0, 360), spin=self.rng.uniform(-3.2, 3.2))

    def slice_object(self, f, p1, p2, whole_r=None):
        f["sliced"] = True
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        mag = math.hypot(dx, dy)
        if mag < 1e-3:
            nx, ny = 1.0, 0.0
        else:
            nx, ny = -dy / mag, dx / mag
        kind = f["type"]
        if kind != "bomb":
            face = f["cut"] if "cut" in f else self.assets.cut[kind]
            rot = pygame.transform.rotozoom(face, f.get("angle", 0.0), 1.0)
            for (sub, off), sg in zip(split_surface(rot, nx, ny), (1, -1)):
                self.halves.append(dict(surf=sub, x=f["x"] + off[0], y=f["y"] + off[1],
                                        vx=f.get("vx", 0.0) + sg * nx * HALF_SPEED,
                                        vy=f.get("vy", 0.0) + sg * ny * HALF_SPEED - 1.0,
                                        rot=0.0, rs=sg * self.rng.uniform(1.6, 4.2)))
            col = JUICE_COLORS[kind]
            ang0 = math.atan2(ny, nx)
            n = self.rng.randint(JUICE_MIN, JUICE_MAX)
            for _ in range(n):
                a = ang0 + self.rng.choice([-1, 1]) * self.rng.uniform(0, math.pi / 2.6)
                sp = self.rng.uniform(2.0, 8.0)
                self.particles.append(dict(x=f["x"], y=f["y"], vx=math.cos(a) * sp, vy=math.sin(a) * sp - 2.0,
                                           color=col, life=self.rng.randint(26, 52), max=52,
                                           r=self.rng.uniform(2.0, 5.5)))
            if len(self.splats) < 14:
                self.splats.append(dict(surf=make_splat(col, self.rng), x=f["x"], y=f["y"], life=240, max=240,
                                        rot=self.rng.uniform(0, 360)))
        else:
            for _ in range(70):
                a = self.rng.uniform(0, 2 * math.pi)
                sp = self.rng.uniform(3.0, 13.0)
                self.particles.append(dict(x=f["x"], y=f["y"], vx=math.cos(a) * sp, vy=math.sin(a) * sp,
                                           color=self.rng.choice([(255, 110, 30), (255, 210, 60), (70, 70, 70), (255, 255, 200)]),
                                           life=self.rng.randint(30, 70), max=70, r=self.rng.uniform(2.5, 6.5)))

    def update(self):
        now_ms = time.time() * 1000.0
        if self.combo >= 2 and now_ms - self.last_slice_ms > COMBO_WINDOW_MS:
            self.combo = 0
        slice_seg = None
        if self.blade is not None and self.prev_blade is not None:
            dist = math.hypot(self.blade[0] - self.prev_blade[0], self.blade[1] - self.prev_blade[1])
            spd = dist / max(1e-3, self.frame_dt)      # px per second -> same feel at any FPS
            if spd >= MIN_SLICE_SPEED:
                slice_seg = (self.prev_blade, (self.blade[0], self.blade[1]))
            if spd > SWISH_SPEED and time.time() - self.last_swish > 0.28:
                self.last_swish = time.time()
                self.play("swish")

        if self.state == "playing":
            self.spawn_timer += 1
            interval = max(MIN_INTERVAL, self.next_spawn - self.score // DIFFICULTY_STEP)
            if self.spawn_timer >= interval:
                self.spawn_timer = 0
                self.next_spawn = self.rng.randint(SPAWN_INTERVAL_MIN, SPAWN_INTERVAL_MAX)
                hi = min(SPAWN_COUNT_MAX + self.score // 400, 5)
                for _ in range(self.rng.randint(SPAWN_COUNT_MIN, hi)):
                    self.fruits.append(self.make_fruit())
            if slice_seg:
                for f in self.fruits:
                    if f["sliced"]:
                        continue
                    if segment_circle_intersect(slice_seg[0], slice_seg[1], (f["x"], f["y"]), f["radius"] * 0.95):
                        bomb = f["type"] == "bomb"
                        self.slice_object(f, *slice_seg)
                        if bomb:
                            self.shake, self.flash = SHAKE_FRAMES, FLASH_FRAMES
                            self.combo = 0
                            self.play("bomb")
                            self.popups.append(dict(text="BOOM!", x=f["x"], y=f["y"], life=60, color=(255, 110, 70), size=1.4))
                            self.end_game()
                            break
                        gain = 10
                        if now_ms - self.last_slice_ms < COMBO_WINDOW_MS:
                            self.combo += 1
                            bonus = (self.combo - 1) * COMBO_BONUS_BASE
                            gain += bonus
                            self.combo_pulse = 1.6
                            if self.combo >= 2:
                                self.play("combo")
                        else:
                            self.combo = 1
                        self.last_slice_ms = now_ms
                        self.score += gain
                        self.play("slice")
                        self.popups.append(dict(text=f"+{gain}", x=f["x"], y=f["y"] - 20, life=48,
                                                color=COMBO_COLOR if self.combo >= 2 else (255, 255, 255), size=1.0 + 0.1 * min(self.combo, 5)))
        elif slice_seg and self.menu_target and not self.menu_target["hit"]:
            m = self.menu_target
            if segment_circle_intersect(slice_seg[0], slice_seg[1], (m["x"], m["y"]), m["r"]):
                m["hit"] = True
                fake = dict(type=m["kind"], x=m["x"], y=m["y"], vx=0.0, vy=-2.0, angle=m["angle"], cut=m["cut"], sliced=False)
                self.slice_object(fake, *slice_seg)
                self.play("slice")
                self.pending_start = 28

        if getattr(self, "pending_start", 0) > 0:
            self.pending_start -= 1
            if self.pending_start == 0:
                self.start_game()

        # physics
        for f in self.fruits:
            f["vy"] += GRAVITY
            f["x"] += f["vx"]
            f["y"] += f["vy"]
            f["angle"] += f["spin"]
        for h in self.halves:
            h["vy"] += GRAVITY * HALF_GRAVITY_FACTOR
            h["x"] += h["vx"]
            h["y"] += h["vy"]
            h["rot"] += h["rs"]
        self.halves = [h for h in self.halves if h["y"] < HEIGHT + 160 and -240 < h["x"] < WIDTH + 240]
        for p in self.particles:
            p["vy"] += PARTICLE_GRAVITY
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]
        for s in self.splats:
            s["life"] -= 1
        self.splats = [s for s in self.splats if s["life"] > 0]
        for pp in self.popups:
            pp["life"] -= 1
            pp["y"] -= 0.8
        self.popups = [p for p in self.popups if p["life"] > 0]

        # bomb sparks
        for f in self.fruits:
            if f["type"] == "bomb" and not f["sliced"] and self.rng.random() < 0.7:
                tx, ty = self.assets.bomb_tip
                self.particles.append(dict(x=f["x"] + tx, y=f["y"] + ty, vx=self.rng.uniform(-1.4, 1.4),
                                           vy=self.rng.uniform(-2.8, -0.3), color=self.rng.choice([(255, 220, 90), (255, 140, 40)]),
                                           life=self.rng.randint(8, 16), max=16, r=self.rng.uniform(1.5, 3.0)))

        if self.state == "playing":
            keep = []
            for f in self.fruits:
                if f["sliced"]:
                    continue
                if f["y"] > DESPAWN_Y and f["vy"] > 0:
                    if f["type"] != "bomb":
                        self.lives -= 1
                        self.combo = 0
                        self.play("miss")
                        if self.lives <= 0:
                            self.end_game()
                else:
                    keep.append(f)
            self.fruits = keep
        else:
            self.fruits = [f for f in self.fruits if not f["sliced"] and f["y"] < DESPAWN_Y]

        # trail
        tnow = time.time()
        if self.blade is not None:
            self.trail.append((self.blade[0], self.blade[1], tnow))
        while self.trail and tnow - self.trail[0][2] > TRAIL_LIFE:
            self.trail.popleft()

        if self.combo_pulse > 1.0:
            self.combo_pulse = max(1.0, self.combo_pulse - 0.04)
        self.shake = max(0, self.shake - 1)
        self.flash = max(0, self.flash - 1)

        # menu target idle motion
        if self.menu_target and not self.menu_target["hit"] and self.state != "playing":
            self.menu_target["angle"] += 0.6
            self.menu_target["yy"] = self.menu_target["y"] + math.sin(self.t * 2.2) * 9

    # -------------------------------------------------------------- drawing --
    def draw_background(self, ox, oy):
        self.screen.blit(self.bg, (ox, oy))
        if self.cam_mode == 2 and self.snap.frame is not None:
            self._cam_surface()
            cam = pygame.transform.smoothscale(self.cam_surface, (WIDTH, HEIGHT))
            cam.set_alpha(120)
            self.screen.blit(cam, (ox, oy))
        for s in self.splats:
            a = int(150 * min(1.0, s["life"] / 80.0))
            img = pygame.transform.rotozoom(s["surf"], s["rot"], 1.0)
            img.set_alpha(a)
            self.screen.blit(img, img.get_rect(center=(int(s["x"]), int(s["y"]))))

    def _cam_surface(self):
        fr = self.snap.frame
        if fr is not None:
            self.cam_surface = pygame.image.frombuffer(fr.tobytes(), (fr.shape[1], fr.shape[0]), "RGB")

    def draw_fruit(self, f):
        kind = f["type"]
        sh = self.assets.shadow
        self.screen.blit(sh, sh.get_rect(center=(int(f["x"] + 12), int(f["y"] + 16))))
        img = self.assets.whole[kind]
        if kind != "bomb":
            img = pygame.transform.rotozoom(img, f["angle"], 1.0)
        self.screen.blit(img, img.get_rect(center=(int(f["x"]), int(f["y"]))))
        if kind == "bomb":
            tx, ty = self.assets.bomb_tip
            cx, cy = int(f["x"] + tx), int(f["y"] + ty)
            r = self.rng.randint(5, 9)
            pygame.draw.circle(self.screen, (255, 120, 30), (cx, cy), r)
            pygame.draw.circle(self.screen, (255, 230, 120), (cx, cy), max(2, r - 3))

    def draw_halves(self):
        for h in self.halves:
            sh = self.assets.shadow
            img = pygame.transform.rotozoom(h["surf"], h["rot"], 1.0)
            self.screen.blit(img, img.get_rect(center=(int(h["x"]), int(h["y"]))))

    def draw_particles(self):
        self.psurf.fill((0, 0, 0, 0))
        for p in self.particles:
            a = max(0, min(255, int(255 * p["life"] / p["max"]) + 40))
            col = (*p["color"], a)
            r = max(1, int(p["r"] * (0.5 + 0.5 * p["life"] / p["max"])))
            x, y = int(p["x"]), int(p["y"])
            sp = math.hypot(p["vx"], p["vy"])
            if sp > 3.5:
                pygame.draw.line(self.psurf, col, (x, y), (int(x - p["vx"] * 1.6), int(y - p["vy"] * 1.6)), max(1, r))
            pygame.draw.circle(self.psurf, col, (x, y), r)
            if r >= 3:
                pygame.draw.circle(self.psurf, (255, 255, 255, a // 2), (x - 1, y - 1), max(1, r // 3))
        self.screen.blit(self.psurf, (0, 0))

    def draw_trail(self):
        pts = list(self.trail)
        if len(pts) < 2:
            return
        pts = _chaikin(_chaikin(pts))    # round off the corners -> silky ribbon
        now = time.time()
        self.glow.fill((0, 0, 0, 0))
        n = len(pts)
        for i in range(n - 1):
            age = (now - pts[i][2]) / TRAIL_LIFE
            k = max(0.0, 1.0 - age)
            w = max(1, int(2 + 11 * k))
            a = (pts[i][0], pts[i][1])
            b = (pts[i + 1][0], pts[i + 1][1])
            pygame.draw.line(self.glow, (90, 190, 255, int(70 * k)), a, b, w * 3)
            pygame.draw.line(self.glow, (150, 220, 255, int(150 * k)), a, b, w * 2)
        self.screen.blit(self.glow, (0, 0))
        for i in range(n - 1):
            age = (now - pts[i][2]) / TRAIL_LIFE
            k = max(0.0, 1.0 - age)
            w = max(1, int(1 + 9 * k))
            col = (int(180 + 75 * k), int(225 + 30 * k), 255)
            pygame.draw.line(self.screen, col, pts[i][:2], pts[i + 1][:2], w)
        tip = pts[-1]
        pygame.draw.circle(self.screen, (255, 255, 255), (int(tip[0]), int(tip[1])), 7)
        pygame.draw.circle(self.screen, (150, 220, 255), (int(tip[0]), int(tip[1])), 4)

    def draw_pip(self):
        fr = self.snap.frame
        w, h = 256, 144
        x0, y0 = WIDTH - w - 14, HEIGHT - h - 14
        box = pygame.Surface((w, h))
        if fr is not None:
            self._cam_surface()
            box.blit(pygame.transform.smoothscale(self.cam_surface, (w, h)), (0, 0))
            lms = self.snap.lms
            if lms:
                P = [(int(px * w), int(py * h)) for px, py in lms]
                for a, b in HAND_LINKS:
                    pygame.draw.line(box, (80, 255, 170), P[a], P[b], 2)
                for p in P:
                    pygame.draw.circle(box, (255, 255, 255), p, 2)
                pygame.draw.circle(box, (255, 80, 80), P[8], 5)
            elif self.snap.tip is not None:
                pygame.draw.circle(box, (255, 80, 80), (int(self.snap.tip[0] * w), int(self.snap.tip[1] * h)), 6)
            # active-area frame
            pygame.draw.rect(box, (255, 255, 255), (int(MARGIN_X * w), int(MARGIN_Y * h), int(w * (1 - 2 * MARGIN_X)),
                                                    int(h * (1 - 2 * MARGIN_Y))), 1)
            box.set_alpha(215)
        else:
            box.fill((20, 20, 24))
            msg = self.f_tiny.render("no camera image", True, (180, 180, 190))
            box.blit(msg, msg.get_rect(center=(w // 2, h // 2)))
        self.screen.blit(box, (x0, y0))
        pygame.draw.rect(self.screen, (255, 255, 255), (x0, y0, w, h), 2, border_radius=2)
        ok = self.snap.cam_ok and self.snap.tip is not None
        pygame.draw.circle(self.screen, (80, 235, 120) if ok else (235, 80, 80), (x0 + 12, y0 + 12), 6)

    def draw_hud(self):
        self.text(self.f_big, f"{self.score}", (22, 12))
        self.text(self.f_small, f"BEST {self.best}", (24, 54), (220, 205, 180))
        for i in range(MAX_LIVES):
            ic = self.hearts[0] if i < self.lives else self.hearts[1]
            self.screen.blit(ic, (WIDTH - 44 - i * 36, 14))
        if self.combo >= 2:
            img = self.f_big.render(f"COMBO x{self.combo}", True, COMBO_COLOR)
            sc = self.combo_pulse
            img = pygame.transform.smoothscale(img, (max(1, int(img.get_width() * sc)), max(1, int(img.get_height() * sc))))
            sh = pygame.transform.smoothscale(self.f_big.render(f"COMBO x{self.combo}", True, (0, 0, 0)), img.get_size())
            r = img.get_rect(center=(WIDTH // 2, 44))
            self.screen.blit(sh, r.move(2, 2))
            self.screen.blit(img, r)
        for pp in self.popups:
            a = max(0, min(255, pp["life"] * 8))
            font = self.f_mid if pp["size"] < 1.3 else self.f_big
            self.text(font, pp["text"], (int(pp["x"]), int(pp["y"])), pp["color"], "center", alpha=a)

    def draw_status(self):
        s = self.snap
        if self.mouse_mode:
            self.text(self.f_tiny, "Mouse mode - hold left button to slice", (14, HEIGHT - 24), (220, 220, 220))
            return
        if self.debug and s.cam_ok:
            line = (f"{s.tracker}  |  cam {self.vision.cam_index if self.vision else '-'}  |  "
                    f"{s.fps:.0f} fps tracking   |   game {self.clock.get_fps():.0f} fps")
        else:
            line = s.status
        self.text(self.f_tiny, line, (14, HEIGHT - 24), (225, 225, 225))
        if self.state == "playing" and s.cam_ok and self.no_hand > 40:
            self.text(self.f_mid, "Show your index finger to the camera", (WIDTH // 2, HEIGHT // 2), (240, 240, 240), "center")
        if not s.cam_ok and self.state != "menu":
            self.text(self.f_mid, "Waiting for camera...   (you can also hold the left mouse button)",
                      (WIDTH // 2, HEIGHT // 2 + 40), (255, 210, 160), "center")

    def draw_menu_target(self, label):
        m = self.menu_target
        if not m or m["hit"]:
            return
        yy = m.get("yy", m["y"])
        sh = pygame.transform.smoothscale(self.assets.shadow, (250, 250))
        self.screen.blit(sh, sh.get_rect(center=(int(m["x"] + 14), int(yy + 20))))
        img = pygame.transform.rotozoom(m["whole"], m["angle"], 1.0)
        self.screen.blit(img, img.get_rect(center=(int(m["x"]), int(yy))))
        self.text(self.f_mid, label, (int(m["x"]), int(yy + m["r"] + 34)), (255, 236, 200), "center")

    def draw_overlay_menu(self):
        t = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        t.fill((0, 0, 0, 70))
        self.screen.blit(t, (0, 0))
        self.text(self.f_huge, "HAND NINJA", (WIDTH // 2, 118), (255, 240, 215), "center")
        self.text(self.f_mid, "Your index finger is the blade", (WIDTH // 2, 178), (255, 210, 130), "center")
        self.draw_menu_target("slice the watermelon to start")
        self.text(self.f_tiny, "SPACE start   C camera view   M mute   F fullscreen   P pause   ESC quit",
                  (WIDTH // 2, HEIGHT - 52), (230, 230, 230), "center")

    def draw_overlay_gameover(self):
        t = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        t.fill((10, 0, 0, 150))
        self.screen.blit(t, (0, 0))
        self.text(self.f_huge, "GAME OVER", (WIDTH // 2, 112), (245, 80, 80), "center")
        self.text(self.f_big, f"Score {self.score}", (WIDTH // 2, 178), HUD_COLOR, "center")
        new = self.score >= self.best and self.score > 0
        self.text(self.f_small, "NEW BEST!" if new else f"Best {self.best}", (WIDTH // 2, 214), COMBO_COLOR, "center")
        self.draw_menu_target("slice the apple to play again")

    def render(self):
        ox = oy = 0
        if self.shake > 0:
            ox, oy = self.rng.randint(-7, 7), self.rng.randint(-7, 7)
        self.draw_background(ox, oy)
        for f in self.fruits:
            if not f["sliced"]:
                self.draw_fruit(f)
        self.draw_halves()
        self.draw_particles()
        if self.state == "menu":
            self.draw_overlay_menu()
        elif self.state == "gameover":
            self.draw_overlay_gameover()
        self.draw_trail()
        if self.state != "menu":
            self.draw_hud()
        self.screen.blit(self.vignette, (0, 0))
        if self.cam_mode == 1 and not self.mouse_mode:
            self.draw_pip()
        self.draw_status()
        if self.flash > 0:
            fl = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            fl.fill((235, 40, 30, int(190 * self.flash / FLASH_FRAMES)))
            self.screen.blit(fl, (0, 0))
        if self.paused:
            self.text(self.f_huge, "PAUSED", (WIDTH // 2, HEIGHT // 2), HUD_COLOR, "center")
        pygame.display.flip()

    # ----------------------------------------------------------------- loop --
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return False
                if e.key == pygame.K_SPACE and self.state in ("menu", "gameover"):
                    self.pending_start = 0
                    self.start_game()
                elif e.key == pygame.K_c:
                    self.cam_mode = (self.cam_mode + 1) % 3
                elif e.key == pygame.K_m:
                    self.muted = not self.muted
                elif e.key == pygame.K_d:
                    self.debug = not self.debug
                elif e.key == pygame.K_p and self.state == "playing":
                    self.paused = not self.paused
                elif e.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
        return True

    def run(self):
        self.pending_start = 0
        shots = {int(x) for x in (self.args.shots or "").split(",") if x.strip().isdigit()}
        t_start = time.time()
        last_t = t_start
        running = True
        while running:
            self.clock.tick(FPS)
            now = time.time()
            self.frame_dt = max(1e-3, min(0.05, now - last_t))
            last_t = now
            self.t += self.frame_dt
            self.frame_no += 1
            running = self.handle_events()
            if not self.paused:
                self.poll_input()
                self.update()
            self.render()
            if self.frame_no in shots:
                pygame.image.save(self.screen, os.path.join(self.args.shot_dir, f"shot_{self.frame_no:04d}.png"))
            if self.args.selftest and self.frame_no >= self.args.selftest:
                break
        if self.args.selftest:
            el = max(1e-6, time.time() - t_start)
            print(f"[selftest] frames={self.frame_no} avg_fps={self.frame_no / el:.1f} "
                  f"score={self.score} state={self.state}")
        if self.vision:
            self.vision.stop_flag = True
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description="Hand Ninja - slice fruit with your hand")
    ap.add_argument("--camera", type=int, default=None, help="force camera index (default: auto-detect)")
    ap.add_argument("--list-cameras", action="store_true", help="list working cameras and exit")
    ap.add_argument("--no-camera", action="store_true", help="don't use the camera")
    ap.add_argument("--mouse", action="store_true", help="mouse blade (move the mouse to slice)")
    ap.add_argument("--model", default=None, help="path to hand_landmarker.task")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--autoplay", action="store_true", help="demo mode: the game plays itself")
    ap.add_argument("--selftest", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--shots", default="", help=argparse.SUPPRESS)
    ap.add_argument("--shot-dir", default=".", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.list_cameras:
        found = False
        for i in candidate_cameras():
            cap, w, h = open_camera(i)
            if cap is not None:
                print(f"camera {i}: {w}x{h}  OK")
                cap.release()
                found = True
        if not found:
            print("No working cameras found.")
        return

    Game(args).run()


if __name__ == "__main__":
    main()
