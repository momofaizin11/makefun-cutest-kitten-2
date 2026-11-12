#!/usr/bin/env python3
"""Offline studio for the make.fun entry: a tiny kitten in a teacup.

Storybook / children's-book look: soft warm palette, clean rounded shapes,
gentle dark outlines.  Usage:  python3 tools/render.py [out.png]
"""
import sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S, SS = 1024, 3
D = S * SS
INK = np.array([86, 60, 49], np.float32)
OUT = sys.argv[1] if len(sys.argv) > 1 else "entry.png"
YY, XX = np.mgrid[0:S, 0:S].astype(np.float32)
CX = 505.0                                     # shared axis of cup + kitten

# ---------------------------------------------------------------- mask plumbing
def nm():
    return Image.new("L", (D, D), 0)

def fin(m):
    return np.asarray(m.resize((S, S), Image.Resampling.LANCZOS), np.float32) / 255.0

def _box(a, r, ax):
    if r < 1:
        return a
    pad = [(0, 0), (0, 0)]
    pad[ax] = (r, r)
    p = np.pad(a, pad, mode="edge")
    cs = np.cumsum(p, axis=ax, dtype=np.float32)
    z = np.zeros_like(cs.take([0], axis=ax))
    cs = np.concatenate([z, cs], axis=ax)
    n = a.shape[ax]
    hi = cs.take(range(2 * r + 1, n + 2 * r + 1), axis=ax)
    lo = cs.take(range(0, n), axis=ax)
    return (hi - lo) / float(2 * r + 1)

def blur(a, r):
    """Separable box blur (x3) — a gaussian stand-in that works on float masks."""
    r = max(1, int(round(r * 0.75)))
    b = a.astype(np.float32)
    for _ in range(3):
        b = _box(b, r, 0)
        b = _box(b, r, 1)
    return b

def edge_of(a, r=14.0, thr=0.30):
    """Antialiased silhouette grown outward by ~r/2 px (line-art outline)."""
    return np.clip((blur(a, r) - thr) / 0.12, 0, 1)

def poly(m, P):
    ImageDraw.Draw(m).polygon([(float(x) * SS, float(y) * SS) for x, y in P], fill=255)

def stroke(m, P, w, w2=None, ws=None, wob=0.07, ph=0.3):
    """Round-cap stroke with width taper; segment-wise resampling so it never dots."""
    P = np.asarray(P, float)
    n = len(P)
    d = ImageDraw.Draw(m)

    def wid(i):
        t = i / max(1, n - 1)
        if ws is not None:
            ww = float(ws[min(i, len(ws) - 1)])
        else:
            ww = w + ((w if w2 is None else w2) - w) * t
        return ww * (1.0 + wob * math.sin(i * 0.4 + ph))

    for i in range(n - 1):
        x0, y0 = P[i] * SS
        x1, y1 = P[i + 1] * SS
        seg = max(1e-6, math.hypot(x1 - x0, y1 - y0))
        w0, w1 = wid(i), wid(i + 1)
        k = max(1, int(seg / max(0.7, 0.30 * min(w0, w1) * SS)))
        for j in range(k + 1):
            t = j / k
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            r = max(0.6, (w0 + (w1 - w0) * t) * 0.5 * SS)
            d.ellipse((x - r, y - r, x + r, y + r), fill=255)

# ---------------------------------------------------------------- geometry helpers
def catmull(P, samples=14, closed=False):
    P = np.asarray(P, float)
    Q = np.vstack([P[-1], P, P[0], P[1]]) if closed else np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(len(Q) - 3):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        for t in np.linspace(0, 1, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                              (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(Q[-2])
    return np.array(out)

def quad(p0, p1, p2, n=18):
    p0, p1, p2 = np.asarray(p0, float), np.asarray(p1, float), np.asarray(p2, float)
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

def ellm(cx, cy, rx, ry, rot=0.0, n=96):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(th), ry * np.sin(th)
    ca, sa = math.cos(rot), math.sin(rot)
    m = nm()
    poly(m, np.stack([cx + x * ca - y * sa, cy + x * sa + y * ca], 1))
    return fin(m)

def smask(P, w, w2=None, ws=None, closed=False):
    m = nm()
    P = np.asarray(P, float)
    if closed:
        P = np.vstack([P, P[:1]])
    stroke(m, P, w, w2, ws)
    return fin(m)

def soft(cx, cy, rx, ry, r, rot=0.0):
    return blur(ellm(cx, cy, rx, ry, rot), r)

def vgrad(stops):
    ys = np.linspace(0.0, 1.0, S, np.float32)
    out = np.zeros((S, S, 3), np.float32)
    for c in range(3):
        out[..., c] = np.interp(ys, [s[0] for s in stops],
                                [float(s[1][c]) for s in stops])[:, None]
    return out

def leaf_hw(t, H, p=3.0, q=0.55, serr=0.055):
    """Half-width profile of the leaf blade: broad middle, sharp tips."""
    w = H * np.clip(1 - np.abs(t) ** p, 0.0, 1.0) ** q
    return w * (1.0 + serr * np.sin(t * 4.5 * np.pi))

def leaf(cx, cy, L, H, ang, p=3.0, q=0.55):
    """Bold pointed leaf contour (sharp tips, fat middle)."""
    t = np.linspace(-1, 1, 96)
    yy = leaf_hw(t, H, p, q)
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    up = np.stack([cx + t * L * ca - yy * sa, cy + t * L * sa + yy * ca], 1)
    dn = np.stack([cx + t[::-1] * L * ca + yy[::-1] * sa,
                   cy + t[::-1] * L * sa - yy[::-1] * ca], 1)
    return np.vstack([up, dn])

def superell(cx, cy, a, b, n=5.0, rot=0.0, N=110):
    th = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = a * np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2.0 / n)
    y = b * np.sign(np.sin(th)) * np.abs(np.sin(th)) ** (2.0 / n)
    ca, sa = math.cos(rot), math.sin(rot)
    return np.stack([cx + x * ca - y * sa, cy + x * sa + y * ca], 1)

class Img:
    def __init__(self, base):
        self.c = base

    def over(self, a, col):
        a = np.clip(a, 0, 1)[..., None]
        self.c = self.c * (1 - a) + np.asarray(col, np.float32) * a

    def shade(self, m, lx, ly, r0, r1, amt=0.3, col=(210, 170, 130), power=1.5):
        g = np.clip((np.hypot(XX - lx, YY - ly) - r0) / (r1 - r0), 0, 1) ** power
        self.over(m * g * amt, np.asarray(col, np.float32))

    def shape(self, m, col, r=6.0, ink=INK, ink_a=0.93):
        if r:
            self.over(edge_of(m, r) * ink_a, ink)
        self.over(m, col)
        return m

# ---------------------------------------------------------------- the drawing
def draw():
    img = Img(vgrad([(0.0, (236, 201, 168)), (0.42, (244, 217, 189)), (1.0, (226, 183, 150))]))
    d = np.hypot((XX - 512) / 445.0, (YY - 706) / 342.0)
    img.over(np.clip((1 - d) / 0.30, 0, 1) ** 0.9 * 0.92, np.array([255, 247, 236], np.float32))
    for x, y, rr, aa in [(150, 190, 5, .5), (212, 302, 3.5, .45), (880, 172, 4, .5),
                         (830, 322, 3, .4), (120, 470, 4, .4), (905, 520, 5, .45),
                         (176, 700, 3.5, .3), (934, 762, 4, .3), (242, 118, 3, .35),
                         (760, 108, 3.5, .4), (78, 862, 4, .3), (966, 930, 3, .28)]:
        img.over(soft(x, y, rr, rr, 1.2) * aa, np.array([255, 238, 218], np.float32))

    # --- saucer -------------------------------------------------------------
    img.over(soft(512, 884, 352, 46, 20) * 0.30, np.array([176, 134, 104], np.float32))
    sa = ellm(512, 824, 348, 92)
    img.shape(sa, np.array([255, 251, 243], np.float32), r=14)
    img.shade(sa, 372, 690, 170, 520, 0.22, (204, 166, 132))
    ring = np.clip(ellm(512, 810, 296, 72) - ellm(512, 810, 284, 66), 0, 1)
    img.over(ring * 0.85, np.array([236, 205, 175], np.float32))
    mint = np.clip(ellm(512, 806, 252, 60) - ellm(512, 806, 245, 56), 0, 1)
    img.over(mint * 0.9, np.array([150, 206, 184], np.float32))

    # --- cup body + handle (one silhouette: no internal seams) --------------
    img.over(soft(512, 786, 196, 42, 15) * 0.32, np.array([168, 126, 96], np.float32))
    m = nm()
    top_y, bot_y, rx_t, rx_b = 548.0, 748.0, 214.0, 152.0
    right, left = [], []
    for t in np.linspace(0, 1, 40):
        y = top_y + (bot_y - top_y) * t
        rx = rx_t - (rx_t - rx_b) * t ** 0.85
        right.append((CX + rx, y))
        left.append((CX - rx, y))
    bottom = [(CX + 152 * np.cos(a), bot_y + 34 * np.sin(a)) for a in np.linspace(0, np.pi, 30)]
    poly(m, np.array(right + bottom + left[::-1]))
    stroke(m, catmull([(700, 600), (784, 608), (830, 664), (812, 722), (718, 748)], 14), 42)
    cup = fin(m)
    img.shape(cup, vgrad([(0.0, (253, 243, 228)), (0.55, (247, 231, 209)), (1.0, (233, 207, 180))]), r=14)
    img.shade(cup, 392, 520, 150, 520, 0.34, (192, 148, 112))
    light = np.clip(1 - np.hypot(XX - 320, YY - 610) / 250, 0, 1) ** 2 * 0.22
    img.over(cup * light, np.array([255, 255, 255], np.float32))
    band = np.clip(ellm(CX, 688, 178, 42) - ellm(CX, 688, 170, 35), 0, 1) * cup
    img.over(band * 0.8, np.array([160, 214, 192], np.float32))

    # --- rim + tea ----------------------------------------------------------
    tea_o = ellm(CX, 548, 196, 48)
    img.shape(ellm(CX, 548, 218, 62), np.array([245, 192, 165], np.float32), r=13)
    img.over(edge_of(tea_o, 4) * 0.9, np.array([186, 128, 92], np.float32))
    img.over(tea_o, vgrad([(0.42, (232, 186, 142)), (0.60, (200, 141, 100))]))
    img.over(tea_o * np.clip((545 - YY) / 55, 0, 1) * 0.4, np.array([152, 98, 66], np.float32))

    # --- steam: three separated curls, drawn behind the kitten --------------
    for bx, amp, top in ((340, 24, 168), (CX, 26, 140), (672, 24, 168)):
        P = catmull([(bx, 556), (bx - amp, 486), (bx + amp * 0.85, 420),
                     (bx - amp * 0.95, 352), (bx + amp * 0.55, top + 38),
                     (bx - amp * 1.45, top)], 11)
        img.over(blur(smask(P, 17.0, 9.0), 9) * 0.35, np.array([255, 253, 248], np.float32))
        img.over(smask(P, 12.0, 6.5) * 0.95, np.array([255, 255, 255], np.float32))

    # --- kitten (head + ears + chest as one silhouette) ---------------------
    km = nm()
    def ear(A, T, B, bow=1.0, s=1.0):
        A, T, B = np.asarray(A, float), np.asarray(T, float), np.asarray(B, float)
        c = (A + B + T) / 3.0
        A, T, B = c + (A - c) * s, c + (T - c) * s, c + (B - c) * s
        m1 = A + 0.45 * (T - A) + np.array([-9.0 * bow, -3.0]) * s
        m2 = T + 0.55 * (B - T) + np.array([12.0 * bow, -6.0]) * s
        return np.vstack([quad(A, m1, T, 16), quad(T, m2, B, 16)])
    le = ear((362, 376), (328, 230), (454, 288), 1.0)
    re = ear((648, 376), (682, 230), (556, 288), -1.0)
    poly(km, le); poly(km, re)
    poly(km, catmull([(CX - 120, 566), (CX - 100, 496), (CX, 474), (CX + 100, 496),
                      (CX + 120, 566), (CX + 58, 600), (CX - 58, 600)], 12, closed=True))
    th = np.linspace(0, 2 * np.pi, 150, endpoint=False)
    r = np.ones_like(th)
    for ta, dep, wid in [(24, .055, 11), (156, .055, 11), (204, .05, 10), (336, .05, 10),
                         (80, .04, 8), (100, .04, 8)]:
        d = (np.degrees(th) - ta + 180) % 360 - 180
        r += dep * np.exp(-(d / wid) ** 2)
    poly(km, np.stack([CX + 152 * r * np.cos(th), 417 + 138 * r * np.sin(th)], 1))
    K = fin(km)
    img.shape(K, vgrad([(0.0, (255, 251, 242)), (0.85, (252, 240, 222))]), r=14)
    img.shade(K, 396, 300, 110, 500, 0.26, (227, 199, 167))
    for pts in (ear((376, 366), (344, 268), (440, 300), 1.0, 0.62),
                ear((634, 366), (666, 268), (570, 300), -1.0, 0.62)):
        mm = nm(); poly(mm, pts)
        img.shape(fin(mm), np.array([246, 189, 185], np.float32), r=3.0, ink=INK, ink_a=0.35)

    # fur strokes (matte tufts, no gloss)
    for p in (catmull([(CX - 34, 306), (CX - 22, 324), (CX - 10, 306)], 8),
              catmull([(CX + 16, 306), (CX + 28, 324), (CX + 40, 308)], 8),
              catmull([(CX - 108, 452), (CX - 92, 466), (CX - 78, 468)], 8),
              catmull([(CX + 82, 470), (CX + 96, 464), (CX + 110, 450)], 8)):
        img.over(smask(p, 5.0, 3.0) * 0.42, np.array([216, 186, 152], np.float32))
    # face
    for sx in (-1, 1):
        ex = CX + sx * 52
        p = catmull([(ex - 19, 436), (ex - 7, 421), (ex + 7, 421), (ex + 19, 436)], 9)
        h = len(p) // 2
        ws = np.concatenate([np.linspace(5.0, 9.0, h), np.linspace(9.0, 5.0, len(p) - h)])
        img.over(smask(p, 0, ws=ws), INK)
    img.over(soft(CX - 96, 470, 30, 17, 7) * 0.5, np.array([247, 163, 158], np.float32))
    img.over(soft(CX + 96, 470, 30, 17, 7) * 0.5, np.array([247, 163, 158], np.float32))
    mm = nm(); poly(mm, catmull([(CX - 12, 446), (CX + 12, 446), (CX, 464)], 11, closed=True))
    img.shape(fin(mm), np.array([242, 158, 152], np.float32), r=4.5, ink=INK, ink_a=0.9)
    for p in (catmull([(CX, 465), (CX - 8, 477), (CX - 20, 473)], 8),
              catmull([(CX, 465), (CX + 8, 477), (CX + 20, 473)], 8)):
        img.over(smask(p, 5.6, 3.4) * 0.95, INK)
    for sx in (-1, 1):
        for dy, ln, w0 in ((2, 74, 4.4), (16, 80, 4.0)):
            p = catmull([(CX + sx * 58, 456 + dy), (CX + sx * 96, 450 + dy - 5),
                         (CX + sx * (58 + ln), 442 + dy - 8)], 8)
            img.over(smask(p, w0, 2.2) * 0.85, np.array([190, 154, 122], np.float32))

    # --- cup front rim over the kitten --------------------------------------
    thf = np.linspace(0, np.pi, 60)
    om = [(CX + 218 * np.cos(a), 548 + 62 * np.sin(a)) for a in thf]
    im = [(CX + 196 * np.cos(a), 548 + 48 * np.sin(a)) for a in thf[::-1]]
    m = nm(); poly(m, np.array(om + im))
    img.shape(fin(m), vgrad([(0.50, (247, 197, 173)), (0.62, (240, 178, 148))]), r=13)

    # --- paws on the rim (3 lobes + toe creases) ----------------------------
    PAW = [(-46, 4), (-36, -14), (-28, -26), (-21, -20), (-14, -17), (-7, -21),
           (0, -27), (7, -21), (14, -17), (21, -20), (28, -26), (36, -14),
           (46, 4), (40, 28), (0, 40), (-40, 28)]
    for sx in (-1, 1):
        p = catmull([(CX + sx * 60 + u * sx, 556 + v) for u, v in PAW], 10, closed=True)
        m = nm(); poly(m, p)
        f = fin(m)
        img.over(edge_of(f, 11) * 0.95, INK)
        img.over(f, np.array([255, 250, 241], np.float32))
        img.shade(f, CX + sx * 46, 498, 26, 190, 0.26, (224, 194, 160))
        img.over(soft(CX + sx * 60, 602, 52, 14, 7) * 0.28, np.array([178, 138, 106], np.float32))
        for k in (-1, 1):
            q = catmull([(CX + sx * (60 + k * 14), 538), (CX + sx * (60 + k * 15), 552),
                         (CX + sx * (60 + k * 16), 566)], 8)
            img.over(smask(q, 5.6, 3.4) * 0.5, INK)

    # --- teabag tag: rounded rectangle + string + hole ----------------------
    m = nm(); poly(m, superell(296, 712, 44, 52, 4.6, math.radians(-8)))
    f = fin(m)
    img.shape(f, np.array([255, 252, 245], np.float32), r=10)
    img.shade(f, 236, 648, 20, 170, 0.18, (206, 172, 138))
    img.over(smask(catmull([(332, 574), (322, 616), (306, 652)], 10), 4.6, 3.8) * 0.95,
             np.array([176, 140, 108], np.float32))
    img.over(ellm(296, 668, 6.5, 6.5) * 0.75, np.array([140, 110, 84], np.float32))
    m2 = nm(); poly(m2, leaf(296, 724, 30, 14, -18))
    img.over(fin(m2) * 0.85, np.array([124, 192, 166], np.float32))

    # --- bold pointed mint leaf on the saucer -------------------------------
    m = nm(); poly(m, leaf(622, 868, 50, 32, 22))
    img.shape(fin(m), np.array([174, 224, 202], np.float32), r=9)
    LX, LY, LL, LH, LA = 704.0, 838.0, 98.0, 55.0, -24.0
    m = nm(); poly(m, leaf(LX, LY, LL, LH, LA)); f = fin(m)
    img.shape(f, vgrad([(0.78, (158, 220, 195)), (0.88, (122, 196, 168))]), r=11)
    caL, saL = math.cos(math.radians(LA)), math.sin(math.radians(LA))
    def lpt(u, v):
        return (LX + u * LL * caL - v * saL, LY + u * LL * saL + v * caL)
    img.over(smask(catmull([lpt(-0.86, 0), lpt(0, 0), lpt(0.88, 0)], 12), 4.6, 2.6) * 0.85,
             np.array([92, 172, 146], np.float32))
    for tv in (-0.42, 0.02, 0.46):
        for sgn in (1, -1):
            u1 = min(tv + 0.3, 0.97)
            v1 = sgn * leaf_hw(u1, LH) * 0.52
            img.over(smask(catmull([lpt(tv, 0), lpt((tv + u1) / 2, v1 * 0.45), lpt(u1, v1)], 6),
                           3.2, 1.8) * 0.55, np.array([92, 172, 146], np.float32))

    # --- vignette -----------------------------------------------------------
    v = np.clip((np.hypot(XX - 512, YY - 500) - 360) / 430, 0, 1) ** 1.4 * 0.22
    img.over(v, np.array([132, 92, 66], np.float32))
    return img.c

def main():
    a = draw()
    rgb = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")
    rgb.save(OUT, optimize=True)
    rgb.resize((300, 300), Image.Resampling.LANCZOS).save(OUT.replace(".png", "_300.png"), optimize=True)
    print("wrote", OUT, rgb.size, rgb.mode)

if __name__ == "__main__":
    main()
