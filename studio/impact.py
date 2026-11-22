"""
Anime impact frames and speed lines, applied to a finished clip (frames = uint8 (T,H,W,3)).

impact_frame : the 1-2 frame high-contrast black/white "hit" (threshold + optional invert + radial lines).
insert_impact: replaces frames at the given indices (hit-stop style, clip length unchanged).
speed_lines  : animated radial or horizontal lines over a frame range, redrawn every 2 frames (animation on twos).
"""
import numpy as np
from PIL import Image, ImageDraw


def _lines(h, w, rnd, radial=True, n=110, inner=0.22):
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    if radial:
        cx, cy = w / 2 + rnd.uniform(-0.06, 0.06) * w, h / 2 + rnd.uniform(-0.06, 0.06) * h
        r_out = np.hypot(w, h)
        for _ in range(n):
            a = rnd.uniform(0, 2 * np.pi)
            wid = rnd.uniform(0.004, 0.022)
            r_in = inner * min(w, h) * rnd.uniform(0.8, 1.6)
            p0 = (cx + np.cos(a) * r_in, cy + np.sin(a) * r_in)
            p1 = (cx + np.cos(a - wid) * r_out, cy + np.sin(a - wid) * r_out)
            p2 = (cx + np.cos(a + wid) * r_out, cy + np.sin(a + wid) * r_out)
            d.polygon([p0, p1, p2], fill=255)
    else:
        for _ in range(n // 2):
            y = rnd.uniform(0, h)
            th = rnd.uniform(1, 5)
            x0 = rnd.uniform(-0.2, 0.6) * w
            d.rectangle([x0, y, x0 + rnd.uniform(0.3, 0.9) * w, y + th], fill=255)
    return np.asarray(im, np.float32) / 255.0


def impact_frame(frame, invert=True, lines=True, seed=0, level=0.0):
    g = np.asarray(Image.fromarray(frame).convert("L"), np.float32)
    thr = g.mean() + level * 255
    bw = (g > thr).astype(np.float32)
    if invert:
        bw = 1 - bw
    if lines:
        m = _lines(*frame.shape[:2], np.random.default_rng(seed))
        line_col = 0.0 if bw.mean() > 0.5 else 1.0                          # lines contrast with the dominant tone
        bw = bw * (1 - m) + line_col * m
    return np.repeat((bw * 255).astype(np.uint8)[..., None], 3, -1)


def insert_impact(frames, at, hold=2, invert=True, lines=True):
    out = frames.copy()
    for k, i in enumerate(at):
        for j in range(hold):
            if 0 <= i + j < len(out):
                out[i + j] = impact_frame(frames[min(i + j, len(frames) - 1)], invert=(invert if j % 2 == 0 else not invert),
                                          lines=lines, seed=k * 7 + j)
    return out


def speed_lines(frames, start, end, radial=True, color=255, opacity=0.85, seed=1, on=2):
    out = frames.copy()
    rnd = np.random.default_rng(seed)
    m = None
    for i in range(max(start, 0), min(end, len(out))):
        if m is None or (i - start) % on == 0:
            m = _lines(*frames.shape[1:3], rnd, radial=radial)[..., None] * opacity
        out[i] = np.clip(out[i] * (1 - m) + color * m, 0, 255).astype(np.uint8)
    return out
