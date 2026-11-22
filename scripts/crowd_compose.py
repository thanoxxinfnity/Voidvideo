#!/usr/bin/env python3
"""
Layered crowd compositor: every person is generated alone (one prompt, one character card, solid green background), then
placed into the scene here. Faces cannot mix because no two people ever share a generation.

    python scripts/crowd_compose.py layout.json out.mp4

layout.json:
{
  "size": [720, 480], "fps": 24, "frames": 49,
  "background": "bg.mp4"            # a generated empty-street clip, an image, or "#202030" for a flat colour
  "layers": [
    {"video": "hero1.mp4", "x": 0.30, "y": 0.95, "height": 0.62,   # x = centre, y = feet (0..1), height = fraction of frame
     "key": "green", "flip": false, "offset": 0, "tint": [1,1,1], "shadow": true, "dx": 0.0}   # dx: walk, frame-widths per second
  ]
}
Layers are drawn far to near (smaller y first). Foreground = anything that is not the key colour (estimated from the clip
border), soft edge + green despill. Works best on flat-colour anime frames.
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image


def read_frames(path, w=None, h=None, n=None):
    """All frames as uint8 (T,H,W,3). Size is probed when not given."""
    if w is None:
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                             "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()
        w, h = map(int, pr.split(","))
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)] + (["-frames:v", str(n)] if n else []) + \
          ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def key_alpha(frames, key="green"):
    """alpha (T,H,W) float in [0,1] and despilled rgb (T,H,W,3)."""
    ch = {"green": 1, "blue": 2}[key]
    other = [i for i in range(3) if i != ch]
    f = frames.astype(np.float32) / 255.0
    score = f[..., ch] - f[..., other].max(-1)                     # how much more key-coloured than anything else
    border = np.concatenate([score[:, :6].ravel(), score[:, -6:].ravel(), score[:, :, :6].ravel(), score[:, :, -6:].ravel()])
    k = float(np.median(border))
    k = max(k, 0.25)                                               # a clip with no clean green border still keys sanely
    alpha = np.clip(1.0 - (score - 0.30 * k) / (0.35 * k), 0.0, 1.0)
    rgb = f.copy()
    rgb[..., ch] = np.minimum(rgb[..., ch], f[..., other].max(-1))  # despill
    return alpha, rgb


def make_sprite(path, key, height_px, flip, frames_n):
    fr = read_frames(path, n=frames_n)
    alpha, rgb = key_alpha(fr, key)
    mask = alpha.max(0) > 0.5
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise SystemExit(f"{path}: nothing left after keying (is the background really {key}?)")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sprites = []
    scale = height_px / (y1 - y0)
    size = (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale)))
    for t in range(len(fr)):
        rgba = np.dstack([rgb[t, y0:y1, x0:x1], alpha[t, y0:y1, x0:x1, None]])
        im = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").resize(size, Image.LANCZOS)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        sprites.append(np.asarray(im).astype(np.float32) / 255.0)
    return sprites


def background(spec, w, h, n):
    if isinstance(spec, str) and spec.startswith("#"):
        c = np.array([int(spec[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
        return np.broadcast_to(c, (n, h, w, 3)).copy()
    if str(spec).lower().endswith((".png", ".jpg", ".jpeg")):
        im = np.asarray(Image.open(spec).convert("RGB").resize((w, h), Image.LANCZOS))
        return np.broadcast_to(im, (n, h, w, 3)).copy()
    fr = [np.asarray(Image.fromarray(f).resize((w, h), Image.LANCZOS)) for f in read_frames(spec, n=n)]
    while len(fr) < n:
        fr.append(fr[-1])
    return np.stack(fr[:n])


def composite(layout):
    w, h = layout.get("size", [720, 480])
    n, fps = layout.get("frames", 49), layout.get("fps", 24)
    canvas = background(layout["background"], w, h, n).astype(np.float32) / 255.0
    layers = sorted(layout["layers"], key=lambda L: L["y"])        # far (small y) first
    for L in layers:
        spr = make_sprite(L["video"], L.get("key", "green"), L.get("height", 0.5) * h, L.get("flip", False), n)
        off, tint = L.get("offset", 0), np.array(L.get("tint", [1, 1, 1]), np.float32)
        for t in range(n):
            s = spr[(t + off) % len(spr)]
            sh, sw = s.shape[:2]
            cx = (L["x"] + L.get("dx", 0.0) * t / fps) * w
            x0, y0 = int(round(cx - sw / 2)), int(round(L["y"] * h - sh))
            if L.get("shadow", True):                              # soft ellipse under the feet
                yy, xx = np.ogrid[:h, :w]
                e = ((xx - cx) / (sw * 0.45)) ** 2 + ((yy - L["y"] * h) / (sh * 0.05 + 2)) ** 2
                canvas[t] *= (1 - 0.28 * np.clip(1 - e, 0, 1))[..., None]
            cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + sw, w), min(y0 + sh, h)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            piece = s[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            a = piece[..., 3:4]
            canvas[t, cy0:cy1, cx0:cx1] = piece[..., :3] * tint * a + canvas[t, cy0:cy1, cx0:cx1] * (1 - a)
    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8), fps


def write(frames, fps, out):
    t, h, w, _ = frames.shape
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    p.communicate(frames.tobytes())


def main():
    layout_path, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    layout = json.load(open(layout_path))
    os.chdir(os.path.dirname(layout_path))                          # clip paths are relative to the layout file
    frames, fps = composite(layout)
    write(frames, fps, out)
    print(f"wrote {sys.argv[2]}: {len(frames)} frames, {len(layout['layers'])} layers")


if __name__ == "__main__":
    main()
