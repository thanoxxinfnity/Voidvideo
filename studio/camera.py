"""
Camera tools on a finished clip (frames = uint8 array (T,H,W,3)).

2D path   : spline of (x, y, zoom, rotation) keyframes -> pan, dolly-in/out (zoom), roll, handheld, FPV-ish sweeps.
Parallax  : with a depth map (e.g. Depth-Anything-V2-Small, runs on CPU) near pixels move more than far ones, which
            gives a believable small orbit / dolly / slide (2.5D). Large moves need real 3D and will show holes.
Shake     : decaying random offsets on chosen frames (impact hits).
DOF       : depth-based blur with a focus distance that can be animated (focus pull) - bokeh-style, layered gaussian.
"""
import numpy as np
from PIL import Image, ImageFilter


def spline(keys, n):
    """keys: [{"t": 0..1, "x": frac of width, "y": frac of height, "zoom": 1.0, "rot": degrees}, ...] -> dict of arrays (n,)."""
    keys = sorted(keys, key=lambda k: k["t"])
    ts = np.array([k["t"] for k in keys], np.float64)
    out = {}
    tt = np.linspace(0, 1, n)
    for name, default in (("x", 0.0), ("y", 0.0), ("zoom", 1.0), ("rot", 0.0)):
        v = np.array([k.get(name, default) for k in keys], np.float64)
        if len(keys) < 3:
            out[name] = np.interp(tt, ts, v)
            continue
        # Catmull-Rom through the keys (clamped ends)
        p = np.concatenate([[v[0]], v, [v[-1]]])
        tp = np.concatenate([[ts[0]], ts, [ts[-1]]])
        res = np.empty(n)
        for i, t in enumerate(tt):
            j = min(np.searchsorted(ts, t, side="right") - 1, len(ts) - 2)
            j = max(j, 0)
            u = (t - ts[j]) / max(ts[j + 1] - ts[j], 1e-9)
            p0, p1, p2, p3 = p[j], p[j + 1], p[j + 2], p[j + 3]
            res[i] = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
        out[name] = res
    return out


def apply_path(frames, path):
    """Warp each frame by the path (zoom around the centre, rotate, translate). Zoom is raised automatically so no
    black borders show (the crop-in is the cost of the move)."""
    t, h, w, _ = frames.shape
    out = np.empty_like(frames)
    for i in range(t):
        z, r = float(path["zoom"][i]), np.deg2rad(float(path["rot"][i]))
        tx, ty = float(path["x"][i]) * w, float(path["y"][i]) * h
        need = 1 + 2 * max(abs(tx) / w, abs(ty) / h) + abs(np.sin(r)) * max(w / h, h / w)
        z = max(z, need)
        c, s = np.cos(r) / z, np.sin(r) / z                              # output -> input mapping
        cx, cy = w / 2, h / 2
        a, b = c, s
        d, e = -s, c
        cc = cx - a * (cx + tx) - b * (cy + ty)
        f = cy - d * (cx + tx) - e * (cy + ty)
        out[i] = np.asarray(Image.fromarray(frames[i]).transform((w, h), Image.AFFINE, (a, b, cc, d, e, f), Image.BICUBIC))
    return out


def shake(frames, hits, strength=0.025, length=8, seed=0):
    """hits: frame indices of impacts. Offsets decay over `length` frames (fraction of frame size)."""
    t = frames.shape[0]
    rnd = np.random.default_rng(seed)
    x, y, rot = np.zeros(t), np.zeros(t), np.zeros(t)
    for hit in hits:
        for k in range(length):
            if 0 <= hit + k < t:
                decay = (1 - k / length) ** 2
                x[hit + k] += rnd.uniform(-1, 1) * strength * decay
                y[hit + k] += rnd.uniform(-1, 1) * strength * decay
                rot[hit + k] += rnd.uniform(-1, 1) * 1.5 * decay
    return apply_path(frames, {"x": x, "y": y, "zoom": np.ones(t), "rot": rot})


def parallax(frames, depth, path_x, path_y=None, strength=0.04):
    """2.5D camera slide. depth: (H,W) or (T,H,W) float, 1 = near, 0 = far. path_x/path_y: (T,) in [-1, 1]. Near pixels shift
    `strength` * path of the frame width, far pixels stay (a sideways dolly / small orbit)."""
    t, h, w, _ = frames.shape
    path_y = np.zeros(t) if path_y is None else path_y
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.empty_like(frames)
    for i in range(t):
        d = depth[i] if depth.ndim == 3 else depth
        d = d - d.mean()                                                  # keep the focus plane still
        sx = xx - d * path_x[i] * strength * w
        sy = yy - d * path_y[i] * strength * h
        sx, sy = np.clip(np.rint(sx).astype(int), 0, w - 1), np.clip(np.rint(sy).astype(int), 0, h - 1)
        out[i] = frames[i][sy, sx]
    return out


def depth_of_field(frames, depth, focus, max_blur=9, levels=4):
    """focus: float or (T,) depth value in focus (0..1). Blur grows with |depth - focus|; layered gaussian blends."""
    t, h, w, _ = frames.shape
    focus = np.broadcast_to(np.asarray(focus, np.float32), (t,))
    out = np.empty_like(frames)
    for i in range(t):
        d = depth[i] if depth.ndim == 3 else depth
        coc = np.clip(np.abs(d - focus[i]) * 2.0, 0, 1)                   # circle of confusion 0..1
        im = Image.fromarray(frames[i])
        stack = [frames[i].astype(np.float32)] + [np.asarray(im.filter(ImageFilter.GaussianBlur(max_blur * (k + 1) / levels)), np.float32)
                                                   for k in range(levels)]
        pos = coc * levels
        lo = np.clip(np.floor(pos).astype(int), 0, levels - 1)
        frac = (pos - lo)[..., None]
        a = np.take_along_axis(np.stack(stack), lo[None, ..., None].repeat(3, -1), 0)[0]
        b = np.take_along_axis(np.stack(stack), (lo + 1)[None, ..., None].repeat(3, -1), 0)[0]
        out[i] = np.clip(a * (1 - frac) + b * frac, 0, 255).astype(np.uint8)
    return out


PRESETS = {
    "static": [{"t": 0}, {"t": 1}],
    "dolly_in": [{"t": 0, "zoom": 1.0}, {"t": 1, "zoom": 1.25}],
    "dolly_out": [{"t": 0, "zoom": 1.25}, {"t": 1, "zoom": 1.0}],
    "pan_right": [{"t": 0, "x": -0.06, "zoom": 1.15}, {"t": 1, "x": 0.06, "zoom": 1.15}],
    "pan_left": [{"t": 0, "x": 0.06, "zoom": 1.15}, {"t": 1, "x": -0.06, "zoom": 1.15}],
    "tilt_up": [{"t": 0, "y": 0.06, "zoom": 1.15}, {"t": 1, "y": -0.06, "zoom": 1.15}],
    "roll": [{"t": 0, "rot": -3, "zoom": 1.1}, {"t": 1, "rot": 3, "zoom": 1.1}],
    "fpv_sweep": [{"t": 0, "x": -0.05, "y": 0.03, "zoom": 1.05, "rot": -4}, {"t": 0.35, "x": 0.04, "y": -0.04, "zoom": 1.2, "rot": 5},
                  {"t": 0.7, "x": -0.03, "y": 0.02, "zoom": 1.3, "rot": -3}, {"t": 1, "x": 0.05, "y": -0.02, "zoom": 1.4, "rot": 2}],
}
