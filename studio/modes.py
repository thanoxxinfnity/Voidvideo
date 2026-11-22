"""
Input builders for the VACE modes (pure numpy/PIL, CPU-testable). VACE convention (diffusers WanVACEPipeline): `video` holds the
frames that are kept plus gray (127) placeholders; `mask` is white (255) where new content must be generated, black (0) where
the given frames are kept. Everything here returns (video: list[PIL], mask: list[PIL]).

keyframes : i2v / start+middle+end frames -> smooth animation between them
extend    : continue a clip (keeps the last `keep` frames, generates the rest) -> chain shots into 30-60 s
inpaint   : regenerate a masked region (remove / replace an object or a character) while everything else stays
outpaint  : place a clip (e.g. 9:16) on a wider canvas and generate the missing sides
control   : edge / depth / pose "control video" for video-to-video restyle and motion transfer
"""
import numpy as np
from PIL import Image

GRAY = 127


def fit(img, w, h):
    """Cover-resize + centre crop a PIL image or array to (w,h) -> PIL."""
    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    s = max(w / im.width, h / im.height)
    im = im.resize((max(w, round(im.width * s)), max(h, round(im.height * s))), Image.LANCZOS)
    l, t = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((l, t, l + w, t + h))


def _pil(arr):
    return [Image.fromarray(a) for a in arr]


def keyframes(keys, n, w, h):
    """keys: [(frame_index, image), ...] e.g. [(0, start), (24, middle), (48, end)] -> video, mask."""
    video = np.full((n, h, w, 3), GRAY, np.uint8)
    mask = np.full((n, h, w), 255, np.uint8)
    for idx, img in keys:
        idx = min(max(idx, 0), n - 1)
        video[idx] = np.asarray(fit(img, w, h))
        mask[idx] = 0
    return _pil(video), [Image.fromarray(m) for m in mask]


def extend(prev, n, w, h, keep=17):
    """prev: frames (T,H,W,3) of the clip to continue. Keeps its last `keep` frames as the first `keep` of the new clip."""
    tail = prev[-keep:]
    video = np.full((n, h, w, 3), GRAY, np.uint8)
    mask = np.full((n, h, w), 255, np.uint8)
    for i, f in enumerate(tail):
        video[i] = np.asarray(fit(f, w, h))
        mask[i] = 0
    return _pil(video), [Image.fromarray(m) for m in mask]


def box_mask(n, w, h, box, soft=0):
    """box = (x0, y0, x1, y1) as fractions of the frame; constant over time. white = regenerate."""
    m = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = box
    m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 255
    return np.repeat(m[None], n, 0)


def inpaint(frames, mask, w, h):
    """frames (T,H,W,3), mask (T,H,W) or (H,W) uint8 white = replace. The masked area is grayed in the video input."""
    n = len(frames)
    vid = np.stack([np.asarray(fit(f, w, h)) for f in frames])
    m = mask if mask.ndim == 3 else np.repeat(mask[None], n, 0)
    m = np.stack([np.asarray(fit(Image.fromarray(x).convert("RGB"), w, h).convert("L")) for x in m])
    m = ((m > 127) * 255).astype(np.uint8)
    vid = np.where(m[..., None] > 0, GRAY, vid).astype(np.uint8)
    return _pil(vid), [Image.fromarray(x) for x in m]


def outpaint(frames, w, h):
    """Place the clip on a (w,h) canvas, keep its aspect ratio, centred; the empty sides are generated."""
    n, fh, fw, _ = frames.shape
    s = min(w / fw, h / fh)
    nw, nh = round(fw * s), round(fh * s)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    vid = np.full((n, h, w, 3), GRAY, np.uint8)
    mask = np.full((n, h, w), 255, np.uint8)
    for i in range(n):
        vid[i, y0:y0 + nh, x0:x0 + nw] = np.asarray(Image.fromarray(frames[i]).resize((nw, nh), Image.LANCZOS))
    mask[:, y0:y0 + nh, x0:x0 + nw] = 0
    return _pil(vid), [Image.fromarray(x) for x in mask]


def control(frames, kind, w, h):
    """Control video for VACE restyle / motion transfer, with the whole clip regenerated (mask all white).
    kind: 'edges' (CPU), 'depth' (Depth-Anything via transformers, GPU/CPU), 'pose' (controlnet_aux OpenPose incl. hands)."""
    src = [fit(f, w, h) for f in frames]
    if kind == "edges":
        out = []
        for im in src:
            g = np.asarray(im.convert("L"), np.float32)
            gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
            gy = np.abs(np.diff(g, axis=0, prepend=g[:1]))
            e = (np.clip((gx + gy) * 4, 0, 255) > 60) * 255
            out.append(Image.fromarray(np.repeat(e.astype(np.uint8)[..., None], 3, -1)))
    elif kind == "depth":
        from transformers import pipeline
        pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
        out = [pipe(im)["depth"].convert("RGB") for im in src]
    elif kind == "pose":
        from controlnet_aux import OpenposeDetector
        det = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        out = [det(im, include_hand=True, include_face=True).convert("RGB").resize((w, h)) for im in src]
    else:
        raise ValueError(kind)
    white = [Image.fromarray(np.full((h, w), 255, np.uint8)) for _ in src]
    return out, white
