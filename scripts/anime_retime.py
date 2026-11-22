#!/usr/bin/env python3
"""
Re-time a generated clip into anime-style 24 fps with holds.

The training clips were normalised to 8 fps, so the LoRA learned one new drawing every 1/8 s
(anime "on threes"). Played back at 16 fps (the Wan default) the motion runs 2x too fast. This
script puts every generated frame back on its real 1/8 s slot (3 frames at 24 fps) and, like
limited animation, holds the previous drawing wherever the picture barely changes, so the faint
AI "boil" between near-identical frames turns into a clean hold. Fast action keeps every drawing.

    python scripts/anime_retime.py in.mp4 out.mp4 [--src-fps 8] [--hold 0.6]

--hold scales the hold threshold: higher holds more (more limited-animation feel), 0 disables holds.

--adaptive (for 24 fps clips from the 24 fps LoRA): pick the timing per moment like an animator would:
fast motion stays on ones, medium motion goes on twos, slow motion on threes, near-static holds longer.
--fast/--medium/--slow are the frame-difference thresholds (grey levels on a 1/4-size frame).
"""
import argparse
import json
import subprocess

import numpy as np


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "json", path], capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    return s["width"], s["height"]


def read_frames(path, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def small_gray(f):
    g = f[::4, ::4].astype(np.float32).mean(axis=2)
    return g


def plan_drawings(frames, hold_scale):
    """For each source frame, the index of the drawing to show (holding earlier drawings on small changes)."""
    grays = [small_gray(f) for f in frames]
    steps = np.array([np.abs(grays[i + 1] - grays[i]).mean() for i in range(len(grays) - 1)])
    # Threshold relative to this clip's own motion: a change well below its typical step is "boil", not motion.
    thresh = hold_scale * float(np.median(steps)) if len(steps) else 0.0
    shown, last = [0], 0
    for i in range(1, len(frames)):
        if np.abs(grays[i] - grays[last]).mean() >= thresh:
            last = i
        shown.append(last)
    return shown, steps, thresh


def plan_adaptive(frames, fast, medium, slow, max_hold=6):
    """Walk the 24 fps frames and choose ones/twos/threes/holds from the motion just ahead."""
    grays = [small_gray(f) for f in frames]
    out, i, mix = [], 0, {1: 0, 2: 0, 3: 0, "hold": 0}
    while i < len(frames):
        ahead = [np.abs(grays[j + 1] - grays[j]).mean() for j in range(i, min(i + 2, len(frames) - 1))]
        m = max(ahead) if ahead else 0.0
        if m >= fast:
            k = 1
        elif m >= medium:
            k = 2
        elif m >= slow:
            k = 3
        else:  # near-static: hold this drawing until the picture really changes (capped)
            k = 1
            while i + k < len(frames) and k < max_hold and np.abs(grays[i + k] - grays[i]).mean() < slow:
                k += 1
            k = max(k, 3)
        k = min(k, len(frames) - i)
        mix["hold" if m < slow else k] += 1
        out += [i] * k
        i += k
    return out, mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--src-fps", type=float, default=8.0, help="real-time spacing of the generated frames")
    ap.add_argument("--hold", type=float, default=0.6)
    ap.add_argument("--adaptive", action="store_true", help="24 fps input: choose ones/twos/threes/holds by motion")
    ap.add_argument("--fast", type=float, default=12.0)
    ap.add_argument("--medium", type=float, default=4.0)
    ap.add_argument("--slow", type=float, default=1.2)
    args = ap.parse_args()

    w, h = probe(args.src)
    frames = read_frames(args.src, w, h)
    if args.adaptive:
        shown, mix = plan_adaptive(frames, args.fast, args.medium, args.slow)
        print(f"{len(frames)} frames at 24 fps -> {len(set(shown))} drawings "
              f"({len(set(shown)) / (len(frames) / 24):.1f}/s); timing choices: ones={mix[1]} twos={mix[2]} "
              f"threes={mix[3]} holds={mix['hold']}")
        write(args.dst, frames, shown, 1, w, h)
        return
    shown, steps, thresh = plan_drawings(frames, args.hold)
    per = round(24 / args.src_fps)  # output frames per source frame (3 for 8 fps)
    drawings = len(set(shown))
    print(f"{len(frames)} generated frames -> {len(frames) * per} frames at 24 fps "
          f"({len(frames) * per / 24:.2f}s), {drawings} distinct drawings, "
          f"{len(frames) - drawings} held (threshold {thresh:.2f})")

    write(args.dst, frames, shown, per, w, h)


def write(dst, frames, shown, per, w, h):
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", dst],
                           stdin=subprocess.PIPE)
    for idx in shown:
        for _ in range(per):
            enc.stdin.write(frames[idx].tobytes())
    enc.stdin.close()
    enc.wait()


if __name__ == "__main__":
    main()
