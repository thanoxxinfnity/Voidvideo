#!/usr/bin/env python3
"""
Cut long videos (fan animations, episodes, AMVs) into clean training windows.

Why not fixed 4 s pieces: a piece that straddles a hard cut teaches the model a bad jump cut. So this finds the shot cuts
(ffmpeg scene detection), drops intro / credits / too-short shots, optionally crops away a watermark, and writes one segment per
shot. scripts/prepare_24fps_windows.py then cuts 49-frame windows inside each segment and tags the timing (ones / twos / threes).

    python scripts/prepare_fan_video.py VIDEO_OR_DIR --out work/fan --category fan-animation \
        [--threshold 0.3] [--skip-start 5] [--skip-end 20] [--crop w:h:x:y] [--min-seconds 2.1]
    python scripts/prepare_24fps_windows.py --list work/fan/listing.json --raw work/fan/raw --out work/fan/ds24 --only-24 --allow-resample
"""
import argparse
import json
import re
import subprocess
from pathlib import Path


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    num, den = j["streams"][0]["r_frame_rate"].split("/")
    return float(j["format"]["duration"]), float(num) / float(den), j["streams"][0]["width"], j["streams"][0]["height"]


def cuts(path, threshold):
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vf", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    return sorted({float(m.group(1)) for m in re.finditer(r"pts_time:([0-9.]+)", r.stderr)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--category", default="fan-animation")
    ap.add_argument("--threshold", type=float, default=0.3, help="scene-change score 0..1 (lower = more cuts)")
    ap.add_argument("--min-seconds", type=float, default=2.1, help="drop shots shorter than this (need >= 49 frames at 24 fps)")
    ap.add_argument("--skip-start", type=float, default=0.0)
    ap.add_argument("--skip-end", type=float, default=0.0)
    ap.add_argument("--crop", default="", help="ffmpeg crop w:h:x:y to cut a watermark / letterbox away")
    ap.add_argument("--caption", default="")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    srcs = sorted(Path(a.src).glob("*")) if Path(a.src).is_dir() else [Path(a.src)]
    srcs = [s for s in srcs if s.suffix.lower() in (".mp4", ".mkv", ".mov", ".webm", ".avi")]
    listing, n_seg, short = {a.category: []}, 0, 0
    for s in srcs:
        dur, fps, w, h = probe(s)
        bounds = [0.0] + [c for c in cuts(s, a.threshold) if 0 < c < dur] + [dur]
        lo, hi = a.skip_start, dur - a.skip_end
        for k, (t0, t1) in enumerate(zip(bounds, bounds[1:])):
            t0, t1 = max(t0, lo), min(t1, hi)
            t0 += 0.08                                      # step past the cut frame itself (fades / flashes)
            t1 -= 0.04
            if t1 - t0 < a.min_seconds:
                short += 1
                continue
            name = f"{a.category}__{re.sub(r'[^A-Za-z0-9]+', '-', s.stem)[:30]}-s{k:03d}"
            vf = (f"crop={a.crop}," if a.crop else "") + "null"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(s), "-vf", vf, "-an",
                            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out / "raw" / f"{name}.mp4")], check=True)
            listing[a.category].append({"publicId": f"fan/{name.split('__', 1)[1]}", "caption": a.caption or "An anime fight scene from a fan animation."})
            n_seg += 1
        print(f"{s.name}: {dur:.0f}s, {fps:.1f} fps, {len(bounds) - 1} shots")
    json.dump(listing, open(out / "listing.json", "w"), indent=1)
    print(f"{n_seg} segments written to {out}/raw ({short} shots dropped as too short)")


if __name__ == "__main__":
    main()
