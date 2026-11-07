#!/usr/bin/env python3
"""
Build a 24 fps training set that keeps anime timing (ones / twos / threes / holds).

The first dataset was resampled to 8 fps, which throws away the held frames that make anime
timing. This script cuts each original 24 fps clip into consecutive windows WITHOUT resampling,
keeps portrait clips portrait (480 wide x 720 tall) and landscape clips landscape (720 x 480),
measures how many distinct drawings each window has, and appends a timing tag to its caption
("animated on ones" / "on twos" / "on threes", plus "with held frames"), so timing can later be
asked for in the prompt.

    python scripts/prepare_24fps_windows.py --list vault_list.json --raw raw_dir --out data/processed_24fps

`--list` is the JSON saved from the VoidVideo Vault's /api/list (category -> clips with captions);
raw files are named "<category>__<public-id-basename>.mp4".
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

WINDOW = 49          # frames per training sample (4k+1 for the Wan VAE): ~2 s at 24 fps
LONG_SIDE, SHORT_SIDE = 720, 480
SAME_FRAME = 0.6     # mean abs grey difference below which two frames are the same drawing


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), int(s.get("nb_frames") or 0), float(num) / float(den)


def cut_window(src, dst, start, w, h, src_fps):
    # Scale to cover the target box, centre-crop, keep the original frames (fps filter only if not 24).
    vf = [f"select='between(n\\,{start}\\,{start + WINDOW - 1})'", "setpts=N/24/TB"]
    if abs(src_fps - 24) > 0.01:
        vf = ["fps=24"] + vf
    vf += [f"scale={w}:{h}:force_original_aspect_ratio=increase", f"crop={w}:{h}"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", ",".join(vf), "-frames:v", str(WINDOW),
                    "-r", "24", "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(dst)], check=True)


def timing_tag(path, w, h):
    sw, sh = w // 5, h // 5
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={sw}:{sh},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, sh, sw).astype(np.float32)
    same = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2)) < SAME_FRAME
    runs, c = [], 1
    for s in same:
        if s:
            c += 1
        else:
            runs.append(c)
            c = 1
    runs.append(c)
    per_sec = len(runs) / (len(fr) / 24)
    if per_sec >= 16:
        tag = "animated on ones"
    elif per_sec >= 9.5:
        tag = "animated on twos"
    else:
        tag = "animated on threes"
    if max(runs) >= 5:
        tag += " with held frames"
    return tag, per_sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    listing = json.load(open(args.list))
    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rows, stats = [], {}
    for category, clips in listing.items():
        for c in clips:
            base = c["publicId"].split("/")[-1]
            src = Path(args.raw) / f"{category}__{base}.mp4"
            if not src.exists():
                print("missing", src)
                continue
            sw, sh, n, fps = probe(src)
            portrait = sh > sw
            w, h = (SHORT_SIDE, LONG_SIDE) if portrait else (LONG_SIDE, SHORT_SIDE)
            n24 = int(n * 24 / fps) if n else 96
            starts = [0] + ([n24 - WINDOW] if n24 - WINDOW >= WINDOW - 4 else [])
            caption = (c.get("caption") or f"An anime clip showing {category.replace('-', ' ')}.").strip()
            for k, start in enumerate(starts):
                name = f"{category}__{base}_w{k}.mp4"
                dst = out / "clips" / name
                cut_window(src, dst, start, w, h, fps)
                tag, per_sec = timing_tag(dst, w, h)
                stats[tag.split(" with")[0]] = stats.get(tag.split(" with")[0], 0) + 1
                rows.append({"video": f"clips/{name}", "caption": f"{caption.rstrip('.')}, {tag}.",
                             "width": w, "height": h})
                print(f"{name}: {w}x{h}, {per_sec:.1f} drawings/s -> {tag}", flush=True)
    with open(out / "metadata.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} windows written; timing mix: {stats}")


if __name__ == "__main__":
    main()
