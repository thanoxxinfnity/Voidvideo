#!/usr/bin/env python3
"""
Turn clips made by another AI video model (10 s, 24 fps, portrait or landscape) into training windows.

    python scripts/prepare_ai_clips.py CLIPS_DIR --map map.json --out work/teacher [--crop 720:1120:0:0] [--stride 36]

map.json: {"<clip file stem>": {"name": "fire", "caption": "<the prompt used>"}, ...}. For each clip: crop away the visible
corner watermark (--crop = w:h:x:y), cut into 49-frame windows every --stride frames (default 36 = 13 frames overlap),
tag the timing, and use the prompt as the provisional caption (the VLM re-captions later: a 10 s clip changes content over time, so a
single prompt only fits its first window). Output: OUT/metadata.jsonl + OUT/clips/*.mp4 (windows named teacher__<name>_wK.mp4).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", default="")
    ap.add_argument("--stride", type=int, default=36)
    ap.add_argument("--category", default="teacher")
    a = ap.parse_args()
    mp = json.load(open(a.map))
    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rows = []
    for stem, info in mp.items():
        src = next(Path(a.clips).glob(f"*{stem}*.mp4"), None)
        if src is None:
            print("missing", stem)
            continue
        w = out / "_work" / info["name"]
        shutil.rmtree(w, ignore_errors=True)
        w.mkdir(parents=True)
        shutil.copy(src, w / f"{info['name']}.mp4")
        cmd = [sys.executable, str(HERE / "prepare_fan_video.py"), str(w / f"{info['name']}.mp4"), "--out", str(w / "seg"), "--category", a.category,
               "--threshold", "0.5", "--min-seconds", "2.06", "--caption", info["caption"]]
        if a.crop:
            cmd += ["--crop", a.crop]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, str(HERE / "prepare_24fps_windows.py"), "--list", str(w / "seg" / "listing.json"), "--raw", str(w / "seg" / "raw"),
                        "--out", str(w / "ds"), "--only-24", "--allow-resample", "--stride", str(a.stride)], check=True, stdout=subprocess.DEVNULL)
        for line in open(w / "ds" / "metadata.jsonl"):
            r = json.loads(line)
            name = Path(r["video"]).name
            shutil.copy(w / "ds" / r["video"], out / "clips" / name)
            r["video"] = f"clips/{name}"
            r["source"] = "ai-teacher"
            rows.append(r)
        shutil.rmtree(w, ignore_errors=True)
        print(f"{info['name']}: {sum(1 for r in rows if info['name'] in r['video'])} windows")
    shutil.rmtree(out / "_work", ignore_errors=True)
    with open(out / "metadata.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} windows -> {out}")


if __name__ == "__main__":
    main()
