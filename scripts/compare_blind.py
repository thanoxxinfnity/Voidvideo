#!/usr/bin/env python3
"""
Blind side-by-side comparison of two sets of clips (e.g. our model vs Google Veo) made from the SAME prompts.

    python scripts/compare_blind.py OURS_DIR OTHER_DIR OUT_DIR [--seconds 2.04] [--seed 1]
    python scripts/compare_blind.py --reveal OUT_DIR

Files with the same base name in both folders are paired (monster.mp4 / monster.mp4). Each pair becomes OUT_DIR/<name>.mp4:
left = A, right = B, order randomised and hidden (the key is in OUT_DIR/key.json, shown only by --reveal). Both clips are cut to the
first --seconds (default 2.04 s = our 49 frames), scaled to the same height, labelled A / B, no audio.
Score each pair first (docs/COMPARE_VS_VEO.md has the sheet), then run --reveal.
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

EXT = (".mp4", ".mov", ".webm", ".mkv")


def pair_up(a, b):
    fa = {p.stem: p for p in Path(a).iterdir() if p.suffix.lower() in EXT}
    fb = {p.stem: p for p in Path(b).iterdir() if p.suffix.lower() in EXT}
    return {k: (fa[k], fb[k]) for k in sorted(fa) if k in fb}, sorted(set(fa) ^ set(fb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=49 / 24)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reveal", action="store_true")
    args = ap.parse_args()
    if args.reveal:
        out = Path(args.out)
        for k, v in json.load(open(out / "key.json")).items():
            print(f"{k}: A = {v['A']}, B = {v['B']}")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs, unmatched = pair_up(args.a, args.b)
    if unmatched:
        print("no partner for:", ", ".join(unmatched))
    rnd = random.Random(args.seed)
    key = {}
    for name, (pa, pb) in pairs.items():
        left, right = (("ours", pa), ("other", pb)) if rnd.random() < 0.5 else (("other", pb), ("ours", pa))
        key[name] = {"A": left[0], "B": right[0]}
        h = args.height
        vf = (f"[0:v]trim=duration={args.seconds:.3f},setpts=PTS-STARTPTS,scale=-2:{h},fps=24,drawtext=text='A':x=14:y=10:fontsize=36:"
              f"fontcolor=white:box=1:boxcolor=black@0.6[l];[1:v]trim=duration={args.seconds:.3f},setpts=PTS-STARTPTS,scale=-2:{h},fps=24,"
              f"drawtext=text='B':x=14:y=10:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.6[r];[l][r]hstack=inputs=2[v]")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(left[1]), "-i", str(right[1]), "-filter_complex", vf, "-map", "[v]", "-an",
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out / f"{name}.mp4")], check=True)
        print("wrote", out / f"{name}.mp4")
    json.dump(key, open(out / "key.json", "w"), indent=1)
    print(f"{len(key)} blind pairs written; the key is in {out}/key.json (do not open it until you have scored).")


if __name__ == "__main__":
    main()
