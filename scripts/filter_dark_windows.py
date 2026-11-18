#!/usr/bin/env python3
"""Drop (near-)black windows (fade-outs, black title screens) from a prepared dataset dir (metadata.jsonl + clips/).
    python scripts/filter_dark_windows.py DATASET_DIR [--min-luma 14]"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

W, H = 64, 36


def median_luma(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select='not(mod(n\\,6))',scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    px = np.frombuffer(raw, np.uint8)
    n = px.size // (W * H)
    return float(np.median(px[: n * W * H].reshape(n, W * H).mean(1))) if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--min-luma", type=float, default=14.0)
    a = ap.parse_args()
    d = Path(a.dir)
    rows = [json.loads(l) for l in open(d / "metadata.jsonl") if l.strip()]
    keep, drop = [], []
    for r in rows:
        (keep if median_luma(d / r["video"]) > a.min_luma else drop).append(r)
    for r in drop:
        (d / r["video"]).unlink(missing_ok=True)
    with open(d / "metadata.jsonl", "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    print(f"kept {len(keep)}, dropped {len(drop)} dark windows")


if __name__ == "__main__":
    main()
