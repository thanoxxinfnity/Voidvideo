#!/usr/bin/env python3
"""
Training mix for a top-up run: a stratified sample of the old windows + the NEW windows repeated, so a short run actually sees them.

    python scripts/build_topup_metadata.py BASE.jsonl NEW.jsonl OUT.jsonl [--base-n 1000] [--fan-repeat 2] [--teacher-repeat 5]

Windows of the same clip are never cut: repeats just add the same row several times (the trainer caches each clip's latent once).
Prints the mix. Category = file name part before "__" (fighting, creatures, vfx-fire ... fan-animation, teacher, teacher-vfx).
"""
import argparse
import collections
import json
import random


def cat(r):
    n = r["video"].split("/")[-1]
    return n.split("__")[0] if "__" in n else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("out")
    ap.add_argument("--base-n", type=int, default=1000)
    ap.add_argument("--fan-repeat", type=int, default=2)
    ap.add_argument("--teacher-repeat", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    base = [json.loads(l) for l in open(a.base) if l.strip()]
    new = [json.loads(l) for l in open(a.new) if l.strip()]
    by = collections.defaultdict(list)
    for r in base:
        by[cat(r)].append(r)
    keep = []
    for c, rows in by.items():
        k = max(1, round(a.base_n * len(rows) / len(base)))          # proportional, at least one per category
        keep += rnd.sample(rows, min(k, len(rows)))
    out = list(keep)
    for r in new:
        c = cat(r)
        rep = a.teacher_repeat if c.startswith("teacher") else a.fan_repeat if c.startswith("fan") else 1
        out += [r] * rep
    rnd.shuffle(out)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    mix = collections.Counter("new" if cat(r).startswith(("fan", "teacher")) else "old" for r in out)
    print(f"{len(out)} rows: {dict(mix)}; new windows {len(new)} (unique), old kept {len(keep)} of {len(base)}")


if __name__ == "__main__":
    main()
