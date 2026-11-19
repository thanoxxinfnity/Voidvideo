#!/usr/bin/env python3
"""
Pick a balanced training subset from the v4 24 fps windows that fits a weekly Kaggle quota.

Keeps every window whose latents are already cached (so they are not re-encoded) and adds a fixed number of
NEW windows per category (monsters, animals, crowds, each VFX type, some synthetic motion graphics).

    python scripts/select_training_subset.py --meta v4_metadata.jsonl --cached cached_a.jsonl cached_b.jsonl --out subset.jsonl
"""
import argparse
import collections
import json
import random

NEW_PER_CATEGORY = {"animals": 150, "creatures": 200, "crowd": 120, "vfx-motion-graphics": 80,
                    "vfx-beam": 45, "vfx-explosion": 45, "vfx-fire": 45, "vfx-ice": 45, "vfx-impact": 45,
                    "vfx-lightning": 45, "vfx-morph": 45, "vfx-smoke-sparks": 45, "vfx-water": 45,
                    "vfx-wind-debris": 45}


def category(video: str) -> str:
    return video.split("/")[-1].split("__")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--cached", nargs="+", required=True, help="metadata files of the windows whose latents are cached")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.meta) if line.strip()]
    cached = {json.loads(line)["video"] for f in args.cached for line in open(f) if line.strip()}
    rnd = random.Random(args.seed)
    by_cat = collections.defaultdict(list)
    for r in rows:
        if r["video"] not in cached:
            by_cat[category(r["video"])].append(r)
    picked = [r for r in rows if r["video"] in cached]
    for cat, n in NEW_PER_CATEGORY.items():
        pool = by_cat.get(cat, [])
        # Take whole source clips where possible (both windows) so a clip's two phases stay together.
        rnd.shuffle(pool)
        picked += pool[:n]
    with open(args.out, "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    c = collections.Counter(category(r["video"]) for r in picked)
    print(f"{len(picked)} windows ({len(cached)} cached + {len(picked) - len(cached)} new)")
    print(dict(sorted(c.items())))


if __name__ == "__main__":
    main()
