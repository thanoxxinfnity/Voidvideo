#!/usr/bin/env python3
"""
Category-balance the training metadata without touching any clip file.

Reads data/processed/metadata_full.jsonl (written by prepare_dataset.py) and
writes data/processed/metadata.jsonl, which is what training reads. The category
is the filename prefix before "__" (expressions, gestures, locomotion,
secondary-motion, fighting, vfx); clips without a prefix go into "other".

For each category:
  - more than --cap rows: a seeded random subset of --cap rows is kept
  - fewer than --cap rows: rows are repeated, at most --max-repeat times each,
    so tiny categories do not get memorised

Usage:
    python scripts/prepare_dataset.py --task t2v
    python scripts/balance_metadata.py --cap 110 --max-repeat 2
    python scripts/balance_metadata.py --target 110   # exactly 110 rows per category
"""
import argparse
import collections
import json
import random
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def category(row: dict) -> str:
    name = Path(row["video"]).name
    return name.split("__")[0] if "__" in name else "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=110, help="max rows per category per run")
    parser.add_argument("--max-repeat", type=int, default=2, help="max times a clip is repeated")
    parser.add_argument("--target", type=int, default=0,
                        help="if set, every category (incl. other) gets exactly this many rows: "
                             "larger ones are sampled, smaller ones repeated (whole copies + random remainder), "
                             "ignoring --cap/--max-repeat")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    full = PROCESSED_DIR / "metadata_full.jsonl"
    rows = [json.loads(line) for line in full.read_text().splitlines() if line.strip()]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[category(r)].append(r)

    rng = random.Random(args.seed)
    out = []
    print(f"{'category':18} {'clips':>6} -> {'rows':>5}")
    for cat in sorted(groups):
        items = groups[cat]
        if args.target:
            full, rem = divmod(args.target, len(items))
            chosen = items * full + rng.sample(items, rem) if full else rng.sample(items, args.target)
        elif len(items) > args.cap:
            chosen = rng.sample(items, args.cap)
        else:
            chosen = list(items)
            reps = 1 if cat == "other" else min(args.max_repeat, max(1, args.cap // len(items)))
            chosen = items * reps
        out += chosen
        print(f"{cat:18} {len(items):6d} -> {len(chosen):5d}")
    rng.shuffle(out)

    with open(PROCESSED_DIR / "metadata.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(out)} balanced rows to {PROCESSED_DIR / 'metadata.jsonl'} (full list kept in metadata_full.jsonl)")


if __name__ == "__main__":
    main()
