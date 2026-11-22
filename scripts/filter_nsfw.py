#!/usr/bin/env python3
"""
Keep adult / sexual material out of the training set.

    python scripts/filter_nsfw.py IN.jsonl OUT.jsonl [--report removed.jsonl]

Drops every row whose caption (or VLM fields, when present) mentions nudity / sexual content / fetish wording, or whose
VLM `nsfw` flag is true. Sakugabooru picks are additionally restricted to rating "Safe" when they are selected
(see docs/VFX_MASTER_PLAN.md); this script is the second net for Vault uploads and for anything the VLM describes.
Run it on every metadata file before it goes to training. Removed rows are listed in the report for a quick eyeball.
"""
import argparse
import json
import re
import sys

BLOCK = re.compile(
    r"\b(nude|nudity|naked|topless|bare[- ]?breast\w*|breasts?|nipples?|genitals?|sex\w*|erotic\w*|hentai|ecchi|lewd|"
    r"fetish|lingerie|underwear|panties|panty|pantsu|upskirt|cleavage|bikini|swimsuit|undress\w*|strip(?:per|tease|ping)|seductive|"
    r"provocative|fanservice|fan[- ]service|nsfw|explicit|porn\w*|moan\w*|groping|bath(?:ing)? scene)\b", re.I)


def text_of(r):
    """Caption + the descriptive VLM values (never the JSON keys: the key "nsfw" would match the word list)."""
    parts = [r.get("caption", "")]
    v = r.get("vlm") or {}
    for k in ("subjects", "action", "setting", "camera", "effects"):
        x = v.get(k)
        parts.append(" ".join(map(str, x)) if isinstance(x, list) else str(x or ""))
    return " ".join(parts)


def is_bad(r):
    v = r.get("vlm") or {}
    if v.get("nsfw") in (True, "true", "yes") or v.get("text") in (True, "true", "yes"):   # adult content, or burned-in text / logos
        return True
    return bool(BLOCK.search(text_of(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    keep = drop = 0
    rep = open(a.report, "w") if a.report else None
    with open(a.out, "w") as fo:
        for line in open(a.inp):
            if not line.strip():
                continue
            r = json.loads(line)
            if is_bad(r):
                drop += 1
                if rep:
                    rep.write(json.dumps({"video": r["video"], "caption": r.get("caption", "")[:300]}) + "\n")
            else:
                keep += 1
                fo.write(line if line.endswith("\n") else line + "\n")
    print(f"kept {keep}, removed {drop}", file=sys.stderr)


if __name__ == "__main__":
    main()
