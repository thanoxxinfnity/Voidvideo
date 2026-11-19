#!/usr/bin/env python3
"""
Pick Sakugabooru clips for the "dark action" set (blood, death, brutal fights).

Sakugabooru tags techniques, not gore, so we pick fight clips from series known for violent scenes and let the VLM
(`violence` field in scripts/vlm_caption_clips.py) decide which windows really show blood or death. Only rating "Safe"
clips are taken (the sexual material sits in Questionable/Explicit).

    python scripts/select_dark_action.py videos.json OUT.json --exclude picked.json picked2.json ... [--total 1500]
"""
import argparse
import collections
import json

SERIES = ["kimetsu no yaiba", "shingeki no kyojin", "tokyo ghoul", "jujutsu kaisen", "devilman", "dororo", "vinland saga",
          "afro samurai", "akame ga kill", "chainsaw man", "berserk", "hellsing", "black lagoon", "claymore", "parasyte",
          "elfen lied", "jigokuraku", "goblin slayer", "ajin", "gantz", "fate/zero", "mononoke", "bleach", "naruto",
          "fullmetal alchemist", "hunter x hunter", "ghost in the shell", "akira", "spriggan", "kaiju no. 8", "overlord",
          "made in abyss", "dorohedoro", "ninja scroll", "vampire hunter d", "blood-c", "jormungand", "psycho-pass",
          "samurai champloo", "kill la kill", "solo leveling", "dungeon meshi", "evangelion"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("out")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--total", type=int, default=1500)
    ap.add_argument("--per-series", type=int, default=90)
    ap.add_argument("--only-series", default="", help="comma list: use only these series (e.g. \"jujutsu kaisen\")")
    ap.add_argument("--cat", default="dark-action")
    a = ap.parse_args()
    used = set()
    for f in a.exclude:
        used |= {p["post_id"] for p in json.load(open(f))}
    series = [x.strip() for x in a.only_series.split(",") if x.strip()] or SERIES
    per, picks = collections.Counter(), []
    for r in sorted(json.load(open(a.index)), key=lambda r: -r[3]):            # most favourited first
        pid, tags, rating, fav, _w, _h, cr = r
        if rating != "Safe" or "fighting" not in tags or pid in used:
            continue
        crs = " ".join(cr).lower()
        s = next((s for s in series if s in crs), None)
        if not s or per[s] >= a.per_series:
            continue
        per[s] += 1
        picks.append({"post_id": pid, "cat": a.cat, "tags": tags, "copyright": cr, "fav": fav, "tar": f"{pid // 1000}.tar"})
        if len(picks) >= a.total:
            break
    json.dump(picks, open(a.out, "w"))
    print(len(picks), "clips;", dict(per.most_common(12)))


if __name__ == "__main__":
    main()
