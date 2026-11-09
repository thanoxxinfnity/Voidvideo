#!/usr/bin/env python3
"""
Turn Sakugabooru clips (trojblue/sakugabooru2025 on Hugging Face) into the listing format that
scripts/prepare_24fps_windows.py reads, with captions built from the booru tags (the clips have no
written captions), and link the media files under the "<category>__<id>.mp4" names it expects.

    python scripts/sakuga_to_listing.py picked.json clips_dir raw_dir listing.json
"""
import json
import os
import sys

PHRASES = {
    "fighting": "an anime fight scene", "impact frames": "with impact frames", "smears": "with smear frames",
    "effects": "with hand-drawn effects", "explosions": "explosions", "fire": "fire", "lightning": "lightning",
    "beams": "energy beams", "sparks": "sparks", "smoke": "billowing smoke", "debris": "flying debris",
    "liquid": "splashing liquid", "running": "a character running", "walk cycle": "a character walking",
    "hair": "hair moving", "fabric": "fabric and clothes fluttering", "wind": "in the wind",
    "character acting": "expressive character acting", "crying": "a character crying",
    "dancing": "a character dancing", "falling": "falling", "flying": "flying", "rotation": "turning around",
    "animals": "animals moving", "creatures": "a fantasy creature", "mecha": "a giant robot", "sports": "sports action", "crowd": "a crowd",
    "background animation": "with an animated background", "ice": "ice", "morphing": "morphing shapes",
    "henshin": "a transformation sequence",
}
LEAD = {"fighting": "Anime fight scene", "vfx": "Anime effects animation", "locomotion": "Anime character motion",
        "secondary-motion": "Anime secondary motion", "expressions": "Anime character acting close to the camera",
        "creatures": "Anime monster and creature animation", "animals": "Anime animal animation",
        "crowd": "Anime crowd scene with many characters"}


VFX_LEAD = {"vfx-morph": "Anime transformation and morphing effects", "vfx-ice": "Anime ice and freezing effects",
            "vfx-lightning": "Anime lightning effects", "vfx-beam": "Anime energy beam effects",
            "vfx-impact": "Anime impact frames and hit effects", "vfx-water": "Anime water and liquid effects",
            "vfx-fire": "Anime fire effects", "vfx-explosion": "Anime explosion effects",
            "vfx-wind-debris": "Anime wind and flying debris effects", "vfx-smoke-sparks": "Anime smoke and sparks effects"}
LEAD.update(VFX_LEAD)


def caption(cat, tags):
    words = [PHRASES[t] for t in tags if t in PHRASES]
    words = [w for w in dict.fromkeys(words)][:6]
    return f"{LEAD[cat]}: " + ", ".join(words) + ", traditional 2D hand-drawn animation." if words else \
        f"{LEAD[cat]}, traditional 2D hand-drawn animation."


def main():
    picked, clips_dir, raw_dir, out = sys.argv[1:5]
    os.makedirs(raw_dir, exist_ok=True)
    listing = {}
    for p in json.load(open(picked)):
        src = None
        for ext in (".mp4", ".webm"):
            f = os.path.join(clips_dir, f"sakuga_{p['post_id']}{ext}")
            if os.path.exists(f):
                src = f
        if not src:
            continue
        dst = os.path.join(raw_dir, f"{p['cat']}__sakuga_{p['post_id']}.mp4")
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
        listing.setdefault(p["cat"], []).append({"publicId": f"sakuga/sakuga_{p['post_id']}",
                                                 "caption": caption(p["cat"], p["tags"])})
    json.dump(listing, open(out, "w"), indent=1)
    print({k: len(v) for k, v in listing.items()})


if __name__ == "__main__":
    main()
