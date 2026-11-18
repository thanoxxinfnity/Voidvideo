#!/usr/bin/env python3
"""
Plan a crowd scene as layers (see crowd_compose.py): one background plate, one clip per hero (from character cards), and a
small pool of "extra" clips that are re-used with flips / time offsets / tints to fill rows behind the heroes.

    python scripts/crowd_plan.py --scene "ruined city street at dusk, orange sky" --heroes kaito,ryu,mina \
        --extras 14 --pool 4 --action "cheering with fists raised" --out plan/

Writes plan/jobs.json (what to generate: name, prompt, width, height, seed) and plan/layout.json (where to put it, with
clip paths layers/<name>.mp4 and bg.mp4 relative to plan/). Generate the jobs (kaggle_wan24_crowd_entrypoint.py does it),
then run crowd_compose.py plan/layout.json out.mp4.
"""
import argparse
import json
import random
from pathlib import Path

CARDS = Path(__file__).resolve().parent.parent / "configs" / "characters.json"
GREEN = ("isolated on a solid flat pure green screen background (#00ff00), full body visible from head to feet, "
         "no ground, no shadow, no scenery, static camera")
STYLE = "Traditional 2D hand-drawn anime."
HAIR = ["short black hair", "long brown hair", "spiky blond hair", "short red hair", "blue ponytail", "grey bob cut",
        "curly dark hair", "green short hair", "long silver hair", "messy brown hair"]
CLOTHES = ["a grey hoodie and jeans", "a school uniform", "a red jacket", "a white shirt and black trousers",
           "a yellow raincoat", "a green sweater and skirt", "a dark coat with a scarf", "a blue tracksuit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--heroes", default="", help="comma list of character-card names")
    ap.add_argument("--extras", type=int, default=12, help="how many background people in the final frame")
    ap.add_argument("--pool", type=int, default=4, help="how many different extra clips to generate (re-used with flips)")
    ap.add_argument("--action", default="standing and watching, subtle breathing")
    ap.add_argument("--hero-actions", default="", help="semicolon list, one action per hero (default: --action)")
    ap.add_argument("--cards", default=str(CARDS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    cards = {k.lower(): v for k, v in json.load(open(a.cards)).items() if not k.startswith("_")}
    heroes = [h.strip().lower() for h in a.heroes.split(",") if h.strip()]
    h_actions = [x.strip() for x in a.hero_actions.split(";")] if a.hero_actions else []
    out = Path(a.out)
    (out / "layers").mkdir(parents=True, exist_ok=True)

    jobs = [{"name": "bg", "width": 720, "height": 480, "seed": a.seed,
             "prompt": f"Wide static shot of {a.scene}, empty, no people, no characters, no figures. {STYLE}", "out": "bg.mp4"}]
    layers = []
    n = len(heroes)
    for i, name in enumerate(heroes):
        c = cards[name]
        act = h_actions[i] if i < len(h_actions) and h_actions[i] else a.action
        jobs.append({"name": name, "width": 480, "height": 720, "seed": a.seed + i,
                     "prompt": f"{c['look'].capitalize()}. The character is {act}. {GREEN.capitalize()}. {STYLE}", "out": f"layers/{name}.mp4"})
        layers.append({"video": f"layers/{name}.mp4", "x": round((i + 1) / (n + 1), 3), "y": 0.97, "height": 0.62,
                       "flip": False, "offset": 0, "tint": [1, 1, 1], "shadow": True})
    pool = []
    for k in range(a.pool):
        look = f"a person with {rnd.choice(HAIR)}, wearing {rnd.choice(CLOTHES)}"
        jobs.append({"name": f"extra{k}", "width": 480, "height": 720, "seed": a.seed + 100 + k,
                     "prompt": f"{look.capitalize()}. The person is {a.action}. {GREEN.capitalize()}. {STYLE}", "out": f"layers/extra{k}.mp4"})
        pool.append(f"layers/extra{k}.mp4")
    rows = [(0.80, 0.30, 0.82), (0.74, 0.24, 0.72), (0.68, 0.19, 0.64)]          # (feet y, height, brightness) far = dimmer
    for e in range(a.extras):
        row = rows[e % len(rows)]
        per = max(1, -(-a.extras // len(rows)))
        slot = e // len(rows)
        x = (slot + 0.5 + rnd.uniform(-0.25, 0.25)) / per
        b = row[2]
        layers.append({"video": pool[e % len(pool)] if pool else None, "x": round(min(max(x, 0.04), 0.96), 3),
                       "y": round(row[0] + rnd.uniform(-0.015, 0.015), 3), "height": round(row[1] * rnd.uniform(0.92, 1.08), 3),
                       "flip": rnd.random() < 0.5, "offset": rnd.randint(0, 24),
                       "tint": [round(b * rnd.uniform(0.92, 1.05), 2) for _ in range(3)], "shadow": True})
    layers = [L for L in layers if L["video"]]
    json.dump(jobs, open(out / "jobs.json", "w"), indent=1)
    json.dump({"size": [720, 480], "fps": 24, "frames": 49, "background": "bg.mp4", "layers": layers},
              open(out / "layout.json", "w"), indent=1)
    print(f"{len(jobs)} clips to generate ({len(heroes)} heroes + {len(pool)} extra looks + background), "
          f"{len(layers)} people in the frame -> {out}/jobs.json, {out}/layout.json")


if __name__ == "__main__":
    main()
