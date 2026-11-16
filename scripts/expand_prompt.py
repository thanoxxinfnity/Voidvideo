#!/usr/bin/env python3
"""
Expand @character mentions into full visual descriptions (the "character card" jugaad).

Big closed models know named characters because their training data links names to looks, and many products also
rewrite the prompt before it reaches the model. Our Wan LoRA only knows words, so we do the rewriting ourselves:

    python scripts/expand_prompt.py "@kaito faces @ryu in a ruined street, kaito raises two fingers, ryu lunges"

-> "Medium shot, static camera. On the left, a tall young man with spiky white hair ... On the right, a muscular man
    with pink-red spiky hair ... Left: kaito raises two fingers. Right: ryu lunges. Traditional 2D hand-drawn anime."
Names are matched case-insensitively; the first mentioned card goes left, the second right (third = center background).
Cards live in configs/characters.json (add your own, or pass --cards).
"""
import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_CARDS = Path(__file__).resolve().parent.parent / "configs" / "characters.json"
POSITIONS = {1: ["in the center"], 2: ["on the left", "on the right"], 3: ["on the left", "on the right", "in the center background"]}
STYLE = "Traditional 2D hand-drawn anime."


def expand(prompt: str, cards: dict, camera: str = "Medium shot, static camera.") -> str:
    names = [m.group(1).lower() for m in re.finditer(r"@(\w+)", prompt)]
    seen = []
    for n in names:
        if n not in cards:
            raise KeyError(f"unknown character @{n}; known: {', '.join(k for k in cards if not k.startswith('_'))}")
        if n not in seen:
            seen.append(n)
    if not seen:
        return prompt if prompt.rstrip().endswith(".") else prompt.rstrip() + "."
    if len(seen) > 3:
        print("warning: 4+ characters in one shot mix up; compose them in layers instead", file=sys.stderr)
    pos = POSITIONS.get(len(seen), POSITIONS[3] + ["in the background"] * (len(seen) - 3))
    intro = " ".join(f"{p.capitalize()}, {cards[n]['look']}." + (f" {cards[n]['body'].capitalize()}." if cards[n].get("body") else "")
                     for n, p in zip(seen, pos))
    where = dict(zip(seen, pos))
    # Keep the user's action text, but tie every @name to its position so the model does not swap the roles.
    action = re.sub(r"@(\w+)", lambda m: f"the character {where[m.group(1).lower()]}" if len(seen) > 1 else "the character", prompt)
    action = re.sub(r"(^|[.!?] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), action)
    return f"{camera} {intro} {action.rstrip('.')}. {STYLE}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--cards", default=str(DEFAULT_CARDS))
    ap.add_argument("--camera", default="Medium shot, static camera.")
    args = ap.parse_args()
    cards = {k.lower(): v for k, v in json.load(open(args.cards)).items()}
    print(expand(args.prompt, cards, args.camera))


if __name__ == "__main__":
    main()
