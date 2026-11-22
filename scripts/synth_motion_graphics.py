#!/usr/bin/env python3
"""
Synthetic motion-graphics / quiz / transition clips for the VFX LoRA: unlimited data, no copyright issues,
captions that are exact because we know every parameter used to render the clip.

    python scripts/synth_motion_graphics.py --out DIR --n 700 [--seed 0]

Writes DIR/clips/*.mp4 (49 frames, 24 fps, 720x480 or 480x720, every frame a new drawing = "on ones") and
DIR/metadata.jsonl rows {video, caption, width, height}, the same layout as the 24 fps anime sets.
Needs node + playwright (headless Chromium, WebGL via SwiftShader) and ffmpeg.
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMES = 49
COLORS = {0: "red", 20: "orange", 45: "yellow", 90: "green", 160: "teal", 200: "blue", 260: "purple", 310: "pink"}
QUIZ = [
    ("Which planet is the Red Planet?", ["Venus", "Mars", "Jupiter", "Saturn"], 1),
    ("How many continents are there?", ["5", "6", "7", "8"], 2),
    ("Which animal is the largest on Earth?", ["Elephant", "Blue whale", "Giraffe", "Shark"], 1),
    ("What is the capital of Japan?", ["Seoul", "Osaka", "Kyoto", "Tokyo"], 3),
    ("Which gas do plants absorb?", ["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"], 2),
    ("Who painted the Mona Lisa?", ["Picasso", "Da Vinci", "Van Gogh", "Monet"], 1),
    ("How many sides does a hexagon have?", ["5", "6", "7", "8"], 1),
    ("Which ocean is the biggest?", ["Atlantic", "Indian", "Pacific", "Arctic"], 2),
]
VS_LABELS = ["TEAM A", "TEAM B", "TEAM C", "TEAM D", "PLAYER 1", "PLAYER 2", "RED", "BLUE", "TIGER", "DRAGON"]


def color_name(h):
    return COLORS[min(COLORS, key=lambda k: min(abs(k - h % 360), 360 - abs(k - h % 360)))]


def make_job(k, rnd):
    scene = rnd.choice(["countdown", "quiz", "vs", "geo", "waves", "coins", "tunnel"])
    portrait = rnd.random() < 0.4
    w, h = (480, 720) if portrait else (720, 480)
    hue = rnd.randrange(360)
    cn, shape = color_name(hue), ("vertical" if portrait else "horizontal")
    p = {"seed": rnd.randrange(10**6), "hue": hue}
    if scene == "countdown":
        p["numbers"] = rnd.choice([[3, 2, 1], [5, 4, 3], [10, 9, 8], [3, 2, 1, "GO"]])
        cap = (f"Countdown motion graphic: the numbers {', '.join(map(str, p['numbers']))} pop in one after another inside a "
               f"glowing progress ring with flying sparks, neon {cn} colours on a dark gradient background, static camera, "
               f"fast energetic motion")
    elif scene == "quiz":
        q, opts, c = rnd.choice(QUIZ)
        p.update(question=q, options=opts, correct=c)
        cap = (f"Quiz video motion graphic: the question \"{q}\" appears at the top, four answer boxes A to D slide in from "
               f"the sides, then the correct answer \"{opts[c]}\" lights up green while the others dim and a timer bar drains, "
               f"{cn} tinted gradient background, static camera")
    elif scene == "vs":
        n_panels = rnd.choice([2, 4])
        p.update(panels=n_panels, hues=[rnd.randrange(360) for _ in range(4)], labels=rnd.sample(VS_LABELS, 4))
        cap = (f"Versus split-screen motion graphic: the screen is divided into {n_panels} coloured panels labelled "
               f"{', '.join(p['labels'][:n_panels])} that slide in, a shockwave ring expands and a glowing VS logo pulses "
               f"with yellow sparks, static camera")
    elif scene == "geo":
        p.update(layers=rnd.randint(5, 12), sides=rnd.choice([3, 4, 5, 6, 8]), speed=rnd.choice([0.5, 1, 2]))
        cap = (f"Abstract geometric loop: {p['layers']} concentric glowing {p['sides']}-sided outlines rotate in opposite "
               f"directions with a {cn} to rainbow colour gradient, neon glow on a dark background, static camera, "
               f"{'slow' if p['speed'] < 1 else 'fast' if p['speed'] > 1 else 'medium'} rotation")
    elif scene == "waves":
        p.update(bands=rnd.randint(4, 9), speed=rnd.choice([0.5, 1, 2]))
        cap = (f"Abstract colourful wave background: {p['bands']} smooth flowing wave bands in {cn} and shifting hues drift "
               f"across the screen, seamless loop, static camera, {'slow' if p['speed'] < 1 else 'fast' if p['speed'] > 1 else 'calm'} motion")
    elif scene == "coins":
        p.update(count=rnd.randint(30, 120), gravity=rnd.choice([9, 14, 22]))
        cap = (f"3D coin rain: {p['count']} shiny golden coins fall from above, spinning and bouncing on a dark floor, warm "
               f"directional light with metallic highlights, static camera, "
               f"{'slow floaty' if p['gravity'] < 10 else 'fast heavy' if p['gravity'] > 15 else 'natural'} falling speed")
    else:
        p.update(shape=rnd.choice(["round", "square"]), speed=rnd.choice([0.6, 1, 1.8]), roll=rnd.choice([0, 0.15, 0.4]))
        cap = (f"Endless neon tunnel loop: glowing {'circular' if p['shape'] == 'round' else 'square'} rings in rainbow colours "
               f"fly towards the camera through darkness with fog, {'slight camera roll, ' if p['roll'] else ''}"
               f"{'slow' if p['speed'] < 1 else 'very fast' if p['speed'] > 1.5 else 'fast'} forward motion")
    cap += f", {shape} video, vfx-motion-graphics, animated on ones."
    return {"id": f"synth{k:04d}_{scene}", "scene": scene, "width": w, "height": h, "frames": FRAMES, "params": p}, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir(exist_ok=True)
    rnd = random.Random(args.seed)
    jobs, caps = zip(*[make_job(k, rnd) for k in range(args.n)])
    (out / "jobs.json").write_text(json.dumps(jobs))
    subprocess.run(["node", str(HERE / "synth_motion" / "render.js"), str(out / "jobs.json"), str(out / "frames")], check=True)
    with open(out / "metadata.jsonl", "w") as meta:
        for job, cap in zip(jobs, caps):
            clip = out / "clips" / f"vfx-motion-graphics__{job['id']}.mp4"
            if not clip.exists():
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", str(out / "frames" / job["id"] / "f%04d.png"),
                                "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(clip)], check=True)
            meta.write(json.dumps({"video": f"clips/{clip.name}", "caption": cap, "width": job["width"], "height": job["height"]}) + "\n")
    print(f"{len(jobs)} clips in {out}/clips")


if __name__ == "__main__":
    main()
