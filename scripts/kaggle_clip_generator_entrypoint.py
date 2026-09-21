#!/usr/bin/env python3
"""
Runs INSIDE a Kaggle GPU kernel. Generates short anime-style clips with the
base CogVideoX-2B text-to-video model (the same model this project trains a
LoRA on) and uploads each one straight to the VoidVideo Vault site
(voidvideo-vault.vercel.app) as it finishes, tagged with the category the
prompt was written for and captioned from that same prompt.

Uploads happen one clip at a time as soon as each is generated, so a kernel
that gets cut off by Kaggle's session time limit still keeps everything
generated up to that point -- nothing is lost by stopping early.

This is a stopgap for padding out under-filled categories when real footage
is running low on time, not a replacement for real hand-drawn reference
clips: CogVideoX-2b here is the un-tuned base model, so output quality and
anime-style adherence will be inconsistent. Treat generated clips as filler,
review them before training on them if you can.

Pushed to Kaggle by scripts/push_clip_generator_to_kaggle.py.
"""
import random
import subprocess
import sys
import time
from pathlib import Path


def pip_install():
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "diffusers>=0.30.0", "transformers>=4.44", "accelerate>=0.33",
            "imageio[ffmpeg]", "sentencepiece", "requests",
        ],
        check=True,
    )


pip_install()

import numpy as np  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
from diffusers import CogVideoXPipeline  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402

SITE = "https://voidvideo-vault.vercel.app"
MODEL_ID = "THUDM/CogVideoX-2b"
NUM_FRAMES = 49  # matches configs/training_config.yaml (t2v) exactly
FPS = 8
HEIGHT = 480
WIDTH = 720
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 6.0
MAX_RETRIES_PER_PROMPT = 2  # re-roll with a new seed if output looks blank
MAX_SECONDS = 5.5 * 3600  # user needs quota left over for the real LoRA training run today

OUT_DIR = Path("/kaggle/working/generated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STYLE_SUFFIX = (
    ", traditional 2D hand-drawn anime animation, natural line-art, "
    "smooth timing, cel shading, consistent character design"
)

# Base prompts per category, matching the motion sub-types in data/README.md.
# Repeated N times below to weight generation toward whichever categories
# were furthest from their target when this was written -- edit the weights
# in build_queue() if the site's real gaps have shifted since.
PROMPTS = {
    "locomotion": [
        "a hand-drawn anime boy walking briskly down a city sidewalk",
        "a hand-drawn anime girl running through a park",
        "a hand-drawn anime character idly standing and breathing calmly",
        "a hand-drawn anime boy turning around to look behind him",
        "a hand-drawn anime girl walking, filmed from the side profile",
        "a hand-drawn anime character jogging along a beach at sunset",
        "a hand-drawn anime boy walking up a staircase",
        "a hand-drawn anime boy in a fighting stance, dodging and weaving side to side",
        "a hand-drawn anime girl sprinting forward then leaping into a spinning kick",
        "a hand-drawn anime character charging forward aggressively into battle",
        "a hand-drawn anime boy dashing sideways fast to dodge an incoming attack",
    ],
    "gestures": [
        "a hand-drawn anime girl waving her hand cheerfully at the camera",
        "a hand-drawn anime boy pointing forward with his finger",
        "a hand-drawn anime character reaching out to pick up a cup from a table",
        "a hand-drawn anime girl opening a wooden door",
        "a hand-drawn anime boy checking his phone and typing",
        "a hand-drawn anime character picking up a book from a shelf",
        "a hand-drawn anime boy throwing a powerful punch forward",
        "a hand-drawn anime girl kicking high with a fierce battle cry",
        "a hand-drawn anime character blocking an incoming attack with crossed arms",
        "a hand-drawn anime boy drawing a sword from its sheath",
        "a hand-drawn anime character clenching a fist in preparation to strike",
    ],
    "expressions": [
        "a hand-drawn anime girl blinking slowly in a close-up shot",
        "a hand-drawn anime boy smiling warmly in a close-up shot",
        "a hand-drawn anime character frowning sadly in a close-up shot",
        "a hand-drawn anime girl looking surprised with wide eyes in a close-up shot",
        "a hand-drawn anime boy talking animatedly in a close-up shot",
        "a hand-drawn anime character looking angry with furrowed eyebrows in a close-up shot",
        "a hand-drawn anime character shouting furiously with gritted teeth in a close-up shot",
        "a hand-drawn anime boy glaring with intense rage in a close-up shot",
        "a hand-drawn anime girl smirking confidently before a fight in a close-up shot",
        "a hand-drawn anime character screaming in battle fury in a close-up shot",
    ],
    # A character is kept visible in every one of these (per user request) --
    # the ambient element is still the point of the shot, the character is
    # just present in frame rather than the clip being empty of people.
    "secondary-motion": [
        "a hand-drawn anime girl standing quietly as autumn leaves fall gently around her along a forest path",
        "a hand-drawn anime girl's long hair swaying gently in the wind, her face visible in frame",
        "a hand-drawn anime character standing by an open window as the curtains flutter beside them",
        "a hand-drawn anime character sitting by a calm pond as the water ripples gently near them",
        "a hand-drawn anime character's coat and scarf blowing in a strong wind as they stand still",
        "a hand-drawn anime character standing amid sparks and embers flying from a nearby sword fight",
        "a hand-drawn anime character standing as dust and debris swirl around them from a powerful impact",
    ],
}


def build_queue():
    # Weighted toward categories that were furthest behind target at write
    # time (locomotion 9/50, gestures 8/30, expressions 15/40,
    # secondary-motion 2/20) -- secondary-motion and gestures get more copies.
    weights = {"secondary-motion": 4, "gestures": 3, "locomotion": 2, "expressions": 2}
    queue = []
    for cat, w in weights.items():
        for _ in range(w):
            queue.extend((cat, p) for p in PROMPTS[cat])
    random.shuffle(queue)
    return queue


def caption_from_prompt(base_prompt: str) -> str:
    c = base_prompt.strip()
    if not c.endswith("."):
        c += "."
    return c[0].upper() + c[1:]


def upload_clip(path: Path, category: str, caption: str) -> dict:
    sign_res = requests.post(f"{SITE}/api/sign", json={"tag": category, "caption": caption}, timeout=30)
    sign_res.raise_for_status()
    sd = sign_res.json()

    with open(path, "rb") as f:
        files = {"file": f}
        data = {
            "api_key": sd["apiKey"],
            "timestamp": sd["timestamp"],
            "signature": sd["signature"],
            "folder": sd["folder"],
            "tags": sd["tags"],
        }
        if sd.get("context"):
            data["context"] = sd["context"]
        upload_res = requests.post(
            f"https://api.cloudinary.com/v1_1/{sd['cloudName']}/video/upload",
            data=data, files=files, timeout=180,
        )
    upload_res.raise_for_status()
    return upload_res.json()


def frame_looks_blank(frames) -> bool:
    """CogVideoX-2b in fp16 has a known failure mode on older GPUs (e.g.
    Kaggle's T4) where the VAE decode produces near-uniform white/gray
    frames instead of an actual image -- silent garbage, not an exception.
    Catch it by checking pixel variance on a couple of sample frames."""
    for idx in (len(frames) // 2, len(frames) - 1):
        arr = np.asarray(frames[idx], dtype=np.float32)
        if arr.std() < 8.0:  # a real rendered frame has far more spread than this
            return True
    return False


def main():
    print(f"Loading {MODEL_ID} (this can take a few minutes on first run)...")
    pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    # Known mitigation for CogVideoX-2b producing blank/white output on
    # T4-class GPUs: fp16 VAE decode overflows to NaN/white. Keeping the VAE
    # in float32 while the transformer stays fp16 avoids it.
    pipe.vae.to(dtype=torch.float32)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    queue = build_queue()
    print(f"Queue built: {len(queue)} clips planned across {len(PROMPTS)} categories")

    done = 0
    failed = 0
    blanked = 0
    start = time.time()

    for i, (category, base_prompt) in enumerate(queue):
        elapsed = time.time() - start
        if elapsed > MAX_SECONDS:
            print(f"\nApproaching the session time limit ({elapsed/3600:.1f}h elapsed), stopping here.")
            break

        prompt = base_prompt + STYLE_SUFFIX
        print(f"\n[{i + 1}/{len(queue)}] category={category} elapsed={elapsed/60:.1f}min")
        print(f"  prompt: {prompt}")

        out_path = OUT_DIR / f"clip_{i:04d}.mp4"
        try:
            frames = None
            for attempt in range(1, MAX_RETRIES_PER_PROMPT + 2):
                seed = random.randint(0, 2**31 - 1)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                candidate = pipe(
                    prompt=prompt,
                    num_frames=NUM_FRAMES,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                ).frames[0]

                if not frame_looks_blank(candidate):
                    frames = candidate
                    break
                print(f"  attempt {attempt} came out blank (seed={seed}), retrying..." if attempt <= MAX_RETRIES_PER_PROMPT
                      else f"  attempt {attempt} came out blank (seed={seed}), giving up on this prompt.")

            if frames is None:
                blanked += 1
                continue

            export_to_video(frames, str(out_path), fps=FPS)

            caption = caption_from_prompt(base_prompt)
            result = upload_clip(out_path, category, caption)
            print(f"  uploaded: {result.get('public_id')}")
            done += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
        finally:
            if out_path.exists():
                out_path.unlink()
            torch.cuda.empty_cache()

    total_min = (time.time() - start) / 60
    print(f"\n=== DONE: {done} uploaded, {blanked} blanked (skipped), {failed} failed, {total_min:.1f} min elapsed ===")


if __name__ == "__main__":
    main()
