#!/usr/bin/env python3
"""
Prepare raw anime clips for LoRA training.

Reads every video in `data/raw_clips/`, validates/normalizes it to the target
spec (resolution, fps, frame count) defined in configs/training_config.yaml,
writes the processed clip to `data/processed/clips/`, and builds
`data/processed/metadata.jsonl` with one {"video": ..., "caption": ...} row
per clip.

Captions: if `data/captions/<clip_stem>.txt` exists, it is used verbatim.
Otherwise, when --auto-caption is passed, BLIP-2 generates a caption from a
sampled frame (rough starting point only — hand-written captions give much
better LoRA steerability, especially for describing motion).

Usage:
    python scripts/prepare_dataset.py --task t2v
    python scripts/prepare_dataset.py --task t2v --auto-caption
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw_clips"
CAPTIONS_DIR = REPO_ROOT / "data" / "captions"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CLIPS_OUT_DIR = PROCESSED_DIR / "clips"
METADATA_PATH = PROCESSED_DIR / "metadata.jsonl"
CONFIG_PATH = REPO_ROOT / "configs" / "training_config.yaml"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def load_task_config(task: str) -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if task not in cfg:
        raise ValueError(f"Unknown task '{task}', expected one of {list(cfg.keys())}")
    return cfg[task]


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def normalize_clip(src: Path, dst: Path, width: int, height: int, fps: int, num_frames: int) -> None:
    """
    Resize with center-crop to exactly (width, height), resample to `fps`,
    and trim/pad to exactly `num_frames` frames using ffmpeg.
    """
    target_duration = num_frames / fps
    src_duration = ffprobe_duration(src)

    # Center-crop-then-scale filter: crop to target aspect ratio first, then scale.
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"fps={fps}"
    )

    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-t", str(min(target_duration, src_duration)),
        "-vf", vf,
        "-frames:v", str(num_frames),
        "-an",  # strip audio — silent animation pipeline
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def get_caption(stem: str, auto_caption: bool, blip=None) -> str:
    manual = CAPTIONS_DIR / f"{stem}.txt"
    if manual.exists():
        return manual.read_text().strip()
    if auto_caption and blip is not None:
        return blip(CLIPS_OUT_DIR / f"{stem}.mp4")
    return ""  # empty captions are valid — trains unconditional/style-only signal


def build_blip_captioner():
    """Lazy import so prepare_dataset.py works without transformers/torch
    when captions are provided manually."""
    import cv2
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    def caption_video(video_path: Path) -> str:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return ""
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, return_tensors="pt").to(device, model.dtype)
        out = model.generate(**inputs, max_new_tokens=40)
        text = processor.decode(out[0], skip_special_tokens=True).strip()
        return f"hand-drawn anime, traditional 2D animation, {text}"

    return caption_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["t2v", "i2v"], default="t2v",
                         help="Which config block (resolution/frame target) to normalize clips against.")
    parser.add_argument("--auto-caption", action="store_true",
                         help="Generate captions with BLIP-2 for clips with no manual caption file.")
    args = parser.parse_args()

    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(f"No clips found in {RAW_DIR}. Drop your .mp4 files there first.", file=sys.stderr)
        sys.exit(1)

    task_cfg = load_task_config(args.task)
    width, height = task_cfg["width"], task_cfg["height"]
    fps, num_frames = task_cfg["fps"], task_cfg["num_frames"]

    CLIPS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    clip_paths = sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    print(f"Found {len(clip_paths)} raw clips. Normalizing to {width}x{height} @ {fps}fps, {num_frames} frames...")

    blip = build_blip_captioner() if args.auto_caption else None

    rows = []
    skipped = []
    for src in clip_paths:
        stem = src.stem
        dst = CLIPS_OUT_DIR / f"{stem}.mp4"
        try:
            normalize_clip(src, dst, width, height, fps, num_frames)
        except subprocess.CalledProcessError as e:
            skipped.append((src.name, e.stderr.decode(errors="ignore")[-300:]))
            continue

        caption = get_caption(stem, args.auto_caption, blip)
        rows.append({"video": f"clips/{dst.name}", "caption": caption})
        print(f"  ok   {src.name} -> {dst.name}  caption={'yes' if caption else 'EMPTY'}")

    with open(METADATA_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(rows)} clips to {CLIPS_OUT_DIR}")
    print(f"Wrote metadata to {METADATA_PATH}")
    if skipped:
        print(f"\n{len(skipped)} clips failed ffmpeg processing:")
        for name, err in skipped:
            print(f"  - {name}: {err.strip().splitlines()[-1] if err.strip() else 'unknown error'}")

    no_caption = sum(1 for r in rows if not r["caption"])
    if no_caption and not args.auto_caption:
        print(
            f"\n{no_caption} clips have no caption. Either add data/captions/<clip_stem>.txt files "
            "or re-run with --auto-caption for a BLIP-2 first pass."
        )


if __name__ == "__main__":
    main()
