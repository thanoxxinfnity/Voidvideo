#!/usr/bin/env python3
"""
Push a LoRA checkpoint to Kaggle as a private dataset, so the NEXT training
kernel run can attach it and resume instead of starting over.

Why this exists: Kaggle kernels have a 12-hour hard session limit, and a
single CogVideoX-2B LoRA step on a T4 (gradient checkpointing, native
49-frame/480x720, fp16) took ~200s in practice -- only ~150-200 steps fit in
one session. Training needs several relaunches to reach max_train_steps, and
without this relay each relaunch would start from scratch, discarding every
prior run's progress.

Usage (after a training kernel run has produced a checkpoint):
    python scripts/fetch_trained_weights.py          # pulls kernel output locally
    python scripts/push_checkpoint_to_kaggle.py --task t2v

Then scripts/push_training_kernel_to_kaggle.py will attach the checkpoint
dataset automatically on the next push (if KAGGLE_CHECKPOINT_DATASET_SLUG is
set in .env), and kaggle_entrypoint.py resumes from it.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


def latest_checkpoint_dir(output_dir: Path) -> Path:
    final = output_dir / "final"
    if final.is_dir() and (final / "lora_weights.pt").exists():
        return final
    candidates = [
        d for d in output_dir.glob("checkpoint-*")
        if d.is_dir() and (d / "lora_weights.pt").exists()
    ]
    if not candidates:
        raise RuntimeError(
            f"No checkpoint with lora_weights.pt found under {output_dir}. "
            "Run scripts/fetch_trained_weights.py first."
        )
    return max(candidates, key=lambda d: int(d.name.split("-")[-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["t2v", "i2v"], required=True)
    parser.add_argument("--message", default="Update LoRA checkpoint for resume")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    slug = os.environ.get("KAGGLE_CHECKPOINT_DATASET_SLUG")
    if not slug:
        print("KAGGLE_CHECKPOINT_DATASET_SLUG not set in .env.", file=sys.stderr)
        sys.exit(1)

    # fetch_trained_weights.py moves the kernel's outputs/lora_<task>/ into
    # models/lora_<task>/ locally -- match that, not outputs/.
    output_dir = REPO_ROOT / f"models/lora_{args.task}"
    if not output_dir.exists():
        print(f"{output_dir} not found. Run scripts/fetch_trained_weights.py first.", file=sys.stderr)
        sys.exit(1)

    ckpt_dir = latest_checkpoint_dir(output_dir)
    print(f"Using checkpoint: {ckpt_dir}")

    staging = REPO_ROOT / "kaggle_checkpoint_staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(ckpt_dir, staging)

    dataset_metadata = {
        "title": slug.split("/")[-1],
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2))

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    try:
        api.dataset_status(slug)
        exists = True
    except Exception:
        exists = False

    if exists:
        print(f"Dataset {slug} exists, pushing new version...")
        api.dataset_create_version(folder=str(staging), version_notes=args.message, dir_mode="zip")
    else:
        print(f"Dataset {slug} does not exist yet, creating it...")
        api.dataset_create_new(folder=str(staging), public=False, dir_mode="zip")

    print(f"Done. View at https://www.kaggle.com/datasets/{slug}")


if __name__ == "__main__":
    main()
