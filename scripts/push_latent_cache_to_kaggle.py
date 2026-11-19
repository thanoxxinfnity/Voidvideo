#!/usr/bin/env python3
"""
Push the VAE latent cache to Kaggle as a private dataset so the next training
kernel run can attach it and skip re-encoding clips.

Encoding one clip through the fp32 tiled VAE takes ~70s on a T4, so a full
dataset pass can consume a whole 12h session. train_lora.py writes every
encoded latent to outputs/latent_cache/ and reads any previously pushed cache
from the attached dataset; this script is the relay for that cache, the same
way push_checkpoint_to_kaggle.py is for LoRA checkpoints.

Usage (after scripts/fetch_trained_weights.py has moved the kernel output):
    python scripts/push_latent_cache_to_kaggle.py
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SLUG_SUFFIX = "voidvideo-latent-cache-t2v"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", default="Update latent cache")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    slug = os.environ.get("KAGGLE_LATENT_CACHE_DATASET_SLUG")
    if not slug:
        user = os.environ.get("KAGGLE_USERNAME")
        if not user:
            print("Set KAGGLE_LATENT_CACHE_DATASET_SLUG or KAGGLE_USERNAME in .env.", file=sys.stderr)
            sys.exit(1)
        slug = f"{user}/{DEFAULT_SLUG_SUFFIX}"

    cache_dir = REPO_ROOT / "models" / "latent_cache"
    files = sorted(cache_dir.glob("*.pt")) if cache_dir.exists() else []
    if not files:
        print(f"No latents under {cache_dir}; nothing to push.")
        return
    print(f"Pushing {len(files)} cached latents to {slug}")

    staging = REPO_ROOT / "kaggle_latent_cache_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for f in files:
        shutil.copy2(f, staging / f.name)
    (staging / "dataset-metadata.json").write_text(json.dumps({
        "title": slug.split("/")[-1], "id": slug, "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    try:
        api.dataset_status(slug)
        exists = True
    except Exception:
        exists = False

    if exists:
        print("Dataset exists, pushing new version...")
        api.dataset_create_version(folder=str(staging), version_notes=args.message, dir_mode="zip")
    else:
        print("Dataset does not exist yet, creating it (private)...")
        api.dataset_create_new(folder=str(staging), public=False, dir_mode="zip")
    shutil.rmtree(staging, ignore_errors=True)
    print(f"Done. View at https://www.kaggle.com/datasets/{slug}")


if __name__ == "__main__":
    main()
