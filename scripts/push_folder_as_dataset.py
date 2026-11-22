#!/usr/bin/env python3
"""Push a local folder to Kaggle as a private dataset (create it, or add a new version if it exists).

    python scripts/push_folder_as_dataset.py <folder> <owner/slug> [--message "..."]

Used for the Wan relay: the LoRA checkpoint dir and the latent cache dir each go up as their own dataset.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("slug")
    ap.add_argument("--message", default="Update")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    # Stage with symlink-free copies of regular files only, plus the metadata file Kaggle needs.
    staging = Path(tempfile.mkdtemp(prefix="kaggle_ds_"))
    for f in Path(args.folder).iterdir():
        if f.is_file():
            shutil.copy(f, staging / f.name)
    (staging / "dataset-metadata.json").write_text(json.dumps({
        "title": args.slug.split("/")[-1], "id": args.slug, "licenses": [{"name": "CC0-1.0"}]}, indent=2))

    try:
        api.dataset_status(args.slug)
        exists = True
    except Exception:
        exists = False
    if exists:
        api.dataset_create_version(folder=str(staging), version_notes=args.message, dir_mode="skip")
    else:
        api.dataset_create_new(folder=str(staging), public=False, dir_mode="skip")
    shutil.rmtree(staging, ignore_errors=True)
    print(f"Done: https://www.kaggle.com/datasets/{args.slug}")


if __name__ == "__main__":
    main()
