#!/usr/bin/env python3
"""
Push the processed dataset (data/processed/) to Kaggle as a private dataset,
creating it on first run and pushing a new version on subsequent runs.

Requires scripts/setup_kaggle_credentials.py to have been run first.

Usage:
    python scripts/setup_kaggle_credentials.py
    python scripts/prepare_dataset.py --task t2v
    python scripts/push_dataset_to_kaggle.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--message", default="Update VoidVideo training clips",
        help="Version notes for this dataset push.",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    slug = os.environ.get("KAGGLE_DATASET_SLUG")
    if not slug:
        print("KAGGLE_DATASET_SLUG not set in .env (format: <username>/<dataset-name>).", file=sys.stderr)
        sys.exit(1)

    if not (PROCESSED_DIR / "metadata.jsonl").exists():
        print(
            f"{PROCESSED_DIR}/metadata.jsonl not found. Run scripts/prepare_dataset.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import here so this script's --help works without Kaggle creds configured.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dataset_metadata = {
        "title": slug.split("/")[-1],
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (PROCESSED_DIR / "dataset-metadata.json").write_text(json.dumps(dataset_metadata, indent=2))

    owner, name = slug.split("/")
    exists = False
    try:
        api.dataset_status(slug)
        exists = True
    except Exception:
        exists = False

    if exists:
        print(f"Dataset {slug} exists, pushing new version...")
        api.dataset_create_version(
            folder=str(PROCESSED_DIR),
            version_notes=args.message,
            dir_mode="zip",
        )
    else:
        print(f"Dataset {slug} does not exist yet, creating it...")
        api.dataset_create_new(
            folder=str(PROCESSED_DIR),
            public=False,
            dir_mode="zip",
        )

    print(f"Done. View at https://www.kaggle.com/datasets/{slug}")


if __name__ == "__main__":
    main()
