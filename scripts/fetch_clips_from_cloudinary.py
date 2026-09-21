#!/usr/bin/env python3
"""
Download all clips uploaded through upload-portal/ (tagged locomotion,
gestures, expressions, secondary-motion in Cloudinary) into data/raw_clips/,
ready for scripts/prepare_dataset.py.

Requires CLOUDINARY_CLOUD_NAME / CLOUDINARY_API_KEY / CLOUDINARY_API_SECRET
in .env (the same values set on the Vercel project).

Usage:
    python scripts/fetch_clips_from_cloudinary.py
"""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw_clips"
TAGS = ["locomotion", "gestures", "expressions", "secondary-motion"]


def main():
    load_dotenv(REPO_ROOT / ".env")
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
    api_key = os.environ.get("CLOUDINARY_API_KEY")
    api_secret = os.environ.get("CLOUDINARY_API_SECRET")
    if not all([cloud_name, api_key, api_secret]):
        print(
            "CLOUDINARY_CLOUD_NAME / CLOUDINARY_API_KEY / CLOUDINARY_API_SECRET must be set in .env "
            "(same values you set in the Vercel project's environment variables).",
            file=sys.stderr,
        )
        sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    auth = (api_key, api_secret)
    total = 0

    for tag in TAGS:
        url = f"https://api.cloudinary.com/v1_1/{cloud_name}/resources/video/tags/{tag}"
        resp = requests.get(url, params={"max_results": 200}, auth=auth, timeout=30)
        resp.raise_for_status()
        resources = resp.json().get("resources", [])
        print(f"[{tag}] {len(resources)} clips found")

        for r in resources:
            name = Path(r["public_id"]).name
            dest = RAW_DIR / f"{tag}__{name}.mp4"
            if dest.exists():
                continue
            video_resp = requests.get(r["secure_url"], timeout=120)
            video_resp.raise_for_status()
            dest.write_bytes(video_resp.content)
            total += 1
            print(f"  downloaded {dest.name}")

    print(f"\nDone. {total} new clips saved to {RAW_DIR}")
    print("Next: python scripts/prepare_dataset.py --task t2v")


if __name__ == "__main__":
    main()
