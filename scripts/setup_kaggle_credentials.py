#!/usr/bin/env python3
"""
Write ~/.kaggle/kaggle.json from KAGGLE_USERNAME / KAGGLE_KEY in the
environment (or a .env file), with correct 0600 permissions. The Kaggle
API client refuses to run without this file in place.

This script never prints or logs the key, and never writes it anywhere
inside the repo.

Usage:
    python scripts/setup_kaggle_credentials.py
"""
import json
import os
import stat
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    load_dotenv(REPO_ROOT / ".env")

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        print(
            "KAGGLE_USERNAME / KAGGLE_KEY not set. Copy .env.example to .env "
            "and fill in your Kaggle API token (from kaggle.com/settings).",
            file=sys.stderr,
        )
        sys.exit(1)

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(mode=0o700, exist_ok=True)
    creds_path = kaggle_dir / "kaggle.json"

    creds_path.write_text(json.dumps({"username": username, "key": key}))
    creds_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600

    print(f"Wrote Kaggle credentials to {creds_path} (0600).")


if __name__ == "__main__":
    main()
