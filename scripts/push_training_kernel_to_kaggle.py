#!/usr/bin/env python3
"""
Push training/kaggle_entrypoint.py to Kaggle as a GPU-enabled kernel and
trigger a run. The kernel git-clones this repo at the current branch/commit
at runtime, mounts the dataset pushed by push_dataset_to_kaggle.py, and runs
training/train_lora.py via accelerate.

Usage:
    python scripts/setup_kaggle_credentials.py
    python scripts/push_dataset_to_kaggle.py
    python scripts/push_training_kernel_to_kaggle.py --task t2v
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT_SRC = REPO_ROOT / "training" / "kaggle_entrypoint.py"
STAGING_DIR = REPO_ROOT / "kaggle_kernel_staging"


def current_git_ref() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "main"


def current_git_remote_url() -> str:
    url = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:"):]
    if not url.endswith(".git"):
        url += ".git"
    return url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--ref", default=None,
                         help="Git branch/tag/commit the Kaggle kernel should clone. Defaults to current branch.")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    kernel_slug = os.environ.get("KAGGLE_KERNEL_SLUG")
    dataset_slug = os.environ.get("KAGGLE_DATASET_SLUG")
    if not kernel_slug or not dataset_slug:
        print("KAGGLE_KERNEL_SLUG and KAGGLE_DATASET_SLUG must be set in .env.", file=sys.stderr)
        sys.exit(1)

    ref = args.ref or current_git_ref()
    remote_url = current_git_remote_url()

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir()

    entrypoint_code = ENTRYPOINT_SRC.read_text()
    entrypoint_code = entrypoint_code.replace(
        'REPO_GIT_URL = "https://github.com/thanoxxinfnity/Voidvideo.git"  # rewritten at push time',
        f'REPO_GIT_URL = "{remote_url}"  # rewritten at push time',
    ).replace(
        'REPO_GIT_REF = "claude/voidvideo-anime-model-bl0lor"  # rewritten at push time',
        f'REPO_GIT_REF = "{ref}"  # rewritten at push time',
    ).replace(
        'TRAIN_TASK = "t2v"  # rewritten at push time; "t2v" or "i2v"',
        f'TRAIN_TASK = "{args.task}"  # rewritten at push time; "t2v" or "i2v"',
    )
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        # Same pattern as push_gradio_generator_to_kaggle.py: injected only
        # into the staged (gitignored) copy, never the tracked source.
        # Unauthenticated HF requests are slow/rate-limited and this kernel
        # downloads T5-XXL + the CogVideoX transformer + VAE on every run.
        entrypoint_code = f'import os as _os; _os.environ["HF_TOKEN"] = {hf_token!r}\n' + entrypoint_code
        print("HF_TOKEN found in .env -- injecting into the staged kernel for faster/authenticated downloads.")
    else:
        print("No HF_TOKEN in .env -- proceeding with unauthenticated (slower, rate-limited) HF downloads.")
    (STAGING_DIR / "kaggle_entrypoint.py").write_text(entrypoint_code)

    # Kaggle's kernel-metadata.json schema has no field for arbitrary env vars,
    # so task selection is baked into the entrypoint source above instead.
    kernel_metadata = {
        "id": kernel_slug,
        "title": kernel_slug.split("/")[-1],
        "code_file": "kaggle_entrypoint.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,  # required for pip install + git clone
        "dataset_sources": [dataset_slug],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (STAGING_DIR / "kernel-metadata.json").write_text(json.dumps(kernel_metadata, indent=2))

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Pushing kernel {kernel_slug} (task={args.task}, ref={ref}) to Kaggle...")
    api.kernels_push(str(STAGING_DIR))
    print(
        f"Pushed. Kaggle will now build and run the kernel with a GPU.\n"
        f"Track progress at https://www.kaggle.com/code/{kernel_slug}\n"
        f"Poll status / download output with:\n"
        f"  python scripts/fetch_trained_weights.py --wait"
    )


if __name__ == "__main__":
    main()
