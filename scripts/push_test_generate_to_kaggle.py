#!/usr/bin/env python3
"""
Push scripts/kaggle_test_generate_entrypoint.py as a GPU kernel that generates one
clip with the trained T2V LoRA (and the same seed without it) for a visual check.
Fetch the clips afterwards:  kaggle kernels output <user>/voidvideo-test-generate -p <dir>

Usage:
    python scripts/push_test_generate_to_kaggle.py
"""
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT_SRC = REPO_ROOT / "scripts" / "kaggle_test_generate_entrypoint.py"
STAGING_DIR = REPO_ROOT / "kaggle_kernel_staging_testgen"
KERNEL_SLUG_DEFAULT = "voidvideo-test-generate"


def main():
    load_dotenv(REPO_ROOT / ".env")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    username = os.environ.get("KAGGLE_USERNAME") or api.get_config_value("username")
    kernel_slug = f"{username}/{KERNEL_SLUG_DEFAULT}"
    ckpt_slug = os.environ.get("KAGGLE_CHECKPOINT_DATASET_SLUG") or f"{username}/voidvideo-lora-checkpoint-t2v"

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir()
    code = ENTRYPOINT_SRC.read_text()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        code = f'import os as _os; _os.environ["HF_TOKEN"] = {hf_token!r}\n' + code
    (STAGING_DIR / "kaggle_entrypoint.py").write_text(code)
    (STAGING_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_slug,
        "title": KERNEL_SLUG_DEFAULT,
        "code_file": "kaggle_entrypoint.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [ckpt_slug],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))

    print(f"Pushing kernel {kernel_slug} (checkpoint dataset {ckpt_slug}) to Kaggle...")
    resp = api.kernels_push(str(STAGING_DIR))
    err = getattr(resp, "error", None)
    if err:
        raise SystemExit(f"Push rejected: {err}")
    print(f"Pushed. https://www.kaggle.com/code/{kernel_slug}")


if __name__ == "__main__":
    main()
