#!/usr/bin/env python3
"""
Push scripts/kaggle_gradio_generator_entrypoint.py to Kaggle as a GPU-enabled
kernel. Unlike the fully-automated clip generator, this one launches a
Gradio app with a public share link -- open the kernel's page on kaggle.com
(Kaggle streams stdout live there) to find the printed https://xxxxx.gradio.live
URL, then use the UI yourself: pick/type a prompt, hit Generate, watch the
result, download what's good.

Usage:
    python scripts/setup_kaggle_credentials.py
    python scripts/push_gradio_generator_to_kaggle.py
"""
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT_SRC = REPO_ROOT / "scripts" / "kaggle_gradio_generator_entrypoint.py"
STAGING_DIR = REPO_ROOT / "kaggle_kernel_staging_gradio"

KERNEL_SLUG_DEFAULT = "voidvideo-gradio-studio"


def main():
    import os
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    username = os.environ.get("KAGGLE_USERNAME")
    kernel_slug = f"{username}/{KERNEL_SLUG_DEFAULT}" if username else None

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    if not kernel_slug:
        kernel_slug = f"{api.get_config_value('username')}/{KERNEL_SLUG_DEFAULT}"

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir()

    entrypoint_code = ENTRYPOINT_SRC.read_text()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        # Injected only into the staged copy (this whole directory is
        # gitignored and rebuilt fresh on every push) -- never written into
        # the tracked source file in scripts/. Authenticated HF requests
        # avoid the download throttling unauthenticated requests were
        # hitting after repeated same-day restarts.
        entrypoint_code = f'import os as _os; _os.environ["HF_TOKEN"] = {hf_token!r}\n' + entrypoint_code
        print("HF_TOKEN found in .env -- injecting into the staged kernel for faster/authenticated downloads.")
    else:
        print("No HF_TOKEN in .env -- proceeding with unauthenticated (slower, rate-limited) HF downloads.")
    (STAGING_DIR / "kaggle_entrypoint.py").write_text(entrypoint_code)

    kernel_metadata = {
        "id": kernel_slug,
        "title": KERNEL_SLUG_DEFAULT,
        "code_file": "kaggle_entrypoint.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,  # required: pip install, HF model download, Gradio's share tunnel
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (STAGING_DIR / "kernel-metadata.json").write_text(json.dumps(kernel_metadata, indent=2))

    print(f"Pushing kernel {kernel_slug} to Kaggle...")
    api.kernels_push(str(STAGING_DIR))
    print(
        f"Pushed. Kaggle will now build and run it with a GPU.\n"
        f"Open https://www.kaggle.com/code/{kernel_slug} in your browser -- "
        f"Kaggle streams the live output there. Once you see \"Model loaded. "
        f"Launching Gradio...\" followed by a line ending in .gradio.live, "
        f"open that link. Pick a prompt, hit Generate, wait ~8-10 min, watch "
        f"the result, download what's good."
    )


if __name__ == "__main__":
    main()
