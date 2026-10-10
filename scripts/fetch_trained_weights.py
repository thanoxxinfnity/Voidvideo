#!/usr/bin/env python3
"""
Poll a Kaggle training kernel until it finishes, then download its output
(LoRA checkpoints under outputs/) into models/.

Usage:
    python scripts/fetch_trained_weights.py --wait
    python scripts/fetch_trained_weights.py            # single status check, no waiting
"""
import argparse
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

TERMINAL_STATUSES = {"complete", "error", "cancelAcknowledged"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true", help="Poll until the kernel finishes instead of checking once.")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between status checks when --wait.")
    parser.add_argument("--timeout", type=int, default=6 * 3600, help="Max seconds to wait before giving up.")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    kernel_slug = os.environ.get("KAGGLE_KERNEL_SLUG")
    if not kernel_slug:
        print("KAGGLE_KERNEL_SLUG not set in .env.", file=sys.stderr)
        sys.exit(1)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    start = time.time()
    while True:
        # kernels_status() returns an ApiGetKernelSessionStatusResponse object
        # (a KernelWorkerStatus enum under .status), not a dict -- .get(...)
        # raises AttributeError and was silently never reached before.
        status_resp = api.kernels_status(kernel_slug)
        status = status_resp.status.name
        print(f"Kernel {kernel_slug} status: {status}")

        if status == "COMPLETE":
            break
        if status == "CANCEL_ACKNOWLEDGED":
            # Expected outcome, not a failure: Kaggle's 12h hard session limit
            # kills long training runs before they reach max_train_steps, but
            # checkpointing_steps still saved intermediate checkpoints worth
            # downloading and resuming from (see push_checkpoint_to_kaggle.py).
            print(f"Kernel was cancelled (likely the 12h session limit), not finished cleanly. "
                  f"Downloading anyway to check for a usable checkpoint.")
            break
        if status == "ERROR":
            print(f"Kernel errored out (status={status}). Check logs at "
                  f"https://www.kaggle.com/code/{kernel_slug}", file=sys.stderr)
            sys.exit(1)
        if not args.wait:
            print("Not complete yet. Re-run with --wait to poll, or check again later.")
            return
        if time.time() - start > args.timeout:
            print("Timed out waiting for kernel to finish.", file=sys.stderr)
            sys.exit(1)

        time.sleep(args.poll_interval)

    MODELS_DIR.mkdir(exist_ok=True)
    download_dir = MODELS_DIR / "_kaggle_output_tmp"
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir()

    print("Downloading kernel output...")
    api.kernels_output(kernel_slug, path=str(download_dir))

    for zpath in download_dir.glob("*.zip"):
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(download_dir)
        zpath.unlink()

    # kaggle_entrypoint.py mirrors REPO_DIR/outputs to WORKING_DIR/outputs
    # (a top-level "outputs") only after training finishes normally -- a run
    # killed mid-training by the 12h session limit never reaches that step,
    # so the real checkpoints are still wherever training wrote them:
    # nested at <cloned-repo-dir>/outputs/. Search for either.
    # A session can also end with only a latent cache and no checkpoint yet
    # (long encode phase), so a dir with latents alone counts too.
    candidates = [
        d for d in download_dir.rglob("outputs")
        if d.is_dir() and (any(d.glob("lora_*/*/lora_weights.pt")) or any(d.glob("latent_cache/*.pt")))
    ]
    src_outputs = min(candidates, key=lambda d: len(d.parts)) if candidates else download_dir / "outputs"
    if src_outputs.exists():
        for item in src_outputs.iterdir():
            dest = MODELS_DIR / item.name
            if item.name == "latent_cache" and item.is_dir():
                # Merge: keep latents from earlier sessions, add the new ones.
                dest.mkdir(exist_ok=True)
                for f in item.glob("*.pt"):
                    shutil.move(str(f), str(dest / f.name))
                continue
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(item), str(dest))
        print(f"Moved trained checkpoints into {MODELS_DIR}")
    else:
        print(
            f"No 'outputs/' directory found in kernel output. Raw output left at {download_dir} for inspection.",
            file=sys.stderr,
        )
        return

    shutil.rmtree(download_dir, ignore_errors=True)
    print("Done. Point app/gradio_app.py at the new checkpoint under models/.")


if __name__ == "__main__":
    main()
