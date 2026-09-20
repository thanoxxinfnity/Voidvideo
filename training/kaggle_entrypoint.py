#!/usr/bin/env python3
"""
Entrypoint that runs INSIDE the Kaggle kernel/notebook environment.

This script is pushed to Kaggle by scripts/push_training_kernel_to_kaggle.py
as the kernel's source file. On Kaggle it:
  1. installs this repo's pinned dependencies,
  2. locates the mounted training-clips dataset under /kaggle/input/,
  3. symlinks it to data/processed/ so training/train_lora.py finds it,
  4. runs LoRA training via `accelerate launch`,
  5. leaves the resulting checkpoints in /kaggle/working/outputs/ so they
     appear in the kernel's Output tab for scripts/fetch_trained_weights.py
     to download afterwards.

Kaggle notebooks execute with /kaggle/working as the CWD and persist
everything under it as kernel output automatically -- no extra upload step
needed on the Kaggle side.

This file is pushed to Kaggle AS-IS as the kernel's single source file (Kaggle
script kernels only accept one code file via the API), so it clones the repo
from GitHub at runtime rather than relying on bundled sibling files. The two
constants below are rewritten by scripts/push_training_kernel_to_kaggle.py
immediately before staging, so they always match the branch you're pushing
from -- editing them here has no effect on the next push.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_GIT_URL = "https://github.com/thanoxxinfnity/Voidvideo.git"  # rewritten at push time
REPO_GIT_REF = "claude/voidvideo-anime-model-bl0lor"  # rewritten at push time
TRAIN_TASK = "t2v"  # rewritten at push time; "t2v" or "i2v"

KAGGLE_INPUT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
REPO_DIR = WORKING_DIR / "voidvideo"


def run(cmd: list[str], **kwargs) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def find_dataset_dir() -> Path:
    candidates = [p for p in KAGGLE_INPUT.iterdir() if p.is_dir()] if KAGGLE_INPUT.exists() else []
    if not candidates:
        raise RuntimeError(
            "No dataset mounted under /kaggle/input. Attach the dataset pushed by "
            "scripts/push_dataset_to_kaggle.py as a data source on this kernel."
        )
    if len(candidates) > 1:
        print(f"Multiple datasets mounted, using the first: {[c.name for c in candidates]}")
    return candidates[0]


def main():
    task = os.environ.get("TRAIN_TASK", TRAIN_TASK)
    assert task in ("t2v", "i2v"), f"Unknown TRAIN_TASK={task}"

    if not REPO_DIR.exists():
        run(["git", "clone", "--branch", REPO_GIT_REF, "--depth", "1", REPO_GIT_URL, str(REPO_DIR)])
    os.chdir(REPO_DIR)
    sys.path.insert(0, str(REPO_DIR))

    run(["pip", "install", "-q", "-r", "requirements.txt"])

    dataset_dir = find_dataset_dir()
    processed_link = REPO_DIR / "data" / "processed"
    if processed_link.exists() or processed_link.is_symlink():
        processed_link.unlink() if processed_link.is_symlink() else None
    processed_link.parent.mkdir(parents=True, exist_ok=True)
    if not processed_link.exists():
        processed_link.symlink_to(dataset_dir)
    print(f"Linked dataset {dataset_dir} -> {processed_link}")

    num_gpus = int(subprocess.run(
        ["python", "-c", "import torch;print(torch.cuda.device_count())"],
        capture_output=True, text=True,
    ).stdout.strip() or "0")
    print(f"Detected {num_gpus} GPU(s).")

    run([
        "accelerate", "launch",
        "--num_processes", str(max(1, num_gpus)),
        "--mixed_precision", "bf16",
        "training/train_lora.py",
        "--task", task,
        "--config", "configs/training_config.yaml",
    ])

    # Mirror outputs to /kaggle/working root so they show up in the kernel's Output tab
    # even if REPO_DIR itself isn't surfaced.
    src = REPO_DIR / "outputs"
    dst = WORKING_DIR / "outputs"
    if src.exists() and src.resolve() != dst.resolve():
        run(["cp", "-r", str(src), str(dst)])

    print("Training complete. Checkpoints are in /kaggle/working/outputs/")


if __name__ == "__main__":
    main()
