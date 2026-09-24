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
import shutil
import subprocess
import sys
import zipfile
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
    # Kaggle now mounts attached datasets nested as
    # /kaggle/input/datasets/<owner>/<slug>/ rather than /kaggle/input/<slug>/,
    # so the top-level child is just a parent folder with no metadata.jsonl
    # in it. Search for the file itself instead of assuming a depth.
    matches = sorted(KAGGLE_INPUT.rglob("metadata.jsonl")) if KAGGLE_INPUT.exists() else []
    if not matches:
        raise RuntimeError(
            "No metadata.jsonl found anywhere under /kaggle/input. Attach the dataset "
            "pushed by scripts/push_dataset_to_kaggle.py as a data source on this kernel."
        )
    if len(matches) > 1:
        print(f"Multiple metadata.jsonl found, using the first: {[str(m) for m in matches]}")
    return matches[0].parent


def link_or_extract_dataset(dataset_dir: Path, processed: Path) -> None:
    """Make `processed` hold metadata.jsonl + clips/. Kaggle usually
    auto-extracts the uploaded clips.zip into clips/, in which case a symlink
    is enough; if it's still a zip, /kaggle/input is read-only so extract a
    real copy under the working dir instead."""
    if processed.is_symlink():
        processed.unlink()
    elif processed.exists():
        shutil.rmtree(processed)
    processed.parent.mkdir(parents=True, exist_ok=True)

    if (dataset_dir / "clips").is_dir():
        processed.symlink_to(dataset_dir.resolve())
        print(f"Linked dataset {dataset_dir} -> {processed}")
        return

    zip_path = dataset_dir / "clips.zip"
    if not zip_path.exists():
        raise RuntimeError(f"{dataset_dir} has metadata.jsonl but neither clips/ nor clips.zip.")
    processed.mkdir()
    shutil.copy(dataset_dir / "metadata.jsonl", processed / "metadata.jsonl")
    clips_dir = processed / "clips"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(clips_dir)
    nested = clips_dir / "clips"
    if nested.is_dir() and not any(clips_dir.glob("*.mp4")):
        for f in nested.iterdir():
            f.rename(clips_dir / f.name)
        nested.rmdir()
    print(f"Extracted {zip_path} -> {clips_dir} ({len(list(clips_dir.glob('*.mp4')))} clips)")


def main():
    task = os.environ.get("TRAIN_TASK", TRAIN_TASK)
    assert task in ("t2v", "i2v"), f"Unknown TRAIN_TASK={task}"

    if not REPO_DIR.exists():
        run(["git", "clone", "--branch", REPO_GIT_REF, "--depth", "1", REPO_GIT_URL, str(REPO_DIR)])
    os.chdir(REPO_DIR)
    sys.path.insert(0, str(REPO_DIR))

    run(["pip", "install", "-q", "-r", "requirements.txt"])
    # Kaggle's base image ships torchao 0.10.0. A recent peft release probes
    # is_torchao_available() while dispatching LoRA layers -- even though we
    # never asked for torchao-backed quantized LoRA -- and raises ImportError
    # outright on any *incompatible* (not just missing) version, crashing
    # training before a single step ran. Adding torchao>=0.16.0 to
    # requirements.txt did not take effect in this environment (still 0.10.0
    # after install, likely a Kaggle-side constraint pinning it), so instead
    # uninstall it entirely: is_torchao_available() treats "not installed" as
    # a clean, exception-free False rather than a version check that can fail.
    subprocess.run(["pip", "uninstall", "-y", "-q", "torchao"], check=False)

    dataset_dir = find_dataset_dir()
    link_or_extract_dataset(dataset_dir, REPO_DIR / "data" / "processed")

    num_gpus = int(subprocess.run(
        ["python", "-c", "import torch;print(torch.cuda.device_count())"],
        capture_output=True, text=True,
    ).stdout.strip() or "0")
    print(f"Detected {num_gpus} GPU(s).")
    if num_gpus > 1:
        print(
            f"{num_gpus} GPUs available, but forcing --num_processes 1 anyway: "
            "data-parallel launch means every process loads its own full copy "
            "of the model during the CPU-side deserialization phase before "
            "anything moves to a GPU, and two of those at once on a Kaggle "
            "instance's system RAM triggered an OOM-kill (SIGKILL, no Python "
            "traceback) partway through loading on a 2-GPU run. This LoRA "
            "fine-tune on ~140 clips doesn't need multi-GPU scaling anyway."
        )

    run([
        "accelerate", "launch",
        "--num_processes", "1",
        "--mixed_precision", "fp16",
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
