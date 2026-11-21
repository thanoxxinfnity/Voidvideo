"""
Kaggle kernel: re-caption the 24 fps windows (dataset attached as input, e.g. voidvideo-anime-clips-24fps-v3)
with Qwen3-VL-4B on every visible GPU in parallel. Writes /kaggle/working/metadata_vlm.jsonl (same rows as the
input metadata, captions replaced; windows the VLM could not describe keep their old caption) plus the raw
answers in captions_raw.jsonl. Resumable only within a session; ~5 s per window per GPU.
"""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_GIT_URL = "https://github.com/thanoxxinfnity/Voidvideo.git"
REPO_GIT_REF = "claude/claude-md-jfha3c"
LIMIT = 0   # >0: only the first N windows per GPU (quick quality test)
STRIDE = 1  # >1: every Nth window instead of all (spread sample for quick tests)
WORK = Path("/kaggle/working")
REPO = WORK / "voidvideo"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers==4.57.3", "accelerate>=0.33",
                "pillow", "imageio-ffmpeg"], check=True)
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_GIT_REF, REPO_GIT_URL, str(REPO)], check=True)

meta = sorted(Path("/kaggle/input").rglob("metadata.jsonl"))
assert meta, "attach the 24 fps clips dataset (metadata.jsonl + clips/ or clips.zip)"
src = meta[0].parent
data = src
if (src / "clips" / "clips").is_dir():                       # Kaggle extracted a zip that held a clips/ folder to clips/clips/
    data = WORK / "data"
    data.mkdir(parents=True, exist_ok=True)
    if not (data / "clips").exists():
        (data / "clips").symlink_to(src / "clips" / "clips")
    (data / "metadata.jsonl").write_bytes((src / "metadata.jsonl").read_bytes())
elif not (src / "clips").is_dir():
    data = WORK / "data"
    (data / "clips").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src / "clips.zip") as z:
        z.extractall(data / "clips")
    nested = data / "clips" / "clips"
    if nested.is_dir():
        for f in nested.iterdir():
            f.rename(data / "clips" / f.name)
    (data / "metadata.jsonl").write_bytes((src / "metadata.jsonl").read_bytes())
# Optional: a small extra dataset holding subset.jsonl (the windows to caption, with their old captions).
subset = sorted(Path("/kaggle/input").rglob("subset.jsonl"))
meta_file = subset[0] if subset else data / "metadata.jsonl"
n_rows = sum(1 for _ in open(meta_file))
print(f"{n_rows} windows listed in {meta_file}, clips under {data}", flush=True)

import torch  # noqa: E402
n_gpu = max(1, torch.cuda.device_count())
procs = []
for g in range(n_gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
    procs.append(subprocess.Popen(
        [sys.executable, str(REPO / "scripts" / "vlm_caption_clips.py"), "--data", str(data),
         "--out", str(WORK / f"cap_shard{g}.jsonl"), "--shard", str(g), "--nshards", str(n_gpu), "--meta", str(meta_file)]
        + (["--limit", str(LIMIT)] if LIMIT else []) + ["--stride", str(STRIDE)], env=env))
codes = [p.wait() for p in procs]
print("shard exit codes:", codes, flush=True)

new = {}
with open(WORK / "captions_raw.jsonl", "w") as raw:
    for g in range(n_gpu):
        f = WORK / f"cap_shard{g}.jsonl"
        if f.exists():
            for line in open(f):
                r = json.loads(line)
                new[r["video"]] = r
                raw.write(line)
ok = bad = non_anime = 0
with open(WORK / "metadata_vlm.jsonl", "w") as out, open(WORK / "metadata_vlm_anime_only.jsonl", "w") as only:
    for line in open(meta_file):
        r = json.loads(line)
        n = new.get(r["video"])
        is_anime = True
        if n and n.get("vlm"):
            r["caption"] = n["caption"]
            r["style"] = n["vlm"].get("style")
            st = str(r["style"] or "").lower()
            is_anime = ("2d" in st or "anime" in st or "hand" in st)
            ok += 1
        else:
            bad += 1
        out.write(json.dumps(r) + "\n")
        if is_anime:
            only.write(json.dumps(r) + "\n")
        else:
            non_anime += 1
print(f"DONE: {ok} re-captioned, {bad} kept their old caption, {non_anime} windows are NOT 2D anime "
      f"(left out of metadata_vlm_anime_only.jsonl)", flush=True)
