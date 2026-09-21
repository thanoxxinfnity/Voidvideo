"""VoidVideo Caption Studio.

Local Gradio dashboard: bulk-upload anime training clips, auto-caption them
with a vision-language model (via Hugging Face's hosted Inference Providers),
edit captions by hand, then package everything into a zip ready to upload to
Kaggle for LoRA training.

Run with: python app.py
"""
import base64
import io
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import imageio.v2 as imageio
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN", "")
# Qwen2-VL-7B-Instruct itself has no hosted inference provider right now, and
# Qwen2.5-VL-7B-Instruct's only provider (featherless-ai) is frequently at
# capacity on the free tier. Qwen2.5-VL-72B-Instruct via ovhcloud is the
# closest verified-working substitute -- override via HF_CAPTION_MODEL if
# your account has a provider you prefer.
HF_CAPTION_MODEL = os.environ.get("HF_CAPTION_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud")
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

# The ovhcloud-hosted Qwen endpoint caps requests at 5 images; this is also
# generally plenty to describe motion across a 4-6s clip.
NUM_SAMPLE_FRAMES = 5
FRAME_MAX_WIDTH = 220

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
VIDEOS_DIR = DATASET_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = BASE_DIR / "dataset.zip"
METADATA_PATH = DATASET_DIR / "dataset-metadata.json"

CAPTION_PROMPT = (
    "You are writing a short training caption for an anime video LoRA dataset. "
    "These images are frames sampled evenly across ONE short video clip, in time order -- "
    "read them as a timeline, not separate pictures. In ONE concise sentence, describe: "
    "what the character is doing (e.g. walking, waving, blinking, hair/cloth moving in the wind), "
    "their facial expression if visible, and the camera shot (close-up/medium/wide, static or panning). "
    "Do not mention 'frame' or 'image' -- describe it as a continuous video clip. "
    "Reply with ONLY the caption sentence, no preamble."
)


def next_clip_index():
    max_idx = 0
    for p in VIDEOS_DIR.glob("clip_*.mp4"):
        try:
            max_idx = max(max_idx, int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max_idx + 1


def extract_sample_frames_b64(video_path, n=NUM_SAMPLE_FRAMES, max_w=FRAME_MAX_WIDTH):
    reader = imageio.get_reader(str(video_path))
    try:
        meta = reader.get_meta_data()
        fps = meta.get("fps") or 24
        try:
            n_frames = reader.count_frames()
            if not n_frames or n_frames == float("inf"):
                raise ValueError
        except Exception:
            n_frames = max(1, int((meta.get("duration") or 3) * fps))

        idxs = sorted({min(int(n_frames * (i + 0.5) / n), n_frames - 1) for i in range(n)})
        frames_b64 = []
        for i in idxs:
            frame = reader.get_data(i)
            img = Image.fromarray(frame)
            w, h = img.size
            if w > max_w:
                img = img.resize((max_w, max(1, round(h * max_w / w))))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=72)
            frames_b64.append(base64.b64encode(buf.getvalue()).decode())
        return frames_b64, meta
    finally:
        reader.close()


def caption_via_hf(frames_b64):
    """Returns (caption, error). error is None on success."""
    if not HF_TOKEN:
        return "", "HF_TOKEN not set -- add it to caption-studio/.env and restart the app."

    content = [{"type": "text", "text": CAPTION_PROMPT}]
    for b64 in frames_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    payload = {
        "model": HF_CAPTION_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 120,
        "temperature": 0.2,
    }

    try:
        resp = requests.post(
            HF_ROUTER_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            json=payload,
            timeout=90,
        )
        data = resp.json()
    except Exception as e:
        return "", f"Request to Hugging Face failed: {e}"

    if resp.status_code != 200:
        msg = (data.get("error") or {}).get("message") or data.get("message") or json.dumps(data)[:300]
        return "", f"HF API error ({resp.status_code}): {msg}"

    try:
        caption = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return "", f"Unexpected HF API response shape: {json.dumps(data)[:300]}"

    return caption, None


def handle_upload(files, records, progress=gr.Progress()):
    if not files:
        return records, "No files selected."

    records = list(records)
    idx = next_clip_index()
    total = len(files)
    failures = []

    for i, f in enumerate(files):
        src_path = Path(getattr(f, "name", f))
        progress(i / total, desc=f"Captioning {i + 1}/{total}: {src_path.name}")

        clip_stem = f"clip_{idx:03d}"
        dest_video = VIDEOS_DIR / f"{clip_stem}.mp4"
        shutil.copy(src_path, dest_video)

        try:
            frames_b64, _meta = extract_sample_frames_b64(dest_video)
            caption, err = caption_via_hf(frames_b64)
        except Exception as e:
            caption, err = "", str(e)

        if err:
            failures.append(f"{clip_stem}.mp4: {err}")
            if not caption:
                caption = "[AI caption failed -- edit this manually]"

        (VIDEOS_DIR / f"{clip_stem}.txt").write_text(caption, encoding="utf-8")
        records.append({"filename": f"{clip_stem}.mp4", "path": str(dest_video), "caption": caption})
        idx += 1

    progress(1.0, desc="Done")
    status = f"✅ Uploaded and captioned {total} clip(s)."
    if failures:
        status += "\n\n⚠️ AI captioning failed for:\n" + "\n".join(f"- {m}" for m in failures)
    return records, status


def make_caption_saver(filename):
    txt_path = VIDEOS_DIR / (Path(filename).stem + ".txt")

    def save(text):
        txt_path.write_text(text or "", encoding="utf-8")

    return save


def make_delete_handler(filename):
    video_path = VIDEOS_DIR / filename
    txt_path = video_path.with_suffix(".txt")

    def delete(records):
        video_path.unlink(missing_ok=True)
        txt_path.unlink(missing_ok=True)
        updated = [r for r in records if r["filename"] != filename]
        return updated, f"🗑️ Deleted {filename}"

    return delete


def load_existing_clips():
    """Runs on every page load so re-opening the app (or a second browser
    tab) shows clips uploaded in an earlier session -- gr.State is otherwise
    per-session and would start empty even though the files are on disk."""
    records = []
    for v in sorted(VIDEOS_DIR.glob("clip_*.mp4")):
        txt = v.with_suffix(".txt")
        caption = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        records.append({"filename": v.name, "path": str(v), "caption": caption})
    return records


def package_dataset(_records):
    videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        return None, "⚠️ No clips uploaded yet -- nothing to package."

    entries = []
    warnings = []
    resolutions = set()

    for v in videos:
        txt = v.with_suffix(".txt")
        caption = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        if not caption:
            warnings.append(f"{v.name}: empty caption")

        width = height = 0
        duration = fps = None
        try:
            reader = imageio.get_reader(str(v))
            meta = reader.get_meta_data()
            width, height = meta.get("size", (0, 0))
            duration = meta.get("duration")
            fps = meta.get("fps")
            reader.close()
        except Exception as e:
            warnings.append(f"{v.name}: could not read video ({e})")

        resolutions.add((width, height))
        entries.append(
            {
                "filename": v.name,
                "caption": caption,
                "width": width,
                "height": height,
                "duration_seconds": round(duration, 2) if duration else None,
                "fps": round(fps, 2) if fps else None,
                "bytes": v.stat().st_size,
            }
        )

    if len(resolutions) > 1:
        warnings.append(
            f"Mixed resolutions found: {sorted(resolutions)} -- LoRA training expects one consistent resolution/orientation across the whole set."
        )

    metadata = {
        "dataset_name": "voidvideo-anime-clips",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "clip_count": len(entries),
        "resolutions_found": [list(r) for r in resolutions],
        "caption_model": HF_CAPTION_MODEL,
        "clips": entries,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for v in videos:
            zf.write(v, arcname=f"videos/{v.name}")
            txt = v.with_suffix(".txt")
            if txt.exists():
                zf.write(txt, arcname=f"videos/{txt.name}")
        zf.write(METADATA_PATH, arcname="dataset-metadata.json")

    log = f"✅ Packaged {len(entries)} clip(s) into `dataset.zip`, ready for Kaggle."
    if warnings:
        log += "\n\n**Warnings:**\n" + "\n".join(f"- {w}" for w in warnings)
    return str(ZIP_PATH), log


with gr.Blocks(title="VoidVideo Caption Studio") as demo:
    gr.Markdown(
        "# 🎬 VoidVideo Caption Studio\n"
        "Bulk-upload anime training clips, let AI draft a caption for each one "
        "(action, expression, camera shot), edit them, then package for Kaggle."
    )
    if not HF_TOKEN:
        gr.Markdown(
            "⚠️ **HF_TOKEN is not set.** Copy `.env.example` to `.env`, add your "
            "Hugging Face token, and restart the app -- uploads will still work, "
            "but captions won't auto-generate until then."
        )

    clip_state = gr.State([])

    upload = gr.File(
        label="Upload video clips (.mp4, drop as many as you like)",
        file_count="multiple",
        file_types=["video"],
    )
    upload_status = gr.Markdown("")

    @gr.render(inputs=clip_state)
    def render_clips(records):
        if not records:
            gr.Markdown("_No clips uploaded yet._")
            return
        for rec in records:
            with gr.Row():
                gr.Video(value=rec["path"], label=rec["filename"], scale=1)
                box = gr.Textbox(
                    value=rec["caption"],
                    label=f"Caption — {rec['filename']}",
                    lines=3,
                    scale=2,
                    interactive=True,
                )
                box.change(fn=make_caption_saver(rec["filename"]), inputs=box, outputs=[])
                with gr.Column(scale=0, min_width=110):
                    del_btn = gr.Button("🗑 Delete", size="sm", variant="stop")
                    del_btn.click(fn=make_delete_handler(rec["filename"]), inputs=clip_state, outputs=[clip_state, upload_status])

    upload.upload(fn=handle_upload, inputs=[upload, clip_state], outputs=[clip_state, upload_status])
    demo.load(fn=load_existing_clips, outputs=clip_state)

    gr.Markdown("---\n## Package for Kaggle")
    gr.Markdown(
        "Verifies every clip's resolution, writes `dataset-metadata.json`, and zips "
        "`videos/<clip>.mp4` + `videos/<clip>.txt` pairs together."
    )
    package_btn = gr.Button("📦 Package Dataset", variant="primary")
    package_file = gr.File(label="dataset.zip")
    package_log = gr.Markdown("")
    package_btn.click(fn=package_dataset, inputs=clip_state, outputs=[package_file, package_log])


if __name__ == "__main__":
    demo.launch()
