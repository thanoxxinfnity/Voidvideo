#!/usr/bin/env python3
"""
Re-caption the 24 fps training windows with a vision-language model (Qwen3-VL-4B-Instruct, Apache-2.0).

The tag-built captions of the Sakugabooru clips say nothing about how many characters or creatures
are in the shot, what they look like, where they stand or how their bodies move. This script shows the
VLM 4 frames of each window and asks for a structured description (count, subjects with kind/look/
position, action and body language, camera, setting, effects), then composes a training caption from it,
keeping the clip's timing tag ("animated on ones/twos/threes ...").

    python scripts/vlm_caption_clips.py --data DIR --out captions_shard0.jsonl [--shard 0 --nshards 2]
    python scripts/vlm_caption_clips.py --data DIR --out /tmp/x.jsonl --dry-run     # no model, checks I/O

DIR holds metadata.jsonl and clips/. Output rows: {"video", "caption", "vlm": {...}, "old_caption"}.
Resumable: windows already present in --out are skipped.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
FRAMES = 4

PROMPT = (
    "You are labeling a 2-second clip from a hand-drawn anime for a video-generation training set. "
    "The {n} images are frames of the clip in time order. Existing rough tags: {hint}.\n"
    "Answer with ONLY a JSON object with these keys:\n"
    '  "count": integer number of distinct characters or creatures visible (use "crowd" if more than 8),\n'
    '  "subjects": list of short descriptions, one per main subject, each starting with its kind '
    "(human / monster / animal / other) and giving appearance (hair, colors, clothes, body shape, limbs, "
    "size) and where it is in the frame (left / center / right / foreground / background),\n"
    '  "action": what each subject does, in order of motion, including body language and facial expression,\n'
    '  "camera": shot size and any camera movement,\n'
    '  "setting": place, time of day and lighting,\n'
    '  "effects": hand-drawn effects such as smoke, fire, impact frames, speed lines, or "none",\n'
    '  "style": exactly one of "2D hand-drawn anime", "3D CGI", "live action or photorealistic", "other".\n'
    "Be literal and concrete. Do not invent character names or series titles."
)


def extract_frames(video: Path, tmp: Path):
    """FRAMES evenly spaced frames as PNG paths (uses the ffmpeg bundled with imageio, or the system one)."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"
    out = []
    for i, n in enumerate((0, 16, 32, 48)[:FRAMES]):
        p = tmp / f"f{i}.png"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(video), "-vf",
                        f"select=eq(n\\,{n}),scale='if(gt(iw,ih),448,-2)':'if(gt(iw,ih),-2,448)'",
                        "-frames:v", "1", str(p)], check=True)
        if p.exists():
            out.append(p)
    return out


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def clean(x) -> str:
    if isinstance(x, list):
        x = "; ".join(clean(i) for i in x)
    return re.sub(r"\s+", " ", str(x or "")).strip().rstrip(".")


def compose(v: dict, old_caption: str) -> str:
    """Training caption from the VLM fields, ending with the clip's existing timing tag."""
    timing = re.search(r"animated on (?:ones|twos|threes)(?: with held frames)?", old_caption)
    count, subjects = v.get("count"), clean(v.get("subjects"))
    parts = []
    if clean(v.get("camera")):
        parts.append(clean(v["camera"]).capitalize())
    if count not in (None, "", 0):
        parts.append(f"{count} subject{'s' if str(count) != '1' else ''} in the shot" if str(count).isdigit()
                     else f"A {clean(count)} of characters")
    if subjects:
        parts.append(subjects)
    for key in ("action", "setting"):
        if clean(v.get(key)):
            parts.append(clean(v[key]))
    eff = clean(v.get("effects"))
    if eff and eff.lower() != "none":
        parts.append(f"effects: {eff}")
    style = clean(v.get("style")).lower()
    if "2d" in style or "anime" in style or "hand" in style:
        look = "Traditional 2D hand-drawn anime"
    elif "3d" in style:
        look = "3D CGI animation"
    elif "live" in style or "photo" in style:
        look = "Live action, photorealistic footage"
    else:
        look = "Animation"
    tail = look + (f", {timing.group(0)}" if timing else "")
    return ". ".join(p[0].upper() + p[1:] for p in parts if p) + f". {tail}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N windows of this shard (testing)")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth window (spread sample for quick tests)")
    ap.add_argument("--dry-run", action="store_true", help="no model: write a placeholder VLM answer")
    ap.add_argument("--max-new-tokens", type=int, default=420)
    ap.add_argument("--batch", type=int, default=4, help="windows generated together per GPU")
    args = ap.parse_args()

    data = Path(args.data)
    rows = [json.loads(line) for line in open(data / "metadata.jsonl") if line.strip()]
    rows = rows[args.shard::args.nshards][::max(1, args.stride)]
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(line)["video"] for line in open(out) if line.strip()}
    todo = [r for r in rows if r["video"] not in done]
    print(f"shard {args.shard}/{args.nshards}: {len(rows)} windows, {len(done)} done, {len(todo)} to do", flush=True)

    model = processor = None
    if not args.dry_run:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda:0")
        model.eval()

    from PIL import Image
    n_ok = n_bad = 0
    dummy = {"count": 2, "subjects": ["human, left, spiky pink hair", "monster, right, four arms"],
             "action": "the human lunges while the monster blocks", "camera": "medium shot, static",
             "setting": "ruined street, daylight", "effects": "dust", "style": "2D hand-drawn anime"}

    def hint_of(r):
        return re.sub(r",? ?animated on [a-z ]+\.?$", "", r.get("caption", "")).strip()

    def conversation(frames, r):
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in frames]
        content.append({"type": "text", "text": PROMPT.format(n=len(frames), hint=hint_of(r)[:200])})
        return [{"role": "user", "content": content}]

    def generate(convs):
        import torch
        inputs = processor.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        return processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    if processor is not None:
        processor.tokenizer.padding_side = "left"
    with open(out, "a") as fout, tempfile.TemporaryDirectory() as td:
        for start in range(0, len(todo), max(1, args.batch)):
            chunk = todo[start:start + max(1, args.batch)]
            convs = []
            for j, r in enumerate(chunk):
                tmp = Path(td) / f"w{j}"
                tmp.mkdir(exist_ok=True)
                for f in tmp.glob("*.png"):
                    f.unlink()
                convs.append(conversation(extract_frames(data / r["video"], tmp), r) if not args.dry_run else None)
            if args.dry_run:
                raws = [json.dumps(dummy)] * len(chunk)
            else:
                try:
                    raws = generate(convs)
                except Exception as e:  # e.g. OOM with a big batch: fall back to one window at a time
                    print(f"batch failed ({type(e).__name__}: {str(e)[:120]}); retrying singly", flush=True)
                    import torch
                    torch.cuda.empty_cache()
                    raws = [generate([c])[0] for c in convs]
            for r, raw in zip(chunk, raws):
                v = parse_json(raw)
                if v is None:
                    n_bad += 1                  # keep the old caption for this window
                    fout.write(json.dumps({"video": r["video"], "caption": r.get("caption", ""), "vlm": None,
                                           "raw": raw[:500], "old_caption": r.get("caption", "")}) + "\n")
                else:
                    n_ok += 1
                    fout.write(json.dumps({"video": r["video"], "caption": compose(v, r.get("caption", "")),
                                           "vlm": v, "old_caption": r.get("caption", "")}) + "\n")
            fout.flush()
            done_n = start + len(chunk)
            if done_n % 24 < len(chunk) or start == 0:
                print(f"[{done_n}/{len(todo)}] ok={n_ok} unparsable={n_bad}", flush=True)
    print(f"DONE ok={n_ok} unparsable={n_bad}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
