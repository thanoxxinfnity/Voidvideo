"""Pipeline loading + generation helpers shared by the Gradio app."""
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import torch
import yaml
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXPipeline
from diffusers.utils import export_to_video
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "training_config.yaml"

_pipelines: dict = {}

# CogVideoX generates at its native 8 fps; the exported clip is upsampled to
# this fps afterwards with ffmpeg motion interpolation (CPU-only post-process,
# no extra GPU memory). Falls back to the native-fps file if ffmpeg fails.
OUTPUT_FPS = 24


def _interpolate_to_output_fps(src_path: str) -> str:
    dst_path = src_path.replace(".mp4", f"_{OUTPUT_FPS}fps.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src_path,
                "-filter:v", f"minterpolate=fps={OUTPUT_FPS}:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                dst_path,
            ],
            check=True, capture_output=True, timeout=180,
        )
        return dst_path
    except Exception as e:
        print(f"{OUTPUT_FPS}fps interpolation failed, returning native-fps clip: {e}", flush=True)
        return src_path


def _load_config(task: str) -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)[task]


def _default_lora_path(task: str) -> Optional[Path]:
    candidate = REPO_ROOT / "models" / f"lora_{task}" / "final" / "lora_weights.safetensors"
    return candidate if candidate.exists() else None


def get_pipeline(task: str, lora_path: Optional[str] = None):
    """
    Lazily loads (and caches) the CogVideoX pipeline for `task` ("t2v" or "i2v"),
    attaching a LoRA checkpoint if one is available. Re-loading with a different
    lora_path replaces the cached adapter.
    """
    cfg = _load_config(task)
    cache_key = task

    if cache_key not in _pipelines:
        pipeline_cls = CogVideoXImageToVideoPipeline if task == "i2v" else CogVideoXPipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        pipe = pipeline_cls.from_pretrained(cfg["base_model_id"], torch_dtype=dtype)
        if device == "cuda":
            pipe.enable_model_cpu_offload()  # keeps VRAM usage manageable, esp. for the 5B I2V model
        else:
            pipe.to(device)
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        _pipelines[cache_key] = {"pipe": pipe, "cfg": cfg, "lora_path": None}

    entry = _pipelines[cache_key]
    resolved_lora = Path(lora_path) if lora_path else _default_lora_path(task)

    if resolved_lora and str(resolved_lora) != entry["lora_path"]:
        if entry["lora_path"] is not None:
            entry["pipe"].unload_lora_weights()
        if resolved_lora.exists():
            entry["pipe"].load_lora_weights(str(resolved_lora.parent), weight_name=resolved_lora.name)
            entry["lora_path"] = str(resolved_lora)
        else:
            entry["lora_path"] = None

    return entry["pipe"], entry["cfg"]


def generate_t2v(prompt: str, negative_prompt: str, num_inference_steps: int,
                  guidance_scale: float, seed: int, lora_path: Optional[str] = None) -> str:
    pipe, cfg = get_pipeline("t2v", lora_path)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_frames=cfg["num_frames"],
        height=cfg["height"],
        width=cfg["width"],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).frames[0]

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    export_to_video(frames, out_path, fps=cfg["fps"])
    return _interpolate_to_output_fps(out_path)


def generate_i2v(image: Image.Image, prompt: str, negative_prompt: str, num_inference_steps: int,
                  guidance_scale: float, seed: int, lora_path: Optional[str] = None) -> str:
    pipe, cfg = get_pipeline("i2v", lora_path)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = image.convert("RGB").resize((cfg["width"], cfg["height"]))

    frames = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_frames=cfg["num_frames"],
        height=cfg["height"],
        width=cfg["width"],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).frames[0]

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    export_to_video(frames, out_path, fps=cfg["fps"])
    return _interpolate_to_output_fps(out_path)
