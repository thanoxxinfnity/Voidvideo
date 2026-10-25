"""
Runs INSIDE a Kaggle GPU kernel. One-shot sanity check of a trained T2V LoRA:
generates the same prompt/seed with and without the LoRA so the difference is
visible, writes the clips to /kaggle/working and exits (no Gradio, no tunnel).
"""
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "diffusers>=0.35.0", "transformers>=4.44", "accelerate>=0.33", "peft>=0.12",
     "imageio[ffmpeg]", "sentencepiece"],
    check=True,
)

# Kaggle ships torchao 0.10.0, which peft rejects as incompatible; uninstalling makes
# is_torchao_available() a clean False (same workaround as training/kaggle_entrypoint.py).
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import torch  # noqa: E402
from diffusers import CogVideoXPipeline  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402
from peft import LoraConfig, set_peft_model_state_dict  # noqa: E402

MODEL_ID = "THUDM/CogVideoX-2b"
LORA_RANK = 64
LORA_ALPHA = 64
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
FPS = 8
OUT = Path("/kaggle/working")
SEED = 42
STEPS = 30
GUIDANCE = 6.0
# Wording follows the training clips' style (see data/README.md).
PROMPT = (
    "A lean teenage sorcerer in a tattered navy-black high-collar uniform sprinting down a narrow, "
    "rain-slicked Tokyo back-alley at night. Low-angle tracking shot on boots splashing through shallow "
    "puddles. 2D traditional cel animation, sharp ink lines, hand-drawn splash in-betweens, flat shadows, "
    "modern 2D anime film style, fully colored, clean inked outlines."
)
NEGATIVE = "photorealistic, 3D render, CGI, blurry, distorted, deformed, black and white, sketch, watermark"


def find_lora() -> Path:
    matches = sorted(Path("/kaggle/input").rglob("lora_weights.pt"))
    assert matches, "no lora_weights.pt found under /kaggle/input -- attach the checkpoint dataset"
    print("Using LoRA:", matches[0], flush=True)
    return matches[0]


def interpolate(src: Path) -> Path:
    import imageio_ffmpeg
    dst = src.with_name(src.stem + "_24fps.mp4")
    try:
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(src), "-filter:v",
             "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(dst)],
            check=True, capture_output=True, timeout=300,
        )
        return dst
    except Exception as e:
        print("24fps interpolation failed, keeping 8fps:", e, flush=True)
        return src


def decode_latents(pipe, latents):
    # Decode with the VAE in fp32 regardless of the denoising dtype.
    vae = pipe.vae
    vae.to("cuda", dtype=torch.float32)
    with torch.no_grad():
        z = latents.to("cuda", dtype=torch.float32).permute(0, 2, 1, 3, 4) / vae.config.scaling_factor
        video = vae.decode(z).sample
    vae.to("cpu")
    torch.cuda.empty_cache()
    return video.cpu()


def run(pipe, tag: str, steps: int, save: bool = True):
    t0 = time.time()
    latents = pipe(
        prompt=PROMPT, negative_prompt=NEGATIVE, num_frames=49, height=480, width=720,
        num_inference_steps=steps, guidance_scale=GUIDANCE,
        generator=torch.Generator("cpu").manual_seed(SEED), output_type="latent",
    ).frames
    video = decode_latents(pipe, latents)
    spread = video.float().std().item()
    print(f"[{tag}] latents nan={torch.isnan(latents).any().item()} std={latents.float().std().item():.3f}; "
          f"decoded min={video.min().item():.2f} max={video.max().item():.2f} std={spread:.3f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    path = None
    if save:
        frames = pipe.video_processor.postprocess_video(video=video, output_type="pil")[0]
        raw = OUT / f"{tag}.mp4"
        export_to_video(frames, str(raw), fps=FPS)
        path = interpolate(raw)
    return spread, path


def load_pipe(dtype):
    pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def attach_lora(pipe, dtype):
    pipe.transformer.add_adapter(LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS, init_lora_weights=True))
    result = set_peft_model_state_dict(pipe.transformer, lora_state)
    print("LoRA loaded. unexpected keys:", len(getattr(result, "unexpected_keys", None) or []), flush=True)
    for p in pipe.transformer.parameters():
        if p.dtype != dtype:
            p.data = p.data.to(dtype)


lora_state = torch.load(find_lora(), map_location="cpu")

# Stage 1: quick 6-step probe per dtype (no LoRA); the first one that decodes to a non-flat image wins.
chosen = None
for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
    pipe = load_pipe(dtype)
    spread, _ = run(pipe, f"probe_{name}", 6, save=False)
    if spread > 0.1:
        chosen = (name, dtype, pipe)
        break
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
assert chosen, "both fp16 and bf16 decode to a flat image -- see probe lines in the log"
name, dtype, pipe = chosen
print("Using dtype", name, flush=True)

# Stage 2: full-quality clips; LoRA first so it is saved even if the second run runs out of time.
attach_lora(pipe, dtype)
run(pipe, "with_lora", STEPS)
pipe.transformer.disable_adapters()
run(pipe, "base_no_lora", STEPS)
print("ALL_DONE", flush=True)
