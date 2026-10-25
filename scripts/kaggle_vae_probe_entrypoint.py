"""Kaggle kernel: denoises a short 25-step clip with a pinned diffusers and prints per-stage stats."""
import subprocess
import sys
import time

DIFFUSERS_PIN = "0.35.2"  # latest (0.42.0) gave noise-like latents for CogVideoX-2b
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers==" + DIFFUSERS_PIN, "transformers>=4.44",
                "accelerate>=0.33", "imageio[ffmpeg]", "sentencepiece"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import diffusers  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import AutoencoderKLCogVideoX, CogVideoXPipeline  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402

print("diffusers", diffusers.__version__, "torch", torch.__version__, flush=True)
MODEL_ID = "THUDM/CogVideoX-2b"
OUT = Path("/kaggle/working")


def stats(tag, x):
    x = x.float()
    print(f"[{tag}] shape={tuple(x.shape)} nan={torch.isnan(x).any().item()} min={x.min().item():.3f} "
          f"max={x.max().item():.3f} mean={x.mean().item():.3f} std={x.std().item():.3f}", flush=True)


# Real denoised latents (9 frames, 25 steps, fp16) decoded with each VAE mode.
pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
t0 = time.time()
lat = pipe(
    prompt="A girl in a school uniform walking down a sunlit street, 2D anime, hand-drawn, clean ink lines, flat colors",
    num_frames=9, height=480, width=720, num_inference_steps=25, guidance_scale=6.0,
    generator=torch.Generator("cpu").manual_seed(1), output_type="latent",
    callback_on_step_end=lambda p, i, t, kw: (print(f"step {i} std={kw['latents'].float().std().item():.3f}", flush=True) if i % 5 == 0 else None, kw)[1],
    callback_on_step_end_tensor_inputs=["latents"],
).frames
stats(f"denoised latents ({(time.time() - t0) / 60:.1f} min)", lat)
for tiling in (True,):
    v = AutoencoderKLCogVideoX.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32).to("cuda")
    if tiling:
        v.enable_tiling()
    with torch.no_grad():
        z = lat.to("cuda", torch.float32).permute(0, 2, 1, 3, 4) / v.config.scaling_factor
        dec = v.decode(z).sample.cpu()
    stats(f"real decode tiling={tiling}", dec)
    frames = pipe.video_processor.postprocess_video(video=dec, output_type="pil")[0]
    export_to_video(frames, str(OUT / f"probe_tiling_{tiling}.mp4"), fps=8)
    del v
    torch.cuda.empty_cache()
print("PROBE_DONE", flush=True)
