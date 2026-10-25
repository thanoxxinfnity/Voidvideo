"""Kaggle kernel: checks whether the CogVideoX-2b VAE round-trips a synthetic video, and whether
decoding with tiling on/off differs; then decodes real 25-step latents both ways."""
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.35.0", "transformers>=4.44",
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


vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32).to("cuda")
print("scaling_factor", vae.config.scaling_factor, flush=True)

# Synthetic structured video in [-1, 1]: moving gradient + a bright square.
T, H, W = 9, 480, 720
yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
vid = np.zeros((T, 3, H, W), dtype=np.float32)
for t in range(T):
    vid[t, 0] = xx
    vid[t, 1] = yy
    vid[t, 2] = np.sin(3 * (xx + t * 0.1))
    vid[t, :, 100 + 20 * t:200 + 20 * t, 200:300] = 1.0
x = torch.from_numpy(vid).permute(1, 0, 2, 3).unsqueeze(0).to("cuda")  # B,C,T,H,W
stats("input", x)

for tiling in (True,):
    if tiling:
        vae.enable_tiling()
    else:
        vae.disable_tiling()
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
        stats(f"latent tiling={tiling}", z)
        rec = vae.decode(z).sample
    stats(f"roundtrip tiling={tiling}", rec)
    print(f"[roundtrip tiling={tiling}] mean abs err = {(rec - x).abs().mean().item():.4f}", flush=True)
    stats(f"decode(randn) tiling={tiling}", vae.decode(torch.randn_like(z)).sample)
del vae
torch.cuda.empty_cache()

# Real denoised latents (9 frames, 25 steps, fp16) decoded with each VAE mode.
pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
t0 = time.time()
lat = pipe(
    prompt="A girl in a school uniform walking down a sunlit street, 2D anime, hand-drawn, clean ink lines, flat colors",
    num_frames=9, height=480, width=720, num_inference_steps=25, guidance_scale=6.0,
    generator=torch.Generator("cpu").manual_seed(1), output_type="latent",
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
