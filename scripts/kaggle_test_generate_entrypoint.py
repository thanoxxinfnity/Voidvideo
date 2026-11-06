"""
Kaggle kernel: (1) canonical diffusers-docs CogVideoX-2b generation as a pipeline sanity baseline,
(2) the same call with the trained LoRA on an anime prompt. Both decoded with an fp32 VAE.
"""
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers==0.35.2", "transformers==4.57.3",
                "accelerate>=0.33", "peft>=0.12", "imageio[ffmpeg]", "sentencepiece"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import torch  # noqa: E402
from diffusers import AutoencoderKLCogVideoX, CogVideoXPipeline  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402
from peft import LoraConfig, set_peft_model_state_dict  # noqa: E402

MODEL_ID = "THUDM/CogVideoX-2b"
OUT = Path("/kaggle/working")
PANDA = ("A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo "
         "forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes.")
ANIME = ("A lean teenage sorcerer in a tattered navy-black high-collar uniform sprinting down a narrow, "
         "rain-slicked Tokyo back-alley at night, boots splashing through shallow puddles. 2D traditional cel "
         "animation, sharp ink lines, hand-drawn in-betweens, flat shadows, modern 2D anime film style.")


def stats(tag, x):
    x = x.float()
    print(f"[{tag}] min={x.min().item():.3f} max={x.max().item():.3f} mean={x.mean().item():.3f} "
          f"std={x.std().item():.3f} nan={torch.isnan(x).any().item()}", flush=True)


pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
vae32 = AutoencoderKLCogVideoX.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
vae32.enable_slicing()
vae32.enable_tiling()


def gen(tag, prompt):
    t0 = time.time()
    lat = pipe(prompt=prompt, num_videos_per_prompt=1, num_inference_steps=50, num_frames=49,
               guidance_scale=6, generator=torch.Generator("cuda").manual_seed(42), output_type="latent").frames
    stats(f"{tag} latents", lat)
    vae32.to("cuda")
    with torch.no_grad():
        video = vae32.decode(lat.to("cuda", torch.float32).permute(0, 2, 1, 3, 4) / vae32.config.scaling_factor).sample
    vae32.to("cpu")
    torch.cuda.empty_cache()
    stats(f"{tag} decoded", video)
    frames = pipe.video_processor.postprocess_video(video=video.cpu(), output_type="pil")[0]
    export_to_video(frames, str(OUT / f"{tag}.mp4"), fps=8)
    print(f"[{tag}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


gen("baseline_panda", PANDA)

ckpt = sorted(Path("/kaggle/input").rglob("lora_weights.pt"))[0]
print("Using LoRA:", ckpt, flush=True)
pipe.transformer.add_adapter(LoraConfig(r=64, lora_alpha=64, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                                        init_lora_weights=True))
res = set_peft_model_state_dict(pipe.transformer, torch.load(ckpt, map_location="cpu"))
print("LoRA loaded, unexpected keys:", len(getattr(res, "unexpected_keys", None) or []), flush=True)
for p in pipe.transformer.parameters():
    if p.dtype != torch.float16:
        p.data = p.data.to(torch.float16)
gen("lora_anime", ANIME)
print("ALL_DONE", flush=True)
