"""
Kaggle kernel: LoRA-driven shots (slash, charge, blast), two seeds each, so the best take can be picked and stitched locally.
Settings are the verified ones: fp16 pipeline, 50 steps, no negative prompt, fp32 VAE decode.
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
STYLE = "2D anime film style, hand-drawn cel animation, clean ink outlines, flat colors, smooth motion"
# One character and one action per shot: the 2B model garbles scenes with two interacting characters.
BASE_SHOTS = [
    ("slash",
     "Medium shot, static camera. A shirtless muscular man with spiky pink hair and black tattoo markings on his face "
     "swings his right arm in one fast diagonal slash, thin white-red slash lines streak through the air in front of "
     "him, ruined city street behind him. " + STYLE),
    ("charge",
     "Close-up, static camera. A young man with white hair and a black blindfold holds two fingers of his right hand "
     "forward. A blue glowing orb and a red glowing orb float in from the left and right and slowly merge in front of "
     "his fingers into one bright purple glowing orb. " + STYLE),
    ("blast",
     "Wide shot, static camera. A huge glowing purple energy sphere shoots forward down an empty ruined city street "
     "as a giant purple beam, destroying buildings, debris and dust flying, bright white flash. " + STYLE),
]
SEEDS = [11, 22]
SHOTS = [(f"{name}_s{seed}", prompt, seed) for name, prompt in BASE_SHOTS for seed in SEEDS]


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


def interpolate(src):
    import imageio_ffmpeg
    dst = src.with_name(src.stem + "_24fps.mp4")
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(src), "-filter:v",
                        "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:vsbmc=1", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", str(dst)], check=True, capture_output=True, timeout=400)
        return dst
    except Exception as e:
        print("24fps interpolation failed, keeping 8fps:", e, flush=True)
        return None


def gen(tag, prompt, seed):
    t0 = time.time()
    lat = pipe(prompt=prompt, num_videos_per_prompt=1, num_inference_steps=50, num_frames=49,
               guidance_scale=6, generator=torch.Generator("cuda").manual_seed(seed), output_type="latent").frames
    vae32.to("cuda")
    with torch.no_grad():
        video = vae32.decode(lat.to("cuda", torch.float32).permute(0, 2, 1, 3, 4) / vae32.config.scaling_factor).sample
    vae32.to("cpu")
    torch.cuda.empty_cache()
    stats(f"{tag} decoded", video)
    frames = pipe.video_processor.postprocess_video(video=video.cpu(), output_type="pil")[0]
    raw = OUT / f"{tag}.mp4"
    export_to_video(frames, str(raw), fps=8)
    print(f"[{tag}] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return interpolate(raw) or raw


ckpt = sorted(Path("/kaggle/input").rglob("lora_weights.pt"))[0]
print("Using LoRA:", ckpt, flush=True)
pipe.transformer.add_adapter(LoraConfig(r=64, lora_alpha=64, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                                        init_lora_weights=True))
res = set_peft_model_state_dict(pipe.transformer, torch.load(ckpt, map_location="cpu"))
print("LoRA loaded, unexpected keys:", len(getattr(res, "unexpected_keys", None) or []), flush=True)
for p in pipe.transformer.parameters():
    if p.dtype != torch.float16:
        p.data = p.data.to(torch.float16)
for tag, prompt, seed in SHOTS:
    gen(tag, prompt, seed)
print("ALL_DONE", flush=True)
