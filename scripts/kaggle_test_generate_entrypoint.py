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


def run(pipe, tag: str, steps: int, save: bool = True, frames_n: int = 49, embeds=None, verbose=False):
    t0 = time.time()

    def log_step(_pipe, i, t, kw):
        lat = kw["latents"]
        print(f"[{tag}] step {i}: latents std={lat.float().std().item():.3f} "
              f"absmax={lat.float().abs().max().item():.2f} nan={torch.isnan(lat).any().item()}", flush=True)
        return kw

    kwargs = dict(prompt_embeds=embeds[0], negative_prompt_embeds=embeds[1]) if embeds else \
        dict(prompt=PROMPT, negative_prompt=NEGATIVE)
    if verbose:
        kwargs.update(callback_on_step_end=log_step, callback_on_step_end_tensor_inputs=["latents"])
    latents = pipe(
        num_frames=frames_n, height=480, width=720,
        num_inference_steps=steps, guidance_scale=GUIDANCE,
        generator=torch.Generator("cpu").manual_seed(SEED), output_type="latent", **kwargs,
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

# Stage 1: tiny probes (9 frames, 4 steps, no LoRA) over text-encoder / transformer dtype combinations.
# CogVideoX-2B in plain fp16 gave flat (all ~-1) frames, bf16 OOMs at full length on a T4 (no
# memory-efficient attention), so find which combination actually denoises.
pipe = load_pipe(torch.float16)


def embeds_for(te_dtype, out_dtype):
    pipe.text_encoder.to(te_dtype)
    pipe.text_encoder.to("cuda")
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(PROMPT, NEGATIVE, do_classifier_free_guidance=True, device="cuda")
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    print(f"embeds te={te_dtype}: nan={torch.isnan(pe).any().item()} inf={torch.isinf(pe).any().item()} "
          f"absmax={pe.float().abs().max().item():.1f} std={pe.float().std().item():.3f}", flush=True)
    return pe.to(out_dtype), ne.to(out_dtype)


variants = [
    ("fp16_te16", torch.float16, torch.float16),
    ("fp16_tebf16", torch.bfloat16, torch.float16),
    ("bf16_tebf16", torch.bfloat16, torch.bfloat16),
    ("fp32_tebf16", torch.bfloat16, torch.float32),
]
chosen = None
for name, te_dtype, tf_dtype in variants:
    try:
        emb = embeds_for(te_dtype, tf_dtype)
        pipe.transformer.to(tf_dtype)
        spread, _ = run(pipe, f"probe_{name}", 4, save=False, frames_n=9, embeds=emb, verbose=True)
    except Exception as e:  # keep probing the remaining variants
        print(f"[probe_{name}] FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        torch.cuda.empty_cache()
        continue
    if spread > 0.1 and chosen is None:
        chosen = (name, te_dtype, tf_dtype)
assert chosen, "no dtype combination decodes to a real image -- see probe lines in the log"
name, te_dtype, dtype = chosen
print("Using", name, flush=True)
if dtype != torch.float16:
    # bf16/fp32 attention falls back to a ~70GB score matrix at 49 frames on a T4, so the full
    # clip cannot run in this setup; stop here and let the probe lines in the log decide the next step.
    print("PROBES_DONE: only a non-fp16 variant works; full-length run would OOM, stopping.", flush=True)
    sys.exit(0)
# Stage 2: full-quality clips; LoRA first so it is saved even if the second run runs out of time.
emb = embeds_for(te_dtype, dtype)
pipe.transformer.to(dtype)
attach_lora(pipe, dtype)
run(pipe, "with_lora", STEPS, embeds=emb)
pipe.transformer.disable_adapters()
run(pipe, "base_no_lora", STEPS, embeds=emb)
print("ALL_DONE", flush=True)
