"""
Kaggle kernel: which open model should make GENERAL video (space, nature, animals, people) on a free T4? Same prompts, same seed,
three candidates, with time and peak GPU memory per clip so the choice is made on evidence:

  wan22_5b   Wan2.2-TI2V-5B   (Apache-2.0, 24 fps, newer/bigger than our 1.3B)       832x480, 49 frames
  wan21_1b   Wan2.1-T2V-1.3B  (base model without our anime LoRA, the current fallback) 832x480, 49 frames
  cogx_5b    CogVideoX-5B     (general, fp16-friendly)                                  720x480, 49 frames

No LoRA anywhere. Outputs /kaggle/working/<model>_<prompt>.mp4 and bench.json. A failed model is reported, not fatal.
Needs GPU + internet. About 1.5-2 h of GPU; run it once, after the weekly quota reset.
"""
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers==0.35.2", "transformers==4.57.3", "accelerate>=0.33",
                "ftfy", "sentencepiece", "imageio[ffmpeg]"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import torch  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402

OUT = Path("/kaggle/working")
PROMPTS = {
    "space": "A slow cinematic flight over the rings of Saturn, the golden planet glowing, tiny ice particles sparkling in the sunlight, "
             "deep black space full of stars, realistic, 35mm film look.",
    "ocean": "Aerial drone shot flying over turquoise ocean waves crashing on a rocky coast at sunrise, white foam, warm golden light, realistic.",
    "fox": "A red fox walks through fresh snow in a pine forest, its breath visible in the cold air, shallow depth of field, natural light, realistic.",
    "chef": "Close-up of a chef tossing vegetables in a flaming wok in a restaurant kitchen, steam and sparks rising, handheld camera, realistic.",
}
NEG = ("Bright tones, overexposed, static, blurred details, worst quality, low quality, JPEG artifacts, ugly, deformed, extra fingers, "
       "poorly drawn hands, poorly drawn faces, fused fingers, three legs")
RESULTS = []


def run(tag, pipe, w, h, fps, steps, guidance, neg=NEG, **kw):
    for name, prompt in PROMPTS.items():
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        try:
            frames = pipe(prompt=prompt, negative_prompt=neg, height=h, width=w, num_frames=49, num_inference_steps=steps,
                          guidance_scale=guidance, generator=torch.Generator("cuda").manual_seed(7), output_type="np", **kw).frames[0]
            bad = bool(torch.isnan(torch.tensor(frames)).any())
            export_to_video(frames, str(OUT / f"{tag}_{name}.mp4"), fps=fps)
            r = {"model": tag, "prompt": name, "minutes": round((time.time() - t0) / 60, 1), "nan": bad,
                 "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "std": round(float(frames.std()), 3)}
        except Exception as e:
            r = {"model": tag, "prompt": name, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            torch.cuda.empty_cache()
        print(r, flush=True)
        RESULTS.append(r)
        json.dump(RESULTS, open(OUT / "bench.json", "w"), indent=1)


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# 1) Wan2.2 5B (new VAE with 16x spatial compression: sizes must be multiples of 32)
try:
    from diffusers import AutoencoderKLWan, WanPipeline
    mid = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.float16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.enable_model_cpu_offload()
    run("wan22_5b", pipe, 832, 480, 24, 30, 5.0)
    free(pipe, vae)
except Exception as e:
    print("wan22_5b FAILED:", type(e).__name__, str(e)[:300], flush=True)
    RESULTS.append({"model": "wan22_5b", "error": str(e)[:200]})

# 2) Wan2.1 1.3B base
try:
    from diffusers import AutoencoderKLWan, WanPipeline
    mid = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.float16)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.enable_model_cpu_offload()
    run("wan21_1b", pipe, 832, 480, 24, 30, 5.0)
    free(pipe, vae)
except Exception as e:
    print("wan21_1b FAILED:", type(e).__name__, str(e)[:300], flush=True)
    RESULTS.append({"model": "wan21_1b", "error": str(e)[:200]})

# 3) CogVideoX-5B (the settings verified earlier in this project: fp16, fp32 VAE decode, 50 steps, no negative prompt)
try:
    from diffusers import CogVideoXPipeline
    pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.float16)
    pipe.vae.to(torch.float32)
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()
    run("cogx_5b", pipe, 720, 480, 8, 50, 6.0, neg=None)
    free(pipe)
except Exception as e:
    print("cogx_5b FAILED:", type(e).__name__, str(e)[:300], flush=True)
    RESULTS.append({"model": "cogx_5b", "error": str(e)[:200]})

json.dump(RESULTS, open(OUT / "bench.json", "w"), indent=1)
print("ALL_DONE", flush=True)
