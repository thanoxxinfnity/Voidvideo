"""
Kaggle kernel: layered crowd scene with the 24 fps Wan LoRA. Plans the layers (scripts/crowd_plan.py), generates every
layer alone (background plate, each hero from its character card, a pool of extras on a green screen), then composites
them (scripts/crowd_compose.py) so no two people share a generation. Edit the constants below. Attach the 24 fps
checkpoint dataset (KAGGLE_CHECKPOINT_DATASET_SLUG). ~11 min per layer on a T4: heroes + pool + 1 background.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

SCENE = "a ruined city street at dusk with an orange sky and drifting smoke"
HEROES = "kaito,ryu,mina"
EXTRAS, POOL = 14, 4
ACTION = "cheering with fists raised"
HERO_ACTIONS = "raising two fingers and smiling;lunging forward with a punch;jumping and waving both arms"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers==0.35.2", "transformers==4.57.3",
                "accelerate>=0.33", "ftfy", "sentencepiece", "imageio[ffmpeg]"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
REPO = Path("/kaggle/working/voidvideo")
subprocess.run(["git", "clone", "--depth", "1", "--branch", "claude/claude-md-jfha3c",
                "https://github.com/thanoxxinfnity/Voidvideo.git", str(REPO)], check=True)
PLAN = Path("/kaggle/working/crowd_plan")
subprocess.run([sys.executable, str(REPO / "scripts/crowd_plan.py"), "--scene", SCENE, "--heroes", HEROES,
                "--extras", str(EXTRAS), "--pool", str(POOL), "--action", ACTION, "--hero-actions", HERO_ACTIONS,
                "--out", str(PLAN)], check=True)

import torch  # noqa: E402
from diffusers import AutoencoderKLWan, WanPipeline  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402
from peft import LoraConfig, set_peft_model_state_dict  # noqa: E402

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
NEGATIVE = ("Bright tones, overexposed, static, blurred details, worst quality, low quality, JPEG artifacts, "
            "ugly, deformed, extra fingers, poorly drawn hands, poorly drawn faces, fused fingers, three legs")
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.float16)
ckpt = sorted(Path("/kaggle/input").rglob("lora_weights.pt"))[0]
print("Using LoRA:", ckpt, flush=True)
pipe.transformer.add_adapter(LoraConfig(r=64, lora_alpha=64, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                                        init_lora_weights=True))
set_peft_model_state_dict(pipe.transformer, torch.load(ckpt, map_location="cpu"))
for p in pipe.transformer.parameters():
    if p.dtype != torch.float16:
        p.data = p.data.to(torch.float16)
pipe.text_encoder.to(torch.bfloat16)
pipe.enable_model_cpu_offload()

for job in json.load(open(PLAN / "jobs.json")):
    t0 = time.time()
    dst = PLAN / job["out"]
    if dst.exists():
        continue
    try:
        with torch.no_grad():
            pe, ne = pipe.encode_prompt(job["prompt"], NEGATIVE, do_classifier_free_guidance=True, device="cuda")
        frames = pipe(prompt_embeds=pe.to(torch.float16), negative_prompt_embeds=ne.to(torch.float16),
                      height=job["height"], width=job["width"], num_frames=49, num_inference_steps=30,
                      guidance_scale=5.0, generator=torch.Generator("cuda").manual_seed(job["seed"]),
                      output_type="np").frames[0]
        dst.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(dst), fps=24)
        print(f"[{job['name']}] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    except Exception as e:
        print(f"[{job['name']}] FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        torch.cuda.empty_cache()

subprocess.run([sys.executable, str(REPO / "scripts/crowd_compose.py"), str(PLAN / "layout.json"),
                "/kaggle/working/crowd_scene.mp4"], check=True)
for f in PLAN.rglob("*.mp4"):                       # keep the layers next to the result for inspection
    pass
print("ALL_DONE", flush=True)
