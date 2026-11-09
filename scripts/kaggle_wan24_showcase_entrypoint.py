"""
Kaggle kernel: monsters / animals / crowds / VFX showcase for the 24 fps Wan LoRA (task wan_t2v_24). Same prompts with different timing
tags ("animated on ones" / "on threes with held frames") show whether the tag controls how many distinct
drawings the model makes. 49 frames at 24 fps, exported at 24 fps. Attach the 24 fps checkpoint dataset.
"""
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers==0.35.2", "transformers==4.57.3",
                "accelerate>=0.33", "ftfy", "sentencepiece", "imageio[ffmpeg]"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import torch  # noqa: E402
from diffusers import AutoencoderKLWan, WanPipeline  # noqa: E402
from diffusers.utils import export_to_video  # noqa: E402
from peft import LoraConfig, set_peft_model_state_dict  # noqa: E402

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OUT = Path("/kaggle/working")
STYLE = "2D anime film style, hand-drawn cel animation, clean ink outlines, flat colors, smooth motion"
PROMPTS = [  # (tag, prompt, width, height): what the 1244-window 24 fps LoRA should now handle (monsters, animals, crowds, VFX)
    ("monster", "Close-up shot, slight zoom-in as a monster lunges forward. One monster with red and black fur, yellow "
                "eyes and sharp teeth snarls with its mouth open, its body tensed for attack, outdoors in daylight. "
                "Traditional 2D hand-drawn anime, smear frames.", 720, 480),
    ("cheetah", "Wide shot, camera tracks sideways. A cheetah sprints across a golden savanna, its legs stretching "
                "and folding in a fast gallop, its spotted body flexing, dust rising behind it. Traditional 2D "
                "hand-drawn anime.", 720, 480),
    ("crowd", "Wide high-angle shot, static camera. A big crowd of about twenty people in different clothes runs "
              "down a wide city street towards the camera, arms swinging, some looking back in panic. Traditional 2D "
              "hand-drawn anime.", 720, 480),
    ("lightning", "Medium shot, static camera. A young man with spiky black hair raises his fist and bright blue "
                  "lightning crackles around his arm, branching sparks flashing and lighting up his face and the dark "
                  "walls. Anime special effects, hand-drawn.", 720, 480),
    ("ice", "Wide shot, static camera. A girl swings her hand and a wave of ice spreads across the ground, sharp ice "
            "crystals shooting up and freezing a wall, white frost mist drifting. Anime special effects, hand-drawn.",
     720, 480),
    ("fire", "Medium shot, slow push-in. A swordsman swings a sword wrapped in orange flames, the fire trailing "
             "behind the blade in a bright arc, sparks and embers flying, smoke curling up. Anime special effects, "
             "hand-drawn.", 720, 480),
]
NEGATIVE = ("Bright tones, overexposed, static, blurred details, worst quality, low quality, JPEG artifacts, "
            "ugly, deformed, extra fingers, poorly drawn hands, poorly drawn faces, fused fingers, three legs")


def stats(tag, x):
    x = x.float()
    print(f"[{tag}] min={x.min().item():.3f} max={x.max().item():.3f} std={x.std().item():.3f} "
          f"nan={torch.isnan(x).any().item()}", flush=True)


vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.float16)
ckpt = sorted(Path("/kaggle/input").rglob("lora_weights.pt"))[0]
print("Using LoRA:", ckpt, flush=True)
pipe.transformer.add_adapter(LoraConfig(r=64, lora_alpha=64, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                                        init_lora_weights=True))
res = set_peft_model_state_dict(pipe.transformer, torch.load(ckpt, map_location="cpu"))
print("LoRA loaded, unexpected keys:", len(getattr(res, "unexpected_keys", None) or []), flush=True)
for p in pipe.transformer.parameters():
    if p.dtype != torch.float16:
        p.data = p.data.to(torch.float16)
pipe.text_encoder.to(torch.bfloat16)
pipe.enable_model_cpu_offload()
print("Loaded Wan2.1 1.3B", flush=True)

for tag, prompt, width, height in PROMPTS:
    t0 = time.time()
    try:
        with torch.no_grad():
            pe, ne = pipe.encode_prompt(prompt, NEGATIVE, do_classifier_free_guidance=True, device="cuda")
        stats(f"{tag} embeds", pe)
        frames = pipe(prompt_embeds=pe.to(torch.float16), negative_prompt_embeds=ne.to(torch.float16),
                      height=height, width=width, num_frames=49, num_inference_steps=30, guidance_scale=5.0,
                      generator=torch.Generator("cuda").manual_seed(42), output_type="np").frames[0]
        print(f"[{tag}] frames {frames.shape} nan={bool(torch.isnan(torch.tensor(frames)).any())} "
              f"std={float(frames.std()):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        export_to_video(frames, str(OUT / f"wan24show_{tag}.mp4"), fps=24)
    except Exception as e:
        print(f"[{tag}] FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        torch.cuda.empty_cache()
print("ALL_DONE", flush=True)
