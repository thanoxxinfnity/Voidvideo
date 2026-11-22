"""
Kaggle kernel (VFX battery variant): monsters / animals / crowds / VFX showcase for the 24 fps Wan LoRA (task wan_t2v_24). Same prompts with different timing
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
PROMPTS = [  # VFX battery: effect cards (configs/vfx_cards.json via scripts/expand_prompt.py) + the two AI-teacher VFX descriptions
    ('black_flash', 'Medium shot, static camera. In the center, a muscular man with pink-red spiky hair, sharp black tattoo lines on his face and arms, open dark robe, fierce red eyes. Wide aggressive stance, fists clenched. The character punches forward and lands a heavy hit with black flash. Effects: at the instant of the punch a ring of jagged black-and-red lightning cracks outward from the point of impact, the screen flashes white for one frame then black, debris and dust burst away, the background goes dark. Traditional 2D hand-drawn anime.', 720, 480),
    ('cursed_aura', 'Medium shot, static camera. In the center, a tall young man with spiky white hair and a black blindfold covering his eyes, dark high-collared uniform, relaxed confident smile. Loose, casual posture, one hand in his pocket. The character stands still while cursed aura surrounds him. Effects: a swirling dark-purple and blue energy aura wraps around the body and the arms, flickering wisps rise upward like smoke, the outline pulses with light. Traditional 2D hand-drawn anime.', 720, 480),
    ('blue_pull', 'Medium shot, static camera. In the center, a tall young man with spiky white hair and a black blindfold covering his eyes, dark high-collared uniform, relaxed confident smile. Loose, casual posture, one hand in his pocket. The character holds out his open palm with blue pull. Effects: a glowing blue sphere with a bright white core hovers in the palm, rings of light spiral inward, rubble and dust are sucked toward it in a spiral, the air distorts around the sphere. Traditional 2D hand-drawn anime.', 720, 480),
    ('red_blast', 'Medium shot, static camera. In the center, a muscular man with pink-red spiky hair, sharp black tattoo lines on his face and arms, open dark robe, fierce red eyes. Wide aggressive stance, fists clenched. The character opens his hand with red blast. Effects: a glowing red sphere charges in the open hand then bursts, a red shockwave ring expands, the ground cracks and everything near it is thrown outward with speed lines. Traditional 2D hand-drawn anime.', 720, 480),
    ('purple_beam', 'Medium shot, static camera. In the center, a tall young man with spiky white hair and a black blindfold covering his eyes, dark high-collared uniform, relaxed confident smile. Loose, casual posture, one hand in his pocket. The character raises both hands and fires purple beam. Effects: a huge purple-white energy beam fires forward, the edges crackle with violet lightning, the ground is carved into a trench, the screen shakes and the light washes out the background. Traditional 2D hand-drawn anime.', 720, 480),
    ('domain', 'Medium shot, static camera. In the center, a teenage girl with long black hair tied in a ponytail, large brown eyes, white school blouse and red scarf. Light quick steps, arms swinging. The character stands calm while domain expansion happens around her. Effects: the background warps and melts, the scene fills with endless repeating shapes and a dark starry void, hard black-and-white contrast, the character stands calm at the centre as the space around changes. Traditional 2D hand-drawn anime.', 720, 480),
    ('vortex', 'Wide shot, the camera pushes into a ruined night city street. A swirling purple and red energy vortex tears open in the air, lightning branching through it, chunks of road and concrete and building debris spiral into it, dust and smoke billow, a bright purple flash and a white-red shockwave burst. Anime special effects, hand-drawn.', 720, 480),
    ('storm', 'Wide low-angle shot of a dark rain-soaked city street at night. Lightning forks across black storm clouds and strikes, a purple and red energy tornado rises from the ground tearing up the road, debris and flames fly around it, a huge purple shockwave blast erupts and fades into red embers and smoke. Anime special effects, hand-drawn.', 720, 480),
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
