"""
Kaggle kernel: Wan2.1-T2V-1.3B + the VoidVideo LoRA (trained by training/train_wan_lora.py), expression prompts written
in the style of the training captions, at the training resolution (480x720, 33 frames). fp16 transformer, text encoder run in
bf16 (UMT5 overflows in fp16), fp32 VAE. Attach the checkpoint dataset holding lora_weights.pt.
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
# Expression prompts follow the wording of the 110 training captions ("expression shifting from X to Y",
# "eyes widening", "laughing with mouth wide open", close-up, static camera, face towards the camera).
PROMPTS = [
    ("exp_shock", "Close-up of a short-haired girl with large blue eyes facing the camera, static shot. Her expression "
                  "shifts from neutral to shocked: her eyes widen sharply, her eyebrows shoot up and her mouth drops open."),
    ("exp_shock_style", "Close-up of a short-haired girl with large blue eyes facing the camera, static shot. Her expression "
                        "shifts from neutral to shocked: her eyes widen sharply, her eyebrows shoot up and her mouth drops "
                        "open. " + STYLE),
    ("exp_angry", "Close-up of a spiky-haired black-haired boy facing the camera, static shot. His expression shifts from a "
                  "calm smile to an angry scowl: his eyebrows pull down, his eyes narrow and he clenches his teeth."),
    ("exp_sad", "Close-up of a long-haired girl with brown eyes facing the camera, static shot. Her expression shifts from "
                "a happy smile to sadness: her smile fades, her eyes fill with tears and her lower lip trembles."),
    ("exp_nervous", "Close-up of a brown-haired boy in a red hoodie facing the camera, static shot. He grins nervously with "
                    "wide eyes, then his grin tightens into a worried grimace, a sweat drop on his temple."),
    ("exp_laugh", "Close-up of an elderly white-haired man in a black suit and red bow tie facing the camera, static shot, "
                  "laughing loudly with his mouth wide open, his eyes squeezed shut."),
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

for tag, prompt in PROMPTS:
    t0 = time.time()
    try:
        with torch.no_grad():
            pe, ne = pipe.encode_prompt(prompt, NEGATIVE, do_classifier_free_guidance=True, device="cuda")
        stats(f"{tag} embeds", pe)
        frames = pipe(prompt_embeds=pe.to(torch.float16), negative_prompt_embeds=ne.to(torch.float16),
                      height=480, width=720, num_frames=33, num_inference_steps=30, guidance_scale=5.0,
                      generator=torch.Generator("cuda").manual_seed(42), output_type="np").frames[0]
        print(f"[{tag}] frames {frames.shape} nan={bool(torch.isnan(torch.tensor(frames)).any())} "
              f"std={float(frames.std()):.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        export_to_video(frames, str(OUT / f"wanlora_{tag}.mp4"), fps=16)
    except Exception as e:
        print(f"[{tag}] FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        torch.cuda.empty_cache()
print("ALL_DONE", flush=True)
