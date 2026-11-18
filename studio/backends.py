"""
Generation backends. generate(req, progress(step, total), cancelled()) -> frames uint8 (T,H,W,3).

FakeBackend : synthetic clips, no GPU (tests the server / queue / pipeline).
WanBackend  : Wan2.1-T2V-1.3B + our LoRA for t2v (the settings verified on Kaggle T4), and Wan2.1-VACE-1.3B for keyframes /
              extend / inpaint / outpaint / control (v2v, motion transfer) / reference images. The VACE paths are NOT yet run on a GPU.
"""
import hashlib
from pathlib import Path

import numpy as np

from . import modes

NEGATIVE = ("Bright tones, overexposed, static, blurred details, worst quality, low quality, JPEG artifacts, "
            "ugly, deformed, extra fingers, poorly drawn hands, poorly drawn faces, fused fingers, three legs")


class FakeBackend:
    def generate(self, req, progress, cancelled):
        w, h, n = req.get("width", 360), req.get("height", 240), req.get("num_frames", 49)
        steps = int(req.get("steps", 6))
        col = np.frombuffer(hashlib.md5(req["prompt"].encode()).digest()[:3], np.uint8).astype(int)
        f = np.zeros((n, h, w, 3), np.uint8)
        f[:] = (40, 50, 80)
        for i in range(n):
            x = 20 + i * (w - 80) // n
            f[i, h // 3:h // 3 + 90, x:x + 40] = col
        if req.get("_prev") is not None and req.get("mode") == "extend":
            k = req.get("keep", 17)
            f[:k] = np.stack([np.asarray(modes.fit(p, w, h)) for p in req["_prev"][-k:]])
        for s in range(steps):
            import time
            time.sleep(float(req.get("fake_delay", 0.05)))
            progress(s + 1, steps)
            if cancelled():
                break
        return f


class WanBackend:
    T2V = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    VACE = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"

    def __init__(self, lora_path, use_vace=True, retime_fps=24):
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline
        from peft import LoraConfig, set_peft_model_state_dict
        self.torch = torch
        vae = AutoencoderKLWan.from_pretrained(self.T2V, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanPipeline.from_pretrained(self.T2V, vae=vae, torch_dtype=torch.float16)
        self.pipe.transformer.add_adapter(LoraConfig(r=64, lora_alpha=64, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                                                     init_lora_weights=True))
        set_peft_model_state_dict(self.pipe.transformer, torch.load(lora_path, map_location="cpu"))
        for p in self.pipe.transformer.parameters():
            if p.dtype != torch.float16:
                p.data = p.data.to(torch.float16)
        self.pipe.text_encoder.to(torch.bfloat16)
        self.pipe.enable_model_cpu_offload()
        self.vace = None
        self.use_vace = use_vace
        self.lora_path = lora_path

    def _vace(self):
        if self.vace is None:
            from diffusers import WanVACEPipeline
            torch = self.torch
            self.vace = WanVACEPipeline.from_pretrained(self.VACE, text_encoder=self.pipe.text_encoder, tokenizer=self.pipe.tokenizer,
                                                        vae=self.pipe.vae, torch_dtype=torch.float16)
            try:       # experiment: the T2V LoRA shares the base blocks with VACE; vace_blocks keep their init (no effect)
                from peft import LoraConfig, set_peft_model_state_dict
                self.vace.transformer.add_adapter(LoraConfig(r=64, lora_alpha=64, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                                                             init_lora_weights=True))
                set_peft_model_state_dict(self.vace.transformer, torch.load(self.lora_path, map_location="cpu"))
                for p in self.vace.transformer.parameters():
                    if p.dtype != torch.float16:
                        p.data = p.data.to(torch.float16)
            except Exception as e:
                print("VACE: LoRA not applied:", type(e).__name__, str(e)[:200], flush=True)
            self.vace.enable_model_cpu_offload()
        return self.vace

    def generate(self, req, progress, cancelled):
        torch = self.torch
        mode, steps = req.get("mode", "t2v"), int(req.get("steps", 30))
        w, h, n = int(req.get("width", 720)), int(req.get("height", 480)), int(req.get("num_frames", 49))
        gen = torch.Generator("cuda").manual_seed(int(req.get("seed", 42)))
        with torch.no_grad():
            pe, ne = self.pipe.encode_prompt(req["prompt"], NEGATIVE, do_classifier_free_guidance=True, device="cuda")
        pipe, kw = self.pipe, {}
        if mode != "t2v":
            pipe = self._vace()
            if mode == "keyframes":
                video, mask = modes.keyframes([(k["frame"], k["image"]) for k in req["_keys"]], n, w, h)
            elif mode == "extend":
                video, mask = modes.extend(req["_prev"] if "_prev" in req else req["_frames"], n, w, h, req.get("keep", 17))
            elif mode == "inpaint":
                video, mask = modes.inpaint(req["_frames"][:n], req["_mask"][:n] if req["_mask"].ndim == 3 else req["_mask"], w, h)
            elif mode == "outpaint":
                video, mask = modes.outpaint(req["_frames"][:n], w, h)
            elif mode in ("v2v", "motion_transfer"):
                video, mask = modes.control(req["_frames"][:n], req.get("control", "edges"), w, h)
            elif mode == "ref2v":
                video, mask = None, None
            else:
                raise ValueError(f"unknown mode {mode}")
            kw = {"video": video, "mask": mask, "reference_images": req.get("_refs") or None,
                  "conditioning_scale": float(req.get("control_scale", 1.0))}
        if mode == "ref2v" or (mode != "t2v" and req.get("_refs")):
            kw["reference_images"] = req.get("_refs")

        scale = float(req.get("lora_scale", 1.0 if req.get("style", "anime") == "anime" else 0.0))   # 0 = plain Wan base model
        try:
            pipe.transformer.set_adapters("default", scale)
        except Exception as e:
            print("set_adapters failed:", type(e).__name__, str(e)[:120], flush=True)

        def cb(p, i, t, kwargs):
            progress(i + 1, steps)
            if cancelled():
                p._interrupt = True
            return kwargs

        frames = pipe(prompt_embeds=pe.to(torch.float16), negative_prompt_embeds=ne.to(torch.float16), height=h, width=w,
                      num_frames=n, num_inference_steps=steps, guidance_scale=float(req.get("guidance", 5.0)), generator=gen,
                      output_type="np", callback_on_step_end=cb, **kw).frames[0]
        return (np.clip(frames, 0, 1) * 255).astype(np.uint8) if frames.dtype != np.uint8 else frames
