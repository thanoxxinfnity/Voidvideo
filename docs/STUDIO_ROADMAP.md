# VoidVideo Studio: the 8 systems, honestly

Base: Wan2.1-T2V-1.3B + our 24 fps anime LoRA, free Kaggle T4s. This is NOT above Runway / Kling / Sora: those are 10-100x larger models.
What the studio adds is control, consistency and finishing on top of our model. Status legend: **CPU-tested** = runs and is
tested here without a GPU; **GPU-untested** = code written against the documented API, never run on a GPU yet; **later** = not built.

Code: `studio/` (tests: `python -m studio.selftest`, `python -m studio.test_api`), Kaggle server: `scripts/kaggle_studio_entrypoint.py`.
The training code (`training/`, `train_wan_lora.py`) is untouched.

| # | System | Status | How |
|---|---|---|---|
| 1 | Text-to-video | GPU-verified (earlier kernels) | `WanBackend` t2v, same settings as the showcase clips |
| 1 | Image-to-video, start/middle/end keyframes | GPU-untested | `modes.keyframes` -> Wan2.1-VACE-1.3B (`WanVACEPipeline`, in diffusers 0.35.2) |
| 1 | Video-to-video restyle | GPU-untested | `modes.control` (edges now; depth via Depth-Anything-V2-Small; pose) -> VACE |
| 1 | Adaptive FPS | works (earlier) | `scripts/anime_retime.py --adaptive` runs first in `studio/pipeline.finish` |
| 2 | Motion transfer from a real video | GPU-untested | pose/depth control video + reference image -> VACE (`mode: motion_transfer`) |
| 2 | Motion brush / multi-subject masks | partly | `inpaint` with a mask video keeps the rest locked; per-subject motion needs SAM2 masks (later) |
| 2 | Body + finger pose control | later | `control="pose"` calls controlnet_aux OpenPose with hands, untested; DWPose is not wired |
| 2 | Same face / clothes | GPU-untested | VACE reference images + the memory bank (`studio/memory.py`). IP-Adapter / FaceID are Stable-Diffusion adapters: they do NOT work on Wan, so they are not used |
| 3 | Camera paths, dolly, FPV, shake | CPU-tested | `studio/camera.py`: spline path on the finished clip (crop-in), impact-timed shake |
| 3 | Orbit / parallax | CPU-tested (synthetic depth) | 2.5D parallax from a depth map; small moves only, big moves show holes |
| 3 | Depth of field / focus pull | CPU-tested (synthetic depth) | `camera.depth_of_field`; depth from Depth-Anything-V2-Small (CPU-capable, not run here) |
| 4 | Video inpainting, outpainting 9:16 -> 16:9, character swap | GPU-untested | `modes.inpaint/outpaint` -> VACE; swap = inpaint + reference image |
| 4 | Auto-rotoscoping into layers | later | needs SAM2 video predictor; not written (unverified API) |
| 5 | Impact frames + speed lines | CPU-tested | `studio/impact.py` (manga-style B/W hit frames, radial / horizontal lines, drawn on twos) |
| 5 | Cloth / hair / fire physics | not an engine | it comes from the model's training data (secondary-motion + VFX sets), not a simulator |
| 5 | Auto-relighting | CPU-tested | `post.relight`: a painted directional light + tint, good for mood, no real 3D shadows |
| 5 | Consistency memory bank | CPU-tested | `studio/memory.py` + `remember_as` / `remember_refs` in jobs; refs feed VACE |
| 6 | Foley sound effects | CPU-tested | `studio/foley.py` synthesises basic SFX on impact frames and prompt keywords (stand-ins, not AI Foley; MMAudio / HunyuanVideo-Foley later) |
| 6 | Lip-sync | later | needs TTS first (your own voice data) and a lip-sync model; anime faces are weak for Wav2Lip-type models |
| 6 | Video extend + multi-shot storyboard to 30-60 s | CPU-tested logic, GPU-untested model | `pipeline.run_storyboard`, `mode: extend` keeps 17 frames of the previous shot |
| 7 | Prompt enhancer | CPU-tested | `studio/prompt_enhancer.py`: Hinglish -> English, @character / #effect cards, fills camera, light, effects, timing |
| 7 | Grade / LUT / bloom / grain | CPU-tested | `studio/post.py` (ffmpeg) |
| 7 | Real-ESRGAN upscale, RIFE 60 fps | hooks only | used when `realesrgan-ncnn-vulkan` / `rife-ncnn-vulkan` are installed, else lanczos / skipped. For anime keep RIFE off on character action: it fills the holds that make ones / twos / threes |
| 8 | API /generate /status /cancel + queue + tunnel | CPU-tested (fake backend) | `studio/server.py`: API key, bounded queue (429 when full), cancel, `% progress`, thumbnail preview; Kaggle entrypoint opens a Cloudflare quick tunnel |

## Things you must know
- **GPU quota:** a running Kaggle server burns the 30 h/week quota. The entrypoint shuts itself down after `IDLE_MIN` (30) minutes without a request.
- **Where the URL appears:** the kernel log (live on the notebook page). With `HF_TOKEN` and `HF_ENDPOINT_REPO` set it is also written to a private HF repo file.
- **Speed:** a 2 s clip is ~11 min on a T4, so "live" means progress % and a thumbnail when done, not a real-time stream.
- **LoRA on VACE:** the T2V LoRA shares the base blocks with VACE; whether it improves anime in VACE modes is an experiment.
- **Order for the first GPU test (after training ends):** 1) run `kaggle_studio_entrypoint.py`, 2) t2v through the API, 3) VACE `extend`, `outpaint`, `inpaint`, 4) reference images, 5) motion transfer.
