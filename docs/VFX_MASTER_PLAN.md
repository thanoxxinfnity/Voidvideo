# VoidVideo VFX Master Plan (all-in-one VFX / motion-graphics / cinematic LoRA)

Based on the user's category list (nature elements, destruction, magic/sci-fi, transformations, quiz and motion graphics,
camera, lighting, time/physics). What exists, what is missing, and the order to do it in.

## 0. Corrections to the generic advice that came with the list
- **bf16 is NOT available on our GPUs** (Kaggle T4). We train and sample in **fp16 with fp32 LoRA weights and an fp32 VAE**;
  bf16 attention falls back to a ~70 GB score matrix on a T4. bf16 only makes sense on A100/H100.
- **24 GB+ cards / A100 rentals are optional**, not required: Wan2.1-1.3B trains on a free T4 (10.8 GB, ~154 s/step at 49 frames).
  A paid A100/H100 (~$1-3/h) is only needed for the 14B model.
- **Trainer:** we already have `training/train_wan_lora.py` (flow matching, rank 64, 8-bit AdamW, Kaggle relay). musubi-tuner /
  diffusion-pipe are not needed. **Captioner:** Qwen3-VL-4B (newer than Qwen2-VL), already built.
- **Hindi captions:** Wan's text encoder (UMT5) is multilingual but the LoRA is trained on English captions. Keep captions in English
  and put a Hindi/Hinglish -> English prompt translator in front (the studio app), instead of training on Hindi text.
- **Epochs:** we count steps (batch 1 x accumulation 4); ~500-1500 steps for 2-5k clips, checkpoint every 25.
- **Scraping:** use only data we may train on: Sakugabooru dataset (private use), CC0/CC-BY footage, and **synthetic data we render
  ourselves** (unlimited, perfect captions). Do not scrape paid or copyrighted stock sites.

## 1. Category -> data source map
| Category (user list) | Source | Status |
|---|---|---|
| Fire / water / smoke / lightning / ice / explosions / debris / beams / impact / wind / morph (anime VFX) | Sakugabooru tags, 10 effect classes x 70 clips | batch 3 running (700 clips) |
| Monsters, animals, crowds, fighting, acting | Sakugabooru (creatures 300, animals 250, crowd 150 ...) | extracted (v3, 4072 windows) |
| Magic portals, teleport, shield, aura, black hole, galaxy | Sakuga "effects" + VLM text search (Section 3) | TODO |
| Realistic fire/water/smoke/explosion footage | CC0/CC-BY sources: Wikimedia Commons, Blender open movies, NASA, Pexels/Pixabay (free API key, licence allows training? check each) | TODO (needs the user's free API keys for Pexels/Pixabay) |
| Slow motion, time freeze, hyper-lapse, zero gravity | VLM captions flag "slow motion" etc.; synthetic retime of existing clips (speed ramps) | TODO |
| Camera moves: pan/tilt/dolly/orbit/shake/aerial/low-angle/macro | VLM caption field "camera"; synthetic virtual-camera moves (pan/zoom/shake applied to clips and to rendered scenes) | partly (caption field exists) |
| Quiz, countdown, 3D text, vs/split screen, coin rain, loops, transitions | **Synthetic motion-graphics generator** (Python + ffmpeg, rendered at 24 fps, vertical and horizontal) | TODO (CPU only, no GPU quota) |
| Cyberpunk neon / hologram / glitch | Sakuga + synthetic shader-style generator | TODO |

## 2. One LoRA or several?
One rank-64/128 LoRA covering everything works, but styles dilute. Recommended: **one VFX/motion master LoRA + the anime LoRA**,
combined at inference with weights. Category and trigger tokens go in every caption: `voidvfx, vfx-fire, ...` (the first token
activates the VFX LoRA, the second picks the effect).

## 3. Caption structure (the five required parts)
(a) subject and action, (b) which effect and its physics, (c) camera angle and movement, (d) light, colour, background,
(e) speed. `scripts/vlm_caption_clips.py` already returns count / subjects / action / camera / setting / effects / style;
to add: `speed` (slow motion / normal / fast) and `physics`. Then prefix the trigger and category tokens when assembling the dataset.

## 4. Quota-aware order of work (30 GPU h per week, resets Saturday 00:00Z)
1. (running) 1244-window 24 fps training -> test monsters / crowd / timing.
2. 14 Nov: full VLM captioning of v3 + VFX batch (~4-5 h).
3. Dataset v4 = v3 + VFX batch + synthetic quiz/motion graphics (CPU work done before the reset).
4. Train v4 (cache the encodes; ~2000-window balanced subset first, then the rest).
5. Evaluate per category (6 prompts each), fix weak categories by adding data, repeat.
6. Studio app: Hinglish prompt -> English, take picking, retime, upscale, layers.

## 5. Data status (8 Nov)
Dataset `voidvideo-anime-clips-24fps-v4`: 5835 windows (49 frames, 24 fps, 720x480 or 480x720) = Vault 286 + Sakuga 3786
(creatures 596, animals 463, crowd 292, fighting 578, expressions ~690, locomotion ~470, secondary motion ~400, vfx 524)
+ per-effect VFX 1343 (ice, impact, beam, explosion, morph, fire, wind/debris, lightning, smoke/sparks, water, ~135 each)
+ synthetic motion graphics 420 (countdown, quiz, versus, geometric loops, waves, coin rain, tunnel).

Known gaps: **gestures are thin (58 windows)** - Sakugabooru has no gesture tag. Fix: after VLM captioning, search the
captions of "character acting" / expression windows for hand-gesture words (waves, points, clenches fist, beckons, shrugs)
and re-bucket them; add a synthetic hand-gesture set later. Camera moves, slow motion/time freeze and portals/auras have no
direct tag either: rely on the VLM `camera`/`speed` fields and the Blender virtual-camera set (Section 1, TODO).
