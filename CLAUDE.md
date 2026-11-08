# VoidVideo — Project Memory

Read this first. It captures context from prior sessions so you don't need
to re-derive it or ask the user to re-explain from scratch.

## What this project is
Hand-drawn-style anime video generation pipeline: LoRA fine-tuning on
**CogVideoX-2B** (Text-to-Video) and **CogVideoX-5B-I2V** (Image-to-Video),
trained via Kaggle's free GPU quota, served through a Gradio app. Silent
video only — no audio/voice generation.

## What's already built
- `training/`, `scripts/`, `configs/`, `app/` — full LoRA training +
  Kaggle-push + inference pipeline. See root `README.md` for the full
  end-to-end walkthrough and `data/README.md` for the clip spec (150-300
  clips, categorized: Locomotion/Gestures/Expressions/Secondary Motion).
- `upload-portal/` — a separate static website ("**VoidVideo Vault**") for
  collecting training clips by category, backed by Cloudinary. Two
  interchangeable serverless backends are included: `api/` (Vercel) and
  `netlify/functions/` (Netlify) — deploy to whichever platform works, see
  `upload-portal/README.md`.
- Everything is pushed to branch `claude/voidvideo-anime-model-bl0lor`.

## Current state (Nov 2026) — read this before anything else
**Branch `claude/claude-md-jfha3c`** has everything below. Kaggle account `tejassinghthanos` (token = KAGGLE_API_TOKEN,
HF token in `.env`; the user keeps a backup file of both and re-pastes them if a session starts empty).
- **Base model switched to Wan2.1-T2V-1.3B** (Apache-2.0, HF `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`), far better than CogVideoX-2B
  (hands, motion, expressions). CogVideoX LoRA (1000 steps) still exists as a fallback: dataset `voidvideo-lora-checkpoint-t2v`.
- **Wan LoRAs (rank 64):** `voidvideo-lora-checkpoint-wan-500` (8 fps clips, 500 steps), `voidvideo-lora-checkpoint-wan24`
  (continued on 24 fps windows, step 225). Training: `training/train_wan_lora.py`, tasks `wan_t2v` / `wan_t2v_24` in
  `configs/training_config.yaml`, launched with `scripts/push_training_kernel_to_kaggle.py --task wan_t2v_24 --max-train-steps N`
  (set `KAGGLE_DATASET_SLUG`, `KAGGLE_KERNEL_SLUG`, `KAGGLE_CHECKPOINT_DATASET_SLUG`, `KAGGLE_LATENT_CACHE_DATASET_SLUG` env vars).
  ~154 s/step at 49 frames, 12 h session cap, 2 T4s; latent encode runs on both GPUs in parallel.
- **Data (24 fps, anime timing kept):** `scripts/prepare_24fps_windows.py` cuts 49-frame windows and tags ones/twos/threes.
  Datasets on Kaggle: `voidvideo-anime-clips-24fps` (286, Vault), `-v2` (1244), `-v3` (4072 incl. creatures/animals/crowd from
  Sakugabooru `trojblue/sakugabooru2025` via `scripts/sakuga_to_listing.py`). Vault clips live at `voidvideo-vault.vercel.app`
  (`/api/list?tag=...`, Cloudinary cloud `gttvsok7`); some Vault clips are NOT anime (live action) — the VLM flags them.
- **Captions:** `scripts/vlm_caption_clips.py` (Qwen3-VL-4B, count/subjects/action/camera/effects/style) via
  `scripts/kaggle_vlm_caption_entrypoint.py`; tested OK, full run (~4 h GPU) planned after the 14 Nov quota reset.
- **Inference rules that matter:** fp16 pipeline, fp32 VAE decode, 30-50 steps, explicit motion wording ("left foot lifts...") and
  caption-style prompts ("expression shifts from X to Y"); `scripts/anime_retime.py --adaptive` picks ones/twos/threes/holds.
  Model pack zip (CogVideoX era) was sent to the user; update it with the Wan LoRA when done.
- **Kaggle weekly GPU quota 30 h, resets Saturday 00:00Z.** Never leave stray GPU kernels running; `kaggle kernels delete` stops one.
- User wants: studio-level anime (monsters, animals, crowds, multi-character, correct body language), "ones/twos/threes" timing,
  later a "studio" app (generate → takes → fix → layers → upscale). Plans: original characters only (copyright), own game engine
  is a separate project of the user.

## Known blocker: this sandbox's network policy
The Claude Code sandbox this was built in blocks outbound HTTPS to
`api.vercel.com`, `app.netlify.com`/`api.netlify.com`, and
`www.kaggle.com`/its API (confirmed via curl — "organization policy" CONNECT
rejection through the agent proxy). This means Claude running in a
similarly-restricted environment **cannot**:
- deploy `upload-portal/` to Vercel or Netlify itself
- run `scripts/push_training_kernel_to_kaggle.py` (or any Kaggle API
  script) itself to trigger GPU training or pull results back

**If you're a new session picking this up**: first test whether *your*
environment can reach these hosts (a quick `curl -sS -o /dev/null -w '%{http_code}\n' https://www.kaggle.com`
tells you immediately). If it can, you can run the Kaggle/deploy scripts
directly, exactly as documented in the READMEs. If it's blocked the same
way, say so plainly rather than retrying — the fix is choosing a less
restrictive network policy when the environment/session is created (see
https://code.claude.com/docs/en/claude-code-on-the-web), or the user runs
the relevant commands from their own machine instead.

## Credentials
The user has Kaggle, Cloudinary, and Vercel credentials and has pasted them
directly into chat before. **Never hardcode any of these into a file that
gets committed.** `.env` / `.env.local` are gitignored (and `.netlifyignore`
additionally excludes `.env` from Netlify's manual `deploy` upload, which
bundles straight from local disk) — keep it that way. If a fresh session
needs a credential value, ask the user to paste it again rather than
assuming it's recoverable from anywhere in the repo.

## Communication style
User writes in Hinglish (Hindi transliterated into Latin script, mixed with
English) or Hindi/Devanagari, casual tone, calls the assistant "bro"/"bhai".
Reply in matching Hinglish — short, direct, friendly. Avoid long formal
English paragraphs.

## Next steps outstanding
1. User needs to run the Kaggle pipeline scripts (`setup_kaggle_credentials.py`
   -> `push_dataset_to_kaggle.py` -> `push_training_kernel_to_kaggle.py` ->
   `fetch_trained_weights.py`) from a machine/session that can actually
   reach Kaggle's API.
2. User needs to deploy `upload-portal/` via Netlify or Vercel CLI from a
   machine/session that can reach that platform's API.
3. Training clips have not yet been collected/uploaded — that's the
   long pole before any training run can start.
