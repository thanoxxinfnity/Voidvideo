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
