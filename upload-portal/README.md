# VoidVideo Vault — Clip Upload Portal

A tiny static website + serverless functions for collecting VoidVideo
training clips, organized by motion/expression category, with per-clip delete
(like a YouTube Studio uploads list). Clips are stored in Cloudinary; nothing
is stored in this repo or on the hosting platform itself.

A "Smart Upload" tab lets you drop clips without picking a category first — an
NVIDIA NIM vision model looks at a frame from the middle of each clip and
guesses locomotion/gestures/expressions/secondary-motion, shown as an editable
per-file dropdown so you can confirm or correct it before uploading.

Two equivalent backends are included — deploy to whichever platform works for
you, they don't need to coexist:

- **Vercel**: `api/sign.js`, `api/list.js`, `api/delete.js`, `api/classify.js`
- **Netlify**: `netlify/functions/sign.js`, `list.js`, `delete.js`, `classify.js`
  + `netlify.toml` (routes `/api/*` to the functions so the frontend code is
  identical either way)

`index.html` / `style.css` / `app.js` is the whole frontend, plain JS, no build step,
shared by both.

## Deploy option A: Vercel dashboard (needs your GitHub connected to Vercel)

1. Go to https://vercel.com/new and import the `thanoxxinfnity/Voidvideo` GitHub repo.
2. When configuring the project:
   - **Root Directory**: `upload-portal` (important — this is a subfolder of the repo)
   - **Framework Preset**: "Other"
3. Under **Environment Variables**, add (paste your real values directly into the
   dashboard, not into any file in this repo):
   - `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET`
   - `NVIDIA_API_KEY` (free key from https://build.nvidia.com — only needed for
     the Smart Upload tab's AI category detection; everything else works without it)
   - `HF_TOKEN` (free token from https://huggingface.co/settings/tokens — only
     needed for auto-captioning on upload; everything else works without it)
4. Click **Deploy**.

## Deploy option B: CLI, no GitHub needed at all (Vercel or Netlify)

Works even if your GitHub account isn't connected to the hosting platform —
this deploys straight from your local folder.

**Vercel:**
```bash
cd upload-portal
npm i -g vercel
vercel login
vercel link
vercel env add CLOUDINARY_CLOUD_NAME production
vercel env add CLOUDINARY_API_KEY production
vercel env add CLOUDINARY_API_SECRET production
vercel env add NVIDIA_API_KEY production   # optional, powers AI category detection
vercel env add HF_TOKEN production         # optional, powers auto-captioning
vercel --prod
```

**Netlify:**
```bash
cd upload-portal
npm i -g netlify-cli
netlify login
netlify init          # choose "Create & configure a new site"
                       # when it asks for a site name, try: voidvideo-vault
                       # (if taken, Netlify will tell you -- pick any variant)
netlify env:set CLOUDINARY_CLOUD_NAME <your_cloud_name>
netlify env:set CLOUDINARY_API_KEY <your_api_key>
netlify env:set CLOUDINARY_API_SECRET <your_api_secret>
netlify env:set NVIDIA_API_KEY <your_nvapi_key>   # optional, powers AI category detection
netlify env:set HF_TOKEN <your_hf_token>          # optional, powers auto-captioning
netlify deploy --prod
```

Either way you get a live URL back in the terminal — with the suggested name
that'll be `https://voidvideo-vault.netlify.app` (or whatever variant you pick).

**Security note specific to Netlify's manual `deploy` command**: it uploads
your *local folder* as-is (not what's committed to git), so a stray `.env`
sitting in this folder would otherwise get published as a public static file.
`.netlifyignore` in this folder already excludes `.env`, `.env.local`, and a
few other dev-only files from that upload — don't remove it.

## Local testing (optional)

A `.env` with real Cloudinary values already exists locally in this folder
for local dev (gitignored, never committed, and excluded from deploys via
`.netlifyignore`). To test:

```bash
cd upload-portal
npm i -g vercel        # or: npm i -g netlify-cli
vercel dev             # or: netlify dev
```

## Pulling uploaded clips down for training

Once you've uploaded clips through the site, pull them onto the training
machine with:

```bash
python scripts/fetch_clips_from_cloudinary.py
```

This downloads everything tagged with each category into `data/raw_clips/`,
ready for `scripts/prepare_dataset.py`.

## Categories & targets

| Tag | Label | Target |
|---|---|---|
| `locomotion` | Locomotion | 50 |
| `gestures` | Gestures | 30 |
| `expressions` | Facial Expressions | 40 |
| `secondary-motion` | Secondary Motion | 20 |

Targets are guidance, not hard caps — see the root `data/README.md` for the
full spec (150–300 total clips recommended).
