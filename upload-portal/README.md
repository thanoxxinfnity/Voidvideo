# VoidVideo Clip Upload Portal

A tiny static website + Vercel serverless functions for collecting VoidVideo
training clips, organized by motion/expression category, with per-clip delete
(like a YouTube Studio uploads list). Clips are stored in Cloudinary; nothing
is stored in this repo or on Vercel itself.

- `index.html` / `style.css` / `app.js` — the whole frontend, plain JS, no build step.
- `api/sign.js` — signs a Cloudinary upload request server-side (secret never reaches the browser).
- `api/list.js` — lists uploaded clips for a category tag.
- `api/delete.js` — deletes a clip by public ID.

## Deploy (via Vercel dashboard — no CLI needed)

1. Go to https://vercel.com/new and import the `thanoxxinfnity/Voidvideo` GitHub repo.
2. When configuring the project:
   - **Root Directory**: `upload-portal` (important — this is a subfolder of the repo)
   - **Framework Preset**: "Other"
3. Under **Environment Variables**, add (paste your real Cloudinary values directly into the
   Vercel dashboard, not into any file in this repo):
   - `CLOUDINARY_CLOUD_NAME`
   - `CLOUDINARY_API_KEY`
   - `CLOUDINARY_API_SECRET`
4. Click **Deploy**. You'll get a `https://<something>.vercel.app` URL.

That's it — open the URL, pick a category tab, drag clips in.

## Local testing (optional)

```bash
cd upload-portal
cp .env.example .env.local   # fill in real Cloudinary values
npm i -g vercel              # if you don't already have it
vercel dev
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
