# VoidVideo Caption Studio

A local Gradio dashboard for building the LoRA training dataset: bulk-upload
clips, get an AI-drafted caption for each one, edit them by hand, then
package everything into a zip ready to upload to Kaggle.

## Setup

```bash
cd caption-studio
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your real Hugging Face token into HF_TOKEN
```

## Run

```bash
python app.py
```

Opens at `http://127.0.0.1:7860`.

## How it works

1. **Upload** — drop as many `.mp4` clips as you like into the file picker.
   Each one is copied into `dataset/videos/` as `clip_001.mp4`,
   `clip_002.mp4`, etc. (numbering continues across upload batches, so you
   can add more clips later without overwriting anything).
2. **Auto-caption** — for each clip, 5 frames are sampled evenly across its
   duration and sent together (so the model sees motion over time, not one
   frozen instant) to a vision-language model via Hugging Face's hosted
   Inference Providers. The caption it writes is saved to
   `dataset/videos/clip_001.txt` and shown next to the video preview.
3. **Edit** — every caption is a normal editable text box. Any edit is saved
   to the matching `.txt` file immediately (no separate "save" step).
4. **Package Dataset** — reads every `clip_*.mp4` / `clip_*.txt` pair fresh
   from disk (so it always reflects your latest edits), checks that all
   clips share one resolution, writes `dataset/dataset-metadata.json`, and
   zips everything into `dataset.zip` with a download link.

## Model note

You asked for `Qwen/Qwen2-VL-7B-Instruct`, but that exact model has **no
hosted inference provider** on Hugging Face right now (checked live via the
API — `inferenceProviderMapping` comes back empty), so it can't be called
through the free hosted API at all. Its successor,
`Qwen/Qwen2.5-VL-7B-Instruct`, does have one provider (`featherless-ai`) but
it was returning "temporarily at capacity" errors when this was built.
`Qwen/Qwen2.5-VL-72B-Instruct` via the `ovhcloud` provider was verified
working end-to-end (including real multi-frame motion reasoning — it
correctly described movement between sampled frames) and is the default in
`.env.example`. Override `HF_CAPTION_MODEL` if you'd rather point at a
different model/provider your account has access to; the code doesn't care
which one, as long as it accepts multiple `image_url` parts in one message
over the OpenAI-compatible `https://router.huggingface.co/v1/chat/completions`
endpoint.

If a clip's caption fails to generate (rate limit, provider hiccup, network),
it's never silently left blank — the textbox shows
`[AI caption failed -- edit this manually]` and the upload status panel lists
exactly which clips failed and why, so you always know what still needs a
manual caption.

## Output layout

```
caption-studio/
├── dataset/
│   ├── videos/
│   │   ├── clip_001.mp4
│   │   ├── clip_001.txt
│   │   ├── clip_002.mp4
│   │   ├── clip_002.txt
│   │   └── ...
│   └── dataset-metadata.json   (written by "Package Dataset")
└── dataset.zip                 (written by "Package Dataset")
```

`dataset.zip` contains `videos/<clip>.mp4` + `videos/<clip>.txt` pairs plus
`dataset-metadata.json` at the root — upload it directly to a Kaggle dataset
for the LoRA training kernel.
