# VoidVideo

A LoRA fine-tuning + inference pipeline for generating **hand-drawn-style
anime video**, silent (no audio/voice), with a Gradio UI supporting both
**Text-to-Video** and **Image-to-Video**.

## How it's built

- **Base models**: [CogVideoX-2B](https://huggingface.co/THUDM/CogVideoX-2b) for
  Text-to-Video, [CogVideoX-5B-I2V](https://huggingface.co/THUDM/CogVideoX-5b-I2V)
  for Image-to-Video. These are separate checkpoints (THUDM never released a 2B
  image-conditioned variant), so they're fine-tuned and stored as **independent
  LoRA adapters** — see `configs/training_config.yaml`'s `t2v:` and `i2v:` blocks.
- **Fine-tuning method**: LoRA on the transformer's attention projections
  (`to_q`/`to_k`/`to_v`/`to_out.0`) via `diffusers` + `peft` + `accelerate`.
  The base model weights are frozen; only the LoRA adapters train, which keeps
  training cheap enough to run on a single Kaggle GPU.
- **Style control**: the Gradio app appends a fixed style-anchor suffix to every
  prompt and pairs it with a negative prompt tuned against glossy/CGI/"AI
  plastic" artifacts (see `app/gradio_app.py`), on top of whatever the LoRA
  itself has learned from your clips.
- **Training compute**: Kaggle's free GPU quota (T4×2 or P100, ~30 GPU-hrs/week),
  driven entirely through the `kaggle` Python API — no manual notebook editing.

## Repository layout

```
configs/
  training_config.yaml     LoRA + dataset hyperparameters for both tasks
data/
  raw_clips/                Your source clips go here (gitignored)
  captions/                 Optional per-clip caption .txt files
  processed/                 Normalized clips + metadata.jsonl (gitignored, generated)
scripts/
  prepare_dataset.py         Normalize raw clips -> training-ready dataset
  setup_kaggle_credentials.py  Write ~/.kaggle/kaggle.json from .env
  push_dataset_to_kaggle.py    Push data/processed/ as a private Kaggle dataset
  push_training_kernel_to_kaggle.py  Push + trigger the GPU training kernel
  fetch_trained_weights.py     Poll the kernel and download trained LoRA weights
training/
  dataset.py                 VideoCaptionDataset (decord-based loader)
  train_lora.py               LoRA training loop (accelerate + diffusers + peft)
  kaggle_entrypoint.py         Runs inside the Kaggle kernel; clones this repo & launches training
app/
  inference.py                Pipeline loading + generation helpers
  gradio_app.py                Gradio UI: Text-to-Video and Image-to-Video tabs
models/                       Downloaded/trained LoRA checkpoints land here (gitignored)
outputs/                       Local training run outputs (gitignored)
```

## End-to-end pipeline

### 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in KAGGLE_USERNAME, KAGGLE_KEY, dataset/kernel slugs
```

Get your Kaggle API token at https://www.kaggle.com/settings → API → *Create New Token*.
`.env` is gitignored — your token never touches the repo.

### 1. Prepare your clips

See `data/README.md` for the full clip specification. Short version: drop
`.mp4`s in `data/raw_clips/`, optionally caption them in `data/captions/`, then:

```bash
python scripts/prepare_dataset.py --task t2v
# or, for image-to-video fine-tuning:
python scripts/prepare_dataset.py --task i2v
```

### 2. Push to Kaggle and train

```bash
python scripts/setup_kaggle_credentials.py
python scripts/push_dataset_to_kaggle.py
python scripts/push_training_kernel_to_kaggle.py --task t2v
python scripts/fetch_trained_weights.py --wait
```

`push_training_kernel_to_kaggle.py` pushes a self-contained entrypoint that
`git clone`s this exact repo/branch inside the Kaggle kernel, installs
`requirements.txt`, mounts the dataset you pushed in step 2 as `/kaggle/input`,
and runs `training/train_lora.py` under `accelerate launch` on Kaggle's GPU.
`fetch_trained_weights.py --wait` polls the kernel and, once it completes,
copies the resulting LoRA checkpoint into `models/lora_<task>/`.

Repeat for `--task i2v` if you want image-to-video fine-tuned too — it's a
fully separate training run against the larger CogVideoX-5B-I2V base model.

### 3. Run the Gradio app

```bash
python app/gradio_app.py
```

It auto-loads `models/lora_t2v/final/lora_weights.safetensors` and
`models/lora_i2v/final/lora_weights.safetensors` if present (or point it at a
specific checkpoint via the "LoRA checkpoint path" field in the UI). With no
LoRA present it still runs — you'll just get the base model's default look
rather than your fine-tuned style.

## Known limitations

- **I2V training doesn't yet inject the conditioning image during the training
  loop** (see the header comment in `training/train_lora.py`) — it trains the
  same style/motion LoRA weights as T2V, which transfers the hand-drawn look
  but isn't specifically taught to track the input image more faithfully.
  Closing this gap means encoding the conditioning frame and concatenating its
  latent onto the noisy input before the transformer call, matching what
  `CogVideoXImageToVideoPipeline` does at inference time.
- LoRA quality is directly bounded by clip count/diversity and caption quality
  — see the checklist in `data/README.md` before investing GPU hours.
- CogVideoX-5B-I2V is a large model; local Gradio inference on it needs a GPU
  with CPU-offload enabled (already on by default in `app/inference.py`) or it
  will be slow/OOM on modest hardware.
