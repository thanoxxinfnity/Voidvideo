#!/usr/bin/env python3
"""
LoRA fine-tuning of Wan2.1-T2V-1.3B (flow matching) on the VoidVideo clip dataset,
using diffusers + PEFT + Accelerate. Mirrors training/train_lora.py (CogVideoX): the same
dataset, caption/latent caching, LoRA-only gradients on the attention projections and
lora_weights.pt / .safetensors checkpoints, so the resume/push/fetch scripts keep working.

Differences from the CogVideoX script:
  * text encoder is UMT5-XXL (512 tokens, zero-padded past the real length, like WanPipeline),
  * the VAE is AutoencoderKLWan and latents are normalised per channel with latents_mean/std,
  * latents are laid out (C, T, H, W) rather than (T, C, H, W),
  * the objective is flow matching: x_t = (1 - s) * x0 + s * noise, target = noise - x0,
    with a logit-normal sigma and Wan's timestep shift.

Usage (under `accelerate launch`, see training/kaggle_entrypoint.py):
    accelerate launch training/train_wan_lora.py --task wan_t2v --config configs/training_config.yaml
"""
import argparse
import gc
import hashlib
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.training_utils import cast_training_params
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from training.dataset import VideoCaptionDataset  # noqa: E402

TEXT_LEN = 512  # WanPipeline's max_sequence_length


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["wan_t2v"], required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "training_config.yaml"))
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help="Path to a previous LoRA checkpoint dir to resume from.")
    return parser.parse_args()


def load_config(path: str, task: str) -> dict:
    with open(path) as f:
        full_cfg = yaml.safe_load(f)
    cfg = full_cfg[task]
    cfg["dataset"] = full_cfg["dataset"]
    # The step counter restarts every Kaggle session; the push script can cap a session.
    if os.environ.get("MAX_TRAIN_STEPS"):
        cfg["max_train_steps"] = int(os.environ["MAX_TRAIN_STEPS"])
    return cfg


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompt: str, device) -> torch.Tensor:
    """(TEXT_LEN, 4096) UMT5 embeddings, zero-padded past the real token count (as WanPipeline does)."""
    tok = tokenizer([prompt], padding="max_length", max_length=TEXT_LEN, truncation=True,
                    add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    n = int(tok.attention_mask.gt(0).sum())
    emb = text_encoder(tok.input_ids.to(device), tok.attention_mask.to(device)).last_hidden_state[0]
    emb = emb[:n]
    return torch.cat([emb, emb.new_zeros(TEXT_LEN - n, emb.shape[1])]).cpu()


@torch.no_grad()
def encode_video(vae, frames: torch.Tensor, mean: torch.Tensor, inv_std: torch.Tensor) -> torch.Tensor:
    # frames: (T, C, H, W) in [-1, 1] -> vae wants (B, C, T, H, W); returns normalised (C, T', H', W').
    px = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
    lat = vae.encode(px).latent_dist.sample()
    return ((lat - mean.to(lat.device)) * inv_std.to(lat.device))[0].cpu()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.task)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        mixed_precision=cfg["mixed_precision"],
        log_with=None,
    )
    set_seed(cfg["seed"])
    weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg["mixed_precision"], torch.float32)
    device = accelerator.device
    n_frames = cfg["num_frames"]

    model_id = cfg["base_model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    # UMT5 overflows in fp16, so it stays bf16 (slow on a T4, but it only runs once per caption).
    text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=weight_dtype)

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    if cfg["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()

    transformer.add_adapter(LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"], init_lora_weights=True,
    ))
    cast_training_params([transformer], dtype=torch.float32)

    if args.resume_from_checkpoint:
        state_dict = torch.load(Path(args.resume_from_checkpoint) / "lora_weights.pt", map_location="cpu")
        set_peft_model_state_dict(transformer, state_dict)
        print(f"Resumed LoRA weights from {args.resume_from_checkpoint}")

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    print(f"Training {sum(p.numel() for p in trainable_params):,} LoRA parameters.")

    if cfg["use_8bit_adam"]:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(trainable_params, lr=cfg["learning_rate"], weight_decay=1e-2)

    # Clips are stored with 49 frames; train on the first n_frames consecutively so motion keeps its natural speed.
    dataset = VideoCaptionDataset(
        metadata_file=str(REPO_ROOT / cfg["dataset"]["metadata_file"]),
        video_root=str(REPO_ROOT / cfg["dataset"]["video_root"]),
        num_frames=max(49, n_frames),
        height=cfg["height"],
        width=cfg["width"],
    )

    # ---- Caption embeddings: T5 goes alone on the GPU (transformer is still on CPU), then is freed. ----
    text_encoder.to(device)
    captions = sorted({row.get("caption", "") for row in dataset.rows} | {""})
    caption_index = {c: i for i, c in enumerate(captions)}
    caption_embeds = torch.stack([encode_prompt(tokenizer, text_encoder, c, device) for c in captions]).to(weight_dtype)
    empty_caption_idx = caption_index[""]
    print(f"Precomputed {len(captions)} caption embeddings {tuple(caption_embeds.shape)}.", flush=True)
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Latent cache (same rules as the CogVideoX script: unique clips, persisted, time-budgeted) ----
    vae.to(device)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    inv_std = (1.0 / torch.tensor(vae.config.latents_std)).view(1, -1, 1, 1, 1)

    t_start = time.time()
    budget = float(os.environ.get("PRECOMPUTE_BUDGET_SEC", 7 * 3600))
    cache_dir = Path(os.environ.get("LATENT_CACHE_DIR", REPO_ROOT / "outputs" / "wan_latent_cache"))
    seed_env = os.environ.get("LATENT_CACHE_SEED_DIR")
    if seed_env:
        seed_dir = Path(seed_env)
    else:
        found = sorted(Path("/kaggle/input").glob("**/voidvideo-latent-cache-wan*")) if Path("/kaggle/input").exists() else []
        seed_dir = found[0] if found else Path("/nonexistent-latent-seed")
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_name(video: str) -> str:
        return hashlib.sha1(f"wan{n_frames}:{video}".encode()).hexdigest()[:16] + ".pt"

    def find_cached(video: str):
        for d in (cache_dir, seed_dir):
            f = d / cache_name(video)
            if f.exists():
                return f
        return None

    def category(video: str) -> str:
        name = Path(video).name
        return name.split("__")[0] if "__" in name else "other"

    unique = {}
    for i, row in enumerate(dataset.rows):
        unique.setdefault(row["video"], i)
    by_cat = {}
    for video, i in unique.items():
        by_cat.setdefault(category(video), []).append((video, i))
    order = []
    while any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat]:
                order.append(by_cat[cat].pop(0))

    latents_by_video = {}
    todo = []
    for video, i in order:
        f = find_cached(video)
        if f is not None:
            latents_by_video[video] = torch.load(f, map_location="cpu").to(weight_dtype)
        else:
            todo.append((video, i))

    # Encode the rest on every visible GPU in parallel (one VAE copy and one worker thread per GPU).
    # Tiling is left off: it roughly halves speed, and the causal Wan VAE fits 33 frames untiled.
    n_gpu = max(1, torch.cuda.device_count())
    vaes = [vae] + [AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
                    for _ in range(n_gpu - 1)]
    for g, v in enumerate(vaes):
        v.requires_grad_(False)
        v.to(f"cuda:{g}")
    print(f"Encoding {len(todo)} uncached clips on {n_gpu} GPU(s).", flush=True)
    lock = threading.Lock()
    counter = {"n": 0}

    def encode_worker(g: int):
        v = vaes[g]
        for video, i in todo[g::n_gpu]:
            if time.time() - t_start > budget:
                return
            frames = dataset[i]["frames"][:n_frames]
            try:
                lat = encode_video(v, frames, mean, inv_std)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                v.enable_tiling()
                lat = encode_video(v, frames, mean, inv_std)
            lat = lat.to(weight_dtype)
            torch.save(lat, cache_dir / cache_name(video))
            with torch.cuda.device(g):
                torch.cuda.empty_cache()
            with lock:
                latents_by_video[video] = lat
                counter["n"] += 1
                if counter["n"] % 10 == 0 or counter["n"] == 1:
                    print(f"Encoded {counter['n']}/{len(todo)} new latents (latent {tuple(lat.shape)}, "
                          f"{(time.time() - t_start) / 60:.1f} min)", flush=True)

    with ThreadPoolExecutor(max_workers=n_gpu) as pool:
        list(pool.map(encode_worker, range(n_gpu)))
    encoded_now = counter["n"]

    cached = [
        (latents_by_video[row["video"]], caption_index[row.get("caption", "")])
        for row in dataset.rows if row["video"] in latents_by_video
    ]
    print(f"Latent cache: {len(latents_by_video)}/{len(order)} unique clips ready ({encoded_now} encoded now); "
          f"training on {len(cached)}/{len(dataset.rows)} rows.", flush=True)
    if not cached:
        raise RuntimeError("No latents available to train on (encode budget too small?).")
    del vae, vaes
    gc.collect()
    torch.cuda.empty_cache()

    caption_dropout = cfg["dataset"]["caption_dropout"]
    dataloader = DataLoader(cached, batch_size=cfg["train_batch_size"], shuffle=True, collate_fn=lambda b: b)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, step / max(1, cfg["lr_warmup_steps"])))
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler)

    output_dir = REPO_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    shift = float(cfg.get("flow_shift", 3.0))
    global_step = 0
    skipped = 0
    t0 = time.time()
    progress_bar = tqdm(total=cfg["max_train_steps"], disable=not accelerator.is_local_main_process)

    while global_step < cfg["max_train_steps"]:
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                latents = torch.stack([lat for lat, _ in batch]).to(device, weight_dtype)  # (B, C, T, H, W)
                cap_ids = [empty_caption_idx if random.random() < caption_dropout else idx for _, idx in batch]
                prompt_embeds = caption_embeds[cap_ids].to(device)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                u = torch.sigmoid(torch.randn(bsz, device=device))            # logit-normal sample
                sigma = shift * u / (1 + (shift - 1) * u)                     # Wan's timestep shift
                s = sigma.view(-1, 1, 1, 1, 1).to(weight_dtype)
                noisy = (1 - s) * latents + s * noise
                pred = transformer(
                    hidden_states=noisy,
                    timestep=(sigma * 1000).to(weight_dtype),
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                loss = F.mse_loss(pred.float(), (noise - latents).float(), reduction="mean")

                if not torch.isfinite(loss):
                    skipped += 1
                    print(f"non-finite loss, skipping micro-batch ({skipped} so far)", flush=True)
                    optimizer.zero_grad()
                    continue
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), step=global_step)
                print(f"step {global_step} loss={loss.item():.4f} "
                      f"({(time.time() - t0) / global_step:.0f}s/step, "
                      f"mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB)", flush=True)
                if accelerator.is_main_process and global_step % cfg["checkpointing_steps"] == 0:
                    save_checkpoint(accelerator, transformer, output_dir, global_step)
            if global_step >= cfg["max_train_steps"]:
                break

    if accelerator.is_main_process:
        save_checkpoint(accelerator, transformer, output_dir, global_step, final=True)
    accelerator.end_training()


def save_checkpoint(accelerator, transformer, output_dir: Path, step: int, final: bool = False):
    unwrapped = accelerator.unwrap_model(transformer)
    state = get_peft_model_state_dict(unwrapped)
    ckpt_dir = output_dir / ("final" if final else f"step-{step}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_dir / "lora_weights.pt")
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in state.items()}, ckpt_dir / "lora_weights.safetensors")
    print(f"Saved LoRA checkpoint to {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
