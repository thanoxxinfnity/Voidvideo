#!/usr/bin/env python3
"""
LoRA fine-tuning for CogVideoX (text-to-video or image-to-video) on a
hand-drawn anime clip dataset, using diffusers + PEFT + Accelerate.

This mirrors the structure of diffusers' official CogVideoX LoRA training
example, trimmed to what VoidVideo needs: single-GPU/DDP training, LoRA-only
gradients on the transformer's attention projections, periodic validation
video export, and safetensors LoRA checkpoints.

Usage (run under `accelerate launch`, see training/kaggle_entrypoint.py):
    accelerate launch training/train_lora.py --task t2v \
        --config configs/training_config.yaml

KNOWN LIMITATION (--task i2v): CogVideoX-5b-I2V's transformer normally receives
the conditioning frame's VAE latents concatenated onto the noisy input during
inference. This script does not yet inject that conditioning signal during
training -- it fine-tunes the same attention-projection LoRA weights that the
t2v path uses, which is enough to transfer the hand-drawn *style/motion* look,
but the LoRA is not specifically taught to follow the conditioning image more
faithfully. If validation videos ignore the input image, that's the gap to
close next: encode the first frame with `vae`, concatenate it onto
`noisy_latents` before the transformer call, matching the channel layout
CogVideoXImageToVideoPipeline uses at inference time.
"""
import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))
from training.dataset import VideoCaptionDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["t2v", "i2v"], required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "training_config.yaml"))
    parser.add_argument("--resume_from_checkpoint", default=None,
                         help="Path to a previous LoRA checkpoint dir to resume from.")
    return parser.parse_args()


def load_config(path: str, task: str) -> dict:
    with open(path) as f:
        full_cfg = yaml.safe_load(f)
    cfg = full_cfg[task]
    cfg["dataset"] = full_cfg["dataset"]
    return cfg


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompts, out_device, max_length=226):
    inputs = tokenizer(
        prompts, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt",
    ).to(text_encoder.device)
    return text_encoder(inputs.input_ids)[0].to(out_device)


@torch.no_grad()
def encode_video(vae, pixel_values, out_device):
    # pixel_values: (B, T, C, H, W) in [-1, 1] -> vae expects (B, C, T, H, W)
    pixel_values = pixel_values.permute(0, 2, 1, 3, 4).to(device=vae.device, dtype=vae.dtype)
    latent_dist = vae.encode(pixel_values).latent_dist
    latents = latent_dist.sample() * vae.config.scaling_factor
    return latents.permute(0, 2, 1, 3, 4).to(out_device)  # (B, T, C, H, W) latent space


def main():
    args = parse_args()
    cfg = load_config(args.config, args.task)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        mixed_precision=cfg["mixed_precision"],
        log_with=None,
    )
    set_seed(cfg["seed"])

    weight_dtype = torch.bfloat16 if cfg["mixed_precision"] == "bf16" else (
        torch.float16 if cfg["mixed_precision"] == "fp16" else torch.float32
    )

    model_id = cfg["base_model_id"]
    tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
    # T5 is notorious for overflowing to NaN in fp16, so it stays bf16 even
    # when training runs fp16; it only encodes ~226 text tokens, so the T4's
    # missing native bf16 support costs speed there, not memory. The VAE stays
    # fp32 for the same overflow reason (fp16 VAE decode went NaN/white on T4
    # in the inference studio) -- it's small, so fp32 is cheap.
    text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    vae.enable_slicing()
    vae.enable_tiling()
    # Frozen base weights in the training compute dtype (the trainable LoRA
    # params are upcast to fp32 below by cast_training_params). fp32 here
    # alone was ~6.8GB, and together with T5-XXL (~9.5GB) that exceeded a
    # single T4 before a single step ran.
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=weight_dtype
    )
    scheduler = CogVideoXDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    if cfg["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        init_lora_weights=True,
    )
    transformer.add_adapter(lora_config)
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

    dataset = VideoCaptionDataset(
        metadata_file=str(REPO_ROOT / cfg["dataset"]["metadata_file"]),
        video_root=str(REPO_ROOT / cfg["dataset"]["video_root"]),
        num_frames=cfg["num_frames"],
        height=cfg["height"],
        width=cfg["width"],
        caption_dropout=cfg["dataset"]["caption_dropout"],
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg["train_batch_size"], shuffle=True,
        num_workers=2, collate_fn=lambda batch: batch,
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / max(1, cfg["lr_warmup_steps"])),
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    # Training is single-process (see kaggle_entrypoint.py), so a second GPU
    # would otherwise sit idle. Put the frozen encoders there and leave the
    # primary GPU entirely to the transformer + its activations. With only
    # one GPU, keep the VAE on it but run T5 on CPU -- T5-XXL alone is ~9.5GB
    # and can't share one T4 with the transformer.
    if torch.cuda.device_count() > 1 and accelerator.num_processes == 1:
        aux_device = torch.device("cuda", (accelerator.device.index or 0) + 1)
        text_encoder.to(aux_device)
        vae.to(aux_device)
    else:
        text_encoder.to("cpu")
        vae.to(accelerator.device)
    print(f"transformer on {accelerator.device}, text encoder on {text_encoder.device}, VAE on {vae.device}")

    output_dir = REPO_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    progress_bar = tqdm(total=cfg["max_train_steps"], disable=not accelerator.is_local_main_process)

    while global_step < cfg["max_train_steps"]:
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                pixel_values = torch.stack([b["frames"] for b in batch])
                captions = [b["caption"] for b in batch]

                latents = encode_video(vae, pixel_values, accelerator.device).to(dtype=weight_dtype)
                prompt_embeds = encode_prompt(tokenizer, text_encoder, captions, accelerator.device)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                model_pred = transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    return_dict=False,
                )[0]

                target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(
                    latents, noise, timesteps
                )
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

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

                if accelerator.is_main_process and global_step % cfg["checkpointing_steps"] == 0:
                    save_checkpoint(accelerator, transformer, output_dir, global_step)

                # validation_steps: 0 disables it. run_validation builds a
                # whole second pipeline with enable_model_cpu_offload() around
                # the *live* training transformer (which would leave it on CPU
                # afterwards) and reloads a second T5 copy -- neither fits
                # alongside training on a T4. Checkpoints are the real output;
                # sample them afterwards in the Gradio studio instead.
                if (accelerator.is_main_process and cfg.get("validation_steps")
                        and global_step % cfg["validation_steps"] == 0):
                    try:
                        run_validation(cfg, model_id, transformer, accelerator, output_dir, global_step, args.task)
                    except Exception as e:
                        print(f"Validation at step {global_step} failed, continuing training: {e}")

            if global_step >= cfg["max_train_steps"]:
                break

    if accelerator.is_main_process:
        save_checkpoint(accelerator, transformer, output_dir, global_step, final=True)
    accelerator.end_training()


def save_checkpoint(accelerator, transformer, output_dir: Path, step: int, final: bool = False):
    unwrapped = accelerator.unwrap_model(transformer)
    lora_state_dict = get_peft_model_state_dict(unwrapped)
    tag = "final" if final else f"step-{step}"
    ckpt_dir = output_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict, ckpt_dir / "lora_weights.pt")
    from safetensors.torch import save_file
    save_file(
        {k: v.contiguous() for k, v in lora_state_dict.items()},
        ckpt_dir / "lora_weights.safetensors",
    )
    print(f"Saved LoRA checkpoint to {ckpt_dir}")


@torch.no_grad()
def run_validation(cfg, model_id, transformer, accelerator, output_dir: Path, step: int, task: str):
    unwrapped = accelerator.unwrap_model(transformer)
    pipeline_cls = CogVideoXImageToVideoPipeline if task == "i2v" else CogVideoXPipeline
    pipe = pipeline_cls.from_pretrained(model_id, transformer=unwrapped, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload(device=accelerator.device)

    generator = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    kwargs = dict(
        prompt=cfg["validation_prompt"],
        num_frames=cfg["num_frames"],
        height=cfg["height"],
        width=cfg["width"],
        num_inference_steps=30,
        generator=generator,
    )
    frames = pipe(**kwargs).frames[0]

    val_dir = output_dir / "validation"
    val_dir.mkdir(exist_ok=True)
    out_path = val_dir / f"step-{step}.mp4"
    export_to_video(frames, str(out_path), fps=cfg["fps"])
    print(f"Wrote validation video to {out_path}")
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
