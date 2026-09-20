#!/usr/bin/env python3
"""
VoidVideo Gradio interface: Text-to-Video and Image-to-Video generation with
the hand-drawn-anime LoRA fine-tunes.

Usage:
    python app/gradio_app.py
    python app/gradio_app.py --share
"""
import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.inference import generate_i2v, generate_t2v  # noqa: E402

STYLE_SUFFIX = (
    ", hand-drawn anime, traditional 2D animation, natural line-art, smooth keyframe timing, "
    "cel shading, no CGI, no 3D render, no glow VFX"
)
DEFAULT_NEGATIVE = (
    "3d render, cgi, plastic skin, waxy, overly smooth, digital airbrush look, glossy, "
    "hyper-glow, lens flare, motion blur artifacts, deformed, extra limbs, watermark, text, subtitles"
)


def t2v_fn(prompt, negative_prompt, steps, guidance_scale, seed, lora_path):
    full_prompt = prompt.strip() + STYLE_SUFFIX
    return generate_t2v(
        prompt=full_prompt,
        negative_prompt=negative_prompt or DEFAULT_NEGATIVE,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        lora_path=lora_path or None,
    )


def i2v_fn(image, prompt, negative_prompt, steps, guidance_scale, seed, lora_path):
    if image is None:
        raise gr.Error("Upload a starting frame/keyframe image first.")
    full_prompt = (prompt or "").strip() + STYLE_SUFFIX
    return generate_i2v(
        image=image,
        prompt=full_prompt,
        negative_prompt=negative_prompt or DEFAULT_NEGATIVE,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        lora_path=lora_path or None,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VoidVideo — Anime Video Generation") as demo:
        gr.Markdown(
            "# VoidVideo\n"
            "Hand-drawn-style anime video generation. Silent output only — no audio/voice.\n"
            "Every prompt is automatically suffixed with style-anchoring tokens and paired with a "
            "negative prompt tuned to push away glossy/CGI/AI-plastic artifacts."
        )

        with gr.Tab("Text-to-Video"):
            with gr.Row():
                with gr.Column():
                    t2v_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="a hand-drawn anime girl walking through a rain-soaked street at night",
                        lines=3,
                    )
                    t2v_negative = gr.Textbox(label="Negative prompt (optional, overrides default)", lines=2)
                    with gr.Accordion("Advanced", open=False):
                        t2v_steps = gr.Slider(10, 100, value=50, step=1, label="Inference steps")
                        t2v_guidance = gr.Slider(1.0, 15.0, value=6.0, step=0.5, label="Guidance scale")
                        t2v_seed = gr.Number(value=42, precision=0, label="Seed")
                        t2v_lora = gr.Textbox(
                            label="LoRA checkpoint path (optional)",
                            placeholder="models/lora_t2v/final/lora_weights.safetensors",
                        )
                    t2v_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    t2v_output = gr.Video(label="Result")
            t2v_btn.click(
                t2v_fn,
                inputs=[t2v_prompt, t2v_negative, t2v_steps, t2v_guidance, t2v_seed, t2v_lora],
                outputs=t2v_output,
            )

        with gr.Tab("Image-to-Video"):
            with gr.Row():
                with gr.Column():
                    i2v_image = gr.Image(label="Starting frame / keyframe", type="pil")
                    i2v_prompt = gr.Textbox(
                        label="Motion prompt (describe how the scene should animate)",
                        placeholder="the character slowly turns her head and blinks, wind moves her hair",
                        lines=3,
                    )
                    i2v_negative = gr.Textbox(label="Negative prompt (optional, overrides default)", lines=2)
                    with gr.Accordion("Advanced", open=False):
                        i2v_steps = gr.Slider(10, 100, value=50, step=1, label="Inference steps")
                        i2v_guidance = gr.Slider(1.0, 15.0, value=6.0, step=0.5, label="Guidance scale")
                        i2v_seed = gr.Number(value=42, precision=0, label="Seed")
                        i2v_lora = gr.Textbox(
                            label="LoRA checkpoint path (optional)",
                            placeholder="models/lora_i2v/final/lora_weights.safetensors",
                        )
                    i2v_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    i2v_output = gr.Video(label="Result")
            i2v_btn.click(
                i2v_fn,
                inputs=[i2v_image, i2v_prompt, i2v_negative, i2v_steps, i2v_guidance, i2v_seed, i2v_lora],
                outputs=i2v_output,
            )

        gr.Markdown(
            "*Image-to-Video uses CogVideoX-5B-I2V (a larger base model than the T2V 2B checkpoint) "
            "since no official 2B image-conditioned variant exists. First run downloads/loads a bigger "
            "model and will take longer.*"
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui()
    demo.queue().launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
