"""VoidVideo 3D: Gradio UI for Pixal3D (image -> textured GLB).

Run:  python app.py            (real GPU backend, see kaggle_kernel for setup)
      PIXAL3D_MOCK=1 python app.py   (CPU mock for UI testing)

Env:  IDLE_MINUTES  auto-exit after this many idle minutes (default 30, 0=off)
      MAX_HOURS      hard session cap (default 11)
      SHARE=1        create a public gradio.live link (needed on Kaggle)
      WARMUP=1       load models in the background at startup
"""
import os
import sys
import threading
import time
import traceback

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend import GenerationError, make_backend  # noqa: E402

IDLE_MINUTES = float(os.environ.get("IDLE_MINUTES", "30"))
MAX_HOURS = float(os.environ.get("MAX_HOURS", "11"))
START = time.time()
backend = make_backend()
_activity = {"last": time.time(), "busy": 0}
_act_lock = threading.Lock()


def _touch(busy_delta=0):
    with _act_lock:
        _activity["last"] = time.time()
        _activity["busy"] += busy_delta


def _shutdown(reason):
    print("SHUTDOWN:", reason, flush=True)
    sys.stdout.flush()
    os._exit(0)


def _watchdog():
    while True:
        time.sleep(20)
        with _act_lock:
            idle = time.time() - _activity["last"]
            busy = _activity["busy"] > 0
        if backend.state == "loading":
            busy = True
        if time.time() - START > MAX_HOURS * 3600:
            _shutdown("max session time reached")
        if IDLE_MINUTES > 0 and not busy and idle > IDLE_MINUTES * 60:
            _shutdown("idle for %.0f min - GPU released" % (idle / 60))


def generate(image, faces, texture_size, seed, remove_bg, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Pehle image upload karo.")
    _touch(+1)
    t0 = time.time()
    try:
        progress(0.02, desc="Image taiyaar ho rahi hai...")
        progress(0.10, desc="3D ban raha hai (T4 pe kuch minute lagte hain)...")
        path, tris = backend.generate(image, int(faces), int(texture_size), int(seed), bool(remove_bg))
    except GenerationError as e:
        raise gr.Error(str(e))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise gr.Error("Unexpected error: %r" % (e,))
    finally:
        _touch(-1)
    size_mb = os.path.getsize(path) / 1e6
    note = "" if abs(tris - faces) <= 0.1 * faces else "  (target se thoda alag - mesh ki bunavat ke hisaab se)"
    status = "Ho gaya in %.0f s | **%s triangles** (target %s)%s | %.1f MB" % (
        time.time() - t0, format(tris, ","), format(int(faces), ","), note, size_mb)
    return path, gr.DownloadButton(value=path, visible=True, label="GLB download (%s tris)" % format(tris, ",")), status


def backend_status():
    icon = {"idle": "", "loading": "Loading", "ready": "Ready", "error": "ERROR"}.get(backend.state, "")
    return "**GPU/model status:** %s - %s" % (icon or backend.state, backend.state_msg)


def quit_now():
    threading.Timer(1.0, _shutdown, args=("user pressed shutdown",)).start()
    return "Band ho raha hai... GPU 1-2 second mein release ho jayegi. Is tab ko close kar sakte ho."


def build_ui():
    with gr.Blocks(title="VoidVideo 3D (Pixal3D)") as demo:
        gr.Markdown("# VoidVideo 3D\nImage upload karo -> textured **GLB** milega. Pixal3D, ~12k triangles default.")
        status_md = gr.Markdown(backend_status())
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Input image (transparent PNG best; warna background auto-remove)",
                                 type="pil", image_mode="RGBA", height=380)
                with gr.Accordion("Settings", open=True):
                    faces = gr.Slider(2000, 50000, value=12000, step=500, label="Triangles (mesh target)")
                    tex = gr.Dropdown([512, 1024, 2048], value=1024, label="Texture size")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    rmbg = gr.Checkbox(value=True, label="Background auto-remove (agar image me transparency na ho)")
                go = gr.Button("Generate 3D (GLB)", variant="primary")
                stop = gr.Button("Band karo (GPU shutdown)", variant="stop")
            with gr.Column():
                viewer = gr.Model3D(label="Preview", height=420, clear_color=(0.9, 0.9, 0.9, 1.0))
                dl = gr.DownloadButton("GLB download", visible=False, variant="primary")
                result_md = gr.Markdown("")
        go.click(generate, [image, faces, tex, seed, rmbg], [viewer, dl, result_md], concurrency_limit=1)
        stop.click(quit_now, None, result_md)
        timer = gr.Timer(5)
        timer.tick(backend_status, None, status_md)
        demo.load(backend_status, None, status_md)
    return demo


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    if os.environ.get("WARMUP", "0") == "1":
        def _warm():
            try:
                backend.warmup()
            except BaseException:  # noqa: BLE001
                traceback.print_exc()
        threading.Thread(target=_warm, daemon=True).start()
    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=8)
    share = os.environ.get("SHARE", "0") == "1"
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), share=share,
                show_error=True, allowed_paths=[os.environ.get("PIXAL3D_OUT", "/kaggle/working/out")])


if __name__ == "__main__":
    main()
