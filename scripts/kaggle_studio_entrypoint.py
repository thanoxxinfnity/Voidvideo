"""
Kaggle kernel: VoidVideo Studio API (see studio/server.py) on the Kaggle GPU, reachable from the phone app through a free
Cloudflare quick tunnel (no account needed). Attach the 24 fps LoRA checkpoint dataset (KAGGLE_CHECKPOINT_DATASET_SLUG),
GPU + internet on.

 - Endpoints: /generate /status/{id} /cancel/{id} /result/{id} /preview/{id} /upload /storyboard /bank (X-API-Key header).
 - The tunnel URL and the API key are printed in the kernel log (visible live on the notebook page) and, when HF_TOKEN is
   present, written to a PRIVATE Hugging Face dataset repo file `studio/endpoint.json` so the app can fetch them.
 - IDLE SHUTDOWN: the server stops after IDLE_MIN minutes without a request, because a running Kaggle GPU session keeps
   burning the 30 h/week quota. Set IDLE_MIN higher only when you are actively using it.
"""
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

IDLE_MIN = int(os.environ.get("IDLE_MIN", 30))
HF_ENDPOINT_REPO = os.environ.get("HF_ENDPOINT_REPO", "")        # e.g. "youruser/voidvideo-endpoint" (private dataset repo)
PORT = 8000

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers==0.35.2", "transformers==4.57.3", "accelerate>=0.33",
                "ftfy", "sentencepiece", "imageio[ffmpeg]", "fastapi", "uvicorn[standard]", "python-multipart"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
REPO = Path("/kaggle/working/voidvideo")
subprocess.run(["git", "clone", "--depth", "1", "--branch", "claude/claude-md-jfha3c",
                "https://github.com/thanoxxinfnity/Voidvideo.git", str(REPO)], check=True)
sys.path.insert(0, str(REPO))

os.environ.setdefault("STUDIO_API_KEY", secrets.token_urlsafe(18))
os.environ["STUDIO_WORK"] = "/kaggle/working/studio_work"
from studio import backends, server  # noqa: E402

ckpt = sorted(Path("/kaggle/input").rglob("lora_weights.pt"))[0]
print("Using LoRA:", ckpt, flush=True)
backend = backends.WanBackend(str(ckpt), use_vace=True)
app = server.create_app(backend, os.environ["STUDIO_WORK"], os.environ["STUDIO_API_KEY"])

# Cloudflare quick tunnel
exe = "/kaggle/working/cloudflared"
subprocess.run(["curl", "-sSL", "-o", exe, "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], check=True)
os.chmod(exe, 0o755)
tun = subprocess.Popen([exe, "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"], stderr=subprocess.PIPE, text=True)
url = None
for line in tun.stderr:
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        url = m.group(0)
        break
print("=" * 60, f"\nSTUDIO URL : {url}\nAPI KEY    : {os.environ['STUDIO_API_KEY']}\n" + "=" * 60, flush=True)
threading.Thread(target=lambda: [None for _ in tun.stderr], daemon=True).start()      # keep draining the tunnel log

if HF_ENDPOINT_REPO and os.environ.get("HF_TOKEN"):
    try:
        from huggingface_hub import HfApi
        Path("endpoint.json").write_text(json.dumps({"url": url, "key": os.environ["STUDIO_API_KEY"], "time": time.time()}))
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj="endpoint.json", path_in_repo="studio/endpoint.json",
                                                         repo_id=HF_ENDPOINT_REPO, repo_type="dataset")
        print("endpoint published to the private HF repo", flush=True)
    except Exception as e:
        print("could not publish endpoint:", type(e).__name__, str(e)[:200], flush=True)


def idle_watch():
    while True:
        time.sleep(30)
        if time.time() - app.state.last_activity > IDLE_MIN * 60 and not any(
                j["state"] in ("queued", "running") for j in app.state.jobs.values()):
            print(f"idle for {IDLE_MIN} min: shutting down to save GPU quota", flush=True)
            os._exit(0)


threading.Thread(target=idle_watch, daemon=True).start()
import uvicorn  # noqa: E402

uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
