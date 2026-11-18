"""
Studio API for the phone app (CHoMU Studio). One GPU worker, a bounded FIFO queue, polling endpoints.

  POST /upload              multipart file -> {"id", "path"}  (input video / mask / key images)
  POST /generate            JSON job -> {"id", "queue_position"}           (429 when the queue is full)
  POST /storyboard          {"shots": [...]} -> {"id"}                      (multi-shot + extend)
  GET  /status/{id}         {state, progress 0-100, stage, queue_position, preview_url, result_url, error}
  POST /cancel/{id}
  GET  /result/{id}  /preview/{id}  /health   (health needs no key)
  GET  /bank, POST /bank/{name}   consistency memory bank (names + upload a reference image)

Every call except /health needs the header  X-API-Key: <STUDIO_API_KEY>  (the Kaggle tunnel URL is public).
The phone only polls /status (a few hundred bytes) and downloads the mp4 once: nothing heavy runs on the device.
"""
import os
import queue
import secrets
import threading
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from . import pipeline
from .memory import Bank

API_KEY = os.environ.get("STUDIO_API_KEY") or secrets.token_urlsafe(18)
WORK = Path(os.environ.get("STUDIO_WORK", "studio_work"))
MAX_QUEUE = int(os.environ.get("STUDIO_MAX_QUEUE", 8))


def create_app(backend, work=WORK, api_key=API_KEY):
    work = Path(work)
    (work / "uploads").mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="VoidVideo Studio")
    bank = Bank(work / "bank")
    jobs, q, order = {}, queue.Queue(), []
    lock = threading.Lock()

    app.state.last_activity = time.time()

    def auth(x_api_key: str = Header(default="")):
        if not secrets.compare_digest(x_api_key, api_key):
            raise HTTPException(401, "bad or missing X-API-Key")
        app.state.last_activity = time.time()                 # the Kaggle entrypoint stops an idle server to save GPU quota

    def position(jid):
        with lock:
            pending = [j for j in order if jobs[j]["state"] == "queued"]
        return pending.index(jid) + 1 if jid in pending else 0

    def worker():
        while True:
            jid = q.get()
            j = jobs[jid]
            if j["state"] == "cancelled":
                continue
            j["state"], j["started"] = "running", time.time()

            def progress(done, total, msg=None):
                j["progress"] = round(100 * done / max(total, 1) * 0.9, 1)          # last 10 % = finishing (grade, foley ...)
                j["stage"] = msg or j["stage"]
                if msg and msg != "generating":
                    j["progress"] = max(j["progress"], 90.0)

            try:
                jd = work / jid
                if j["kind"] == "storyboard":
                    out = pipeline.run_storyboard(j["req"]["shots"], backend, jd, bank, progress, lambda: j["cancel"])
                else:
                    out = pipeline.run_job(j["req"], backend, jd, bank, progress, lambda: j["cancel"])
                j.update(state="done", progress=100.0, stage="done", result=str(out))
                try:
                    from . import preview
                    j["preview"] = preview.mid_frame(out, jd / "preview.jpg")
                except Exception:
                    pass
            except pipeline.Cancelled:
                j.update(state="cancelled", stage="cancelled")
            except Exception as e:                       # keep the worker alive whatever one job does
                j.update(state="failed", error=f"{type(e).__name__}: {str(e)[:300]}", stage="failed")
            j["finished"] = time.time()
            app.state.last_activity = time.time()

    threading.Thread(target=worker, daemon=True).start()

    def submit(kind, req):
        with lock:
            if sum(1 for j in jobs.values() if j["state"] in ("queued", "running")) >= MAX_QUEUE:
                raise HTTPException(429, "queue is full, try again in a minute")
            jid = uuid.uuid4().hex[:10]
            jobs[jid] = {"kind": kind, "req": req, "state": "queued", "progress": 0.0, "stage": "queued", "cancel": False,
                         "created": time.time()}
            order.append(jid)
        q.put(jid)
        return jid

    @app.get("/health")
    def health():
        return {"ok": True, "queued": sum(1 for j in jobs.values() if j["state"] == "queued")}

    @app.post("/upload", dependencies=[Depends(auth)])
    async def upload(file: UploadFile = File(...)):
        uid = uuid.uuid4().hex[:10]
        dst = work / "uploads" / f"{uid}_{Path(file.filename).name}"
        dst.write_bytes(await file.read())
        return {"id": uid, "path": str(dst)}

    @app.post("/generate", dependencies=[Depends(auth)])
    def generate(req: dict):
        if not req.get("prompt"):
            raise HTTPException(400, "prompt is required")
        jid = submit("job", req)
        return {"id": jid, "queue_position": position(jid)}

    @app.post("/storyboard", dependencies=[Depends(auth)])
    def storyboard(req: dict):
        if not req.get("shots"):
            raise HTTPException(400, "shots is required")
        jid = submit("storyboard", req)
        return {"id": jid, "queue_position": position(jid)}

    @app.get("/status/{jid}", dependencies=[Depends(auth)])
    def status(jid: str):
        j = jobs.get(jid) or (_ for _ in ()).throw(HTTPException(404, "unknown job"))
        return {"id": jid, "state": j["state"], "progress": j["progress"], "stage": j["stage"], "queue_position": position(jid),
                "preview_url": f"/preview/{jid}" if j.get("preview") else None,
                "result_url": f"/result/{jid}" if j["state"] == "done" else None, "error": j.get("error")}

    @app.post("/cancel/{jid}", dependencies=[Depends(auth)])
    def cancel(jid: str):
        j = jobs.get(jid) or (_ for _ in ()).throw(HTTPException(404, "unknown job"))
        if j["state"] == "queued":
            j["state"], j["stage"] = "cancelled", "cancelled"
        elif j["state"] == "running":
            j["cancel"] = True
        return {"id": jid, "state": j["state"]}

    @app.get("/result/{jid}", dependencies=[Depends(auth)])
    def result(jid: str):
        j = jobs.get(jid)
        if not j or j["state"] != "done":
            raise HTTPException(404, "not ready")
        return FileResponse(j["result"], media_type="video/mp4", filename=f"{jid}.mp4")

    @app.get("/preview/{jid}", dependencies=[Depends(auth)])
    def preview_img(jid: str):
        j = jobs.get(jid)
        if not j or not j.get("preview"):
            raise HTTPException(404, "no preview")
        return FileResponse(j["preview"], media_type="image/jpeg")

    @app.get("/bank", dependencies=[Depends(auth)])
    def bank_list():
        return {"names": bank.names()}

    @app.post("/bank/{name}", dependencies=[Depends(auth)])
    async def bank_add(name: str, file: UploadFile = File(...), kind: str = "character"):
        tmp = work / "uploads" / f"bank_{uuid.uuid4().hex[:6]}_{Path(file.filename).name}"
        tmp.write_bytes(await file.read())
        Image.open(tmp)                                   # fails (400) on a non-image
        bank.add(name, tmp, kind)
        return {"names": bank.names()}

    app.state.jobs, app.state.bank = jobs, bank
    return app
