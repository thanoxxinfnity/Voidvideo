"""End-to-end API test with the fake backend (no GPU): python -m studio.test_api"""
import io
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from studio.backends import FakeBackend  # noqa: E402
from studio.server import create_app  # noqa: E402


def streams(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,nb_read_frames", "-count_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.split()
    return out


def wait(c, jid, key, secs=120):
    t0 = time.time()
    while time.time() - t0 < secs:
        s = c.get(f"/status/{jid}", headers=key).json()
        if s["state"] in ("done", "failed", "cancelled"):
            return s
        time.sleep(0.2)
    raise SystemExit("timeout")


def main():
    with tempfile.TemporaryDirectory() as td:
        K = {"X-API-Key": "secret"}
        c = TestClient(create_app(FakeBackend(), td, "secret"))
        assert c.get("/health").status_code == 200
        assert c.post("/generate", json={"prompt": "x"}).status_code == 401, "needs a key"
        assert c.post("/generate", json={"prompt": "x"}, headers={"X-API-Key": "wrong"}).status_code == 401
        # full job: enhance + impact + camera + grade + foley
        r = c.post("/generate", headers=K, json={"prompt": "@kaito ladai karta hai raat me", "steps": 5,
                   "finish": {"impact": [20], "camera": "dolly_in", "grade": "neon_night", "bloom": 0.3, "foley": True,
                              "relight": {"direction": "left"}}}).json()
        s = wait(c, r["id"], K)
        assert s["state"] == "done", s
        assert s["progress"] == 100 and s["result_url"] and s["preview_url"], s
        f = Path(td) / r["id"]
        mp4 = next(f.glob("*_sound.mp4"))
        st = streams(mp4)
        assert "audio" in str(st) and "video" in str(st), st
        assert c.get(s["result_url"], headers=K).headers["content-type"] == "video/mp4"
        assert c.get(s["preview_url"], headers=K).headers["content-type"] == "image/jpeg"
        # cancel a running job
        r2 = c.post("/generate", headers=K, json={"prompt": "slow job", "steps": 40, "fake_delay": 0.1}).json()
        time.sleep(0.6)
        assert c.post(f"/cancel/{r2['id']}", headers=K).json()["state"] in ("running", "cancelled")
        assert wait(c, r2["id"], K)["state"] == "cancelled"
        # queue limit
        import studio.server as srv
        ids = [c.post("/generate", headers=K, json={"prompt": f"q{i}", "steps": 20, "fake_delay": 0.2}) for i in range(srv.MAX_QUEUE + 3)]
        codes = [x.status_code for x in ids]
        assert 429 in codes and 200 in codes, codes
        for x in ids:
            if x.status_code == 200:
                c.post(f"/cancel/{x.json()['id']}", headers=K)
        # failed job does not kill the worker
        bad = c.post("/generate", headers=K, json={"prompt": "bad", "steps": 2, "finish": {"camera": "no_such_preset"}}).json()
        assert wait(c, bad["id"], K)["state"] == "failed"
        time.sleep(2)
        ok = c.post("/generate", headers=K, json={"prompt": "after failure", "steps": 2, "finish": {"retime": False}}).json()
        assert wait(c, ok["id"], K, 180)["state"] == "done"
        # storyboard with extend: 49 + (49 - 17) frames
        sb = c.post("/storyboard", headers=K, json={"shots": [
            {"prompt": "a boy runs", "steps": 2, "finish": {"retime": False, "bloom": 0, "grain": 0}},
            {"prompt": "the boy keeps running", "steps": 2, "continue": True}]}).json()
        s = wait(c, sb["id"], K, 180)
        assert s["state"] == "done", s
        n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
                            "-of", "csv=p=0", str(Path(td) / sb["id"] / Path(s["result_url"]).name)], capture_output=True, text=True).stdout
        movie = next((Path(td) / sb["id"]).glob("*_graded.mp4"))
        n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
                            "-of", "csv=p=0", str(movie)], capture_output=True, text=True).stdout.strip()
        assert n == str(49 + 32), f"storyboard frames {n}"
        # memory bank
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, "PNG")
        assert c.post("/bank/kaito", headers=K, files={"file": ("k.png", buf.getvalue(), "image/png")}).json()["names"] == ["kaito"]
        print("STUDIO API TEST OK  (queue codes:", sorted(set(codes)), ")")


if __name__ == "__main__":
    main()
