"""
Studio pipeline: prompt enhance -> generate (backend) -> finish (retime, impact frames, speed lines, camera, shake, relight,
grade, upscale, foley) -> mp4. Also multi-shot storyboards with Video Extend and the consistency memory bank.

Order matters: retime first (so camera moves stay smooth on top of the drawing timing), impact frames before the camera
shake that follows them, grade/upscale last.
"""
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from crowd_compose import read_frames, write  # noqa: E402

from . import camera, foley, impact, modes, post, prompt_enhancer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class Cancelled(Exception):
    pass


def _write_tmp(frames, path, fps=24):
    write(frames, fps, str(path))
    return path


def finish(frames, opts, work, prompt="", progress=None):
    """frames (T,H,W,3) uint8 -> final mp4 path. opts keys (all optional): retime, camera, camera_keys, shake, impact, speed_lines,
    relight, grade, bloom, grain, vignette, upscale, fps60, foley."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    step = lambda msg: progress and progress(msg)          # noqa: E731
    if opts.get("retime", True):
        step("retime")
        raw, rt = _write_tmp(frames, work / f"{tag}_raw.mp4"), work / f"{tag}_rt.mp4"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "anime_retime.py"), str(raw), str(rt), "--adaptive"], check=True,
                       capture_output=True)
        frames = read_frames(rt)
    if opts.get("impact"):
        step("impact frames")
        frames = impact.insert_impact(frames, opts["impact"], hold=opts.get("impact_hold", 2))
    if opts.get("speed_lines"):
        s, e = opts["speed_lines"]
        frames = impact.speed_lines(frames, s, e, radial=opts.get("speed_radial", True))
    cam = opts.get("camera")
    if cam or opts.get("camera_keys"):
        step("camera")
        keys = opts.get("camera_keys") or camera.PRESETS[cam]
        frames = camera.apply_path(frames, camera.spline(keys, len(frames)))
    if opts.get("shake") or opts.get("impact"):
        frames = camera.shake(frames, opts.get("shake") or opts["impact"], strength=opts.get("shake_strength", 0.02))
    if opts.get("relight"):
        step("relight")
        frames = post.relight(frames, **opts["relight"])
    out_mp4 = work / f"{tag}_graded.mp4"
    step("grade")
    src = _write_tmp(frames, work / f"{tag}_pre.mp4")
    post.run_grade(src, out_mp4, opts.get("grade", "cinematic"), lut=opts.get("lut"), bloom=opts.get("bloom", 0.2),
                   grain=opts.get("grain", 0.15), vignette=opts.get("vignette", False))
    final = out_mp4
    if opts.get("upscale"):
        step("upscale")
        up = work / f"{tag}_up.mp4"
        post.upscale(final, up, int(opts["upscale"]))
        final = up
    if opts.get("fps60"):
        step("60 fps")
        f60 = work / f"{tag}_60.mp4"
        if post.to_60fps(final, f60):
            final = f60
    if opts.get("foley"):
        step("foley")
        n = len(frames)
        ev = foley.plan(prompt, opts.get("impact") or [], n)
        wav = foley.render(ev, n / 24 + 0.2, work / f"{tag}.wav")
        mixed = work / f"{tag}_sound.mp4"
        foley.mux(final, wav, mixed)
        final = mixed
    return final


def build_request(req, bank=None):
    """Resolve uploads/paths and memory names into arrays/images the backend uses."""
    r = dict(req)
    if r.get("video"):
        r["_frames"] = read_frames(r["video"])
    if r.get("mask") and Path(r["mask"]).exists():
        m = read_frames(r["mask"])
        r["_mask"] = m.mean(-1).astype(np.uint8)
    if r.get("box"):
        n = len(r["_frames"]) if "_frames" in r else r.get("num_frames", 49)
        r["_mask"] = modes.box_mask(n, r["_frames"].shape[2] if "_frames" in r else r["width"],
                                    r["_frames"].shape[1] if "_frames" in r else r["height"], r["box"])
    if r.get("keys"):
        from PIL import Image
        r["_keys"] = [{"frame": k["frame"], "image": Image.open(k["image"]).convert("RGB")} for k in r["keys"]]
    if bank is not None and r.get("remember_refs"):
        r["_refs"] = bank.refs(r["remember_refs"])
    return r


def run_job(req, backend, work, bank=None, progress=None, cancelled=lambda: False):
    """One shot. progress(done, total, message). Returns the final mp4 path."""
    r = build_request(req, bank)
    if r.get("enhance", True):
        r["prompt"], _ = prompt_enhancer.enhance(r["prompt"])
    frames = backend.generate(r, lambda s, t: progress and progress(s, t, "generating"), cancelled)
    if cancelled():
        raise Cancelled()
    out = finish(frames, r.get("finish", {}), work, r["prompt"], lambda m: progress and progress(1, 1, m))
    if bank is not None and r.get("remember_as"):
        bank.add_from_clip(r["remember_as"], frames, kind=r.get("remember_kind", "character"))
    return out


def run_storyboard(shots, backend, work, bank=None, progress=None, cancelled=lambda: False):
    """shots: list of job dicts. A shot with "continue": true is generated with mode 'extend' from the previous shot's frames and
    its first `keep` frames (copies of the previous tail) are dropped when joining, so a 5 s idea becomes 30-60 s."""
    work = Path(work)
    parts, prev = [], None
    for i, shot in enumerate(shots):
        s = dict(shot)
        if s.pop("continue", False) and prev is not None:
            s["mode"], s["_prev"] = "extend", prev
        r = build_request(s, bank)
        if r.get("enhance", True):
            r["prompt"], _ = prompt_enhancer.enhance(r["prompt"])
        if "_prev" in s:
            r["_prev"] = s["_prev"]
        frames = backend.generate(r, lambda st, t, i=i: progress and progress(i, len(shots), f"shot {i + 1}/{len(shots)} step {st}/{t}"), cancelled)
        if cancelled():
            raise Cancelled()
        prev = frames
        keep = r.get("keep", 17) if r.get("mode") == "extend" else 0
        parts.append(frames[keep:])
        if bank is not None and r.get("remember_as"):
            bank.add_from_clip(r["remember_as"], frames, kind=r.get("remember_kind", "character"))
    movie = np.concatenate(parts)
    return finish(movie, shots[0].get("finish", {}), work, " ".join(s["prompt"] for s in shots), None)
