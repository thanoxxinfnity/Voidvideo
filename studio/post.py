"""
Post-processing lab (ffmpeg): colour grade (+ optional .cube LUT), bloom/glow, film grain, vignette, upscale, 60 fps.

grade_cmd(...) builds one ffmpeg call; upscale and RIFE are optional hooks that are used when the tools are installed
(Real-ESRGAN anime model via `realesrgan-ncnn-vulkan`, RIFE via `rife-ncnn-vulkan`) and fall back to lanczos / skipped.
NOTE for anime: frame interpolation fills the holds that give ones/twos/threes timing their look. Use `fps60` only on
camera-heavy or background-heavy shots; keep it off for hand-drawn character action.
"""
import shutil
import subprocess
from pathlib import Path

GRADES = {  # name -> ffmpeg eq/colorbalance chain
    "none": "",
    "cinematic": "eq=contrast=1.08:saturation=1.1:gamma=0.97,colorbalance=rs=-0.02:bs=0.04:rh=0.04:bh=-0.03",
    "neon_night": "eq=contrast=1.15:saturation=1.3:brightness=-0.03,colorbalance=bs=0.08:rs=0.05:gm=-0.03",
    "sunset": "eq=contrast=1.05:saturation=1.15,colorbalance=rs=0.06:gs=0.01:bs=-0.06:rh=0.05:bh=-0.04",
    "cold_dark": "eq=contrast=1.12:saturation=0.9:brightness=-0.05,colorbalance=bs=0.06:rs=-0.04",
    "bright_anime": "eq=contrast=1.04:saturation=1.2:brightness=0.02",
}


def grade_cmd(src, dst, grade="cinematic", lut=None, bloom=0.0, grain=0.0, vignette=False, scale=None, fps=None, crf=14):
    vf = []
    if GRADES.get(grade):
        vf.append(GRADES[grade])
    if lut:
        vf.append(f"lut3d='{lut}'")
    if vignette:
        vf.append("vignette=PI/5")
    chain = ",".join(vf) or "null"
    parts = [f"[0:v]{chain}[g]"]
    last = "g"
    if bloom > 0:                                    # screen-blend a blurred copy: soft glow on bright parts
        parts.append(f"[{last}]split[a][b];[b]gblur=sigma=18,eq=brightness=-0.15:contrast=1.3[bl];"
                     f"[a][bl]blend=all_mode=screen:all_opacity={bloom}[bo]")
        last = "bo"
    if grain > 0:
        parts.append(f"[{last}]noise=alls={int(grain * 30)}:allf=t[gr]")
        last = "gr"
    tail = []
    if scale:
        tail.append(f"scale={scale[0]}:{scale[1]}:flags=lanczos")
    if fps:
        tail.append(f"fps={fps}")
    if tail:
        parts.append(f"[{last}]{','.join(tail)}[out]")
        last = "out"
    return ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex", ";".join(parts), "-map", f"[{last}]",
            "-an", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(dst)]


def run_grade(*a, **k):
    subprocess.run(grade_cmd(*a, **k), check=True)


def upscale(src, dst, target_h=1080):
    """Real-ESRGAN (anime model) if the ncnn binary exists, else lanczos."""
    exe = shutil.which("realesrgan-ncnn-vulkan")
    if not exe:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"scale=-2:{target_h}:flags=lanczos", "-c:v", "libx264",
                        "-crf", "14", "-pix_fmt", "yuv420p", str(dst)], check=True)
        return "lanczos"
    work = Path(str(dst) + "_frames")
    (work / "in").mkdir(parents=True, exist_ok=True)
    (work / "out").mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), str(work / "in" / "%05d.png")], check=True)
    subprocess.run([exe, "-i", str(work / "in"), "-o", str(work / "out"), "-n", "realesr-animevideov3", "-s", "2"], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", str(work / "out" / "%05d.png"), "-vf",
                    f"scale=-2:{target_h}:flags=lanczos", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(dst)], check=True)
    shutil.rmtree(work, ignore_errors=True)
    return "realesrgan"


def to_60fps(src, dst):
    """RIFE if installed, otherwise None (the caller keeps the original). See the NOTE about anime timing above."""
    exe = shutil.which("rife-ncnn-vulkan")
    if not exe:
        return None
    work = Path(str(dst) + "_frames")
    (work / "in").mkdir(parents=True, exist_ok=True)
    (work / "out").mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), str(work / "in" / "%05d.png")], check=True)
    n = len(list((work / "in").glob("*.png")))
    subprocess.run([exe, "-i", str(work / "in"), "-o", str(work / "out"), "-n", str(int(n * 2.5))], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "60", "-i", str(work / "out" / "%08d.png"), "-c:v", "libx264", "-crf", "14",
                    "-pix_fmt", "yuv420p", str(dst)], check=True)
    shutil.rmtree(work, ignore_errors=True)
    return "rife"


def relight(frames, direction="left", color=(255, 190, 120), intensity=0.45, ambient=0.85, softness=1.2):
    """2D relight on a finished clip (frames (T,H,W,3) uint8): a directional gradient brightens the side the light comes from,
    darkens the far side and tints the lit side. This is a painted-on light (no 3D normals), good for mood changes
    (sun from the left, neon from the right, dark from below); it cannot move real shadows."""
    import numpy as np
    t, h, w, _ = frames.shape
    yy, xx = np.mgrid[0:h, 0:w]
    u = {"left": 1 - xx / w, "right": xx / w, "top": 1 - yy / h, "bottom": yy / h, "center": 1 - np.hypot(xx / w - .5, yy / h - .5) * 1.8}[direction]
    u = np.clip(u, 0, 1) ** softness
    tint = np.array(color, np.float32) / 255.0
    gain = ambient + intensity * u[..., None]
    out = frames.astype(np.float32) * gain
    out = out * (1 - 0.25 * u[..., None]) + out * tint * 0.25 * u[..., None] * 1.6
    return np.clip(out, 0, 255).astype(np.uint8)
