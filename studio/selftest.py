"""CPU self-test of the studio tools on synthetic clips. python -m studio.selftest"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio import camera, foley, impact, memory, modes, post, prompt_enhancer  # noqa: E402
from PIL import Image  # noqa: E402


def clip(t=49, h=240, w=360):
    f = np.zeros((t, h, w, 3), np.uint8)
    f[:] = (60, 70, 110)
    f[:, h // 2:] = (50, 50, 50)
    for i in range(t):
        x = 40 + int(i * 5)
        f[i, 100:200, x:x + 40] = (220, 60, 60)
        f[i, 80:100, x + 5:x + 35] = (250, 215, 175)
    return f


def main():
    f = clip()
    p = camera.spline(camera.PRESETS["fpv_sweep"], len(f))
    assert all(len(v) == len(f) for v in p.values()), "spline length"
    out = camera.apply_path(f, p)
    assert out.shape == f.shape and out.std() > 10, "apply_path"
    sh = camera.shake(f, [10, 30])
    assert (sh[10] != f[10]).mean() > 0.01 and np.array_equal(sh[25], f[25]), "shake moves hit frames only"
    depth = np.tile(np.linspace(0, 1, f.shape[2])[None], (f.shape[1], 1)).astype(np.float32)
    par = camera.parallax(f, depth, np.linspace(-1, 1, len(f)))
    assert par.shape == f.shape and not np.array_equal(par[0], par[-1]), "parallax"
    dof = camera.depth_of_field(f[:5], depth, 0.2)
    assert dof.shape[0] == 5 and np.abs(dof[0].astype(int) - f[0].astype(int)).mean() > 0.5, "dof blurs"
    imp = impact.insert_impact(f, [20], hold=2)
    assert set(np.unique(imp[20])) <= {0, 255} and len(imp) == len(f), "impact frame is pure black/white"
    sl = impact.speed_lines(f, 5, 15)
    assert not np.array_equal(sl[8], f[8]) and np.array_equal(sl[20], f[20]), "speed lines only inside the range"
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "a.mp4", Path(td) / "b.mp4"
        pr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "360x240", "-r", "24", "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], stdin=subprocess.PIPE)
        pr.communicate(f.tobytes())
        post.run_grade(src, dst, "cinematic", bloom=0.35, grain=0.3, vignette=True, scale=(720, 480))
        n = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of",
                            "csv=p=0", str(dst)], capture_output=True, text=True).stdout.strip()
        assert n == "720,480", f"graded output {n}"
        how = post.upscale(dst, Path(td) / "c.mp4", 720)
        print("upscale backend:", how, "| 60fps backend:", post.to_60fps(dst, Path(td) / "d.mp4"))
    # VACE input builders
    key_img = Image.fromarray(f[0])
    v, m = modes.keyframes([(0, key_img), (24, key_img), (48, key_img)], 49, 320, 192)
    mm = np.stack([np.asarray(x) for x in m])
    assert len(v) == 49 and mm[0].max() == 0 and mm[24].max() == 0 and mm[10].min() == 255, "keyframes mask"
    v, m = modes.extend(f, 49, 320, 192, keep=17)
    mm = np.stack([np.asarray(x) for x in m])
    assert mm[:17].max() == 0 and mm[17:].min() == 255, "extend keeps the first 17 frames"
    v, m = modes.inpaint(f, modes.box_mask(49, 360, 240, (0.3, 0.3, 0.6, 0.9)), 360, 240)
    assert np.asarray(m[0]).max() == 255 and np.asarray(v[0])[150, 150, 0] == modes.GRAY, "inpaint grays the hole"
    v, m = modes.outpaint(f[:, :, :180], 360, 240)           # a narrow clip on a wide canvas
    assert np.asarray(m[0])[:, :5].min() == 255 and np.asarray(m[0])[:, 180].max() == 0, "outpaint mask on the sides only"
    v, m = modes.control(f[:3], "edges", 360, 240)
    assert np.asarray(v[0]).max() == 255 and np.asarray(m[0]).min() == 255, "edge control video"
    with tempfile.TemporaryDirectory() as td:
        b = memory.Bank(td)
        b.add_from_clip("kaito", f)
        assert b.names() == ["kaito"] and len(b.refs(["kaito"])) == 1, "memory bank"
        ev = foley.plan("a sword fight in the rain, punch", [20], 49)
        wav = foley.render(ev, 2.2, Path(td) / "s.wav")
        assert wav.stat().st_size > 50000 and {"impact", "whoosh", "rain", "clang"} <= {e[1] for e in ev}, ev
    relit = post.relight(f, "left")
    assert relit[0][120, 5].sum() != relit[0][120, 350].sum(), "relight is directional"
    text, added = prompt_enhancer.enhance("@kaito ladai karta hai @ryu se raat me, #black_flash")
    assert "On the left" in text and "Effects:" in text and "night" in text.lower() and text.endswith("."), text
    print("enhanced:", text[:260], "...", added)
    print("STUDIO SELFTEST OK")


if __name__ == "__main__":
    main()
