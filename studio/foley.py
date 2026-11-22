"""
Procedural Foley: basic synthetic sound effects placed on events (impact frames, keywords), mixed to a wav and muxed into the clip.
This is NOT AI Foley: the sounds are simple synthesised stand-ins (noise/sine/inharmonic partials with envelopes). It gives
every clip usable timing-accurate SFX on CPU; swap in MMAudio / HunyuanVideo-Foley (GPU) later for realistic sound.
"""
import subprocess
import wave

import numpy as np

SR = 44100


def _env(n, attack=0.002, decay=0.2):
    t = np.arange(n) / SR
    return np.minimum(t / max(attack, 1e-4), 1) * np.exp(-t / decay)


def _lowpass(x, k):
    return np.convolve(x, np.ones(k) / k, mode="same")


def impact(dur=0.5, rnd=None):
    rnd = rnd or np.random.default_rng(0)
    n = int(SR * dur)
    t = np.arange(n) / SR
    thud = np.sin(2 * np.pi * (110 * np.exp(-t * 9) + 38) * t) * _env(n, 0.001, 0.18)
    crack = _lowpass(rnd.normal(size=n), 3) * _env(n, 0.0005, 0.04)
    return 0.8 * thud + 0.5 * crack


def whoosh(dur=0.45, rnd=None):
    rnd = rnd or np.random.default_rng(1)
    n = int(SR * dur)
    x = rnd.normal(size=n)
    env = np.sin(np.linspace(0, np.pi, n)) ** 2
    return _lowpass(x, 40) * env * 4


def clang(dur=0.9, rnd=None):
    n = int(SR * dur)
    t = np.arange(n) / SR
    out = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, a, d in
              [(820, 1, 6), (1371, .7, 7), (2210, .5, 9), (3170, .35, 12), (4630, .2, 16)])
    return out * 0.35 * _env(n, 0.0005, 1)


def thunder(dur=1.8, rnd=None):
    rnd = rnd or np.random.default_rng(2)
    n = int(SR * dur)
    crack = _lowpass(rnd.normal(size=n), 2) * _env(n, 0.0003, 0.05)
    rumble = _lowpass(rnd.normal(size=n), 400) * _env(n, 0.05, 0.7) * 8
    return 0.5 * crack + rumble


def rain(dur=3.0, rnd=None):
    rnd = rnd or np.random.default_rng(3)
    n = int(SR * dur)
    return (rnd.normal(size=n) - _lowpass(rnd.normal(size=n), 30)) * 0.12


def footstep(dur=0.2, rnd=None):
    rnd = rnd or np.random.default_rng(4)
    n = int(SR * dur)
    return _lowpass(rnd.normal(size=n), 60) * _env(n, 0.001, 0.05) * 3


def fire(dur=2.0, rnd=None):
    rnd = rnd or np.random.default_rng(5)
    n = int(SR * dur)
    x = _lowpass(rnd.normal(size=n), 120) * 1.5
    clicks = (rnd.random(n) > 0.9993) * rnd.uniform(0.3, 1, n)
    return x * 0.35 + _lowpass(clicks, 4) * 6


SOUNDS = {"impact": impact, "whoosh": whoosh, "clang": clang, "thunder": thunder, "rain": rain, "footstep": footstep, "fire": fire}
KEYWORDS = {"punch": "impact", "hit": "impact", "kick": "impact", "slam": "impact", "explod": "impact", "slash": "whoosh",
            "swing": "whoosh", "sword": "clang", "clash": "clang", "lightning": "thunder", "thunder": "thunder", "rain": "rain",
            "storm": "rain", "walk": "footstep", "run": "footstep", "fire": "fire", "flame": "fire"}


def plan(prompt, impact_frames=(), n_frames=49, fps=24):
    """Events [(time_s, sound, gain)] from impact frames (impact + whoosh just before) and prompt keywords (beds from t=0)."""
    ev = []
    for f in impact_frames:
        ev += [(max(f / fps - 0.12, 0), "whoosh", 0.6), (f / fps, "impact", 1.0)]
    p = prompt.lower()
    for kw, snd in KEYWORDS.items():
        if kw in p and snd in ("rain", "fire", "thunder") and not any(s == snd for _, s, _ in ev):
            ev.append((0.0 if snd != "thunder" else 0.4 * n_frames / fps, snd, 0.5))
        elif kw in p and snd in ("clang", "footstep") and not any(s == snd for _, s, _ in ev):
            if snd == "footstep":
                ev += [(0.15 + 0.42 * k, "footstep", 0.5) for k in range(int(n_frames / fps / 0.42))]
            else:
                ev.append((0.5 * n_frames / fps, "clang", 0.6))
    return ev


def render(events, seconds, path):
    out = np.zeros(int(SR * seconds))
    rnd = np.random.default_rng(7)
    for t, name, gain in events:
        s = SOUNDS[name](rnd=rnd) * gain
        i = int(t * SR)
        seg = s[: max(len(out) - i, 0)]
        out[i:i + len(seg)] += seg
    out = np.tanh(out * 1.2)                                    # soft clip
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((out * 32000).astype(np.int16).tobytes())
    return path


def mux(video, wav, dst):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-shortest", str(dst)], check=True)
