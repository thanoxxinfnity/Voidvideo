"""Pixal3D image -> textured GLB backend.

Two implementations behind one interface:
  * RealBackend  - TencentARC/Pixal3D on a CUDA GPU (Kaggle T4). Needs the
                   checkout at PIXAL3D_REPO, already patched (see patches.py).
  * MockBackend  - CPU-only stand-in (PIXAL3D_MOCK=1) that returns a textured
                   mesh with the requested triangle budget, so the UI can be
                   tested without a GPU.
"""
import os
import sys
import threading
import time

import numpy as np
from PIL import Image

OUT_DIR = os.environ.get("PIXAL3D_OUT", "/kaggle/working/out")


class GenerationError(RuntimeError):
    """A failure whose message is safe and useful to show to the user."""


def to_rgba(img, remove_bg=True, session_factory=None):
    """Return an RGBA image the pipeline can use.

    Pixal3D only calls its background remover when the image has no alpha, and
    the remover model is gated, so we always hand it a real alpha channel.
    """
    if not isinstance(img, Image.Image):
        raise GenerationError("Koi image nahi mili - pehle image upload karo.")
    img = img.convert("RGBA") if img.mode != "RGBA" else img
    alpha = np.asarray(img.getchannel("A"))
    has_cutout = alpha.min() < 250 and (alpha < 250).mean() > 0.01
    if has_cutout:
        return img
    if not remove_bg:
        # Opaque image used as-is: the whole frame becomes the object.
        return img
    try:
        from rembg import new_session, remove
        session = session_factory() if session_factory else new_session("u2net")
        out = remove(img.convert("RGB"), session=session)
        out = out.convert("RGBA")
    except Exception as e:  # noqa: BLE001
        raise GenerationError(
            "Background hatane mein dikkat aayi (%s). Transparent-background PNG upload karo." % e
        )
    a = np.asarray(out.getchannel("A"))
    if (a > 127).mean() < 0.01:
        raise GenerationError(
            "Background remove ke baad object nahi bacha. Transparent-background PNG upload karo."
        )
    return out


def count_triangles(path):
    import trimesh
    scene = trimesh.load(path, force="scene")
    return int(sum(len(g.faces) for g in scene.geometry.values()))


class _Base:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"       # idle | loading | ready | error
        self.state_msg = ""
        self._rembg_session = None

    # -- helpers ----------------------------------------------------------
    def _rembg(self):
        if self._rembg_session is None:
            from rembg import new_session
            self._rembg_session = new_session("u2net")
        return self._rembg_session

    def prepare(self, img, remove_bg):
        return to_rgba(img, remove_bg, self._rembg)

    def warmup(self):
        raise NotImplementedError

    def generate(self, img, faces, texture_size, seed, remove_bg):
        raise NotImplementedError


class MockBackend(_Base):
    name = "mock (CPU)"

    def warmup(self):
        self.state, self.state_msg = "ready", "mock backend ready"

    def generate(self, img, faces, texture_size, seed, remove_bg):
        import trimesh
        with self.lock:
            rgba = self.prepare(img, remove_bg)
            os.makedirs(OUT_DIR, exist_ok=True)
            # uv_sphere(count=[c, c]) has ~4*c*c triangles.
            count = max(4, int(round((faces / 4) ** 0.5)))
            mesh = trimesh.creation.uv_sphere(count=[count, count])
            v = mesh.vertices
            uv = np.stack([(np.arctan2(v[:, 1], v[:, 0]) / (2 * np.pi)) % 1.0,
                           np.arccos(np.clip(v[:, 2], -1, 1)) / np.pi], axis=1)
            tex = rgba.convert("RGB").resize((texture_size, texture_size))
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=tex))
            out = os.path.join(OUT_DIR, "model_%d.glb" % int(time.time() * 1000))
            mesh.export(out)
            return out, count_triangles(out)


class RealBackend(_Base):
    name = "Pixal3D (CUDA)"

    def __init__(self, repo=None):
        super().__init__()
        self.repo = repo or os.environ.get("PIXAL3D_REPO", "/kaggle/tmp/pixal3d")
        self.inference = None

    def _import(self):
        if self.inference is None:
            os.environ["ATTN_BACKEND"] = "sdpa"
            if self.repo not in sys.path:
                sys.path.insert(0, self.repo)
            os.chdir(self.repo)
            import inference  # noqa: WPS433  (patched upstream module)
            self.inference = inference
        return self.inference

    def warmup(self):
        """Build the pipeline and MoGe once so the first request is not the slow one."""
        with self.lock:
            if self.state == "ready":
                return
            self.state, self.state_msg = "loading", "Models download + load ho rahe hain (pehli baar lamba time lagta hai)..."
            try:
                inf = self._import()
                inf.init_pipeline(inf.MODEL_PATH, low_vram=True)
                inf.load_moge_model(device="cpu")
                self.state, self.state_msg = "ready", "Ready"
            except BaseException as e:  # noqa: BLE001
                self.state, self.state_msg = "error", "Model load fail: %r" % (e,)
                raise

    def generate(self, img, faces, texture_size, seed, remove_bg):
        import torch
        rgba = self.prepare(img, remove_bg)
        if self.state != "ready":
            self.warmup()
        with self.lock:
            inf = self._import()
            os.makedirs(OUT_DIR, exist_ok=True)
            stamp = int(time.time() * 1000)
            src = os.path.join(OUT_DIR, "_input_%d.png" % stamp)
            out = os.path.join(OUT_DIR, "model_%d.glb" % stamp)
            rgba.save(src)
            inf.TARGET_FACES = int(faces)
            inf.TEXTURE_SIZE = int(texture_size)
            try:
                inf.run_inference(image_path=src, output_path=out, seed=int(seed),
                                  low_vram=True, resolution=1024)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise GenerationError(
                    "GPU memory khatam ho gayi. Texture size chhota karo (1024) aur dobara try karo.")
            except BaseException as e:  # noqa: BLE001
                torch.cuda.empty_cache()
                raise GenerationError("Generation fail: %r" % (e,))
            finally:
                try:
                    os.remove(src)
                except OSError:
                    pass
            if not os.path.exists(out):
                raise GenerationError("GLB file nahi bani (pipeline ne kuch output nahi diya).")
            return out, count_triangles(out)


def make_backend():
    if os.environ.get("PIXAL3D_MOCK") == "1":
        return MockBackend()
    return RealBackend()
