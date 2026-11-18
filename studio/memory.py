"""
Consistency memory bank: reference images per character / location, reused as VACE `reference_images` in every later shot so
looks stay the same across a 10-20 scene film. Stored on disk (a folder + index.json); works without any model.
"""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


class Bank:
    def __init__(self, root="studio_bank"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.idx_file = self.root / "index.json"
        self.idx = json.load(open(self.idx_file)) if self.idx_file.exists() else {}

    def _save(self):
        json.dump(self.idx, open(self.idx_file, "w"), indent=1)

    def add(self, name, image, kind="character", note=""):
        """image: PIL.Image or path. Several images per name are kept (front, side, expression ...)."""
        d = self.root / name
        d.mkdir(exist_ok=True)
        i = len(list(d.glob("*.png")))
        dst = d / f"{i:02d}.png"
        if isinstance(image, (str, Path)):
            Image.open(image).convert("RGB").save(dst)
        else:
            image.convert("RGB").save(dst)
        e = self.idx.setdefault(name, {"kind": kind, "note": note, "files": []})
        e["files"].append(dst.name)
        self._save()
        return dst

    def add_from_clip(self, name, frames, kind="character", pick="sharpest"):
        """Store the sharpest frame (highest gradient energy) of a generated clip (T,H,W,3) as a reference."""
        g = frames.astype(np.float32).mean(-1)
        score = [np.abs(np.diff(x, axis=0)).mean() + np.abs(np.diff(x, axis=1)).mean() for x in g]
        k = int(np.argmax(score)) if pick == "sharpest" else len(frames) // 2
        return self.add(name, Image.fromarray(frames[k]), kind)

    def refs(self, names, max_total=3):
        """PIL reference images for the given names (VACE takes a few; more dilutes each)."""
        out = []
        for n in names:
            for f in self.idx.get(n, {}).get("files", [])[:2]:
                out.append(Image.open(self.root / n / f).convert("RGB"))
        return out[:max_total]

    def names(self):
        return list(self.idx)

    def remove(self, name):
        shutil.rmtree(self.root / name, ignore_errors=True)
        self.idx.pop(name, None)
        self._save()
