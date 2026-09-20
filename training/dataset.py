"""Video dataset for CogVideoX LoRA fine-tuning."""
import json
import random
from pathlib import Path

import decord
import numpy as np
import torch
from torch.utils.data import Dataset

decord.bridge.set_bridge("torch")


class VideoCaptionDataset(Dataset):
    """
    Reads a metadata.jsonl of {"video": "clips/x.mp4", "caption": "..."}
    (paths relative to `video_root`) and returns fixed-length, fixed-resolution
    frame tensors normalized to [-1, 1] plus the caption string.

    Clips are expected to already be normalized to the target resolution/fps/
    frame-count by scripts/prepare_dataset.py, so this loader mostly just reads
    and validates — with a defensive center-crop/pad in case a clip drifted.
    """

    def __init__(self, metadata_file: str, video_root: str, num_frames: int,
                 height: int, width: int, caption_dropout: float = 0.0):
        self.video_root = Path(video_root)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.caption_dropout = caption_dropout

        self.rows = []
        with open(metadata_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

        if not self.rows:
            raise ValueError(f"No rows found in {metadata_file}. Run prepare_dataset.py first.")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_frames(self, video_path: Path) -> torch.Tensor:
        vr = decord.VideoReader(str(video_path), width=self.width, height=self.height)
        total = len(vr)
        if total >= self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames).astype(int)
        else:
            # Pad short clips by repeating the last frame.
            indices = np.concatenate([np.arange(total), np.full(self.num_frames - total, total - 1)])
        frames = vr.get_batch(indices.tolist()).float()  # (T, H, W, C), 0..255
        frames = frames.permute(0, 3, 1, 2) / 127.5 - 1.0  # (T, C, H, W), -1..1
        return frames

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        video_path = self.video_root / row["video"]
        frames = self._load_frames(video_path)

        caption = row.get("caption", "")
        if caption and random.random() < self.caption_dropout:
            caption = ""

        return {"frames": frames, "caption": caption}
