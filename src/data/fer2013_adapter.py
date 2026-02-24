"""
FER-2013 dataset adapter.

Loads the FER-2013 CSV (48×48 grayscale), converts to the canonical
FaceCrop contract (256×256 RGB float32 [0,1]), and preserves original
train / PublicTest / PrivateTest splits.

Expected file layout:
    data/fer2013/fer2013.csv

CSV columns:  emotion, pixels, Usage
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal, Optional

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.data.contracts import (
    CANONICAL_RESOLUTION,
    ExpressionLabel,
    FaceCrop,
    FaceCropMeta,
    gray_to_rgb,
    uint8_to_float,
)

SPLIT_MAP: dict[str, str] = {
    "train": "Training",
    "val": "PublicTest",
    "test": "PrivateTest",
}


class FER2013Dataset(Dataset):
    """
    PyTorch-compatible dataset that yields FaceCrop objects.

    Parameters
    ----------
    root : str | Path
        Directory containing ``fer2013.csv``.
    split : {"train", "val", "test"}
        Which split to load.
    resolution : int
        Target spatial size (default = 256, the canonical contract).
    transform : callable, optional
        Additional torchvision-style transform applied *after* conversion to
        float32 RGB numpy array.  Receives and returns np.ndarray (H, W, 3).
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        resolution: int = CANONICAL_RESOLUTION,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.resolution = resolution
        self.transform = transform

        csv_path = self.root / "fer2013.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"FER-2013 CSV not found at {csv_path}. "
                f"Download from https://www.kaggle.com/datasets/msambare/fer2013 "
                f"and place it in {self.root}"
            )

        df = pd.read_csv(csv_path)
        usage = SPLIT_MAP[split]
        self.df = df[df["Usage"] == usage].reset_index(drop=True)

        # Pre-parse pixel strings into arrays for faster __getitem__
        self._pixels: list[np.ndarray] = []
        self._labels: list[int] = []
        for _, row in self.df.iterrows():
            # np.fromstring(sep=...) still works but np.array(str.split()) is
            # more robust across NumPy versions and avoids deprecation warnings.
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            self._pixels.append(pixels)
            self._labels.append(int(row["emotion"]))

    # ── Dataset protocol ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> FaceCrop:
        gray48 = self._pixels[idx]                           # (48, 48) uint8
        label = self._labels[idx]

        # Up-sample to canonical resolution
        img = cv2.resize(
            gray48,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_CUBIC,
        )                                                    # (256, 256) uint8

        # Grayscale → RGB, uint8 → float32 [0,1]
        img_rgb = gray_to_rgb(img)                           # (256, 256, 3) uint8
        img_f32 = uint8_to_float(img_rgb)                    # float32 [0,1]

        if self.transform is not None:
            img_f32 = self.transform(img_f32)

        meta = FaceCropMeta(
            dataset="fer2013",
            split=self.split,
            image_id=f"fer2013_{self.split}_{idx:05d}",
            expression_label=label,
            alignment_method="none",          # FER-2013 is pre-cropped
            detector="none",
        )

        return FaceCrop(image=img_f32, meta=meta)

    # ── Convenience ────────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        """Return the number of expression classes."""
        return len(ExpressionLabel)

    @property
    def class_names(self) -> list[str]:
        """Return human-readable names for each expression class."""
        return [e.name.capitalize() for e in ExpressionLabel]

    def label_distribution(self) -> dict[str, int]:
        """Return per-class sample count."""
        from collections import Counter
        counts = Counter(self._labels)
        return {ExpressionLabel(k).name: v for k, v in sorted(counts.items())}

    def __repr__(self) -> str:
        return (
            f"FER2013Dataset(split={self.split!r}, "
            f"n={len(self)}, res={self.resolution})"
        )
