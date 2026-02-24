"""
Pins Face Recognition dataset adapter.

Loads the Pins Face Recognition dataset (105 celebrity identities, ~17 534 images)
and converts to the canonical FaceCrop contract (256×256 RGB float32 [0,1]).

Expected file layout:
    data/pins_face_recognition/
        pins_Adam Sandler/
            image_001.jpg
            ...
        pins_Adriana Lima/
            ...
        ...  (105 identity folders)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Literal, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from src.data.contracts import (
    CANONICAL_RESOLUTION,
    FaceCrop,
    FaceCropMeta,
    uint8_to_float,
)


class PinsFaceDataset(Dataset):
    """
    PyTorch-compatible dataset that yields FaceCrop objects for identity
    evaluation.

    Parameters
    ----------
    root : str | Path
        Root directory of the Pins dataset
        (e.g. ``data/pins_face_recognition``).
    split : {"train", "val", "test", "all"}
        Dataset split.  ``"all"`` returns every image.
        Splits are created deterministically by hashing filenames:
        70 % train / 15 % val / 15 % test.
    resolution : int
        Target spatial size (default = 256).
    transform : callable, optional
        Extra transform applied after conversion to float32 RGB.
    """

    SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test", "all"] = "all",
        resolution: int = CANONICAL_RESOLUTION,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.resolution = resolution
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"Pins dataset not found at {self.root}. "
                f"Download from https://www.kaggle.com/datasets/hereisburak/pins-face-recognition"
            )

        # Discover identities and images
        self._identity_dirs: list[Path] = sorted(
            p for p in self.root.iterdir() if p.is_dir()
        )
        self._id_to_idx: dict[str, int] = {
            p.name: i for i, p in enumerate(self._identity_dirs)
        }

        # Build full file list
        all_entries: list[tuple[Path, int, str]] = []  # (path, id_idx, id_name)
        for id_dir in self._identity_dirs:
            id_idx = self._id_to_idx[id_dir.name]
            id_name = id_dir.name.replace("pins_", "")
            for img_path in sorted(id_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    all_entries.append((img_path, id_idx, id_name))

        # Apply split
        if split == "all":
            self._entries = all_entries
        else:
            self._entries = [
                e for e in all_entries if self._assign_split(e[0]) == split
            ]

    @staticmethod
    def _assign_split(path: Path) -> str:
        """Deterministic split assignment based on filename hash.

        Uses SHA-256 (not Python's built-in hash) so that splits are
        reproducible across Python sessions and versions.
        """
        h = int(hashlib.sha256(path.name.encode("utf-8")).hexdigest(), 16) % 100
        if h < 70:
            return "train"
        elif h < 85:
            return "val"
        else:
            return "test"

    # ── Dataset protocol ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> FaceCrop:
        img_path, id_idx, id_name = self._entries[idx]

        # Read image as BGR, convert to RGB
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Cannot read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to canonical resolution
        img_resized = cv2.resize(
            img_rgb,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_AREA
            if img_rgb.shape[0] > self.resolution
            else cv2.INTER_CUBIC,
        )

        img_f32 = uint8_to_float(img_resized)

        if self.transform is not None:
            img_f32 = self.transform(img_f32)

        meta = FaceCropMeta(
            dataset="pins",
            split=self.split,
            image_id=f"pins_{id_idx:03d}_{img_path.stem}",
            source_path=str(img_path),
            identity_label=id_idx,
            identity_name=id_name,
            alignment_method="none",   # raw crops — real alignment in Phase 1.5
            detector="none",
        )

        return FaceCrop(image=img_f32, meta=meta)

    # ── Convenience ────────────────────────────────────────────────────

    @property
    def num_identities(self) -> int:
        """Return the total number of unique identities in the dataset."""
        return len(self._identity_dirs)

    @property
    def identity_names(self) -> list[str]:
        """Return human-readable names for each identity."""
        return [d.name.replace("pins_", "") for d in self._identity_dirs]

    def identity_distribution(self) -> dict[str, int]:
        """Return per-identity sample count."""
        from collections import Counter
        counts = Counter(e[2] for e in self._entries)
        return dict(sorted(counts.items()))

    def __repr__(self) -> str:
        return (
            f"PinsFaceDataset(split={self.split!r}, "
            f"n={len(self)}, identities={self.num_identities}, "
            f"res={self.resolution})"
        )
