"""
CelebAMask-HQ dataset adapter.

Loads high-resolution celebrity face images and their 19-class semantic
parsing masks.  Used for face-parsing pre-training validation and as a
reference distribution for FID/realism diagnostics.

Expected file layout:
    data/CelebAMask-HQ/
        CelebA-HQ-img/
            0.jpg  ...  29999.jpg
        CelebAMask-HQ-mask-anno/
            0/
                00000_skin.png
                00000_l_eye.png
                ...
            1/
            ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from src.data.contracts import (
    CANONICAL_RESOLUTION,
    FaceCrop,
    FaceCropMeta,
    uint8_to_float,
)

# 19 semantic part classes used in CelebAMask-HQ
PARSING_CLASSES: list[str] = [
    "background",   # 0
    "skin",          # 1
    "l_brow",        # 2
    "r_brow",        # 3
    "l_eye",         # 4
    "r_eye",         # 5
    "eye_g",         # 6  (eyeglasses)
    "l_ear",         # 7
    "r_ear",         # 8
    "ear_r",         # 9  (earring)
    "nose",          # 10
    "mouth",         # 11
    "u_lip",         # 12
    "l_lip",         # 13
    "neck",          # 14
    "neck_l",        # 15 (necklace)
    "cloth",         # 16
    "hair",          # 17
    "hat",           # 18
]

NUM_PARSING_CLASSES = len(PARSING_CLASSES)


class CelebAHQDataset(Dataset):
    """
    PyTorch-compatible dataset for CelebAMask-HQ.

    Yields FaceCrop objects with ``parsing_mask`` populated when masks
    are available.

    Parameters
    ----------
    root : str | Path
        Root directory (``data/CelebAMask-HQ``).
    split : {"train", "val", "test", "all"}
        Deterministic split by image index: 0-23999 train, 24000-27999 val,
        28000-29999 test (following common convention).
    resolution : int
        Target spatial size (default 256).
    load_masks : bool
        Whether to load and merge the per-part semantic masks.
    transform : callable, optional
        Extra transform on the float32 RGB image.
    """

    TRAIN_END = 24000
    VAL_END = 28000
    TOTAL = 30000

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test", "all"] = "all",
        resolution: int = CANONICAL_RESOLUTION,
        load_masks: bool = True,
        transform=None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.resolution = resolution
        self.load_masks = load_masks
        self.transform = transform

        self.img_dir = self.root / "CelebA-HQ-img"
        self.mask_dir = self.root / "CelebAMask-HQ-mask-anno"

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"CelebA-HQ images not found at {self.img_dir}. "
                f"Download from https://github.com/switchablenorms/CelebAMask-HQ"
            )

        # Build index list for the requested split
        all_ids = list(range(self.TOTAL))
        if split == "train":
            self._ids = all_ids[: self.TRAIN_END]
        elif split == "val":
            self._ids = all_ids[self.TRAIN_END : self.VAL_END]
        elif split == "test":
            self._ids = all_ids[self.VAL_END :]
        else:
            self._ids = all_ids

    # ── Dataset protocol ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> FaceCrop:
        img_id = self._ids[idx]

        # ── Load image ──────────────────────────────────────────────
        img_path = self.img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            img_path = self.img_dir / f"{img_id}.png"
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Cannot read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(
            img_rgb,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_AREA,
        )
        img_f32 = uint8_to_float(img_resized)

        if self.transform is not None:
            img_f32 = self.transform(img_f32)

        # ── Load parsing mask ───────────────────────────────────────
        parsing_mask: Optional[np.ndarray] = None
        if self.load_masks and self.mask_dir.exists():
            parsing_mask = self._load_merged_mask(img_id)
            if parsing_mask is not None:
                parsing_mask = cv2.resize(
                    parsing_mask,
                    (self.resolution, self.resolution),
                    interpolation=cv2.INTER_NEAREST,
                )

        meta = FaceCropMeta(
            dataset="celebahq",
            split=self.split,
            image_id=f"celebahq_{img_id:05d}",
            source_path=str(img_path),
            alignment_method="none",
            detector="none",
        )

        crop = FaceCrop(image=img_f32, meta=meta, parsing_mask=parsing_mask)
        return crop

    # ── Mask loading ───────────────────────────────────────────────────

    def _load_merged_mask(self, img_id: int) -> Optional[np.ndarray]:
        """
        Merge individual per-part binary masks into one (H, W) int label map.
        Masks live in subdirectories 0/, 1/, ... 14/ (each holding 2000 images).
        """
        subdir = self.mask_dir / str(img_id // 2000)
        if not subdir.exists():
            return None

        prefix = f"{img_id:05d}"
        h = w = 512  # CelebAMask-HQ masks are 512×512
        merged = np.zeros((h, w), dtype=np.int64)

        for cls_idx, cls_name in enumerate(PARSING_CLASSES):
            if cls_name == "background":
                continue
            mask_path = subdir / f"{prefix}_{cls_name}.png"
            if mask_path.exists():
                part = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if part is not None:
                    merged[part > 128] = cls_idx

        return merged

    # ── Convenience ────────────────────────────────────────────────────

    @staticmethod
    def class_names() -> list[str]:
        return PARSING_CLASSES.copy()

    def __repr__(self) -> str:
        return (
            f"CelebAHQDataset(split={self.split!r}, "
            f"n={len(self)}, res={self.resolution}, masks={self.load_masks})"
        )
