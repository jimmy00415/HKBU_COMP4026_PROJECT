"""
Core data contracts for the face anonymization pipeline.

Every module (dataset adapters, anonymizers, evaluators) speaks through these
dataclasses so that resolution, colour-space, and alignment assumptions are
enforced in exactly one place.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ── Constants ───────────────────────────────────────────────────────────────

CANONICAL_RESOLUTION = 256          # width = height = 256
CANONICAL_COLOR_SPACE = "RGB"       # always RGB float32 [0, 1]
CANONICAL_ALIGNMENT = "arcface_5pt" # 5-point affine alignment


class ExpressionLabel(Enum):
    """FER-2013 7-class expression taxonomy."""
    ANGRY    = 0
    DISGUST  = 1
    FEAR     = 2
    HAPPY    = 3
    SAD      = 4
    SURPRISE = 5
    NEUTRAL  = 6


EXPRESSION_NAMES: list[str] = [e.name.capitalize() for e in ExpressionLabel]


# ── Face-crop contract ─────────────────────────────────────────────────────

@dataclass
class FaceCropMeta:
    """Per-image metadata that travels with every aligned face crop."""

    dataset: str                          # e.g. "fer2013", "pins", "celebahq"
    split: str                            # "train" / "val" / "test"
    image_id: str                         # unique within dataset + split
    source_path: str = ""                 # original file path (for debugging)

    # Labels (may be None when a dataset doesn't provide them)
    identity_label: Optional[int] = None  # integer identity index
    identity_name: Optional[str] = None   # human-readable identity string
    expression_label: Optional[int] = None  # ExpressionLabel.value

    # Alignment info
    alignment_method: str = CANONICAL_ALIGNMENT
    affine_matrix: Optional[np.ndarray] = field(default=None, repr=False)

    # Detector info
    detector: str = "retinaface"
    detection_score: float = 0.0
    bbox: Optional[np.ndarray] = None     # [x1, y1, x2, y2] in original image


@dataclass
class FaceCrop:
    """
    A single aligned face crop conforming to the canonical contract:
      • image: np.ndarray  shape (H, W, 3)  dtype float32  range [0, 1]  RGB
      • H = W = CANONICAL_RESOLUTION (256)
    Optional auxiliary tensors are attached but may be None.
    """

    image: np.ndarray                     # (256, 256, 3) float32 RGB [0,1]
    meta: FaceCropMeta

    # ── Optional conditioning signals ───────────────────────────────────
    landmarks_5pt: Optional[np.ndarray] = field(default=None, repr=False)
    # (5, 2) float32 — eyes, nose, mouth corners

    landmarks_68pt: Optional[np.ndarray] = field(default=None, repr=False)
    # (68, 2) float32 — full landmark set

    landmark_heatmaps: Optional[np.ndarray] = field(default=None, repr=False)
    # (K, H, W) float32 — Gaussian heatmaps around each landmark

    parsing_mask: Optional[np.ndarray] = field(default=None, repr=False)
    # (H, W) int64 — 19-class CelebAMask-HQ semantic labels

    def validate(self) -> None:
        """Quick runtime assertion that this crop conforms to contract."""
        assert self.image.ndim == 3, f"Expected 3D, got {self.image.ndim}D"
        h, w, c = self.image.shape
        assert h == w == CANONICAL_RESOLUTION, (
            f"Expected {CANONICAL_RESOLUTION}x{CANONICAL_RESOLUTION}, got {h}x{w}"
        )
        assert c == 3, f"Expected 3 channels (RGB), got {c}"
        assert self.image.dtype == np.float32, (
            f"Expected float32, got {self.image.dtype}"
        )
        assert 0.0 <= self.image.min() and self.image.max() <= 1.0, (
            f"Expected [0,1] range, got [{self.image.min():.3f}, {self.image.max():.3f}]"
        )


@dataclass
class FaceCropBatch:
    """A list of FaceCrop objects, with convenience accessors."""

    crops: list[FaceCrop]

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> FaceCrop:
        return self.crops[idx]

    @property
    def images(self) -> np.ndarray:
        """Stacked (N, H, W, 3) array."""
        return np.stack([c.image for c in self.crops], axis=0)

    @property
    def expression_labels(self) -> list[Optional[int]]:
        return [c.meta.expression_label for c in self.crops]

    @property
    def identity_labels(self) -> list[Optional[int]]:
        return [c.meta.identity_label for c in self.crops]


# ── Anonymized output ──────────────────────────────────────────────────────

@dataclass
class AnonymizedFace:
    """Output of an anonymizer for a single face."""

    image: np.ndarray                     # (256, 256, 3) float32 RGB [0,1]
    original: FaceCrop                    # keep reference to input
    anonymizer_name: str = ""
    anonymizer_params: dict = field(default_factory=dict)
    mask: Optional[np.ndarray] = None     # optional edit-region mask
    aux: dict = field(default_factory=dict)  # any extra outputs


@dataclass
class AnonymizedBatch:
    """Batch of anonymized faces."""

    faces: list[AnonymizedFace]

    def __len__(self) -> int:
        return len(self.faces)

    def __getitem__(self, idx: int) -> AnonymizedFace:
        return self.faces[idx]

    @property
    def images(self) -> np.ndarray:
        return np.stack([f.image for f in self.faces], axis=0)

    @property
    def original_images(self) -> np.ndarray:
        return np.stack([f.original.image for f in self.faces], axis=0)


# ── Helpers ─────────────────────────────────────────────────────────────────

def uint8_to_float(img: np.ndarray) -> np.ndarray:
    """Convert uint8 [0,255] → float32 [0,1]."""
    return img.astype(np.float32) / 255.0


def float_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float32 [0,1] → uint8 [0,255]."""
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert single-channel grayscale to 3-channel RGB by replication."""
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[-1] == 1:
        return np.concatenate([img, img, img], axis=-1)
    return img  # already multi-channel
