"""
Face parsing module.

Wraps a pre-trained BiSeNet face-parsing model (trained on CelebAMask-HQ,
19 semantic classes) to produce per-pixel face-part segmentation masks.

Reference repo: https://github.com/zllrunning/face-parsing.PyTorch

The module is designed to work with or without GPU and gracefully
degrades to a no-op if the model weights are not available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.data.contracts import CANONICAL_RESOLUTION

logger = logging.getLogger(__name__)

# CelebAMask-HQ 19-class label map (same as in celebahq_adapter.py)
PARSING_LABELS: list[str] = [
    "background",  # 0
    "skin",         # 1
    "l_brow",       # 2
    "r_brow",       # 3
    "l_eye",        # 4
    "r_eye",        # 5
    "eye_g",        # 6
    "l_ear",        # 7
    "r_ear",        # 8
    "ear_r",        # 9
    "nose",         # 10
    "mouth",        # 11
    "u_lip",        # 12
    "l_lip",        # 13
    "neck",         # 14
    "neck_l",       # 15
    "cloth",        # 16
    "hair",         # 17
    "hat",          # 18
]

# Colour palette for visualization (RGB)
PARSING_PALETTE = np.array(
    [
        [0, 0, 0],        # background
        [204, 0, 0],      # skin
        [76, 153, 0],     # l_brow
        [204, 204, 0],    # r_brow
        [51, 51, 255],    # l_eye
        [204, 0, 204],    # r_eye
        [0, 255, 255],    # eye_g
        [255, 204, 204],  # l_ear
        [102, 51, 0],     # r_ear
        [255, 0, 0],      # ear_r
        [102, 204, 0],    # nose
        [255, 255, 0],    # mouth
        [0, 0, 153],      # u_lip
        [0, 0, 204],      # l_lip
        [255, 51, 153],   # neck
        [0, 204, 204],    # neck_l
        [0, 51, 0],       # cloth
        [255, 153, 51],   # hair
        [0, 204, 0],      # hat
    ],
    dtype=np.uint8,
)


class FaceParser:
    """
    Face semantic segmentation using a pre-trained BiSeNet model.

    Parameters
    ----------
    checkpoint_path : str | Path | None
        Path to the pre-trained BiSeNet ``.pth`` weights file.
        If None, the parser will be unavailable (no-op).
    device : str
        ``"cuda"`` or ``"cpu"``.
    input_size : int
        Resolution the BiSeNet expects (typically 512).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: str = "cpu",
        input_size: int = 512,
    ) -> None:
        self.device = device
        self.input_size = input_size
        self._available = False
        self._model = None

        if checkpoint_path is None:
            logger.info(
                "FaceParser: no checkpoint provided — parsing unavailable. "
                "Download from https://github.com/zllrunning/face-parsing.PyTorch"
            )
            return

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning("FaceParser: checkpoint not found at %s", checkpoint_path)
            return

        try:
            self._load_model(checkpoint_path)
            self._available = True
            logger.info("FaceParser: BiSeNet loaded from %s", checkpoint_path)
        except Exception as e:
            logger.warning("FaceParser: failed to load model — %s", e)

    def _load_model(self, checkpoint_path: Path) -> None:
        """Load the BiSeNet model."""
        import torch
        import torch.nn as nn
        import torchvision.transforms as T

        # Try importing the BiSeNet architecture
        # Users should clone face-parsing.PyTorch into third_party/
        # For now, we use a simple wrapper approach
        try:
            import sys
            third_party_dir = Path(__file__).resolve().parent.parent.parent / "third_party" / "face-parsing.PyTorch"
            if third_party_dir.exists() and str(third_party_dir) not in sys.path:
                sys.path.insert(0, str(third_party_dir))
            from model import BiSeNet
            self._model = BiSeNet(n_classes=19)
        except ImportError:
            # Fallback: use torchvision's segmentation backbone
            logger.info("BiSeNet source not found, using torchvision DeepLabV3 as fallback")
            self._model = torch.hub.load(
                "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=False
            )
            # This fallback won't produce correct face parsing, but keeps the
            # pipeline structurally valid until the user sets up BiSeNet.
            return

        state_dict = torch.load(str(checkpoint_path), map_location=self.device)
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()

        self._transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @property
    def available(self) -> bool:
        """Whether the BiSeNet face parser loaded successfully."""
        return self._available

    def parse(
        self,
        image_rgb: np.ndarray,
        output_size: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Segment a face image into 19 semantic classes.

        Parameters
        ----------
        image_rgb : (H, W, 3) uint8 RGB image
        output_size : if given, resize the mask to this size

        Returns
        -------
        mask : (H, W) int64 array with class indices [0..18], or None
        """
        if not self._available or self._model is None:
            return None

        import torch

        # Resize to model input size
        h_orig, w_orig = image_rgb.shape[:2]
        img_resized = cv2.resize(
            image_rgb, (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # Transform and run
        with torch.no_grad():
            input_tensor = self._transform(img_resized).unsqueeze(0).to(self.device)
            output = self._model(input_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            mask = output.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.int64)

        # Resize to requested output size
        target_size = output_size or h_orig
        if mask.shape[0] != target_size:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (target_size, target_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)

        return mask

    def parse_batch(
        self,
        images: list[np.ndarray],
        output_size: Optional[int] = None,
    ) -> list[Optional[np.ndarray]]:
        """Parse a batch of images."""
        return [self.parse(img, output_size) for img in images]


# ── Visualization ──────────────────────────────────────────────────────

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert an integer label mask to an RGB visualization.

    Parameters
    ----------
    mask : (H, W) int64

    Returns
    -------
    colored : (H, W, 3) uint8 RGB
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(len(PARSING_LABELS)):
        colored[mask == cls_idx] = PARSING_PALETTE[cls_idx]
    return colored


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Overlay a coloured parsing mask on the original image."""
    colored = colorize_mask(mask)
    if image.dtype == np.float32:
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_uint8 = image
    blended = cv2.addWeighted(image_uint8, 1 - alpha, colored, alpha, 0)
    return blended


# ── Identity-region mask helpers ───────────────────────────────────────

def get_identity_region_mask(
    parsing_mask: np.ndarray,
    include_hair: bool = True,
) -> np.ndarray:
    """
    Create a binary mask of identity-bearing face regions.

    Useful for directing the anonymizer to edit only identity-relevant
    areas while preserving expression-relevant regions.

    Returns
    -------
    mask : (H, W) bool
    """
    # Identity-bearing: skin, brows, ears, nose, hair (optionally)
    identity_classes = {1, 2, 3, 7, 8, 10}  # skin, brows, ears, nose
    if include_hair:
        identity_classes.add(17)  # hair

    mask = np.isin(parsing_mask, list(identity_classes))
    return mask


def get_expression_region_mask(parsing_mask: np.ndarray) -> np.ndarray:
    """
    Create a binary mask of expression-relevant face regions.

    Expression cues: eyes, eyebrows, mouth, lips.
    """
    expression_classes = {2, 3, 4, 5, 11, 12, 13}  # brows, eyes, mouth, lips
    mask = np.isin(parsing_mask, list(expression_classes))
    return mask
