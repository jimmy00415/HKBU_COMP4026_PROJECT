"""
Landmark extraction module.

Extracts 5-point and 68-point facial landmarks from aligned face crops
and optionally renders Gaussian heatmaps for use as anonymizer conditioning.

Supported backends:
  • 5-point  — extracted by RetinaFace / MTCNN during detection (pass-through)
  • 68-point — via dlib or face-alignment (pip install face-alignment)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from src.data.contracts import CANONICAL_RESOLUTION

logger = logging.getLogger(__name__)


# ── Heatmap rendering ─────────────────────────────────────────────────

def landmarks_to_heatmaps(
    landmarks: np.ndarray,
    image_size: int = CANONICAL_RESOLUTION,
    sigma: float = 3.0,
) -> np.ndarray:
    """
    Render Gaussian heatmaps from landmark coordinates.

    Parameters
    ----------
    landmarks : (K, 2) float array — (x, y) coordinates.
    image_size : spatial size of the output heatmaps.
    sigma : standard deviation of the Gaussian blobs.

    Returns
    -------
    heatmaps : (K, image_size, image_size) float32 array in [0, 1].
    """
    K = landmarks.shape[0]
    heatmaps = np.zeros((K, image_size, image_size), dtype=np.float32)

    for k in range(K):
        x, y = landmarks[k]
        if np.isnan(x) or np.isnan(y):
            continue

        # Create coordinate grids
        yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        heatmaps[k] = gaussian

    return heatmaps


# ── 68-point landmark extractor ────────────────────────────────────────

class LandmarkExtractor68:
    """
    Extract 68-point facial landmarks using the ``face_alignment`` library
    (by Adrian Bulat).

    Install: ``pip install face-alignment``
    """

    def __init__(self, device: str = "cpu") -> None:
        self._available = False
        try:
            import face_alignment

            self._fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device=device,
                flip_input=False,
            )
            self._available = True
            logger.info("68-point landmark extractor loaded (face_alignment)")
        except ImportError:
            logger.warning(
                "face_alignment not installed — 68-point landmarks unavailable. "
                "Install with: pip install face-alignment"
            )
        except Exception as e:
            logger.warning("Failed to initialise face_alignment: %s", e)

    @property
    def available(self) -> bool:
        return self._available

    def extract(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 68-point landmarks from an RGB image.

        Parameters
        ----------
        image_rgb : (H, W, 3) uint8 RGB image

        Returns
        -------
        landmarks : (68, 2) float32 array or None if no face detected
        """
        if not self._available:
            return None

        preds = self._fa.get_landmarks(image_rgb)
        if preds is None or len(preds) == 0:
            return None

        return preds[0].astype(np.float32)  # (68, 2) — first face only


# ── Combined landmark interface ────────────────────────────────────────

class LandmarkExtractor:
    """
    Unified landmark extraction: 5-point (from detection) + optional 68-point.

    Usage::

        extractor = LandmarkExtractor(device="cuda")
        lm5 = face_box.landmarks_5pt                   # from detector
        lm68 = extractor.get_68pt(aligned_crop_uint8)   # optional
        heatmaps = extractor.to_heatmaps(lm5)           # conditioning signal
    """

    def __init__(self, device: str = "cpu", enable_68pt: bool = True) -> None:
        self._extractor_68 = LandmarkExtractor68(device=device) if enable_68pt else None

    def get_68pt(self, image_rgb_uint8: np.ndarray) -> Optional[np.ndarray]:
        """Extract 68-point landmarks (returns None if unavailable)."""
        if self._extractor_68 is not None and self._extractor_68.available:
            return self._extractor_68.extract(image_rgb_uint8)
        return None

    @staticmethod
    def to_heatmaps(
        landmarks: np.ndarray,
        image_size: int = CANONICAL_RESOLUTION,
        sigma: float = 3.0,
    ) -> np.ndarray:
        """Convert (K, 2) landmarks to (K, H, W) Gaussian heatmaps."""
        return landmarks_to_heatmaps(landmarks, image_size, sigma)

    @staticmethod
    def draw_landmarks(
        image: np.ndarray,
        landmarks: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        radius: int = 2,
    ) -> np.ndarray:
        """Draw landmarks on an image (for visualization)."""
        vis = image.copy()
        for x, y in landmarks.astype(int):
            cv2.circle(vis, (x, y), radius, color, -1)
        return vis
