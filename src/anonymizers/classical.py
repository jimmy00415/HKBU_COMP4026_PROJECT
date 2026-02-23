"""
Classical anonymizer baselines.

Three methods, all operating directly on pixels:

* **Gaussian blur** — applies ``cv2.GaussianBlur`` with a configurable
  kernel size. Larger kernels → stronger privacy but worse utility.
* **Pixelation** — down-samples to a coarse grid then up-samples back.
  Smaller block size → less privacy, larger → more privacy.
* **Blackout** — fills the face region with a solid colour (default black).

All three are deterministic and very fast (CPU-only, no model weights).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from src.anonymizers.base import AnonymizerBase
from src.data.contracts import AnonymizedFace, FaceCrop, float_to_uint8, uint8_to_float

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Gaussian Blur
# ═══════════════════════════════════════════════════════════════════════════

class GaussianBlurAnonymizer(AnonymizerBase):
    """
    Anonymize by applying Gaussian blur to the full face crop.

    Parameters
    ----------
    kernel_size : int
        Side length of the Gaussian kernel (must be odd and ≥ 3).
    mask_region : bool
        If ``True`` and the input ``FaceCrop`` carries a ``parsing_mask``,
        only the *identity-relevant* region is blurred while the background
        is preserved.
    """

    def __init__(
        self,
        kernel_size: int = 31,
        mask_region: bool = False,
    ) -> None:
        # Ensure kernel size is odd and positive
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        self._kernel_size = kernel_size
        self._mask_region = mask_region

    @property
    def name(self) -> str:
        return "blur"

    @property
    def configurable_params(self) -> dict[str, Any]:
        return {"kernel_size": self._kernel_size, "mask_region": self._mask_region}

    def anonymize_single(
        self,
        face: FaceCrop,
        *,
        kernel_size: Optional[int] = None,
        **kwargs: Any,
    ) -> AnonymizedFace:
        import cv2

        ks = kernel_size if kernel_size is not None else self._kernel_size
        # Ensure odd
        if ks % 2 == 0:
            ks += 1

        # Work in uint8 for cv2
        img_u8 = float_to_uint8(face.image)
        blurred = cv2.GaussianBlur(img_u8, (ks, ks), 0)

        if self._mask_region and face.parsing_mask is not None:
            # Identity region: skin(1), l_brow(2), r_brow(3), l_eye(4),
            # r_eye(5), nose(10), u_lip(12), mouth(11), l_lip(13), ear(7,8,9)
            identity_ids = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13}
            mask = np.isin(face.parsing_mask, list(identity_ids)).astype(np.float32)
            # Smooth mask edges to avoid hard seam
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = mask[..., np.newaxis]  # (H, W, 1)
            result = (blurred.astype(np.float32) * mask +
                      img_u8.astype(np.float32) * (1.0 - mask))
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = blurred

        return self._make_result(
            image=uint8_to_float(result),
            original=face,
            kernel_size=ks,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Pixelation
# ═══════════════════════════════════════════════════════════════════════════

class PixelateAnonymizer(AnonymizerBase):
    """
    Anonymize by pixelating the face crop.

    Downscales the image to ``(H // block_size, W // block_size)`` with
    nearest-neighbour sampling, then upscales back to the original
    resolution — producing the characteristic blocky look.

    Parameters
    ----------
    block_size : int
        Side of each "super-pixel" in the output.
    """

    def __init__(self, block_size: int = 12) -> None:
        self._block_size = max(2, block_size)

    @property
    def name(self) -> str:
        return "pixelate"

    @property
    def configurable_params(self) -> dict[str, Any]:
        return {"block_size": self._block_size}

    def anonymize_single(
        self,
        face: FaceCrop,
        *,
        block_size: Optional[int] = None,
        **kwargs: Any,
    ) -> AnonymizedFace:
        import cv2

        bs = block_size if block_size is not None else self._block_size
        bs = max(2, bs)

        h, w = face.image.shape[:2]
        small_h, small_w = max(1, h // bs), max(1, w // bs)

        img_u8 = float_to_uint8(face.image)
        # Down then up with NEAREST to get blocky pixels
        small = cv2.resize(img_u8, (small_w, small_h), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return self._make_result(
            image=uint8_to_float(pixelated),
            original=face,
            block_size=bs,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Blackout
# ═══════════════════════════════════════════════════════════════════════════

class BlackoutAnonymizer(AnonymizerBase):
    """
    Anonymize by replacing the face region with a solid colour.

    By default fills with black ``(0, 0, 0)``.  If a parsing mask is
    available, only the identity-relevant region is filled; otherwise the
    entire crop is replaced.

    Parameters
    ----------
    color : tuple[int, int, int]
        RGB fill colour in [0, 255].
    use_mask : bool
        Use parsing mask to restrict fill to the face region.
    """

    def __init__(
        self,
        color: tuple[int, int, int] = (0, 0, 0),
        use_mask: bool = False,
    ) -> None:
        self._color = color
        self._use_mask = use_mask

    @property
    def name(self) -> str:
        return "blackout"

    @property
    def configurable_params(self) -> dict[str, Any]:
        return {"color": self._color, "use_mask": self._use_mask}

    def anonymize_single(
        self,
        face: FaceCrop,
        **kwargs: Any,
    ) -> AnonymizedFace:
        img = face.image.copy()
        h, w = img.shape[:2]

        fill = np.array(self._color, dtype=np.float32) / 255.0

        if self._use_mask and face.parsing_mask is not None:
            identity_ids = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13}
            mask = np.isin(face.parsing_mask, list(identity_ids))
            img[mask] = fill
        else:
            # fill entire crop
            img[:] = fill

        return self._make_result(
            image=img,
            original=face,
        )
