"""
Face detection and alignment module.

Provides a unified interface over:
  • RetinaFace (via InsightFace)  — primary, high-accuracy
  • MTCNN (via facenet-pytorch)   — fallback

Both detectors output bounding boxes + 5-point landmarks.  The
``align_face`` function warps the original image to a canonical 256×256
crop using a similarity transform derived from the 5 landmarks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.data.contracts import CANONICAL_RESOLUTION, uint8_to_float

logger = logging.getLogger(__name__)

# ── ArcFace-standard reference landmarks for 256×256 ───────────────────
# These are the "ideal" 5-point positions (left-eye, right-eye, nose,
# left-mouth, right-mouth) on a 112×112 ArcFace crop, scaled to 256×256.
_ARCFACE_REF_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Scale from 112 → CANONICAL_RESOLUTION
_SCALE = CANONICAL_RESOLUTION / 112.0
REFERENCE_LANDMARKS_256 = _ARCFACE_REF_112 * _SCALE


# ── Data structures ────────────────────────────────────────────────────

@dataclass
class FaceBox:
    """Detected face bounding box + optional landmarks."""
    bbox: np.ndarray              # [x1, y1, x2, y2]
    score: float
    landmarks_5pt: Optional[np.ndarray] = None   # (5, 2)
    detector: str = "unknown"


# ── Alignment ──────────────────────────────────────────────────────────

def align_face(
    image: np.ndarray,
    landmarks_5pt: np.ndarray,
    output_size: int = CANONICAL_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp ``image`` so that the 5 facial landmarks land on the canonical
    reference positions.

    Parameters
    ----------
    image : (H, W, 3) uint8 or float32
    landmarks_5pt : (5, 2) — detected landmarks in pixel coords
    output_size : target crop size

    Returns
    -------
    aligned : (output_size, output_size, 3) same dtype as input
    M : (2, 3) affine matrix used
    """
    ref = REFERENCE_LANDMARKS_256 * (output_size / CANONICAL_RESOLUTION)
    M = cv2.estimateAffinePartial2D(
        landmarks_5pt.astype(np.float32),
        ref.astype(np.float32),
    )[0]
    if M is None:
        # Fallback: identity crop from bbox center
        M = np.eye(2, 3, dtype=np.float32)
    aligned = cv2.warpAffine(
        image, M, (output_size, output_size),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, M


# ── Detector wrappers ──────────────────────────────────────────────────

class RetinaFaceDetector:
    """
    Face detector using InsightFace's RetinaFace implementation.

    Loads the ``buffalo_l`` model pack which includes RetinaFace + ArcFace.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        try:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name=model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=det_size)
            self._available = True
            logger.info("RetinaFace detector loaded (model=%s)", model_name)
        except Exception as e:
            logger.warning("RetinaFace unavailable: %s — will fall back to MTCNN", e)
            self._available = False

    @property
    def available(self) -> bool:
        """Whether the RetinaFace backend loaded successfully."""
        return self._available

    def detect(self, image_rgb: np.ndarray) -> list[FaceBox]:
        """
        Detect faces in an RGB uint8 image.

        Returns list of FaceBox sorted by score descending.
        """
        if not self._available:
            return []

        # InsightFace expects BGR
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = self._app.get(img_bgr)

        results: list[FaceBox] = []
        for face in faces:
            score = float(face.det_score)
            if score < self.confidence_threshold:
                continue
            results.append(
                FaceBox(
                    bbox=face.bbox.astype(np.float32),
                    score=score,
                    landmarks_5pt=face.kps.astype(np.float32) if face.kps is not None else None,
                    detector="retinaface",
                )
            )
        results.sort(key=lambda f: f.score, reverse=True)
        return results


class MTCNNDetector:
    """
    Fallback face detector using facenet-pytorch's MTCNN.
    """

    def __init__(
        self,
        min_face_size: int = 20,
        confidence_threshold: float = 0.9,
        device: str = "cpu",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        try:
            from facenet_pytorch import MTCNN

            self._mtcnn = MTCNN(
                keep_all=True,
                min_face_size=min_face_size,
                device=device,
            )
            self._available = True
            logger.info("MTCNN detector loaded")
        except Exception as e:
            logger.warning("MTCNN unavailable: %s", e)
            self._available = False

    @property
    def available(self) -> bool:
        """Whether the MTCNN backend loaded successfully."""
        return self._available

    def detect(self, image_rgb: np.ndarray) -> list[FaceBox]:
        """Detect faces using MTCNN.

        Parameters
        ----------
        image_rgb : np.ndarray
            (H, W, 3) uint8 RGB image.

        Returns
        -------
        list[FaceBox]
            Detected faces sorted by confidence descending.
        """
        if not self._available:
            return []

        from PIL import Image

        pil_img = Image.fromarray(image_rgb)
        boxes, probs, landmarks = self._mtcnn.detect(pil_img, landmarks=True)

        if boxes is None:
            return []

        results: list[FaceBox] = []
        for i in range(len(boxes)):
            score = float(probs[i])
            if score < self.confidence_threshold:
                continue
            lm = landmarks[i].astype(np.float32) if landmarks is not None else None
            results.append(
                FaceBox(
                    bbox=boxes[i].astype(np.float32),
                    score=score,
                    landmarks_5pt=lm,
                    detector="mtcnn",
                )
            )
        results.sort(key=lambda f: f.score, reverse=True)
        return results


# ── Unified detector facade ────────────────────────────────────────────

class FaceDetector:
    """
    Unified face detector: tries RetinaFace first, falls back to MTCNN.

    Usage::

        det = FaceDetector()
        boxes = det.detect(image_rgb_uint8)
        if boxes:
            aligned, M = align_face(image_rgb_uint8, boxes[0].landmarks_5pt)
    """

    def __init__(
        self,
        preferred: str = "retinaface",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self._retina = RetinaFaceDetector(confidence_threshold=confidence_threshold)
        self._mtcnn = MTCNNDetector(confidence_threshold=confidence_threshold, device=device)
        self.preferred = preferred

    def detect(self, image_rgb: np.ndarray) -> list[FaceBox]:
        """Return detected faces (best-score first)."""
        if self.preferred == "retinaface" and self._retina.available:
            results = self._retina.detect(image_rgb)
            if results:
                return results

        if self._mtcnn.available:
            results = self._mtcnn.detect(image_rgb)
            if results:
                return results

        # Last fallback: try the other detector
        if self.preferred != "retinaface" and self._retina.available:
            return self._retina.detect(image_rgb)

        logger.warning("No face detected and no detector available")
        return []

    def detect_and_align(
        self,
        image_rgb: np.ndarray,
        output_size: int = CANONICAL_RESOLUTION,
    ) -> list[tuple[np.ndarray, FaceBox, np.ndarray]]:
        """
        Detect faces and return aligned crops.

        Returns
        -------
        List of (aligned_crop, face_box, affine_matrix) tuples.
        """
        boxes = self.detect(image_rgb)
        results = []
        for box in boxes:
            if box.landmarks_5pt is not None:
                aligned, M = align_face(image_rgb, box.landmarks_5pt, output_size)
            else:
                # Fallback: simple crop + resize from bbox
                x1, y1, x2, y2 = box.bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(image_rgb.shape[1], x2)
                y2 = min(image_rgb.shape[0], y2)
                crop = image_rgb[y1:y2, x1:x2]
                aligned = cv2.resize(crop, (output_size, output_size))
                M = np.eye(2, 3, dtype=np.float32)
            results.append((aligned, box, M))
        return results
