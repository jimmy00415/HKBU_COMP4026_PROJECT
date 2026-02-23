"""
ArcFace identity embedder via InsightFace.

Provides a clean wrapper for extracting 512-d identity embeddings and
computing pairwise cosine similarity.  Used both as the **default
privacy attacker** and as the feature space for k-Same anonymization.

Key design decisions
--------------------
* The embedder is decoupled from the detector: it expects *aligned*
  crops (256×256 RGB float32 or uint8), not raw scene images.
* When InsightFace is not available the class degrades gracefully
  (``available == False``); callers must check before using.
* Cosine similarity is the standard metric for ArcFace; we expose it
  directly rather than converting to an ad-hoc distance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class IdentityEmbedder:
    """
    Extract ArcFace identity embeddings via InsightFace.

    Parameters
    ----------
    model_name : str
        InsightFace model pack (default ``"buffalo_l"`` ships ArcFace-R100).
    device : str
        ``"cuda"`` or ``"cpu"``.
    det_size : tuple[int, int]
        Detection input resolution (only used if you call the full pipeline;
        our normal path skips detection and passes pre-aligned crops).
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cuda",
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self._available = False
        self._rec_model = None
        self._app = None

        try:
            from insightface.app import FaceAnalysis

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device == "cuda"
                else ["CPUExecutionProvider"]
            )
            self._app = FaceAnalysis(name=model_name, providers=providers)
            self._app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=det_size)

            # Grab the recognition model directly for embed-only usage
            if hasattr(self._app, "models") and "recognition" in self._app.models:
                self._rec_model = self._app.models["recognition"]
            elif hasattr(self._app, "rec_model"):
                self._rec_model = self._app.rec_model

            self._available = True
            logger.info("IdentityEmbedder loaded (model=%s, device=%s)", model_name, device)
        except Exception as e:
            logger.warning("IdentityEmbedder unavailable: %s", e)

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    @property
    def embedding_dim(self) -> int:
        return 512

    def embed(self, images: np.ndarray) -> np.ndarray:
        """
        Compute identity embeddings for a batch of *aligned* face crops.

        Parameters
        ----------
        images : (N, H, W, 3) or (H, W, 3)
            RGB images.  Accepted dtypes: float32 [0,1] or uint8 [0,255].

        Returns
        -------
        embeddings : (N, 512) float32, L2-normalised.
        """
        if not self._available:
            raise RuntimeError("IdentityEmbedder is not available (InsightFace not installed?)")

        if images.ndim == 3:
            images = images[np.newaxis]  # (H,W,3) → (1,H,W,3)

        # Ensure uint8 for InsightFace
        if images.dtype == np.float32:
            imgs_uint8 = (np.clip(images, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif images.dtype == np.uint8:
            imgs_uint8 = images
        else:
            imgs_uint8 = images.astype(np.uint8)

        embeddings = []
        for img_rgb in imgs_uint8:
            emb = self._embed_single(img_rgb)
            embeddings.append(emb)

        out = np.stack(embeddings, axis=0)  # (N, 512)
        return out

    def _embed_single(self, img_rgb: np.ndarray) -> np.ndarray:
        """Embed a single (H, W, 3) uint8 RGB image."""
        import cv2

        # InsightFace expects BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Try direct recognition model (no detection needed for aligned crops)
        if self._rec_model is not None:
            # rec_model.get() expects a BGR image and a face object
            # We craft a minimal face object with a standard bbox
            emb = self._rec_model.get(img_bgr, _make_dummy_face(img_bgr.shape))
            return _l2_normalise(emb)

        # Fallback: use full FaceAnalysis pipeline (detect + recognise)
        faces = self._app.get(img_bgr)
        if len(faces) == 0:
            logger.warning("No face detected — returning zero embedding")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Pick highest-confidence face
        best = max(faces, key=lambda f: f.det_score)
        return _l2_normalise(best.embedding.astype(np.float32))

    # ── Similarity ─────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
        """
        Pairwise cosine similarity.

        Parameters
        ----------
        e1 : (N, D) or (D,)
        e2 : (M, D) or (D,)

        Returns
        -------
        sim : (N, M) float32 — values in [-1, 1].
              If both inputs are 1-D, returns a scalar float.
        """
        if e1.ndim == 1:
            e1 = e1[np.newaxis]
        if e2.ndim == 1:
            e2 = e2[np.newaxis]

        # L2-normalise (should already be, but be safe)
        e1_n = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-10)
        e2_n = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-10)

        sim = e1_n @ e2_n.T  # (N, M)

        if sim.shape == (1, 1):
            return float(sim[0, 0])
        return sim.astype(np.float32)

    def similarity(self, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
        """Alias for ``cosine_similarity``."""
        return self.cosine_similarity(e1, e2)


# ── Helpers ────────────────────────────────────────────────────────────

def _l2_normalise(x: np.ndarray) -> np.ndarray:
    """L2-normalise a vector (or batch of vectors along last axis)."""
    x = x.astype(np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + 1e-10)


class _DummyFace:
    """Minimal object that satisfies InsightFace rec_model.get() API."""

    def __init__(self, bbox: np.ndarray, kps: np.ndarray) -> None:
        self.bbox = bbox
        self.kps = kps


def _make_dummy_face(img_shape: tuple) -> _DummyFace:
    """Create a dummy face object covering the full aligned crop."""
    h, w = img_shape[:2]
    bbox = np.array([0, 0, w, h], dtype=np.float32)
    # Standard 5-point landmarks for an aligned crop (approximate centres)
    kps = np.array(
        [
            [0.3 * w, 0.35 * h],   # left eye
            [0.7 * w, 0.35 * h],   # right eye
            [0.5 * w, 0.55 * h],   # nose
            [0.35 * w, 0.72 * h],  # left mouth
            [0.65 * w, 0.72 * h],  # right mouth
        ],
        dtype=np.float32,
    )
    return _DummyFace(bbox=bbox, kps=kps)
