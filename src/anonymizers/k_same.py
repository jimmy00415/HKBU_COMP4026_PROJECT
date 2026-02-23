"""
k-Same averaging anonymizer.

For each face in a dataset, find its *k* nearest neighbours in ArcFace
embedding space and replace the face image with the **pixel-wise average**
of those *k* faces.  Larger *k* → stronger anonymisation (the average face
is further from any single identity) but more loss of expression detail.

This is a dataset-level operation: the gallery of candidate faces must be
provided up-front so that nearest-neighbour lookups can be performed.

References
----------
Preserving Privacy by De-identifying Facial Images,
E. Newton, L. Sweeney, B. Malin — Harvard University, 2005.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from src.anonymizers.base import AnonymizerBase
from src.data.contracts import (
    AnonymizedBatch,
    AnonymizedFace,
    FaceCrop,
    FaceCropBatch,
)

logger = logging.getLogger(__name__)


class KSameAnonymizer(AnonymizerBase):
    """
    k-Same face averaging.

    Parameters
    ----------
    k : int
        Number of nearest neighbours to average (including the query face
        itself).  Must be ≥ 2.
    embeddings : np.ndarray | None
        Pre-computed (N, D) ArcFace gallery embeddings.  Can be set later
        via :meth:`set_gallery`.
    gallery_images : np.ndarray | None
        Corresponding (N, H, W, 3) float32 [0,1] face images.
    """

    def __init__(
        self,
        k: int = 5,
        embeddings: Optional[np.ndarray] = None,
        gallery_images: Optional[np.ndarray] = None,
    ) -> None:
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        self._k = k
        self._embeddings: Optional[np.ndarray] = embeddings
        self._gallery_images: Optional[np.ndarray] = gallery_images

    @property
    def name(self) -> str:
        return "k_same"

    @property
    def configurable_params(self) -> dict[str, Any]:
        return {"k": self._k}

    # ── Gallery management ─────────────────────────────────────────────

    def set_gallery(
        self,
        embeddings: np.ndarray,
        gallery_images: np.ndarray,
    ) -> None:
        """
        Set or replace the gallery.

        Parameters
        ----------
        embeddings   : (N, D) float32, L2-normalised.
        gallery_images : (N, H, W, 3) float32 [0, 1].
        """
        assert embeddings.shape[0] == gallery_images.shape[0], (
            f"Embedding count ({embeddings.shape[0]}) != image count "
            f"({gallery_images.shape[0]})"
        )
        # Ensure L2-normalised
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self._embeddings = embeddings / norms
        self._gallery_images = gallery_images
        logger.info(
            "k-Same gallery set: %d faces, embedding dim %d",
            embeddings.shape[0], embeddings.shape[1],
        )

    def _require_gallery(self) -> None:
        if self._embeddings is None or self._gallery_images is None:
            raise RuntimeError(
                "k-Same requires a gallery. Call set_gallery(embeddings, images) first."
            )

    # ── Core ───────────────────────────────────────────────────────────

    def _find_k_nearest(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        Find indices of the *k* nearest gallery faces by cosine similarity.

        Parameters
        ----------
        query_embedding : (D,) L2-normalised.

        Returns
        -------
        indices : (k,) int
        """
        assert self._embeddings is not None
        # Cosine similarity = dot product when both are L2-normalised
        sims = self._embeddings @ query_embedding  # (N,)
        # Take top-k (argsort is ascending, so negate)
        # np.argpartition is O(N) for top-k, faster than full sort
        k_clamped = min(k, len(sims))
        top_k_idx = np.argpartition(-sims, k_clamped)[:k_clamped]
        # Sort those k by descending similarity for deterministic ordering
        top_k_idx = top_k_idx[np.argsort(-sims[top_k_idx])]
        return top_k_idx

    def anonymize_single(
        self,
        face: FaceCrop,
        *,
        k: Optional[int] = None,
        query_embedding: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> AnonymizedFace:
        """
        Anonymize one face by averaging its k nearest gallery neighbours.

        Parameters
        ----------
        face : FaceCrop
        k : int | None
            Override the default *k*.
        query_embedding : (D,) | None
            Pre-computed ArcFace embedding of this face. If ``None``, the
            caller must have set the gallery and this face must be part of it
            — but to avoid this ambiguity, always pass the embedding.
        """
        self._require_gallery()
        assert self._gallery_images is not None

        k_use = k if k is not None else self._k

        if query_embedding is None:
            raise ValueError(
                "query_embedding must be provided for k-Same. "
                "Extract an ArcFace embedding for this face first."
            )

        # L2-normalise
        qnorm = np.linalg.norm(query_embedding)
        if qnorm > 1e-8:
            query_embedding = query_embedding / qnorm

        indices = self._find_k_nearest(query_embedding, k_use)
        # Pixel average of nearest neighbours
        neighbours = self._gallery_images[indices]  # (k, H, W, 3)
        averaged = neighbours.mean(axis=0)           # (H, W, 3)

        return self._make_result(
            image=averaged,
            original=face,
            k=k_use,
            aux={"neighbour_indices": indices.tolist()},
        )

    def anonymize_batch(
        self,
        batch: FaceCropBatch,
        *,
        k: Optional[int] = None,
        query_embeddings: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> AnonymizedBatch:
        """
        Batch version — requires pre-computed embeddings for all faces.

        Parameters
        ----------
        batch : FaceCropBatch
        query_embeddings : (N, D) float32, one per face in the batch.
        """
        if query_embeddings is None:
            raise ValueError("query_embeddings required for batch k-Same.")
        assert query_embeddings.shape[0] == len(batch), (
            f"Expected {len(batch)} embeddings, got {query_embeddings.shape[0]}"
        )

        results = []
        for i, crop in enumerate(batch.crops):
            anon = self.anonymize_single(
                crop, k=k, query_embedding=query_embeddings[i],
            )
            results.append(anon)
        return AnonymizedBatch(faces=results)
