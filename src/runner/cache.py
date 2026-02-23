"""
Disk caching for expensive preprocessing results.

Caches embeddings, landmarks, parsing masks, and expression probabilities
to a per-dataset directory so that repeated experiment runs skip the
compute-heavy steps.

Cache layout
~~~~~~~~~~~~
::

    cache/
      {dataset}_{split}/
        identity_embeddings.npy
        expression_probs.npy
        landmarks.npy
        parsing_masks.npy
        meta.json          ← records how the cache was built

Usage
-----
>>> from src.runner.cache import PreprocessCache
>>> cache = PreprocessCache("cache/fer2013_test")
>>> emb = cache.load("identity_embeddings")  # None if miss
>>> cache.save("identity_embeddings", emb_array)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PreprocessCache:
    """
    Simple numpy-array disk cache with invalidation.

    Parameters
    ----------
    cache_dir : str | Path
        Root directory for this cache partition.
    enabled : bool
        If False, all loads/saves become no-ops.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        enabled: bool = True,
    ) -> None:
        self._dir = Path(cache_dir)
        self._enabled = enabled
        if enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def cache_dir(self) -> Path:
        return self._dir

    # ── Array I/O ─────────────────────────────────────────────────────

    def load(self, key: str) -> Optional[np.ndarray]:
        """
        Load a cached numpy array.

        Returns None on cache miss.
        """
        if not self._enabled:
            return None
        path = self._dir / f"{key}.npy"
        if path.exists():
            try:
                arr = np.load(str(path), allow_pickle=False)
                logger.info("Cache HIT: %s (%s)", key, path)
                return arr
            except Exception as exc:
                logger.warning("Cache load failed for %s: %s", key, exc)
                return None
        logger.debug("Cache MISS: %s", key)
        return None

    def save(self, key: str, array: np.ndarray) -> None:
        """Save a numpy array to cache."""
        if not self._enabled:
            return
        path = self._dir / f"{key}.npy"
        try:
            np.save(str(path), array, allow_pickle=False)
            logger.info("Cache SAVE: %s → %s (%.2f MB)",
                        key, path, path.stat().st_size / 1e6)
        except Exception as exc:
            logger.warning("Cache save failed for %s: %s", key, exc)

    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        if not self._enabled:
            return False
        return (self._dir / f"{key}.npy").exists()

    # ── Metadata ──────────────────────────────────────────────────────

    def save_meta(self, meta: dict[str, Any]) -> None:
        """Save cache metadata (dataset info, timestamps, etc.)."""
        if not self._enabled:
            return
        meta_path = self._dir / "meta.json"
        meta["_saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def load_meta(self) -> Optional[dict]:
        """Load cache metadata."""
        if not self._enabled:
            return None
        meta_path = self._dir / "meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return None

    # ── Invalidation ──────────────────────────────────────────────────

    def invalidate(self, key: Optional[str] = None) -> None:
        """
        Invalidate a specific key or the entire cache.

        Parameters
        ----------
        key : specific key to delete, or None for full wipe.
        """
        if not self._enabled:
            return
        if key is not None:
            path = self._dir / f"{key}.npy"
            if path.exists():
                path.unlink()
                logger.info("Cache invalidated: %s", key)
        else:
            import shutil
            if self._dir.exists():
                shutil.rmtree(self._dir)
                self._dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cache wiped: %s", self._dir)

    def list_keys(self) -> list[str]:
        """List all cached keys."""
        if not self._enabled:
            return []
        return [p.stem for p in self._dir.glob("*.npy")]

    # ── Content-addressable helpers ───────────────────────────────────

    @staticmethod
    def hash_array(arr: np.ndarray, *, max_bytes: int = 100_000) -> str:
        """
        SHA-256 hash of an array (for cache validation).

        Hashes at most ``max_bytes`` for speed.
        """
        data = arr.tobytes()[:max_bytes]
        return hashlib.sha256(data).hexdigest()[:16]


# ── Convenience: cached embedding / probs computation ─────────────────

def get_or_compute_embeddings(
    images: np.ndarray,
    cache: PreprocessCache,
    device: str = "cpu",
    *,
    key: str = "identity_embeddings",
    force: bool = False,
) -> Optional[np.ndarray]:
    """
    Load identity embeddings from cache, or compute and cache them.
    """
    if not force:
        cached = cache.load(key)
        if cached is not None:
            return cached

    from scripts.run_baseline_sweep import _get_identity_embeddings
    embeddings = _get_identity_embeddings(images, device)

    if embeddings is not None:
        cache.save(key, embeddings)
    return embeddings


def get_or_compute_expression_probs(
    images: np.ndarray,
    cache: PreprocessCache,
    device: str = "cpu",
    *,
    key: str = "expression_probs",
    force: bool = False,
) -> Optional[np.ndarray]:
    """
    Load expression probs from cache, or compute and cache them.
    """
    if not force:
        cached = cache.load(key)
        if cached is not None:
            return cached

    from scripts.run_baseline_sweep import _get_expression_probs
    probs = _get_expression_probs(images, device)

    if probs is not None:
        cache.save(key, probs)
    return probs
