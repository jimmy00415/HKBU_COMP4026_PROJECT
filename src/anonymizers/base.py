"""
Anonymizer interface — abstract base class.

Every anonymizer backend (classical, k-Same, GAN-based) **must** subclass
:class:`AnonymizerBase` and implement :meth:`anonymize_single` and/or
:meth:`anonymize_batch`.

The interface guarantees:

* Input : :class:`FaceCrop` or :class:`FaceCropBatch` (256×256 RGB float32 [0,1]).
* Output: :class:`AnonymizedFace` / :class:`AnonymizedBatch` from ``contracts.py``.
* Every backend exposes ``name`` and ``configurable_params`` for sweep support.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

import numpy as np

from src.data.contracts import (
    AnonymizedBatch,
    AnonymizedFace,
    FaceCrop,
    FaceCropBatch,
)

logger = logging.getLogger(__name__)


class AnonymizerBase(abc.ABC):
    """
    Abstract base for all anonymizer backends.

    Subclasses must implement at least :meth:`anonymize_single`.
    :meth:`anonymize_batch` has a default loop-based implementation that can
    be overridden for more efficient batched processing.
    """

    # ── Properties ─────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier used in logs, configs, and result filenames."""
        ...

    @property
    def configurable_params(self) -> dict[str, Any]:
        """
        Return a dict of all tuneable hyper-parameters and their current
        values.  Used to record settings in :class:`AnonymizedFace`.
        """
        return {}

    # ── Core interface ─────────────────────────────────────────────────

    @abc.abstractmethod
    def anonymize_single(
        self,
        face: FaceCrop,
        **kwargs: Any,
    ) -> AnonymizedFace:
        """
        Anonymize a single face crop.

        Parameters
        ----------
        face : FaceCrop
            Canonical 256×256 RGB float32 [0,1].
        **kwargs
            Backend-specific overrides (e.g. ``kernel_size`` for blur).

        Returns
        -------
        AnonymizedFace
        """
        ...

    def anonymize_batch(
        self,
        batch: FaceCropBatch,
        **kwargs: Any,
    ) -> AnonymizedBatch:
        """
        Anonymize a batch of face crops.

        Default implementation loops over :meth:`anonymize_single`.
        Override in subclasses for GPU-batched inference.
        """
        results = [self.anonymize_single(crop, **kwargs) for crop in batch.crops]
        return AnonymizedBatch(faces=results)

    # ── Convenience ────────────────────────────────────────────────────

    def __call__(
        self,
        input_: FaceCrop | FaceCropBatch,
        **kwargs: Any,
    ) -> AnonymizedFace | AnonymizedBatch:
        """Dispatches to single or batch based on input type."""
        if isinstance(input_, FaceCropBatch):
            return self.anonymize_batch(input_, **kwargs)
        return self.anonymize_single(input_, **kwargs)

    def _make_result(
        self,
        image: np.ndarray,
        original: FaceCrop,
        mask: np.ndarray | None = None,
        aux: dict | None = None,
        **param_overrides: Any,
    ) -> AnonymizedFace:
        """
        Helper to build an :class:`AnonymizedFace` with the correct metadata.
        """
        # Clamp to [0, 1] to honour the contract
        image = np.clip(image.astype(np.float32), 0.0, 1.0)

        params = {**self.configurable_params, **param_overrides}
        return AnonymizedFace(
            image=image,
            original=original,
            anonymizer_name=self.name,
            anonymizer_params=params,
            mask=mask,
            aux=aux or {},
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
