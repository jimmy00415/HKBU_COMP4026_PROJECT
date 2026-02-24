"""
Anonymizer registry and factory.

Provides :func:`get_anonymizer` — the single entry point for creating any
anonymizer backend from a name + config dict.  Also re-exports base class
and all concrete backends via lazy imports.

Usage::

    from src.anonymizers import get_anonymizer
    anon = get_anonymizer("blur", kernel_size=51)
    result = anon.anonymize_single(face_crop)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── Registry ───────────────────────────────────────────────────────────────

_REGISTRY: dict[str, str] = {
    # Classical baselines
    "blur":            "src.anonymizers.classical",
    "gaussian_blur":   "src.anonymizers.classical",
    "pixelate":        "src.anonymizers.classical",
    "blackout":        "src.anonymizers.classical",
    # k-Same
    "k_same":          "src.anonymizers.k_same",
    "ksame":           "src.anonymizers.k_same",
    # GAN-based
    "ganonymization":  "src.anonymizers.ganonymization",
    "deep_privacy2":   "src.anonymizers.deep_privacy2",
    "deepprivacy2":    "src.anonymizers.deep_privacy2",
    "ciagan":          "src.anonymizers.ciagan",
}

# Map name → class name (for names that don't follow a simple convention)
_CLASS_MAP: dict[str, str] = {
    "blur":            "GaussianBlurAnonymizer",
    "gaussian_blur":   "GaussianBlurAnonymizer",
    "pixelate":        "PixelateAnonymizer",
    "blackout":        "BlackoutAnonymizer",
    "k_same":          "KSameAnonymizer",
    "ksame":           "KSameAnonymizer",
    "ganonymization":  "GANonymizationAnonymizer",
    "deep_privacy2":   "DeepPrivacy2Anonymizer",
    "deepprivacy2":    "DeepPrivacy2Anonymizer",
    "ciagan":          "CIAGANAnonymizer",
}


def get_anonymizer(name: str, **kwargs: Any) -> "AnonymizerBase":
    """
    Factory function — create an anonymizer by name.

    Parameters
    ----------
    name : str
        One of: ``"blur"``, ``"pixelate"``, ``"blackout"``, ``"k_same"``,
        ``"ganonymization"``, ``"deep_privacy2"``, ``"ciagan"``.
    **kwargs
        Passed to the anonymizer constructor.

    Returns
    -------
    AnonymizerBase subclass instance.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        available = sorted(set(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown anonymizer {name!r}. Available: {available}"
        )

    import importlib
    module = importlib.import_module(_REGISTRY[key])
    cls = getattr(module, _CLASS_MAP[key])
    instance = cls(**kwargs)
    logger.info("Created anonymizer: %s(%s)", cls.__name__, kwargs or "")
    return instance


def list_anonymizers() -> list[str]:
    """Return deduplicated list of registered anonymizer names."""
    return sorted(set(_REGISTRY.keys()))


# ── Lazy module-level imports ──────────────────────────────────────────────

_LAZY_MODULES = {
    "AnonymizerBase":           "src.anonymizers.base",
    "GaussianBlurAnonymizer":   "src.anonymizers.classical",
    "PixelateAnonymizer":       "src.anonymizers.classical",
    "BlackoutAnonymizer":       "src.anonymizers.classical",
    "KSameAnonymizer":          "src.anonymizers.k_same",
    "GANonymizationAnonymizer": "src.anonymizers.ganonymization",
    "DeepPrivacy2Anonymizer":   "src.anonymizers.deep_privacy2",
    "CIAGANAnonymizer":        "src.anonymizers.ciagan",
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        import importlib
        module = importlib.import_module(_LAZY_MODULES[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "get_anonymizer",
    "list_anonymizers",
    "AnonymizerBase",
    "GaussianBlurAnonymizer",
    "PixelateAnonymizer",
    "BlackoutAnonymizer",
    "KSameAnonymizer",
    "GANonymizationAnonymizer",
    "DeepPrivacy2Anonymizer",
    "CIAGANAnonymizer",
]
