"""
Error handling, retry logic, and graceful degradation.

Provides:
  • ``retry`` decorator — retry transient failures (download, OOM, etc.)
  • ``safe_import``     — import with fallback message
  • ``validate_image``  — check image integrity before processing
  • ``handle_oom``      — catch CUDA OOM and retry on CPU
  • ``graceful_fallback`` — decorator to return a default on failure

Usage
-----
>>> from src.runner.error_handling import retry, validate_image, graceful_fallback
>>> @retry(max_retries=3, delay=1.0)
... def download_model():
...     ...
>>> @graceful_fallback(default=None)
... def optional_computation():
...     ...
"""

from __future__ import annotations

import functools
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Retry decorator ───────────────────────────────────────────────────

def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Retry decorator with exponential backoff.

    Parameters
    ----------
    max_retries : maximum number of retry attempts.
    delay       : initial delay between retries (seconds).
    backoff     : multiplier for delay after each attempt.
    exceptions  : tuple of exception types to catch.
    on_retry    : optional callback ``(exception, attempt_number) → None``.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            current_delay = delay
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        logger.warning(
                            "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                            fn.__name__, attempt, max_retries, exc, current_delay,
                        )
                        if on_retry:
                            on_retry(exc, attempt)
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            fn.__name__, max_retries, exc,
                        )
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# ── Graceful fallback decorator ────────────────────────────────────────

def graceful_fallback(
    default: Any = None,
    *,
    log_level: int = logging.WARNING,
    message: Optional[str] = None,
):
    """
    Decorator that catches exceptions and returns a default value.

    Parameters
    ----------
    default   : value to return on failure.
    log_level : logging level for failure messages.
    message   : custom failure message.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                msg = message or f"{fn.__name__} failed: {exc}"
                logger.log(log_level, msg)
                return default
        return wrapper
    return decorator


# ── Safe import ────────────────────────────────────────────────────────

def safe_import(module_name: str, package: Optional[str] = None):
    """
    Import a module, returning None instead of raising ImportError.

    Parameters
    ----------
    module_name : dotted module path.
    package     : pip package name for install instructions.

    Returns
    -------
    The module, or None.
    """
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        pip_hint = f"pip install {package}" if package else ""
        logger.info(
            "Optional module %r not available.%s",
            module_name,
            f" Install: {pip_hint}" if pip_hint else "",
        )
        return None


# ── Image validation ──────────────────────────────────────────────────

class ImageValidationError(ValueError):
    """Raised when an image fails integrity checks."""
    pass


def validate_image(
    image: np.ndarray,
    *,
    expected_shape: Optional[tuple[int, ...]] = None,
    allow_grayscale: bool = False,
    name: str = "image",
) -> np.ndarray:
    """
    Validate image array integrity.

    Checks:
      • dtype is float32 or uint8
      • shape is (H, W, 3) or (H, W) if grayscale allowed
      • values are in expected range ([0,1] for float, [0,255] for uint8)
      • no NaN or Inf values
      • non-zero size

    Returns the validated image (possibly converted from grayscale to RGB).

    Raises
    ------
    ImageValidationError
        If the image fails any check.
    """
    if not isinstance(image, np.ndarray):
        raise ImageValidationError(f"{name}: expected ndarray, got {type(image)}")

    if image.size == 0:
        raise ImageValidationError(f"{name}: empty array")

    if image.ndim not in (2, 3):
        raise ImageValidationError(
            f"{name}: expected 2D or 3D, got shape {image.shape}",
        )

    # Handle grayscale
    if image.ndim == 2:
        if not allow_grayscale:
            raise ImageValidationError(
                f"{name}: grayscale not allowed, got shape {image.shape}",
            )
        image = np.stack([image] * 3, axis=-1)

    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ImageValidationError(
            f"{name}: expected 3 channels, got {image.shape[2]}",
        )

    if image.ndim == 3 and image.shape[2] == 1:
        image = np.concatenate([image] * 3, axis=-1)

    # Check expected shape
    if expected_shape is not None and image.shape != expected_shape:
        raise ImageValidationError(
            f"{name}: expected shape {expected_shape}, got {image.shape}",
        )

    # Check dtype
    if image.dtype == np.uint8:
        pass
    elif image.dtype in (np.float32, np.float64):
        if np.any(np.isnan(image)):
            raise ImageValidationError(f"{name}: contains NaN values")
        if np.any(np.isinf(image)):
            raise ImageValidationError(f"{name}: contains Inf values")
        if image.max() > 1.5:
            logger.warning(
                "%s: float image has max=%.2f (expected [0,1]), "
                "clipping to [0,1].",
                name, float(image.max()),
            )
            image = np.clip(image, 0.0, 1.0)
    else:
        raise ImageValidationError(
            f"{name}: unsupported dtype {image.dtype} "
            "(expected float32 or uint8)",
        )

    return image


def validate_image_batch(
    images: np.ndarray,
    *,
    name: str = "batch",
) -> np.ndarray:
    """
    Validate a batch of images (N, H, W, 3).

    Drops corrupted images and logs warnings.
    """
    if images.ndim != 4:
        raise ImageValidationError(
            f"{name}: expected 4D (N, H, W, C), got shape {images.shape}",
        )

    valid_mask = np.ones(len(images), dtype=bool)
    for i in range(len(images)):
        try:
            validate_image(images[i], name=f"{name}[{i}]")
        except ImageValidationError as exc:
            logger.warning("Dropping %s[%d]: %s", name, i, exc)
            valid_mask[i] = False

    if not valid_mask.any():
        raise ImageValidationError(f"{name}: all images are invalid")

    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        logger.warning(
            "%s: dropped %d/%d invalid images",
            name, n_dropped, len(images),
        )

    return images[valid_mask]


# ── CUDA OOM handling ──────────────────────────────────────────────────

def handle_oom(
    fn: Callable[..., T],
    *args: Any,
    fallback_device: str = "cpu",
    **kwargs: Any,
) -> T:
    """
    Call ``fn(*args, **kwargs)``; if CUDA OOM, retry on CPU.

    Expects ``fn`` to accept a ``device`` keyword argument.
    """
    try:
        return fn(*args, **kwargs)
    except RuntimeError as exc:
        exc_msg = str(exc).lower()
        if "out of memory" in exc_msg or "cuda out of memory" in exc_msg:
            logger.warning(
                "CUDA OOM in %s — retrying on %s",
                fn.__name__, fallback_device,
            )
            # Clear GPU cache
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            kwargs["device"] = fallback_device
            return fn(*args, **kwargs)
        raise


# ── Model download retry ──────────────────────────────────────────────

@retry(max_retries=3, delay=2.0, backoff=2.0)
def download_with_retry(url: str, dest: str | Path) -> Path:
    """Download a file with retry logic."""
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)
    urllib.request.urlretrieve(url, str(dest))
    return dest


# ── Safe file operations ──────────────────────────────────────────────

def safe_load_image(path: str | Path) -> Optional[np.ndarray]:
    """
    Load an image file safely — returns None on failure.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Image not found: %s", path)
        return None

    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to decode image: %s", path)
            return None
        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as exc:
        logger.warning("Error loading image %s: %s", path, exc)
        return None


def safe_json_load(path: str | Path) -> Optional[dict]:
    """Load JSON safely — returns None on failure."""
    try:
        import json
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load JSON %s: %s", path, exc)
        return None
