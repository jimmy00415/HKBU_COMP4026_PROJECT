"""
Mixed-precision (AMP) and memory-optimisation utilities.

Provides:
  • ``amp_inference_context`` — context manager for fp16 inference
  • ``amp_training_context``  — GradScaler + autocast for training
  • ``gradient_checkpoint_wrap`` — wraps a module for gradient checkpointing
  • ``batch_inference``       — generic batched inference with optional AMP

These are *opt-in* utilities.  The rest of the codebase works without them;
callers wrap their hot paths in these when GPU memory or throughput matters.

Usage
-----
>>> from src.runner.amp_utils import amp_inference_context, batch_inference
>>> with amp_inference_context(device="cuda"):
...     output = model(input_tensor)
>>> results = batch_inference(model, data, batch_size=64, device="cuda", use_amp=True)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── AMP availability check ────────────────────────────────────────────

def _amp_available(device: str = "cuda") -> bool:
    """Check whether AMP is available on the current device."""
    if device == "cpu":
        return False
    try:
        import torch
        return torch.cuda.is_available() and hasattr(torch.cuda.amp, "autocast")
    except ImportError:
        return False


# ── Inference context manager ─────────────────────────────────────────

@contextmanager
def amp_inference_context(
    *,
    device: str = "cuda",
    enabled: bool = True,
    dtype: Optional[str] = "float16",
):
    """
    Context manager for AMP inference.

    Parameters
    ----------
    device  : ``"cuda"`` or ``"cpu"``.
    enabled : set to False to make this a no-op.
    dtype   : ``"float16"`` or ``"bfloat16"``.

    Yields inside ``torch.cuda.amp.autocast`` if available.
    """
    use_amp = enabled and _amp_available(device)

    if use_amp:
        import torch

        amp_dtype = torch.float16
        if dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16

        with torch.cuda.amp.autocast(dtype=amp_dtype):
            yield
    else:
        yield


# ── Training context ──────────────────────────────────────────────────

class AMPTrainingContext:
    """
    Wraps training with AMP autocast + GradScaler.

    Usage
    -----
    >>> ctx = AMPTrainingContext(device="cuda")
    >>> for batch in loader:
    ...     with ctx.autocast():
    ...         loss = model(batch)
    ...     ctx.backward(loss, optimizer)
    >>> ctx.close()
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        enabled: bool = True,
        init_scale: float = 2.0 ** 16,
    ) -> None:
        self._enabled = enabled and _amp_available(device)
        self._scaler = None
        if self._enabled:
            import torch
            self._scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
            logger.info("AMP training enabled (init_scale=%.0f)", init_scale)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def autocast(self):
        """Autocast context for the forward pass."""
        if self._enabled:
            import torch
            with torch.cuda.amp.autocast():
                yield
        else:
            yield

    def backward(self, loss, optimizer, *, max_grad_norm: Optional[float] = None) -> None:
        """
        Backward + optimizer step with optional gradient scaling & clipping.
        """
        if self._enabled and self._scaler is not None:
            import torch
            self._scaler.scale(loss).backward()
            if max_grad_norm is not None:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    _get_params(optimizer), max_grad_norm,
                )
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                import torch
                torch.nn.utils.clip_grad_norm_(
                    _get_params(optimizer), max_grad_norm,
                )
            optimizer.step()

    def close(self) -> None:
        """Clean up if needed (currently a no-op)."""
        pass


def _get_params(optimizer):
    """Extract all parameter tensors from an optimizer."""
    params = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    return params


# ── Gradient checkpointing ────────────────────────────────────────────

def enable_gradient_checkpointing(module) -> None:
    """
    Enable gradient checkpointing for a PyTorch ``nn.Module``.

    Works with timm models, torchvision's ResNet, and any model that
    supports ``set_grad_checkpointing`` or ``gradient_checkpointing_enable``.
    """
    # timm models
    if hasattr(module, "set_grad_checkpointing"):
        module.set_grad_checkpointing(True)
        logger.info("Gradient checkpointing enabled (timm API)")
        return

    # HuggingFace-style
    if hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled (HF API)")
        return

    # Manual — wrap all Sequential children
    import torch.utils.checkpoint as cp

    for name, child in module.named_children():
        if isinstance(child, __import__("torch").nn.Sequential) and len(child) > 2:
            original_forward = child.forward

            def _ckpt_forward(*args, _orig=original_forward, **kwargs):
                return cp.checkpoint(_orig, *args, use_reentrant=False, **kwargs)

            child.forward = _ckpt_forward
            logger.debug("  Wrapped %s for gradient checkpointing", name)

    logger.info("Gradient checkpointing enabled (manual wrapping)")


# ── Batched inference utility ──────────────────────────────────────────

def batch_inference(
    fn: Callable,
    data: np.ndarray,
    *,
    batch_size: int = 64,
    device: str = "cpu",
    use_amp: bool = False,
    desc: str = "Inference",
) -> np.ndarray:
    """
    Run ``fn(batch) → result_batch`` in batches with optional AMP.

    Parameters
    ----------
    fn        : callable that takes an ndarray batch and returns an ndarray.
    data      : full dataset (N, ...).
    batch_size: per-batch sample count.
    device    : for AMP context.
    use_amp   : enable fp16 autocast.
    desc      : label for logging.

    Returns
    -------
    Concatenated results (N, ...).
    """
    N = len(data)
    results = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = data[start:end]

        if use_amp and _amp_available(device):
            with amp_inference_context(device=device):
                out = fn(batch)
        else:
            out = fn(batch)

        results.append(out)

        if (start // batch_size) % 10 == 0:
            logger.debug("  %s: %d/%d", desc, end, N)

    return np.concatenate(results, axis=0)


# ── Memory pressure helpers ───────────────────────────────────────────

def clear_gpu_cache() -> None:
    """Free GPU cache if PyTorch CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def estimate_tensor_mb(shape: tuple, dtype: str = "float32") -> float:
    """Estimate the memory footprint of a tensor in MB."""
    dtype_sizes = {
        "float16": 2, "float32": 4, "float64": 8,
        "int8": 1, "int16": 2, "int32": 4, "int64": 8,
        "bfloat16": 2,
    }
    bytes_per_elem = dtype_sizes.get(dtype, 4)
    total_elems = 1
    for s in shape:
        total_elems *= s
    return total_elems * bytes_per_elem / (1024 ** 2)
