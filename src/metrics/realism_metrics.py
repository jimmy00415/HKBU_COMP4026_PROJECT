"""
Realism diagnostic metrics.

Measures how *realistic* the anonymised images are, independent of privacy
or expression utility.

Metrics
-------
1. **FID (Fréchet Inception Distance)**  ↓   — distribution distance between
   anonymised set and a real reference set via Inception v3 features.
2. **LPIPS (Learned Perceptual Image Patch Similarity)** — per-image
   perceptual distance between original and anonymised.
3. **PSNR (Peak Signal-to-Noise Ratio)** ↑ — pixel-level fidelity.
4. **SSIM (Structural Similarity)** ↑ — structure-aware pixel metric.

External library support
------------------------
* ``clean-fid`` or ``pytorch-fid`` for FID.
* ``lpips`` package for LPIPS.
* PSNR and SSIM are implemented in pure numpy to avoid extra deps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class RealismReport:
    """Container for realism-diagnostic results."""

    fid: float = -1.0              # -1 means not computed
    lpips_mean: float = -1.0
    lpips_std: float = -1.0
    psnr_mean: float = 0.0
    psnr_std: float = 0.0
    ssim_mean: float = 0.0
    ssim_std: float = 0.0

    anonymizer_name: str = ""
    num_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize realism metrics to a flat dictionary."""
        return {
            "fid": self.fid,
            "lpips_mean": self.lpips_mean,
            "lpips_std": self.lpips_std,
            "psnr_mean": self.psnr_mean,
            "psnr_std": self.psnr_std,
            "ssim_mean": self.ssim_mean,
            "ssim_std": self.ssim_std,
            "anonymizer_name": self.anonymizer_name,
            "num_samples": self.num_samples,
        }


# ── Pure-numpy pixel metrics ──────────────────────────────────────────────

def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio between two images.

    Parameters
    ----------
    img1, img2 : (H, W, C) float arrays in [0, max_val].

    Returns
    -------
    float — dB.  Higher is better.  Returns ``inf`` if images are identical.
    """
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(max_val ** 2 / mse))


def ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 1.0,
    window_size: int = 7,
) -> float:
    """
    Structural Similarity Index (simplified, per-channel mean).

    Uses a uniform window for simplicity.  For publication-quality results
    prefer ``skimage.metrics.structural_similarity``.

    Returns
    -------
    float in [-1, 1].  Higher is better.
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Channel-wise SSIM
    if img1.ndim == 3:
        vals = [
            _ssim_channel(img1[..., c], img2[..., c], C1, C2, window_size)
            for c in range(img1.shape[2])
        ]
        return float(np.mean(vals))
    return _ssim_channel(img1, img2, C1, C2, window_size)


def _ssim_channel(
    a: np.ndarray, b: np.ndarray,
    C1: float, C2: float, ws: int,
) -> float:
    """SSIM for a single 2-D channel using uniform-window averaging."""
    from scipy.ndimage import uniform_filter

    mu_a = uniform_filter(a, size=ws)
    mu_b = uniform_filter(b, size=ws)
    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = uniform_filter(a ** 2, size=ws) - mu_a_sq
    sigma_b_sq = uniform_filter(b ** 2, size=ws) - mu_b_sq
    sigma_ab = uniform_filter(a * b, size=ws) - mu_ab

    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)
    ssim_map = num / den
    return float(ssim_map.mean())


# ── LPIPS wrapper ──────────────────────────────────────────────────────────

def compute_lpips(
    images_original: np.ndarray,
    images_anonymized: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Per-image LPIPS distance via the ``lpips`` package.

    Parameters
    ----------
    images_original   : (N, H, W, 3) float32 [0, 1].
    images_anonymized : (N, H, W, 3) float32 [0, 1].
    device : str

    Returns
    -------
    distances : (N,) float32 — lower is more similar.
    """
    try:
        import lpips as lpips_pkg
        import torch
    except ImportError:
        logger.warning("lpips/torch not installed — returning NaN distances.")
        return np.full(len(images_original), np.nan, dtype=np.float32)

    loss_fn = lpips_pkg.LPIPS(net="alex", verbose=False).to(device)

    # Convert (N,H,W,3) → (N,3,H,W) and scale to [-1,1]
    orig_t = torch.from_numpy(
        images_original.astype(np.float32)
    ).permute(0, 3, 1, 2) * 2.0 - 1.0
    anon_t = torch.from_numpy(
        images_anonymized.astype(np.float32)
    ).permute(0, 3, 1, 2) * 2.0 - 1.0

    dists = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(orig_t), batch_size):
            o = orig_t[i : i + batch_size].to(device)
            a = anon_t[i : i + batch_size].to(device)
            d = loss_fn(o, a).squeeze().cpu().numpy()
            if d.ndim == 0:
                d = d.reshape(1)
            dists.append(d)

    return np.concatenate(dists).astype(np.float32)


# ── FID wrapper ────────────────────────────────────────────────────────────

def compute_fid(
    real_dir: str | Path,
    fake_dir: str | Path,
    device: str = "cpu",
) -> float:
    """
    Compute FID between two directories of images.

    Tries ``cleanfid`` first, falls back to ``pytorch_fid``.

    Parameters
    ----------
    real_dir, fake_dir : paths to directories of PNG/JPG images.
    device : str

    Returns
    -------
    float — FID score (lower is better).
    """
    # Attempt 1: clean-fid
    try:
        from cleanfid import fid as cleanfid_module

        score = cleanfid_module.compute_fid(
            str(real_dir), str(fake_dir), device=device,
        )
        logger.info("FID (clean-fid) = %.2f", score)
        return float(score)
    except ImportError:
        pass

    # Attempt 2: pytorch-fid
    try:
        from pytorch_fid import fid_score as pyfid

        score = pyfid.calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=50,
            device=device,
            dims=2048,
        )
        logger.info("FID (pytorch-fid) = %.2f", score)
        return float(score)
    except ImportError:
        pass

    logger.warning("Neither clean-fid nor pytorch-fid installed — FID unavailable.")
    return -1.0


def compute_fid_from_arrays(
    images_real: np.ndarray,
    images_fake: np.ndarray,
    device: str = "cpu",
) -> float:
    """
    Compute FID from numpy arrays by saving to temp dirs.

    Parameters
    ----------
    images_real : (N, H, W, 3) float32 [0, 1].
    images_fake : (M, H, W, 3) float32 [0, 1].

    Returns
    -------
    float — FID score.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as real_dir, \
         tempfile.TemporaryDirectory() as fake_dir:

        _save_images(images_real, real_dir)
        _save_images(images_fake, fake_dir)
        return compute_fid(real_dir, fake_dir, device=device)


def _save_images(images: np.ndarray, directory: str | Path) -> None:
    """Save (N, H, W, 3) float32 → PNG files."""
    from PIL import Image

    from src.data.contracts import float_to_uint8

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        pil = Image.fromarray(float_to_uint8(img))
        pil.save(directory / f"{i:06d}.png")


# ── Orchestrator ───────────────────────────────────────────────────────────

def evaluate_realism(
    images_original: np.ndarray,
    images_anonymized: np.ndarray,
    *,
    compute_fid_score: bool = True,
    compute_lpips_score: bool = True,
    device: str = "cpu",
    anonymizer_name: str = "",
) -> RealismReport:
    """
    Compute all realism diagnostic metrics.

    Parameters
    ----------
    images_original   : (N, H, W, 3) float32 [0, 1].
    images_anonymized : (N, H, W, 3) float32 [0, 1].
    compute_fid_score : whether to compute FID (requires saving to disk).
    compute_lpips_score : whether to compute LPIPS (requires lpips + torch).
    device : str
    anonymizer_name : for metadata.

    Returns
    -------
    RealismReport
    """
    N = images_original.shape[0]
    report = RealismReport(anonymizer_name=anonymizer_name, num_samples=N)

    # ── PSNR & SSIM (always, no extra deps) ────────────────────────────
    logger.info("Computing PSNR & SSIM...")
    psnr_vals = np.array([
        psnr(images_original[i], images_anonymized[i])
        for i in range(N)
    ])
    ssim_vals = np.array([
        ssim(images_original[i], images_anonymized[i])
        for i in range(N)
    ])
    # Filter out inf PSNR (identical images)
    finite_psnr = psnr_vals[np.isfinite(psnr_vals)]
    report.psnr_mean = float(finite_psnr.mean()) if len(finite_psnr) > 0 else float("inf")
    report.psnr_std = float(finite_psnr.std()) if len(finite_psnr) > 0 else 0.0
    report.ssim_mean = float(ssim_vals.mean())
    report.ssim_std = float(ssim_vals.std())
    logger.info("  PSNR=%.2f±%.2f  SSIM=%.4f±%.4f",
                report.psnr_mean, report.psnr_std,
                report.ssim_mean, report.ssim_std)

    # ── LPIPS ──────────────────────────────────────────────────────────
    if compute_lpips_score:
        logger.info("Computing LPIPS...")
        lpips_vals = compute_lpips(images_original, images_anonymized, device)
        if not np.isnan(lpips_vals).all():
            report.lpips_mean = float(np.nanmean(lpips_vals))
            report.lpips_std = float(np.nanstd(lpips_vals))
            logger.info("  LPIPS=%.4f±%.4f", report.lpips_mean, report.lpips_std)

    # ── FID ────────────────────────────────────────────────────────────
    if compute_fid_score:
        logger.info("Computing FID...")
        report.fid = compute_fid_from_arrays(
            images_original, images_anonymized, device=device,
        )
        if report.fid >= 0:
            logger.info("  FID=%.2f", report.fid)

    return report
