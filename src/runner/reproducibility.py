"""
Reproducibility utilities.

Ensures deterministic behaviour across runs by:
  1. Seeding Python, NumPy, PyTorch, and CUDA.
  2. Setting PyTorch deterministic mode flags.
  3. Recording git commit hash, Hydra config, and environment info.
  4. Saving a ``reproducibility.json`` manifest alongside results.

Usage
-----
>>> from src.runner.reproducibility import seed_everything, save_manifest
>>> seed_everything(42)
>>> save_manifest("results/run_001")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Seeding ────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42, *, deterministic: bool = True) -> None:
    """
    Seed all RNGs for reproducibility.

    Parameters
    ----------
    seed : global seed.
    deterministic : if True, enable PyTorch deterministic algorithms
        (may reduce performance slightly).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch ≥ 1.8 deterministic mode
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except TypeError:
                    torch.use_deterministic_algorithms(True)

        logger.info(
            "Seeded all RNGs with %d (deterministic=%s)", seed, deterministic,
        )
    except ImportError:
        logger.info("Seeded Python & NumPy with %d (torch not available)", seed)


# ── Git hash ───────────────────────────────────────────────────────────

def get_git_hash(repo_dir: Optional[str | Path] = None) -> Optional[str]:
    """Return the current git commit hash (short), or None."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo_dir) if repo_dir else None,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_git_dirty(repo_dir: Optional[str | Path] = None) -> Optional[bool]:
    """Check if the working tree has uncommitted changes."""
    try:
        cmd = ["git", "diff", "--quiet", "HEAD"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            cwd=str(repo_dir) if repo_dir else None,
            timeout=5,
        )
        return result.returncode != 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ── Environment snapshot ──────────────────────────────────────────────

def get_environment_info() -> dict[str, Any]:
    """Collect system and library version info."""
    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "numpy_version": np.__version__,
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "N/A"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        info["torch_version"] = "not installed"

    try:
        import timm
        info["timm_version"] = timm.__version__
    except ImportError:
        pass

    try:
        import cv2
        info["opencv_version"] = cv2.__version__
    except ImportError:
        pass

    return info


# ── Hydra config capture ──────────────────────────────────────────────

def capture_hydra_config() -> Optional[dict]:
    """
    Capture the current Hydra config (if running inside a Hydra app).

    Returns the resolved config as a dict, or None if Hydra is not active.
    """
    try:
        from omegaconf import OmegaConf
        import hydra

        cfg = hydra.compose(config_name="config")
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return None


def config_to_hash(config: dict) -> str:
    """SHA-256 hash of a config dict for deduplication."""
    serialised = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:12]


# ── Manifest ──────────────────────────────────────────────────────────

def save_manifest(
    output_dir: str | Path,
    *,
    seed: int = 42,
    config: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> Path:
    """
    Save a reproducibility manifest to ``output_dir/reproducibility.json``.

    Parameters
    ----------
    output_dir : directory to write the manifest into.
    seed : the seed used for this run.
    config : the experiment config (e.g. from Hydra).
    extra : any additional key-value metadata.

    Returns
    -------
    Path to the saved manifest.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": seed,
        "git_hash": get_git_hash(),
        "git_dirty": get_git_dirty(),
        "environment": get_environment_info(),
    }

    if config is not None:
        manifest["config"] = config
        manifest["config_hash"] = config_to_hash(config)

    hydra_cfg = capture_hydra_config()
    if hydra_cfg is not None:
        manifest["hydra_config"] = hydra_cfg

    if extra is not None:
        manifest["extra"] = extra

    path = out / "reproducibility.json"
    path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Reproducibility manifest saved to %s", path)
    return path


def load_manifest(path: str | Path) -> dict:
    """Load a previously saved manifest."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_reproducibility(
    manifest_path: str | Path, *, warn_only: bool = True,
) -> bool:
    """
    Check that the current environment matches a saved manifest.

    Compares git hash, Python version, and key library versions.
    Returns True if consistent.
    """
    manifest = load_manifest(manifest_path)
    issues: list[str] = []

    # Git hash
    current_hash = get_git_hash()
    saved_hash = manifest.get("git_hash")
    if current_hash and saved_hash and current_hash != saved_hash:
        issues.append(f"Git hash mismatch: {saved_hash} → {current_hash}")

    # Python version
    saved_py = manifest.get("environment", {}).get("python_version", "")
    if saved_py and not sys.version.startswith(saved_py.split()[0]):
        issues.append(f"Python version mismatch: {saved_py} → {sys.version}")

    # Seed
    saved_seed = manifest.get("seed")
    if saved_seed is not None:
        logger.info("Manifest seed: %d", saved_seed)

    if issues:
        for issue in issues:
            if warn_only:
                logger.warning("Reproducibility: %s", issue)
            else:
                logger.error("Reproducibility: %s", issue)
        return False

    logger.info("Reproducibility check passed.")
    return True
