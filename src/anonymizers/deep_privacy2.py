"""
DeepPrivacy2 backend.

Wraps the DeepPrivacy2 framework from:
    https://github.com/hukkelas/deep_privacy2

The user is expected to clone the repo into ``third_party/deep_privacy2``
and install its dependencies.  This module adds the repo to ``sys.path`` at
runtime.

DeepPrivacy2 has its own face-detection pipeline, so it can operate on
full-scene images.  For our per-crop contract we still feed it the
256 × 256 aligned face and let it re-detect internally, or bypass
detection and pass the crop directly to the anonymiser.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.anonymizers.base import AnonymizerBase
from src.data.contracts import AnonymizedFace, FaceCrop, float_to_uint8, uint8_to_float

logger = logging.getLogger(__name__)

_DEFAULT_REPO_DIR = Path("third_party/deep_privacy2")


class DeepPrivacy2Anonymizer(AnonymizerBase):
    """
    DeepPrivacy2 face anonymizer.

    Parameters
    ----------
    repo_dir : str | Path
        Path to the cloned DeepPrivacy2 repo.
    model : str
        Model variant — ``"face"`` for face-only anonymisation.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        repo_dir: str | Path = _DEFAULT_REPO_DIR,
        model: str = "face",
        device: str = "cpu",
    ) -> None:
        self._repo_dir = Path(repo_dir)
        self._model_name = model
        self._device = device
        self._anonymizer = None  # lazy
        self._available = self._repo_dir.exists()

        if not self._available:
            logger.warning(
                "DeepPrivacy2 repo not found at %s. "
                "Clone https://github.com/hukkelas/deep_privacy2 there.",
                self._repo_dir,
            )

    @property
    def name(self) -> str:
        return "deep_privacy2"

    @property
    def configurable_params(self) -> dict[str, Any]:
        return {"model": self._model_name, "device": self._device}

    # ── Lazy loading ───────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        if self._anonymizer is not None:
            return

        if not self._available:
            raise RuntimeError(
                f"DeepPrivacy2 repo not found at {self._repo_dir}. "
                "Clone https://github.com/hukkelas/deep_privacy2 there."
            )

        repo_str = str(self._repo_dir.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            self._load_model()
        except Exception as e:
            logger.error("Failed to load DeepPrivacy2: %s", e)
            raise

    def _load_model(self) -> None:
        """
        Try import paths known to work with DeepPrivacy2.
        """
        import torch

        # Attempt 1: the high-level API (v2 style)
        try:
            from dp2.anonymizer import build_anonymizer  # type: ignore
            from dp2.utils import load_config  # type: ignore

            cfg_path = self._repo_dir / "configs" / f"{self._model_name}.yaml"
            if not cfg_path.exists():
                # Try finding any face-related config
                configs = list((self._repo_dir / "configs").rglob("*.yaml"))
                face_cfgs = [c for c in configs if "face" in c.stem.lower()]
                if face_cfgs:
                    cfg_path = face_cfgs[0]
                elif configs:
                    cfg_path = configs[0]
                else:
                    raise FileNotFoundError("No config YAML found in DeepPrivacy2 repo.")

            cfg = load_config(str(cfg_path))
            self._anonymizer = build_anonymizer(cfg)
            self._mode = "dp2_api"
            logger.info("DeepPrivacy2 loaded via dp2 API (%s)", cfg_path.name)
            return
        except ImportError:
            pass

        # Attempt 2: command-line wrapper fallback
        self._anonymizer = "cli_fallback"
        self._mode = "cli"
        logger.info("DeepPrivacy2 will use CLI fallback.")

    # ── Inference ──────────────────────────────────────────────────────

    def anonymize_single(
        self,
        face: FaceCrop,
        **kwargs: Any,
    ) -> AnonymizedFace:
        self._ensure_model()

        if self._mode == "dp2_api":
            result = self._infer_dp2_api(face.image)
        else:
            result = self._infer_cli(face.image)

        return self._make_result(image=result, original=face)

    def _infer_dp2_api(self, image: np.ndarray) -> np.ndarray:
        """Anonymize using dp2 Python API."""
        import torch

        img_u8 = float_to_uint8(image)
        # DeepPrivacy2 typically expects (H, W, 3) uint8 BGR (OpenCV format)
        import cv2
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

        anon_bgr = self._anonymizer.anonymize(img_bgr)
        anon_rgb = cv2.cvtColor(anon_bgr, cv2.COLOR_BGR2RGB)
        return uint8_to_float(anon_rgb)

    def _infer_cli(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback: save image to temp file, call DeepPrivacy2 CLI, read back.
        """
        import subprocess
        import tempfile

        import cv2

        img_u8 = float_to_uint8(image)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "input.png"
            out_path = Path(tmpdir) / "output.png"
            cv2.imwrite(str(in_path), img_bgr)

            cmd = [
                sys.executable, str(self._repo_dir / "anonymize.py"),
                "-i", str(in_path),
                "-o", str(out_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(self._repo_dir),
            )
            if result.returncode != 0:
                logger.error("DeepPrivacy2 CLI failed:\n%s", result.stderr)
                # Return original image as fallback
                return image.copy()

            if out_path.exists():
                anon_bgr = cv2.imread(str(out_path))
                anon_rgb = cv2.cvtColor(anon_bgr, cv2.COLOR_BGR2RGB)
                return uint8_to_float(anon_rgb)

        logger.warning("DeepPrivacy2 CLI produced no output; returning original.")
        return image.copy()
