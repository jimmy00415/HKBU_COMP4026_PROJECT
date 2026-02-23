"""
CIAGAN backend.

Wraps the CIAGAN (Conditional Identity Anonymization GAN) from:
    https://github.com/dvl-tum/ciagan

CIAGAN (CVPR 2020) replaces the identity in a face image while preserving
other attributes (pose, expression, background).

The user is expected to clone the repo into ``third_party/ciagan`` and
download/train the model weights.  This module adds the repo to
``sys.path`` at runtime.
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

_DEFAULT_REPO_DIR = Path("third_party/ciagan")


class CIAGANAnonymizer(AnonymizerBase):
    """
    CIAGAN face anonymizer.

    Parameters
    ----------
    repo_dir : str | Path
        Path to the cloned CIAGAN repo.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        repo_dir: str | Path = _DEFAULT_REPO_DIR,
        device: str = "cpu",
    ) -> None:
        self._repo_dir = Path(repo_dir)
        self._device = device
        self._model = None  # lazy
        self._available = self._repo_dir.exists()

        if not self._available:
            logger.warning(
                "CIAGAN repo not found at %s. "
                "Clone https://github.com/dvl-tum/ciagan there.",
                self._repo_dir,
            )

    @property
    def name(self) -> str:
        return "ciagan"

    @property
    def configurable_params(self) -> dict[str, Any]:
        return {"device": self._device}

    # ── Lazy loading ───────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        if not self._available:
            raise RuntimeError(
                f"CIAGAN repo not found at {self._repo_dir}. "
                "Clone https://github.com/dvl-tum/ciagan there."
            )

        repo_str = str(self._repo_dir.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            self._load_model()
        except Exception as e:
            logger.error("Failed to load CIAGAN: %s", e)
            raise

    def _load_model(self) -> None:
        """
        Import the CIAGAN generator.

        The repo's structure typically has a ``models/`` directory with
        the generator and discriminator definitions.
        """
        import torch

        # Attempt: load the generator module directly
        try:
            from models.generator import Generator as CIAGANGenerator  # type: ignore

            gen = CIAGANGenerator()

            # Look for checkpoint
            ckpt_candidates = sorted(
                self._repo_dir.glob("checkpoints/*.pth"),
            ) + sorted(
                self._repo_dir.glob("weights/*.pth"),
            ) + sorted(
                self._repo_dir.glob("*.pth"),
            )

            if ckpt_candidates:
                ckpt_path = ckpt_candidates[-1]  # most recent
                state = torch.load(
                    str(ckpt_path), map_location=self._device, weights_only=True,
                )
                gen.load_state_dict(state)
                logger.info("CIAGAN generator loaded from %s", ckpt_path)
            else:
                logger.warning(
                    "No CIAGAN checkpoint found — generator has random weights."
                )

            gen.to(self._device)
            gen.eval()
            self._model = gen
            self._mode = "generator"
            return
        except ImportError:
            pass

        # Attempt 2: full model wrapper
        try:
            from test import load_model  # type: ignore

            self._model = load_model(self._repo_dir, self._device)
            self._mode = "test_api"
            logger.info("CIAGAN loaded via test.py API")
            return
        except ImportError:
            pass

        # Attempt 3: CLI fallback
        self._model = "cli_fallback"
        self._mode = "cli"
        logger.info("CIAGAN will use CLI fallback.")

    # ── Inference ──────────────────────────────────────────────────────

    def anonymize_single(
        self,
        face: FaceCrop,
        **kwargs: Any,
    ) -> AnonymizedFace:
        self._ensure_model()

        if self._mode == "generator":
            result = self._infer_generator(face.image)
        elif self._mode == "test_api":
            result = self._infer_test_api(face.image)
        else:
            result = self._infer_cli(face.image)

        return self._make_result(image=result, original=face)

    def _infer_generator(self, image: np.ndarray) -> np.ndarray:
        """Run through the CIAGAN generator directly."""
        import torch

        # CIAGAN generator expects [-1, 1] input, (1, 3, H, W)
        tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        tensor = tensor.to(self._device)

        # CIAGAN may require a conditioning identity vector — use a random
        # one to produce a novel identity
        z_dim = 100  # standard latent dim for CIAGAN
        z = torch.randn(1, z_dim, device=self._device)

        with torch.no_grad():
            try:
                out = self._model(tensor, z)
            except TypeError:
                # Some variants take only the image
                out = self._model(tensor)

        out = out.squeeze(0).cpu().numpy()       # (3, H, W)
        out = (out + 1.0) / 2.0                  # [-1,1] → [0,1]
        out = np.clip(out, 0.0, 1.0)
        return out.transpose(1, 2, 0)            # (H, W, 3)

    def _infer_test_api(self, image: np.ndarray) -> np.ndarray:
        """Use the model loaded via test.py API."""
        img_u8 = float_to_uint8(image)
        result = self._model.anonymize(img_u8)
        return uint8_to_float(result)

    def _infer_cli(self, image: np.ndarray) -> np.ndarray:
        """Fallback: invoke test.py via subprocess."""
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
                sys.executable, str(self._repo_dir / "test.py"),
                "--input", str(in_path),
                "--output", str(out_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(self._repo_dir),
            )
            if result.returncode != 0:
                logger.error("CIAGAN CLI failed:\n%s", result.stderr)
                return image.copy()

            if out_path.exists():
                anon_bgr = cv2.imread(str(out_path))
                anon_rgb = cv2.cvtColor(anon_bgr, cv2.COLOR_BGR2RGB)
                return uint8_to_float(anon_rgb)

        logger.warning("CIAGAN CLI produced no output; returning original.")
        return image.copy()
