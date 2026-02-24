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

        The repo houses its source under ``source/arch/arch_unet_flex.py``
        with a ``Generator`` class that expects 6-channel input
        (landmarks + masked face) and a one-hot identity vector.
        """
        import torch

        # The CIAGAN source lives under <repo>/source/
        source_dir = str((self._repo_dir / "source").resolve())
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)

        # Attempt 1: direct generator import from repo source tree
        try:
            from arch.arch_unet_flex import Generator as CIAGANGenerator  # type: ignore

            # CIAGAN Generator: input_nc=6, num_classes=1200, img_size=128
            gen = CIAGANGenerator(
                input_nc=6, num_classes=1200,
                encode_one_hot=True, img_size=128,
            )

            # Look for checkpoint
            ckpt_candidates = (
                sorted(self._repo_dir.glob("modelG*"))
                + sorted(self._repo_dir.glob("checkpoints/*.pth"))
                + sorted(self._repo_dir.glob("*.pth"))
                + sorted(Path("pretrained/ciagan").glob("*.pth"))
            )

            if ckpt_candidates:
                ckpt_path = ckpt_candidates[-1]
                state = torch.load(
                    str(ckpt_path), map_location=self._device, weights_only=True,
                )
                gen.load_state_dict(state)
                logger.info("CIAGAN generator loaded from %s", ckpt_path)
            else:
                logger.warning(
                    "No CIAGAN checkpoint found — generator has random weights. "
                    "Results will be meaningless until weights are provided."
                )

            gen.to(self._device)
            gen.eval()
            self._model = gen
            self._num_classes = 1200
            self._mode = "generator"
            return
        except ImportError as exc:
            logger.warning("CIAGAN generator import failed: %s", exc)

        # Attempt 2: CLI fallback
        self._model = "cli_fallback"
        self._mode = "cli"
        logger.info("CIAGAN will use CLI fallback.")

    # ── Inference ──────────────────────────────────────────────────────

    def anonymize_single(
        self,
        face: FaceCrop,
        **kwargs: Any,
    ) -> AnonymizedFace:
        """Anonymize a single face crop using the CIAGAN generator.

        Parameters
        ----------
        face : FaceCrop
            Canonical 256×256 RGB face crop.
        **kwargs : Any
            Additional keyword arguments (unused).

        Returns
        -------
        AnonymizedFace
            The anonymized face with metadata.
        """
        self._ensure_model()

        if self._mode == "generator":
            result = self._infer_generator(face)
        else:
            result = self._infer_cli(face.image)

        return self._make_result(image=result, original=face)

    def _infer_generator(self, face: FaceCrop) -> np.ndarray:
        """
        Run through the CIAGAN generator.

        CIAGAN expects:
          • 6-channel input: concatenation of landmark image (3ch) and
            masked face (face × (1 − mask), 3ch)
          • one-hot identity vector  (num_classes,)

        When landmarks / mask are not available we approximate:
          • landmarks → zeros (no conditioning)
          • mask → ones  (full face region)

        The output identity is chosen randomly.
        """
        import torch

        image = face.image  # (H, W, 3) float32 [0, 1]

        # Resize to 128×128 for CIAGAN
        import cv2
        img128 = cv2.resize(image, (128, 128)).astype(np.float32)

        # Build landmark image (3-ch) — use zeros if unavailable
        lndm = np.zeros_like(img128)  # (128, 128, 3)

        # Build mask — ones = entire face is identity region
        mask = np.ones((128, 128, 1), dtype=np.float32)

        # 6-ch input: landmarks ‖ face × (1 − mask)
        masked_face = img128 * (1.0 - mask)
        inp = np.concatenate([lndm, masked_face], axis=2)  # (128, 128, 6)

        tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0)  # (1, 6, 128, 128)
        tensor = tensor.to(self._device)

        # Random target identity one-hot
        target_id = np.random.randint(0, self._num_classes)
        onehot = np.zeros((1, self._num_classes), dtype=np.float32)
        onehot[0, target_id] = 1.0
        onehot_t = torch.from_numpy(onehot).to(self._device)

        with torch.no_grad():
            try:
                out = self._model(tensor, onehot=onehot_t)
            except TypeError:
                out = self._model(tensor)

        out = out.squeeze(0).cpu().numpy()       # (3, 128, 128)
        out = np.clip(out, 0.0, 1.0)
        out = out.transpose(1, 2, 0)             # (128, 128, 3)

        # Resize back to original resolution
        if out.shape[:2] != image.shape[:2]:
            out = cv2.resize(out, (image.shape[1], image.shape[0]))

        return out

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
