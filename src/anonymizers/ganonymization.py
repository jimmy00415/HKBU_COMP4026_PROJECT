"""
GANonymization backend.

Wraps the GANonymization model (pix2pix-based face anonymization) from:
    https://github.com/hcmlab/GANonymization

The user is expected to clone the repo into ``third_party/GANonymization``
and download the pre-trained weights.  This module adds the repo to
``sys.path`` at runtime and invokes the model's inference pipeline.

If the third-party code is unavailable the class still loads but raises
an informative error at :meth:`anonymize_single` time.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.anonymizers.base import AnonymizerBase
from src.data.contracts import AnonymizedFace, FaceCrop, float_to_uint8, uint8_to_float

logger = logging.getLogger(__name__)

_DEFAULT_REPO_DIR = Path("third_party/GANonymization")


class GANonymizationAnonymizer(AnonymizerBase):
    """
    GANonymization (pix2pix) face anonymizer.

    Parameters
    ----------
    repo_dir : str | Path
        Path to the cloned GANonymization repo.
    weights : str
        ``"publication"`` (25-epoch) or ``"demo"`` (50-epoch).
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        repo_dir: str | Path = _DEFAULT_REPO_DIR,
        weights: str = "publication",
        device: str = "cpu",
    ) -> None:
        self._repo_dir = Path(repo_dir)
        self._weights = weights
        self._device = device
        self._model = None  # lazy load
        self._available = self._repo_dir.exists()

        if not self._available:
            logger.warning(
                "GANonymization repo not found at %s. "
                "Clone https://github.com/hcmlab/GANonymization there first.",
                self._repo_dir,
            )

    @property
    def name(self) -> str:
        return "ganonymization"

    @property
    def configurable_params(self) -> dict[str, Any]:
        return {"weights": self._weights, "device": self._device}

    # ── Lazy model loading ─────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """Load the pix2pix model on first call."""
        if self._model is not None:
            return

        if not self._available:
            raise RuntimeError(
                f"GANonymization repo not found at {self._repo_dir}. "
                "Please clone https://github.com/hcmlab/GANonymization "
                "into third_party/GANonymization."
            )

        # Add repo to path so we can import its modules
        repo_str = str(self._repo_dir.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            self._load_model()
        except Exception as e:
            logger.error("Failed to load GANonymization model: %s", e)
            raise

    def _load_model(self) -> None:
        """
        Import and instantiate the GANonymization pix2pix generator.

        Supports two loading strategies:

        1. **Lightning checkpoint** (preferred) – load a ``Pix2Pix``
           Lightning checkpoint previously downloaded to
           ``pretrained/ganonymization/<weights>.ckpt`` and extract
           the ``GeneratorUNet`` sub-network.
        2. **Raw generator** – fall back to instantiating
           ``GeneratorUNet`` from the repo and loading a bare
           ``state_dict``.
        """
        import torch

        # Resolve checkpoint path -------------------------------------------
        # Try well-known locations in order of preference.
        _weight_map = {
            "publication": "publication_25ep.ckpt",
            "demo": "demo_50ep.ckpt",
        }
        ckpt_name = _weight_map.get(self._weights, f"{self._weights}.ckpt")

        candidates = [
            Path("pretrained/ganonymization") / ckpt_name,
            self._repo_dir / "checkpoints" / ckpt_name,
            self._repo_dir / ckpt_name,
        ]
        ckpt_path: Optional[Path] = None
        for p in candidates:
            if p.exists():
                ckpt_path = p
                break

        # Attempt 1: full Lightning checkpoint → extract generator -----------
        if ckpt_path is not None:
            try:
                from lib.models.pix2pix import GeneratorUNet  # type: ignore

                ckpt = torch.load(
                    str(ckpt_path), map_location=self._device, weights_only=False,
                )
                state = ckpt.get("state_dict", ckpt)
                gen_state = {
                    k.replace("generator.", ""): v
                    for k, v in state.items()
                    if k.startswith("generator.")
                }
                if not gen_state:
                    # Checkpoint may already contain only generator keys
                    gen_state = state

                netG = GeneratorUNet(in_channels=3, out_channels=3)
                netG.load_state_dict(gen_state)
                netG.to(self._device)
                netG.eval()
                self._model = netG
                self._mode = "raw_generator"
                logger.info(
                    "GANonymization loaded (Lightning ckpt → generator) from %s",
                    ckpt_path,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Lightning checkpoint loading failed (%s), trying fallback", exc,
                )

        # Attempt 2: direct GeneratorUNet with bare state_dict ---------------
        try:
            from lib.models.pix2pix import GeneratorUNet  # type: ignore

            netG = GeneratorUNet(in_channels=3, out_channels=3)
            if ckpt_path is not None:
                bare = torch.load(
                    str(ckpt_path), map_location=self._device, weights_only=True,
                )
                netG.load_state_dict(bare)
            else:
                logger.warning(
                    "No checkpoint found for weights=%s — using random weights",
                    self._weights,
                )
            netG.to(self._device)
            netG.eval()
            self._model = netG
            self._mode = "raw_generator"
            logger.info("GANonymization loaded (raw generator)")
            return
        except ImportError:
            pass

        raise ImportError(
            "Could not import GANonymization model. Ensure the repo is "
            "properly cloned and its dependencies are installed."
        )

    # ── Inference ──────────────────────────────────────────────────────

    def anonymize_single(
        self,
        face: FaceCrop,
        **kwargs: Any,
    ) -> AnonymizedFace:
        """Anonymize a single face crop using the GANonymization pix2pix generator.

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

        result_img = self._infer_raw_generator(face.image)

        return self._make_result(
            image=result_img,
            original=face,
        )

    def _infer_raw_generator(self, image: np.ndarray) -> np.ndarray:
        """Inference using a bare generator network."""
        import torch

        tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        tensor = tensor.to(self._device)

        with torch.no_grad():
            out = self._model(tensor)

        out = out.squeeze(0).cpu().numpy()
        out = (out + 1.0) / 2.0
        out = np.clip(out, 0.0, 1.0)
        return out.transpose(1, 2, 0)
