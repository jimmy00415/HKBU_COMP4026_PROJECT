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

        The exact import path depends on the repo's structure. We try
        the most common entry points.
        """
        import torch

        # Attempt 1: the repo may expose a high-level inference API
        try:
            from models import create_model  # type: ignore
            from options.test_options import TestOptions  # type: ignore

            # Build a minimal options namespace
            opt = TestOptions().parse([
                "--dataroot", ".",
                "--name", self._weights,
                "--model", "pix2pix",
                "--netG", "unet_256",
                "--direction", "AtoB",
                "--dataset_mode", "single",
                "--norm", "batch",
                "--no_dropout",
                "--gpu_ids", "0" if self._device == "cuda" else "-1",
            ])
            opt.num_threads = 0
            opt.batch_size = 1
            opt.serial_batches = True
            opt.no_flip = True
            opt.display_id = -1
            opt.isTrain = False

            model = create_model(opt)
            model.setup(opt)
            model.eval()
            self._model = model
            self._mode = "pix2pix_standard"
            logger.info("GANonymization loaded (pix2pix standard API)")
            return
        except ImportError:
            pass

        # Attempt 2: direct generator checkpoint
        try:
            from models.networks import define_G  # type: ignore

            netG = define_G(
                input_nc=3, output_nc=3, ngf=64,
                netG="unet_256", norm="batch",
                use_dropout=False, init_type="normal",
                init_gain=0.02, gpu_ids=[],
            )
            ckpt_path = self._repo_dir / "checkpoints" / self._weights / "latest_net_G.pth"
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location=self._device, weights_only=True)
                netG.load_state_dict(state)
            else:
                logger.warning(
                    "Checkpoint not found at %s — using random weights", ckpt_path,
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
        self._ensure_model()

        import torch

        if self._mode == "pix2pix_standard":
            result_img = self._infer_pix2pix_standard(face.image)
        else:
            result_img = self._infer_raw_generator(face.image)

        return self._make_result(
            image=result_img,
            original=face,
        )

    def _infer_pix2pix_standard(self, image: np.ndarray) -> np.ndarray:
        """Inference using the full pix2pix model API."""
        import torch

        # Convert to model's expected format: [-1, 1], (1, 3, H, W)
        tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0  # [0,1] → [-1,1]
        tensor = tensor.to(self._device)

        data = {"A": tensor, "A_paths": ""}
        self._model.set_input(data)
        self._model.test()
        visuals = self._model.get_current_visuals()

        # Output is in [-1, 1]
        fake = visuals.get("fake_B", visuals.get("fake", None))
        if fake is None:
            fake = list(visuals.values())[-1]
        out = fake.squeeze(0).cpu().numpy()  # (3, H, W)
        out = (out + 1.0) / 2.0             # [-1,1] → [0,1]
        out = np.clip(out, 0.0, 1.0)
        return out.transpose(1, 2, 0)        # (H, W, 3)

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
