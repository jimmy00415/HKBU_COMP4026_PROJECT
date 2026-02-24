"""
Expression teacher wrapper.

Wraps a pre-trained expression model (e.g. EmoNet or an AffectNet-trained
network) whose **frozen** soft logits serve as the ground-truth signal for
the *expression consistency* evaluation metric:

    consistency = 1  − KL(teacher_logits_original ‖ teacher_logits_anonymized)

The teacher is never trained in this pipeline — only loaded and used for
inference.  Two operating modes are supported:

1. **timm checkpoint** — load any timm backbone + 7-class head from a
   ``.pth`` file (same architecture as ``ExpressionClassifier``).
2. **ONNX model** — load a pre-exported ONNX graph via ``onnxruntime`` for
   maximum portability.

If neither checkpoint nor ONNX path is provided the teacher falls back to
an ImageNet-pretrained backbone and emits a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExpressionTeacher:
    """
    Frozen expression teacher for soft-label consistency evaluation.

    Parameters
    ----------
    backbone : str
        timm model name (only used if ``onnx_path`` is ``None``).
    num_classes : int
        Number of expression classes.
    checkpoint : str | None
        Path to a ``.pth`` state-dict.
    onnx_path : str | None
        Path to a ``.onnx`` export — if provided, ONNX runtime inference is
        used and ``backbone`` / ``checkpoint`` are ignored.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 7,
        checkpoint: Optional[str] = None,
        onnx_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.num_classes = num_classes
        self.device = device
        self._onnx_session = None  # set if ONNX mode
        self._torch_model = None   # set if PyTorch mode

        if onnx_path is not None:
            self._init_onnx(onnx_path)
        else:
            self._init_torch(backbone, num_classes, checkpoint)

    # ── Initialisers ───────────────────────────────────────────────────

    def _init_torch(
        self, backbone: str, num_classes: int, checkpoint: Optional[str],
    ) -> None:
        import timm
        import torch

        model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)

        if checkpoint is not None:
            ckpt_path = Path(checkpoint)
            if not ckpt_path.exists():
                logger.warning(
                    "Teacher checkpoint %s not found — using ImageNet weights only", checkpoint,
                )
            else:
                state_dict = torch.load(str(ckpt_path), map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                logger.info("Teacher loaded from %s", checkpoint)
        else:
            logger.warning(
                "No teacher checkpoint supplied — soft labels come from "
                "ImageNet-pretrained %s (NOT an expression model).",
                backbone,
            )

        model.to(self.device)
        model.eval()
        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad = False

        self._torch_model = model

        # Resolve normalisation config
        self._data_cfg = timm.data.resolve_model_data_config(model)

    def _init_onnx(self, onnx_path: str) -> None:
        import onnxruntime as ort

        onnx_file = Path(onnx_path)
        if not onnx_file.exists():
            raise FileNotFoundError(f"ONNX teacher model not found: {onnx_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._onnx_session = ort.InferenceSession(str(onnx_file), providers=providers)

        inp = self._onnx_session.get_inputs()[0]
        self._onnx_input_name = inp.name
        # Expect (N, 3, H, W) — extract H, W for resize
        self._onnx_input_shape = inp.shape  # may contain dynamic dims ('N')

        logger.info("Teacher loaded ONNX model from %s", onnx_path)

    # ── Input helpers ──────────────────────────────────────────────────

    def _prepare_torch(self, images: np.ndarray):
        """(N, H, W, 3) float32 [0,1] → normalised (N, 3, H, W) tensor."""
        import torch

        if images.ndim == 3:
            images = images[np.newaxis]

        x = torch.from_numpy(images.astype(np.float32)).permute(0, 3, 1, 2)

        mean = torch.tensor(self._data_cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self._data_cfg["std"], dtype=torch.float32).view(1, 3, 1, 1)
        x = (x - mean) / std

        input_size = self._data_cfg["input_size"]  # (C, H, W)
        _, h_target, w_target = input_size
        if x.shape[2] != h_target or x.shape[3] != w_target:
            x = torch.nn.functional.interpolate(
                x, size=(h_target, w_target), mode="bilinear", align_corners=False,
            )
        return x.to(self.device)

    def _prepare_onnx(self, images: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) float32 [0,1] → (N, 3, H, W) float32 with ImageNet norm.

        Resizes to (224, 224) by default — the expected input size for most
        ViT / ResNet ONNX exports (e.g. trpakov/vit-face-expression).
        """
        import cv2

        if images.ndim == 3:
            images = images[np.newaxis]

        # Resize each image to 224×224 (standard for ONNX vision models)
        target_h, target_w = 224, 224
        if images.shape[1] != target_h or images.shape[2] != target_w:
            resized = np.stack([
                cv2.resize(img, (target_w, target_h)) for img in images
            ])
        else:
            resized = images

        # (N, H, W, 3) → (N, 3, H, W)
        x = resized.astype(np.float32).transpose(0, 3, 1, 2)

        # Standard ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        x = (x - mean) / std

        return x

    # ── Inference ──────────────────────────────────────────────────────

    def predict_logits(self, images: np.ndarray) -> np.ndarray:
        """
        Return raw logits from the frozen teacher.

        Parameters
        ----------
        images : (N, H, W, 3) float32 [0, 1]

        Returns
        -------
        logits : (N, num_classes) float32
        """
        if self._onnx_session is not None:
            return self._predict_logits_onnx(images)
        return self._predict_logits_torch(images)

    def _predict_logits_torch(self, images: np.ndarray) -> np.ndarray:
        import torch

        x = self._prepare_torch(images)
        with torch.no_grad():
            logits = self._torch_model(x)
        return logits.cpu().numpy()

    def _predict_logits_onnx(self, images: np.ndarray) -> np.ndarray:
        x = self._prepare_onnx(images)
        outputs = self._onnx_session.run(None, {self._onnx_input_name: x})
        return outputs[0]  # assume first output is logits

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Return softmax probabilities — shape (N, num_classes)."""
        logits = self.predict_logits(images)
        # Stable softmax in numpy
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Return predicted class indices — shape (N,)."""
        return self.predict_logits(images).argmax(axis=1)

    # ── Soft label extraction (main use-case) ──────────────────────

    def soft_labels(
        self,
        images: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Return temperature-scaled softmax probabilities.

        Used as the "ground-truth" soft targets when computing expression
        consistency between original and anonymized faces.

        Parameters
        ----------
        images : (N, H, W, 3) float32 [0, 1]
        temperature : float
            > 1 makes distribution softer (more uniform),
            < 1 makes it peakier.

        Returns
        -------
        probs : (N, num_classes) float32 — sums to 1 along axis 1.
        """
        logits = self.predict_logits(images)
        scaled = logits / temperature
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
