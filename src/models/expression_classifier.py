"""
Expression classifier.

Thin wrapper around a **timm** backbone (default: ResNet-18) with a 7-class
head for the FER-2013 expression taxonomy.

Responsibilities
----------------
* Train / fine-tune on the canonical 256 × 256 face crops.
* Predict logits and probabilities for the 7 expression classes.
* Save / load checkpoints.
* Provide ``extract_features()`` to get penultimate-layer representations
  (useful for e.g. expression-teacher consistency loss via KL-div of logits).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExpressionClassifier:
    """
    7-class facial expression classifier backed by a *timm* model.

    Parameters
    ----------
    backbone : str
        Any valid *timm* model name (e.g. ``"resnet18"``, ``"efficientnet_b0"``).
    num_classes : int
        Number of expression classes (default 7 = FER-2013).
    pretrained : bool
        Use ImageNet-pretrained weights.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 7,
        pretrained: bool = True,
        device: str = "cpu",
    ) -> None:
        import timm
        import torch
        import torch.nn as nn

        self.device = device
        self.num_classes = num_classes
        self.backbone_name = backbone

        # Create timm model with custom head
        self._model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self._model.to(device)

        # Resolve the timm data config for this backbone, so we can use the
        # correct normalisation (mean/std) at inference time.
        self._data_cfg = timm.data.resolve_model_data_config(self._model)
        self._transform = timm.data.create_transform(**self._data_cfg, is_training=False)

        self._criterion = nn.CrossEntropyLoss()
        self._fitted = False

        logger.info(
            "ExpressionClassifier: backbone=%s  num_classes=%d  pretrained=%s",
            backbone, num_classes, pretrained,
        )

    # ── Input helpers ──────────────────────────────────────────────────

    def _prepare_batch(self, images: np.ndarray):
        """
        Convert a numpy batch to a normalised torch tensor.

        Parameters
        ----------
        images : (N, H, W, 3) float32 [0, 1]  **or**  (H, W, 3) single image.

        Returns
        -------
        torch.Tensor : (N, 3, H, W) normalised to backbone mean/std.
        """
        import torch

        if images.ndim == 3:
            images = images[np.newaxis]

        # (N, H, W, 3) → (N, 3, H, W)
        x = torch.from_numpy(images.astype(np.float32)).permute(0, 3, 1, 2)

        # Apply timm normalisation (ImageNet mean/std for most backbones)
        mean = torch.tensor(self._data_cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self._data_cfg["std"], dtype=torch.float32).view(1, 3, 1, 1)
        x = (x - mean) / std

        # Resize to model's expected input size if needed
        input_size = self._data_cfg["input_size"]  # (C, H, W)
        _, h_target, w_target = input_size
        if x.shape[2] != h_target or x.shape[3] != w_target:
            x = torch.nn.functional.interpolate(
                x, size=(h_target, w_target), mode="bilinear", align_corners=False,
            )

        return x.to(self.device)

    # ── Training ───────────────────────────────────────────────────────

    def fit(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        val_images: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Fine-tune the classifier on face crops.

        Parameters
        ----------
        images : (N, H, W, 3) float32 [0, 1]
        labels : (N,) int  — expression label indices in [0, num_classes).
        epochs : int
        batch_size : int
        lr : float
        val_images, val_labels : optional validation arrays.
        verbose : bool

        Returns
        -------
        history : dict with ``"train_loss"``, ``"train_acc"``, and optionally
                  ``"val_acc"``.
        """
        import torch

        N = images.shape[0]
        y_all = torch.from_numpy(labels.astype(np.int64)).to(self.device)

        assert int(y_all.min()) >= 0, f"Labels must be >= 0, got min={int(y_all.min())}"
        assert int(y_all.max()) < self.num_classes, (
            f"Label {int(y_all.max())} >= num_classes {self.num_classes}"
        )

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history: dict[str, list[float]] = {"train_loss": [], "train_acc": []}
        if val_images is not None:
            history["val_acc"] = []

        self._model.train()
        for epoch in range(1, epochs + 1):
            # Shuffle indices each epoch
            perm = np.random.permutation(N)
            total_loss = 0.0
            correct = 0
            total = 0

            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                xb = self._prepare_batch(images[idx])
                yb = y_all[idx]

                optimizer.zero_grad()
                logits = self._model(xb)
                loss = self._criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += xb.size(0)

            scheduler.step()

            epoch_loss = total_loss / max(total, 1)
            epoch_acc = correct / max(total, 1)
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)

            msg = f"  Epoch {epoch:3d}/{epochs}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}"

            if val_images is not None and val_labels is not None:
                val_acc = self.evaluate(val_images, val_labels)
                self._model.train()  # restore train mode after eval
                history["val_acc"].append(val_acc)
                msg += f"  val_acc={val_acc:.4f}"

            if verbose:
                logger.info(msg)

        self._fitted = True
        return history

    # ── Inference ──────────────────────────────────────────────────────

    def predict_logits(self, images: np.ndarray) -> np.ndarray:
        """
        Return raw logits.

        Parameters
        ----------
        images : (N, H, W, 3) float32 [0, 1]

        Returns
        -------
        logits : (N, num_classes) float32
        """
        import torch

        self._model.eval()
        with torch.no_grad():
            x = self._prepare_batch(images)
            logits = self._model(x)
        return logits.cpu().numpy()

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Return softmax probabilities.

        Parameters
        ----------
        images : np.ndarray
            (N, H, W, 3) float32 [0, 1] face crops.

        Returns
        -------
        np.ndarray
            (N, num_classes) float32 probability distribution per sample.
        """
        import torch

        logits_np = self.predict_logits(images)
        logits_t = torch.from_numpy(logits_np)
        return torch.softmax(logits_t, dim=1).numpy()

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Return predicted label indices.

        Parameters
        ----------
        images : np.ndarray
            (N, H, W, 3) float32 [0, 1] face crops.

        Returns
        -------
        np.ndarray
            (N,) int64 predicted expression label indices.
        """
        logits = self.predict_logits(images)
        return logits.argmax(axis=1)

    def evaluate(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 256,
    ) -> float:
        """
        Compute top-1 accuracy over (potentially large) arrays.

        Processes in mini-batches to avoid OOM.

        Parameters
        ----------
        images : np.ndarray
            (N, H, W, 3) float32 [0, 1] face crops.
        labels : np.ndarray
            (N,) int ground-truth expression label indices.
        batch_size : int
            Mini-batch size for inference (default 256).

        Returns
        -------
        float
            Top-1 accuracy in [0, 1].
        """
        N = images.shape[0]
        correct = 0
        for start in range(0, N, batch_size):
            batch_imgs = images[start : start + batch_size]
            batch_lbls = labels[start : start + batch_size]
            preds = self.predict(batch_imgs)
            correct += (preds == batch_lbls).sum()
        return float(correct / max(N, 1))

    # ── Feature extraction ─────────────────────────────────────────────

    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract penultimate-layer features (before the classification head).

        Useful for KL-div / consistency loss computation.

        Returns
        -------
        features : (N, D) float32  where D depends on the backbone
                   (e.g. 512 for ResNet-18).
        """
        import torch

        self._model.eval()
        with torch.no_grad():
            x = self._prepare_batch(images)
            features = self._model.forward_features(x)
            # Global average pool if it's a spatial feature map
            if features.ndim == 4:
                features = features.mean(dim=(2, 3))
            elif features.ndim == 3:
                # Transformer / ViT: take CLS token or mean
                features = features.mean(dim=1)
        return features.cpu().numpy()

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights.

        Parameters
        ----------
        path : str
            Destination file path for the state dict.
        """
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        logger.info("ExpressionClassifier saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights and mark the classifier as fitted.

        Parameters
        ----------
        path : str
            Path to a previously saved state dict.
        """
        import torch
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._fitted = True
        logger.info("ExpressionClassifier loaded from %s", path)

    @property
    def fitted(self) -> bool:
        """Whether the model has been trained or loaded from a checkpoint."""
        return self._fitted
