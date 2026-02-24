"""
Adaptive attacker head.

A lightweight MLP classifier trained *on top of* frozen ArcFace embeddings
from **anonymized** images, to test whether the anonymizer's privacy
protection survives an attacker who has access to anonymized training data.

Workflow
--------
1. Anonymize the gallery (e.g. Pins train split).
2. Extract ArcFace embeddings of the anonymized images.
3. Train ``AdaptiveAttackerHead`` to predict identity from those embeddings.
4. Evaluate on anonymized probe embeddings.
5. If accuracy is high, the anonymizer leaks identity even under distribution
   shift — privacy is weak.

The head is intentionally small (2-layer MLP) so that results reflect
*information leakage* in the embedding space, not the capacity of a huge
classifier.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── PyTorch implementation ─────────────────────────────────────────────

def _build_mlp(
    input_dim: int,
    num_classes: int,
    hidden_dims: list[int],
    dropout: float,
):
    """Build a simple MLP with ReLU + dropout."""
    import torch.nn as nn

    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, h),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ])
        prev_dim = h
    layers.append(nn.Linear(prev_dim, num_classes))
    return nn.Sequential(*layers)


class AdaptiveAttackerHead:
    """
    Lightweight MLP identity classifier on ArcFace embeddings.

    Parameters
    ----------
    embedding_dim : int
        Input embedding dimension (512 for ArcFace).
    num_classes : int
        Number of identities.
    hidden_dims : list[int]
        Hidden layer sizes.
    dropout : float
        Dropout probability.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 105,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
        device: str = "cpu",
    ) -> None:
        import torch
        import torch.nn as nn

        self.device = device
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self._model = _build_mlp(embedding_dim, num_classes, hidden_dims, dropout)
        self._model.to(device)

        self._criterion = nn.CrossEntropyLoss()
        self._fitted = False

    # ── Training ───────────────────────────────────────────────────────

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        val_embeddings: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Train the attacker head.

        Parameters
        ----------
        embeddings : (N, D) float32
        labels     : (N,) int
        epochs     : number of training epochs
        batch_size : mini-batch size
        lr         : learning rate
        val_*      : optional validation set

        Returns
        -------
        history : dict with ``"train_loss"``, ``"train_acc"``, and optionally
                  ``"val_acc"`` per epoch.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = torch.from_numpy(embeddings.astype(np.float32)).to(self.device)
        y = torch.from_numpy(labels.astype(np.int64)).to(self.device)

        # Validate label range
        assert int(y.min()) >= 0, f"Labels must be >= 0, got min={int(y.min())}"
        assert int(y.max()) < self.num_classes, (
            f"Label {int(y.max())} >= num_classes {self.num_classes}"
        )

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history: dict[str, list[float]] = {"train_loss": [], "train_acc": []}
        if val_embeddings is not None:
            history["val_acc"] = []

        self._model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            correct = 0
            total = 0

            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self._model(xb)
                loss = self._criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += xb.size(0)

            scheduler.step()

            epoch_loss = total_loss / total
            epoch_acc = correct / total
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)

            msg = f"  Epoch {epoch:3d}/{epochs}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}"

            if val_embeddings is not None and val_labels is not None:
                val_acc = self.evaluate(val_embeddings, val_labels)
                self._model.train()  # restore train mode after eval
                history["val_acc"].append(val_acc)
                msg += f"  val_acc={val_acc:.4f}"

            if verbose:
                logger.info(msg)

        self._fitted = True
        return history

    # ── Inference ──────────────────────────────────────────────────────

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict identity labels.

        Parameters
        ----------
        embeddings : (N, D) float32

        Returns
        -------
        labels : (N,) int64
        """
        import torch

        self._model.eval()
        X = torch.from_numpy(embeddings.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self._model(X)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict identity probabilities (softmax).

        Returns
        -------
        probs : (N, C) float32
        """
        import torch

        self._model.eval()
        X = torch.from_numpy(embeddings.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self._model(X)
        return torch.softmax(logits, dim=1).cpu().numpy()

    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Return top-1 accuracy.

        Parameters
        ----------
        embeddings : np.ndarray
            (N, D) float32 ArcFace embeddings.
        labels : np.ndarray
            (N,) int ground-truth identity labels.

        Returns
        -------
        float
            Top-1 accuracy in [0, 1].
        """
        preds = self.predict(embeddings)
        return float((preds == labels).mean())

    @property
    def fitted(self) -> bool:
        """Whether the attacker head has been trained or loaded."""
        return self._fitted

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save attacker head weights.

        Parameters
        ----------
        path : str
            Destination file path for the state dict.
        """
        import torch
        torch.save(self._model.state_dict(), path)
        logger.info("Adaptive attacker saved to %s", path)

    def load(self, path: str) -> None:
        """Load attacker head weights and mark as fitted.

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
        logger.info("Adaptive attacker loaded from %s", path)
