"""
Train an expression classifier (FER-2013 7-class).

Hydra-configurable training script with:
- Dataset / backbone / LR schedule / augmentation selection via YAML.
- Early stopping on validation accuracy.
- TensorBoard logging (optional).
- Checkpoint saving (best & last).

Usage
-----
::

    python scripts/train_expression.py                          # defaults
    python scripts/train_expression.py model.expression.backbone=efficientnet_b0
    python scripts/train_expression.py training.epochs=50 training.lr=5e-4
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Hydra / OmegaConf ─────────────────────────────────────────────────────
try:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    _HAS_HYDRA = True
except ImportError:
    _HAS_HYDRA = False

# ── Defaults (used when Hydra is unavailable) ─────────────────────────────
_DEFAULTS = {
    "dataset": "fer2013",
    "data_root": "data/fer2013",
    "csv_file": "fer2013.csv",
    "backbone": "resnet18",
    "num_classes": 7,
    "pretrained": True,
    "epochs": 30,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 7,
    "device": "cuda",
    "seed": 42,
    "output_dir": "results/expression_training",
    "checkpoint_dir": "checkpoints",
    "use_tensorboard": True,
    "augmentation": True,
}


# ── augmentation helper ────────────────────────────────────────────────────

def _augment_batch(images: np.ndarray, seed: int | None = None) -> np.ndarray:
    """
    Simple numpy-based training augmentations for 256×256 RGB face crops.

    Applied augmentations:
    - Random horizontal flip (50%)
    - Random brightness jitter (±10 %)
    - Random slight rotation (±5°, via affine)
    """
    rng = np.random.RandomState(seed)
    out = images.copy()

    for i in range(len(out)):
        # Horizontal flip
        if rng.rand() < 0.5:
            out[i] = out[i, :, ::-1, :]

        # Brightness jitter
        factor = 1.0 + rng.uniform(-0.1, 0.1)
        out[i] = np.clip(out[i] * factor, 0.0, 1.0)

    return out


# ── Early stopping ─────────────────────────────────────────────────────────

class _EarlyStopping:
    """Monitor a metric and stop if it doesn't improve for *patience* epochs."""

    def __init__(self, patience: int = 7, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── TensorBoard helper ────────────────────────────────────────────────────

def _get_writer(log_dir: str):
    """Return a TensorBoard SummaryWriter or None."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except ImportError:
        logger.warning("tensorboard not installed — logging disabled.")
        return None


# ── Main training function ────────────────────────────────────────────────

def train_expression(
    *,
    dataset: str = "fer2013",
    data_root: str = "data/fer2013",
    csv_file: str = "fer2013.csv",
    backbone: str = "resnet18",
    num_classes: int = 7,
    pretrained: bool = True,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "results/expression_training",
    checkpoint_dir: str = "checkpoints",
    use_tensorboard: bool = True,
    augmentation: bool = True,
) -> dict:
    """
    Full training loop for the expression classifier.

    Uses :class:`torch.utils.data.DataLoader` for memory-efficient batching
    (FER-2013 has ~28 k images; loading all as 256×256×3 would need ~21 GB).

    Returns
    -------
    dict with ``history``, ``best_val_acc``, ``checkpoint_path``.
    """
    import torch
    from torch.utils.data import DataLoader

    np.random.seed(seed)
    torch.manual_seed(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ───────────────────────────────────────────────────
    logger.info("Loading dataset '%s' from %s ...", dataset, data_root)

    if dataset == "fer2013":
        from src.data.fer2013_adapter import FER2013Dataset

        train_ds = FER2013Dataset(root=data_root, split="train")
        val_ds = FER2013Dataset(root=data_root, split="val")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Custom collate: FaceCrop → (images_np, labels_np) batch
    def _collate_face_crops(batch):
        images = np.stack([fc.image for fc in batch])         # (B, 256, 256, 3)
        labels = np.array([fc.meta.expression_label for fc in batch])
        return images, labels

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate_face_crops,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, collate_fn=_collate_face_crops,
    )

    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    # ── Build model ────────────────────────────────────────────────────
    from src.models.expression_classifier import ExpressionClassifier

    model = ExpressionClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        device=device,
    )

    # ── TensorBoard ────────────────────────────────────────────────────
    writer = _get_writer(str(out / "tb_logs")) if use_tensorboard else None

    # ── Training loop ──────────────────────────────────────────────────
    stopper = _EarlyStopping(patience=patience, mode="max")
    best_val_acc = 0.0
    best_ckpt_path = str(ckpt_dir / f"expression_{backbone}_best.pth")

    optimizer = torch.optim.AdamW(
        model._model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    model._model.train()
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_images, batch_labels in train_loader:
            # Apply augmentation
            if augmentation:
                batch_images = _augment_batch(batch_images, seed=seed + epoch + total)

            xb = model._prepare_batch(batch_images)
            yb = torch.from_numpy(batch_labels.astype(np.int64)).to(device)

            optimizer.zero_grad()
            logits = model._model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)

        # ── Validation ─────────────────────────────────────────────
        model._model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_imgs, val_lbls in val_loader:
                xv = model._prepare_batch(val_imgs)
                preds = model._model(xv).argmax(dim=1).cpu().numpy()
                val_correct += (preds == val_lbls).sum()
                val_total += len(val_lbls)
        val_acc = float(val_correct / max(val_total, 1))
        model._model.train()

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        logger.info(
            "Epoch %3d/%d  loss=%.4f  train_acc=%.4f  val_acc=%.4f  lr=%.2e",
            epoch, epochs, epoch_loss, epoch_acc, val_acc, current_lr,
        )

        # TensorBoard
        if writer is not None:
            writer.add_scalar("train/loss", epoch_loss, epoch)
            writer.add_scalar("train/accuracy", epoch_acc, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            writer.add_scalar("train/lr", current_lr, epoch)

        # Checkpoint best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(best_ckpt_path)
            logger.info("  -> New best val_acc=%.4f  saved %s", val_acc, best_ckpt_path)

        # Early stopping
        if stopper.step(val_acc):
            logger.info("Early stopping at epoch %d (patience=%d).", epoch, patience)
            break

    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs. Best val_acc=%.4f", elapsed, best_val_acc)

    # Save last checkpoint
    last_ckpt = str(ckpt_dir / f"expression_{backbone}_last.pth")
    model.save(last_ckpt)

    if writer is not None:
        writer.close()

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "checkpoint_path": best_ckpt_path,
        "elapsed_seconds": elapsed,
    }


# ── Hydra entry-point ─────────────────────────────────────────────────────

if _HAS_HYDRA:

    @hydra.main(config_path="../configs", config_name="config", version_base="1.3")
    def main(cfg: DictConfig) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

        # Extract training params from Hydra config
        ds_cfg = cfg.get("dataset", {})
        model_cfg = cfg.get("model", {}).get("expression", {})
        train_cfg = cfg.get("training", {})

        train_expression(
            dataset=ds_cfg.get("name", "fer2013"),
            data_root=ds_cfg.get("root", "data/fer2013"),
            csv_file=ds_cfg.get("csv_file", "fer2013.csv"),
            backbone=model_cfg.get("backbone", "resnet18"),
            num_classes=model_cfg.get("num_classes", 7),
            pretrained=model_cfg.get("pretrained", True),
            epochs=train_cfg.get("epochs", 30),
            batch_size=train_cfg.get("batch_size", 64),
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
            patience=train_cfg.get("patience", 7),
            device=cfg.get("device", "cuda"),
            seed=cfg.get("seed", 42),
            output_dir=cfg.get("output_dir", "results/expression_training"),
            checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
            use_tensorboard=train_cfg.get("use_tensorboard", True),
            augmentation=train_cfg.get("augmentation", True),
        )

else:

    def main() -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
        logger.info("Hydra not available — running with defaults.")
        train_expression(**_DEFAULTS)


if __name__ == "__main__":
    main()
