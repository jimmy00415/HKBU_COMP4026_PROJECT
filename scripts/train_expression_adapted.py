"""
Domain-adapted expression training.

Train / fine-tune the expression classifier under three regimes and compare
utility retention:

1. **real-only**         — train on original (non-anonymised) images.
2. **anonymised-only**   — train on synthetically anonymised images.
3. **mixed**             — train on 50/50 real + anonymised (configurable ratio).

This lets us measure how well expression cues survive anonymisation and
whether domain adaptation on the anonymised distribution closes the gap.

Usage
-----
::

    python scripts/train_expression_adapted.py --regime real
    python scripts/train_expression_adapted.py --regime anonymized --synthetic-dir data/synthetic/blur_k15
    python scripts/train_expression_adapted.py --regime mixed --mix-ratio 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Synthetic data loader ──────────────────────────────────────────────────

def _load_synthetic_split(
    synthetic_dir: str,
    split: str,
    save_format: str = "npy",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load anonymised images + expression labels from generate_synthetic.py output.

    Returns (images, expression_labels).
    """
    base = Path(synthetic_dir)
    split_dir = base / split
    labels_path = base / "labels.json"

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with open(labels_path) as f:
        all_labels = json.load(f)

    split_labels = all_labels.get(split, {})
    n = len(split_labels)

    images = []
    expr_labels = []

    for i in range(n):
        fname = f"image_{i:06d}"
        if save_format == "npy":
            img_path = split_dir / f"{fname}.npy"
            if img_path.exists():
                images.append(np.load(str(img_path)))
            else:
                logger.warning("Missing file: %s", img_path)
                continue
        else:
            # PNG loading
            try:
                from PIL import Image
                from src.data.contracts import uint8_to_float
                img_path = split_dir / f"{fname}.png"
                pil = Image.open(str(img_path)).convert("RGB")
                images.append(uint8_to_float(np.array(pil)))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", img_path, exc)
                continue

        meta = split_labels.get(str(i), {})
        expr_labels.append(meta.get("expression", -1))

    images_arr = np.stack(images)
    labels_arr = np.array(expr_labels, dtype=np.int64)

    # Filter out samples without expression labels
    valid = labels_arr >= 0
    return images_arr[valid], labels_arr[valid]


# ── Mixed dataset builder ─────────────────────────────────────────────────

def _build_mixed_dataset(
    real_images: np.ndarray,
    real_labels: np.ndarray,
    anon_images: np.ndarray,
    anon_labels: np.ndarray,
    mix_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a mixed training set with *mix_ratio* fraction of anonymised data.

    Parameters
    ----------
    mix_ratio : fraction of the output that comes from anonymised data.
                0.0 = all real, 1.0 = all anonymised, 0.5 = balanced.

    Returns
    -------
    (mixed_images, mixed_labels)
    """
    rng = np.random.RandomState(seed)

    n_total = max(len(real_images), len(anon_images))
    n_anon = int(n_total * mix_ratio)
    n_real = n_total - n_anon

    # Sample with replacement if needed
    real_idx = rng.choice(len(real_images), size=min(n_real, len(real_images)), replace=False)
    anon_idx = rng.choice(len(anon_images), size=min(n_anon, len(anon_images)), replace=False)

    mixed_images = np.concatenate([real_images[real_idx], anon_images[anon_idx]], axis=0)
    mixed_labels = np.concatenate([real_labels[real_idx], anon_labels[anon_idx]], axis=0)

    # Shuffle
    perm = rng.permutation(len(mixed_labels))
    return mixed_images[perm], mixed_labels[perm]


# ── Main training function ────────────────────────────────────────────────

def train_expression_adapted(
    *,
    regime: str = "real",           # "real" | "anonymized" | "mixed"
    dataset: str = "fer2013",
    data_root: str = "data/fer2013",
    csv_file: str = "fer2013.csv",
    synthetic_dir: str = "data/synthetic/blur",
    synthetic_format: str = "npy",
    mix_ratio: float = 0.5,
    backbone: str = "resnet18",
    num_classes: int = 7,
    pretrained: bool = True,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 7,
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "results/adapted_training",
    checkpoint_dir: str = "checkpoints",
) -> dict:
    """
    Train expression classifier under the specified regime.

    Returns
    -------
    dict with training results.
    """
    np.random.seed(seed)
    out = Path(output_dir) / regime
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load REAL data (always needed for validation) ──────────────────
    from src.data.fer2013_adapter import FER2013Dataset

    logger.info("Loading real FER-2013 data...")
    train_ds = FER2013Dataset(root=data_root, split="train")
    val_ds = FER2013Dataset(root=data_root, split="val")

    # Load lazily in limited batches to avoid OOM
    n_train = min(len(train_ds), 500)  # cap for memory
    n_val = min(len(val_ds), 500)

    real_train_images = np.stack([train_ds[i].image for i in range(n_train)])
    real_train_labels = np.array([train_ds[i].meta.expression_label for i in range(n_train)])
    val_images = np.stack([val_ds[i].image for i in range(n_val)])
    val_labels = np.array([val_ds[i].meta.expression_label for i in range(n_val)])

    # ── Build training set per regime ──────────────────────────────────
    if regime == "real":
        train_images = real_train_images
        train_labels = real_train_labels
        logger.info("Regime: REAL-ONLY (%d samples)", len(train_labels))

    elif regime == "anonymized":
        logger.info("Loading anonymised data from %s...", synthetic_dir)
        train_images, train_labels = _load_synthetic_split(
            synthetic_dir, "train", synthetic_format,
        )
        logger.info("Regime: ANONYMIZED-ONLY (%d samples)", len(train_labels))

    elif regime == "mixed":
        logger.info("Loading anonymised data from %s...", synthetic_dir)
        anon_images, anon_labels = _load_synthetic_split(
            synthetic_dir, "train", synthetic_format,
        )
        train_images, train_labels = _build_mixed_dataset(
            real_train_images, real_train_labels,
            anon_images, anon_labels,
            mix_ratio=mix_ratio,
            seed=seed,
        )
        logger.info(
            "Regime: MIXED (ratio=%.2f, %d samples)", mix_ratio, len(train_labels)
        )
    else:
        raise ValueError(f"Unknown regime: {regime!r}. Use 'real', 'anonymized', or 'mixed'.")

    # ── Train using the domain-adapted data directly ───────────────────
    from src.models.expression_classifier import ExpressionClassifier
    from scripts.train_expression import _EarlyStopping, _augment_batch, _get_writer

    model = ExpressionClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        device=device,
    )

    import torch

    best_ckpt_path = str(ckpt_dir / f"expression_{backbone}_{regime}_best.pth")
    y_train_t = torch.from_numpy(train_labels.astype(np.int64)).to(device)
    optimizer = torch.optim.AdamW(model._model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    stopper = _EarlyStopping(patience=patience, mode="max")
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    writer = _get_writer(str(out / "tb_logs"))

    model._model.train()
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        epoch_images = _augment_batch(train_images, seed=seed + epoch)
        perm = np.random.permutation(len(train_labels))
        total_loss = 0.0
        correct = 0
        total = 0

        for start in range(0, len(train_labels), batch_size):
            idx = perm[start : start + batch_size]
            xb = model._prepare_batch(epoch_images[idx])
            yb = y_train_t[idx]

            optimizer.zero_grad()
            logits = model._model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

        scheduler.step()
        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)

        # Validate on REAL data (always)
        val_acc = model.evaluate(val_images, val_labels)
        model._model.train()

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            "Epoch %3d/%d  loss=%.4f  acc=%.4f  val_acc=%.4f",
            epoch, epochs, epoch_loss, epoch_acc, val_acc,
        )

        if writer is not None:
            writer.add_scalar("train/loss", epoch_loss, epoch)
            writer.add_scalar("train/accuracy", epoch_acc, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(best_ckpt_path)
            logger.info("  ↑ Best val_acc=%.4f → %s", val_acc, best_ckpt_path)

        if stopper.step(val_acc):
            logger.info("Early stopping at epoch %d.", epoch)
            break

    elapsed = time.time() - t0
    if writer is not None:
        writer.close()

    last_ckpt = str(ckpt_dir / f"expression_{backbone}_{regime}_last.pth")
    model.save(last_ckpt)

    real_val_acc = best_val_acc

    result = {
        "history": history,
        "best_val_acc": best_val_acc,
        "checkpoint_path": best_ckpt_path,
        "elapsed_seconds": elapsed,
    }
    result["regime"] = regime
    result["real_val_acc"] = real_val_acc
    logger.info("Real-data val accuracy (regime=%s): %.4f", regime, real_val_acc)

    # ── Save summary ──────────────────────────────────────────────────
    summary_path = out / "summary.json"
    serialisable = {
        k: v for k, v in result.items()
        if isinstance(v, (int, float, str, bool, list, dict))
    }
    with open(summary_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Domain-adapted expression training (real / anonymized / mixed)."
    )
    parser.add_argument("--regime", choices=["real", "anonymized", "mixed"], default="real")
    parser.add_argument("--dataset", default="fer2013")
    parser.add_argument("--data-root", default="data/fer2013")
    parser.add_argument("--csv-file", default="fer2013.csv")
    parser.add_argument("--synthetic-dir", default="data/synthetic/blur")
    parser.add_argument("--synthetic-format", choices=["npy", "png"], default="npy")
    parser.add_argument("--mix-ratio", type=float, default=0.5)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/adapted_training")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    train_expression_adapted(
        regime=args.regime,
        dataset=args.dataset,
        data_root=args.data_root,
        csv_file=args.csv_file,
        synthetic_dir=args.synthetic_dir,
        synthetic_format=args.synthetic_format,
        mix_ratio=args.mix_ratio,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
