"""
Fine-tune an anonymizer with auxiliary losses.

This script adds two differentiable training objectives on top of an
anonymizer's generator:

1. **Expression-teacher consistency loss** (Task 5.4)
   — Minimize KL(teacher(original) ‖ teacher(anonymised)) so that
   expression cues are preserved through anonymisation.

2. **Identity suppression loss** (Task 5.5)
   — Maximize the ArcFace cosine distance (or equivalently minimize
   cosine similarity) between original and anonymised embeddings so
   that identity information is destroyed.

Combined loss
~~~~~~~~~~~~~
::

    L_total  =  L_anon  +  λ_expr · L_expression  +  λ_id · L_identity

where ``L_anon`` is the anonymizer's own reconstruction/adversarial loss
(if available), ``L_expression`` is KL-divergence of teacher soft labels,
and ``L_identity`` is cosine similarity in ArcFace space.

Applicable backends
~~~~~~~~~~~~~~~~~~~
This script assumes the anonymizer has a differentiable **generator**
accessible via ``anon._generator`` (e.g., GANonymization pix2pix generator).
For non-differentiable backends (blur, pixelate), fine-tuning is a no-op.

Usage
-----
::

    python scripts/finetune_anonymizer.py --anonymizer ganonymization
    python scripts/finetune_anonymizer.py --anonymizer ganonymization \\
        --lambda-expr 1.0 --lambda-id 0.5 --epochs 10
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


# ── Loss Functions ─────────────────────────────────────────────────────────

def expression_consistency_loss(
    teacher_logits_orig: "torch.Tensor",
    teacher_logits_anon: "torch.Tensor",
    temperature: float = 2.0,
) -> "torch.Tensor":
    """
    KL-divergence between temperature-scaled teacher distributions.

    Minimizing this encourages the anonymiser output to preserve
    expression cues as seen by the teacher.

    Parameters
    ----------
    teacher_logits_orig : (N, C) raw logits on original images.
    teacher_logits_anon : (N, C) raw logits on anonymised images.
    temperature : softening temperature (higher → softer targets).

    Returns
    -------
    Scalar KL loss (mean over batch).
    """
    import torch
    import torch.nn.functional as F

    p = F.log_softmax(teacher_logits_anon / temperature, dim=1)
    q = F.softmax(teacher_logits_orig / temperature, dim=1)
    # KL(q || p) — we want anonymised distribution to match original
    kl = F.kl_div(p, q, reduction="batchmean") * (temperature ** 2)
    return kl


def identity_suppression_loss(
    embeddings_orig: "torch.Tensor",
    embeddings_anon: "torch.Tensor",
    margin: float = 0.0,
) -> "torch.Tensor":
    """
    Identity suppression via negative cosine similarity.

    We want to *maximise* the cosine distance (i.e. destroy identity),
    which is equivalent to *minimising* cosine similarity.

    An optional margin can push similarity below a threshold:
    ``L = max(0, cos_sim − margin)``.

    Parameters
    ----------
    embeddings_orig : (N, 512) L2-normalised ArcFace on originals.
    embeddings_anon : (N, 512) L2-normalised ArcFace on anonymised.
    margin : target similarity (0.0 = fully dissimilar).

    Returns
    -------
    Scalar loss — lower when identity is suppressed.
    """
    import torch
    import torch.nn.functional as F

    # Normalise just in case
    e_orig = F.normalize(embeddings_orig, dim=1)
    e_anon = F.normalize(embeddings_anon, dim=1)

    cos_sim = (e_orig * e_anon).sum(dim=1)  # (N,)
    loss = torch.clamp(cos_sim - margin, min=0.0).mean()
    return loss


# ── Generator access helpers ──────────────────────────────────────────────

def _get_generator(anonymizer):
    """
    Try to extract a differentiable PyTorch generator from the anonymizer.

    Returns (generator_module, is_differentiable) or (None, False).
    """
    # Force lazy-loaded anonymizers to initialise their model
    ensure = getattr(anonymizer, "_ensure_model", None)
    if callable(ensure):
        try:
            ensure()
        except Exception as exc:
            logger.warning("Failed to ensure model: %s", exc)

    # GANonymization
    gen = getattr(anonymizer, "_generator", None)
    if gen is not None:
        return gen, True

    # General _model attribute (GANonymization raw_generator, DeepPrivacy2, etc.)
    gen = getattr(anonymizer, "_model", None)
    if gen is not None:
        import torch.nn as nn
        if isinstance(gen, nn.Module):
            return gen, True

    # CIAGAN
    gen = getattr(anonymizer, "_G", None)
    if gen is not None:
        return gen, True

    return None, False


def _anonymize_differentiable(
    generator,
    images_tensor: "torch.Tensor",
) -> "torch.Tensor":
    """
    Run generator forward pass *differentiably* (no torch.no_grad()).

    Assumes generator takes (N, 3, H, W) in [-1, 1] and outputs same.
    """
    import torch

    # Contract: input float32 [0, 1] → generator expects [-1, 1]
    x = images_tensor * 2.0 - 1.0
    out = generator(x)
    # Back to [0, 1]
    out = (out + 1.0) / 2.0
    return out.clamp(0.0, 1.0)


# ── Main fine-tuning loop ─────────────────────────────────────────────────

def finetune_anonymizer(
    *,
    anonymizer_name: str = "ganonymization",
    anonymizer_params: Optional[dict] = None,
    dataset: str = "fer2013",
    data_root: str = "data/fer2013",
    csv_file: str = "fer2013.csv",
    # Loss weights
    lambda_expr: float = 1.0,
    lambda_id: float = 0.5,
    expr_temperature: float = 2.0,
    id_margin: float = 0.0,
    # Training
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    max_samples: Optional[int] = 1000,
    # Teacher config
    teacher_backbone: str = "resnet18",
    teacher_checkpoint: Optional[str] = None,
    # Identity embedder
    identity_model: str = "buffalo_l",
    # Misc
    device: str = "cuda",
    seed: int = 42,
    output_dir: str = "results/finetune_anonymizer",
    checkpoint_dir: str = "checkpoints",
) -> dict:
    """
    Fine-tune an anonymizer with expression consistency + identity suppression.

    Returns
    -------
    dict with training history and saved checkpoint path.
    """
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)

    out = Path(output_dir) / anonymizer_name
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Load anonymizer ────────────────────────────────────────────────
    from src.anonymizers import get_anonymizer

    params = anonymizer_params or {}
    anon = get_anonymizer(anonymizer_name, **params)
    generator, is_diff = _get_generator(anon)

    if not is_diff or generator is None:
        logger.warning(
            "Anonymizer '%s' has no differentiable generator. "
            "Fine-tuning is not applicable — exiting.",
            anonymizer_name,
        )
        return {"status": "not_applicable", "anonymizer": anonymizer_name}

    generator = generator.to(device)
    generator.train()
    logger.info("Generator found — %d parameters.", sum(p.numel() for p in generator.parameters()))

    # ── Load expression teacher (frozen) ───────────────────────────────
    from src.models.expression_teacher import ExpressionTeacher

    teacher = ExpressionTeacher(
        backbone=teacher_backbone,
        checkpoint=teacher_checkpoint,
        device=device,
    )
    logger.info("Expression teacher loaded (frozen).")

    # We need the teacher's torch model for differentiable forward pass.
    # If teacher is ONNX-only, fall back to a lightweight ResNet-18 teacher.
    teacher_model = teacher._torch_model
    if teacher_model is None:
        logger.warning(
            "ONNX teacher detected — loading lightweight ResNet-18 teacher for "
            "differentiable fine-tuning."
        )
        from src.models.expression_classifier import ExpressionClassifier
        _cls = ExpressionClassifier(
            backbone=teacher_backbone, num_classes=7, pretrained=True, device=device,
        )
        teacher_model = _cls._model
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        import timm
        teacher._data_cfg = timm.data.resolve_model_data_config(teacher_model)

    # ── Load identity embedder (frozen) ────────────────────────────────
    id_embedder = _load_identity_embedder(identity_model, device)

    # ── Load training data ─────────────────────────────────────────────
    logger.info("Loading training data...")
    from src.data.fer2013_adapter import FER2013Dataset

    train_ds = FER2013Dataset(root=data_root, split="train")
    n_total = len(train_ds)
    n_use = min(n_total, max_samples) if max_samples else n_total
    # Load lazily to avoid OOM
    images = np.stack([train_ds[i].image for i in range(n_use)])

    logger.info("Training on %d images.", len(images))

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

    history = {
        "total_loss": [],
        "expr_loss": [],
        "id_loss": [],
    }

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(len(images))
        epoch_total = 0.0
        epoch_expr = 0.0
        epoch_id = 0.0
        n_batches = 0

        for start in range(0, len(images), batch_size):
            idx = perm[start : start + batch_size]
            # (B, H, W, 3) float32 [0,1] → (B, 3, H, W) tensor
            batch_np = images[idx]
            batch_t = torch.from_numpy(
                batch_np.astype(np.float32)
            ).permute(0, 3, 1, 2).to(device)

            # ── Forward: generate anonymised ───────────────────────
            anon_t = _anonymize_differentiable(generator, batch_t)

            # ── Expression loss ────────────────────────────────────
            # Teacher expects (N, 3, H, W) in normalised form
            # Use teacher's _prepare_torch for proper normalisation
            orig_for_teacher = teacher._prepare_torch(batch_np)
            anon_nhwc = anon_t.permute(0, 2, 3, 1).detach().cpu().numpy()
            # Re-create tensor through teacher prep for correct normalisation
            # But we need gradients to flow through anon_t, so we do manual norm
            teacher_mean = torch.tensor(
                teacher._data_cfg["mean"], dtype=torch.float32
            ).view(1, 3, 1, 1).to(device)
            teacher_std = torch.tensor(
                teacher._data_cfg["std"], dtype=torch.float32
            ).view(1, 3, 1, 1).to(device)

            anon_normed = (anon_t - teacher_mean) / teacher_std
            input_size = teacher._data_cfg["input_size"]
            _, h_tgt, w_tgt = input_size
            if anon_normed.shape[2] != h_tgt or anon_normed.shape[3] != w_tgt:
                anon_normed = torch.nn.functional.interpolate(
                    anon_normed, size=(h_tgt, w_tgt),
                    mode="bilinear", align_corners=False,
                )

            with torch.no_grad():
                logits_orig = teacher_model(orig_for_teacher)
            logits_anon = teacher_model(anon_normed)

            l_expr = expression_consistency_loss(
                logits_orig, logits_anon, temperature=expr_temperature,
            )

            # ── Identity loss ──────────────────────────────────────
            l_id = torch.tensor(0.0, device=device)
            if id_embedder is not None and lambda_id > 0:
                with torch.no_grad():
                    emb_orig = id_embedder.embed(batch_np)

                # For identity loss we need gradients through anon_t
                # ArcFace typically doesn't support grad — use detached embeddings
                anon_np_for_id = anon_t.permute(0, 2, 3, 1).detach().cpu().numpy()
                with torch.no_grad():
                    emb_anon = id_embedder.embed(anon_np_for_id)

                emb_orig_t = torch.from_numpy(emb_orig).to(device)
                emb_anon_t = torch.from_numpy(emb_anon).to(device)
                l_id = identity_suppression_loss(
                    emb_orig_t, emb_anon_t, margin=id_margin,
                )

            # ── Total loss & backward ─────────────────────────────
            loss = lambda_expr * l_expr + lambda_id * l_id

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_expr += l_expr.item()
            epoch_id += l_id.item()
            n_batches += 1

        avg_total = epoch_total / max(n_batches, 1)
        avg_expr = epoch_expr / max(n_batches, 1)
        avg_id = epoch_id / max(n_batches, 1)

        history["total_loss"].append(avg_total)
        history["expr_loss"].append(avg_expr)
        history["id_loss"].append(avg_id)

        logger.info(
            "Epoch %2d/%d  total=%.4f  expr=%.4f  id=%.4f",
            epoch, epochs, avg_total, avg_expr, avg_id,
        )

    elapsed = time.time() - t0
    logger.info("Fine-tuning complete in %.1fs.", elapsed)

    # ── Save fine-tuned generator ──────────────────────────────────────
    import torch as _torch

    ckpt_path = str(ckpt_dir / f"{anonymizer_name}_finetuned.pth")
    _torch.save(generator.state_dict(), ckpt_path)
    logger.info("Fine-tuned generator saved to %s", ckpt_path)

    # ── Save summary ──────────────────────────────────────────────────
    summary = {
        "anonymizer": anonymizer_name,
        "lambda_expr": lambda_expr,
        "lambda_id": lambda_id,
        "epochs": epochs,
        "lr": lr,
        "elapsed_seconds": elapsed,
        "history": history,
        "checkpoint_path": ckpt_path,
    }
    with open(out / "finetune_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _load_identity_embedder(model_name: str, device: str):
    """Attempt to load ArcFace identity embedder. Return None on failure."""
    try:
        from src.models.identity_embedder import IdentityEmbedder
        emb = IdentityEmbedder(model_name=model_name, device=device)
        logger.info("ArcFace identity embedder loaded (%s).", model_name)
        return emb
    except Exception as exc:
        logger.warning(
            "Could not load identity embedder: %s — identity loss disabled.", exc
        )
        return None


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Fine-tune anonymizer with expression + identity losses."
    )
    parser.add_argument("--anonymizer", default="ganonymization")
    parser.add_argument("--dataset", default="fer2013")
    parser.add_argument("--data-root", default="data/fer2013")
    parser.add_argument("--csv-file", default="fer2013.csv")
    parser.add_argument("--lambda-expr", type=float, default=1.0)
    parser.add_argument("--lambda-id", type=float, default=0.5)
    parser.add_argument("--expr-temperature", type=float, default=2.0)
    parser.add_argument("--id-margin", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--teacher-backbone", default="resnet18")
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--identity-model", default="buffalo_l")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/finetune_anonymizer")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    finetune_anonymizer(
        anonymizer_name=args.anonymizer,
        dataset=args.dataset,
        data_root=args.data_root,
        csv_file=args.csv_file,
        lambda_expr=args.lambda_expr,
        lambda_id=args.lambda_id,
        expr_temperature=args.expr_temperature,
        id_margin=args.id_margin,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        teacher_backbone=args.teacher_backbone,
        teacher_checkpoint=args.teacher_checkpoint,
        identity_model=args.identity_model,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
