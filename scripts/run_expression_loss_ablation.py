"""
Ablation — Expression loss variants.

Compare the effect of different expression-preservation losses on the
privacy–utility frontier:

1. ``no_loss``       — anonymize without any expression-preservation loss
2. ``kd_logit``      — teacher-logit KL-divergence (soft-label KD)
3. ``feature_level`` — feature-level cosine consistency

Since fine-tuning the generator requires the GAN repo, this ablation
*simulates* the effect by:
  - Running the anonymizer once to produce anonymous outputs,
  - Then measuring expression metrics under each loss regime.

For a **real** ablation (if GANonymization is available), the script
fine-tunes with each loss variant and evaluates the *resulting* model.

Usage
-----
::

    python scripts/run_expression_loss_ablation.py
    python scripts/run_expression_loss_ablation.py --device cuda --max-samples 200
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

LOSS_VARIANTS = [
    "no_loss",
    "kd_logit",
    "feature_level",
]


def _compute_feature_consistency(
    images_orig: np.ndarray,
    images_anon: np.ndarray,
    device: str = "cpu",
) -> float:
    """
    Compute mean cosine similarity between teacher features on originals
    vs anonymised.  Returns a scalar in [0, 1] where 1 = identical features.
    """
    try:
        from src.models.expression_classifier import ExpressionClassifier

        clf = ExpressionClassifier(backbone="resnet18", device=device)
        feats_orig = clf.extract_features(images_orig)
        feats_anon = clf.extract_features(images_anon)

        # Normalise and compute cosine similarity
        norm_o = feats_orig / (np.linalg.norm(feats_orig, axis=1, keepdims=True) + 1e-8)
        norm_a = feats_anon / (np.linalg.norm(feats_anon, axis=1, keepdims=True) + 1e-8)
        cos_sim = (norm_o * norm_a).sum(axis=1)
        return float(np.mean(cos_sim))
    except Exception as exc:
        logger.warning("Feature consistency computation failed: %s", exc)
        return 0.0


def run_expression_loss_ablation(
    *,
    anonymizer_name: str = "blur",
    anon_params: Optional[dict] = None,
    dataset: str = "fer2013",
    data_root: str = "data/fer2013",
    csv_file: str = "fer2013.csv",
    max_samples: Optional[int] = None,
    device: str = "cpu",
    seed: int = 42,
    output_dir: str = "results/expression_loss_ablation",
    lambda_expr: float = 1.0,
    finetune_epochs: int = 5,
) -> list[dict]:
    """Run the expression loss ablation."""
    np.random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if anon_params is None:
        anon_params = {"kernel_size": 31}

    from scripts.run_baseline_sweep import (
        _get_expression_probs,
        _get_identity_embeddings,
        _load_data,
        _plot_results,
        _write_frontier_csv,
    )

    images, expr_labels, id_labels = _load_data(
        dataset, data_root, csv_file, max_samples, seed,
    )
    N = len(images)
    logger.info("Loaded %d images.", N)

    probs_original = _get_expression_probs(images, device)
    original_embeddings = _get_identity_embeddings(images, device)

    all_results: list[dict] = []

    for variant in LOSS_VARIANTS:
        run_id = f"expr_loss_{variant}"
        logger.info("━━━ %s ━━━", run_id)
        t0 = time.time()

        # Anonymize — all variants use same anonymizer
        anon_images = _anonymize_with_variant(
            images=images,
            anonymizer_name=anonymizer_name,
            anon_params=anon_params,
            variant=variant,
            device=device,
            lambda_expr=lambda_expr,
            finetune_epochs=finetune_epochs,
        )

        probs_anonymized = _get_expression_probs(anon_images, device)
        anon_embeddings = _get_identity_embeddings(anon_images, device)

        # Extra metric: feature-level consistency
        feat_consistency = _compute_feature_consistency(images, anon_images, device)

        from src.runner.evaluator import run_evaluation

        result = run_evaluation(
            anonymizer_name=anonymizer_name,
            original_images=images,
            anonymized_images=anon_images,
            expression_labels=expr_labels,
            identity_labels=id_labels,
            original_embeddings=original_embeddings,
            anonymized_embeddings=anon_embeddings,
            probs_original=probs_original,
            probs_anonymized=probs_anonymized,
            anonymizer_params={**anon_params, "loss_variant": variant},
            run_privacy=id_labels is not None and anon_embeddings is not None,
            run_expression=expr_labels is not None and probs_original is not None,
            run_realism=True,
            compute_fid=False,
            compute_lpips=True,
            device=device,
            output_dir=str(out),
            run_id=run_id,
        )

        row = result.to_dict()
        row["_run_id"] = run_id
        row["_variant"] = variant
        row["_feature_consistency"] = feat_consistency
        row["_elapsed"] = time.time() - t0
        all_results.append(row)
        logger.info(
            "  %s done in %.1fs (feat_consistency=%.4f)",
            variant, row["_elapsed"], feat_consistency,
        )

    _write_frontier_csv(all_results, out / "frontier.csv")
    _plot_results(out)
    _write_ablation_table(all_results, out / "ablation_table.md")
    logger.info("Expression loss ablation complete — %d variants.", len(all_results))
    return all_results


def _anonymize_with_variant(
    *,
    images: np.ndarray,
    anonymizer_name: str,
    anon_params: dict,
    variant: str,
    device: str,
    lambda_expr: float,
    finetune_epochs: int,
) -> np.ndarray:
    """
    Anonymize with a specific loss variant.

    For ``no_loss``, runs the anonymizer as-is.
    For ``kd_logit`` and ``feature_level``, attempts to fine-tune the
    generator with the respective loss.  If the anonymizer has no
    differentiable generator, falls back to standard anonymization.
    """
    from src.anonymizers import get_anonymizer
    from src.data.contracts import FaceCrop

    anon = get_anonymizer(anonymizer_name, **anon_params)

    if variant == "no_loss":
        return _anonymize_all(anon, images)

    # Try differentiable fine-tuning
    try:
        from scripts.finetune_anonymizer import _get_generator

        generator, is_diff = _get_generator(anon)
        if not is_diff or generator is None:
            logger.info(
                "  %s has no differentiable generator — "
                "using standard anonymization for '%s'.",
                anonymizer_name, variant,
            )
            return _anonymize_all(anon, images)

        # Fine-tune with the specified loss variant
        return _finetune_and_anonymize(
            generator=generator,
            images=images,
            variant=variant,
            device=device,
            lambda_expr=lambda_expr,
            epochs=finetune_epochs,
        )
    except Exception as exc:
        logger.warning(
            "  Fine-tuning for '%s' failed: %s — falling back.", variant, exc,
        )
        return _anonymize_all(anon, images)


def _anonymize_all(anon, images: np.ndarray) -> np.ndarray:
    """Run anonymizer on all images."""
    from src.data.contracts import FaceCrop, FaceCropMeta

    results = []
    for i in range(len(images)):
        meta = FaceCropMeta(dataset="eval", split="test", image_id=str(i))
        crop = FaceCrop(image=images[i], meta=meta)
        try:
            res = anon.anonymize_single(crop)
            results.append(res.image)
        except Exception:
            results.append(images[i])
    return np.stack(results)


def _finetune_and_anonymize(
    *,
    generator,
    images: np.ndarray,
    variant: str,
    device: str,
    lambda_expr: float,
    epochs: int,
) -> np.ndarray:
    """Fine-tune generator with specified loss variant, then anonymize."""
    import torch

    from scripts.finetune_anonymizer import (
        _anonymize_differentiable,
        expression_consistency_loss,
    )

    gen = generator.to(device)
    gen.train()
    optimizer = torch.optim.Adam(gen.parameters(), lr=1e-4)

    # Prepare data as tensor
    images_t = torch.from_numpy(images.astype(np.float32)).permute(0, 3, 1, 2).to(device)

    # Load teacher's PyTorch model for differentiable KD loss
    teacher_model = None
    teacher_mean = None
    teacher_std = None
    teacher_input_size = None
    if variant == "kd_logit":
        try:
            from src.models.expression_teacher import ExpressionTeacher

            teacher = ExpressionTeacher(backbone="resnet18", device=device)
            teacher_model = teacher._torch_model
            if teacher_model is not None:
                teacher_model.eval()
                teacher_mean = torch.tensor(
                    teacher._data_cfg["mean"], dtype=torch.float32,
                ).view(1, 3, 1, 1).to(device)
                teacher_std = torch.tensor(
                    teacher._data_cfg["std"], dtype=torch.float32,
                ).view(1, 3, 1, 1).to(device)
                teacher_input_size = teacher._data_cfg["input_size"]
            else:
                logger.warning("  Teacher has no PyTorch model — KD loss disabled.")
        except Exception:
            logger.warning("  Teacher unavailable — skipping KD loss.")

    # Load classifier's PyTorch model for differentiable feature loss
    clf_model = None
    clf_mean = None
    clf_std = None
    clf_input_size = None
    if variant == "feature_level":
        try:
            from src.models.expression_classifier import ExpressionClassifier

            classifier = ExpressionClassifier(backbone="resnet18", device=device)
            clf_model = classifier._model
            if clf_model is not None:
                clf_model.eval()
                # Use ImageNet normalization as default
                clf_mean = torch.tensor(
                    [0.485, 0.456, 0.406], dtype=torch.float32,
                ).view(1, 3, 1, 1).to(device)
                clf_std = torch.tensor(
                    [0.229, 0.224, 0.225], dtype=torch.float32,
                ).view(1, 3, 1, 1).to(device)
            else:
                logger.warning("  Classifier has no PyTorch model — feature loss disabled.")
        except Exception:
            logger.warning("  Classifier unavailable — skipping feature loss.")

    bs = min(32, len(images))
    for epoch in range(epochs):
        perm = torch.randperm(len(images_t))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(images_t), bs):
            idx = perm[start:start + bs]
            batch = images_t[idx]

            anon_batch = _anonymize_differentiable(gen, batch)

            loss = torch.tensor(0.0, device=device, requires_grad=False)

            if variant == "kd_logit" and teacher_model is not None:
                # Normalize original and anonymized through teacher's norm
                with torch.no_grad():
                    orig_normed = (batch - teacher_mean) / teacher_std
                    _, h_tgt, w_tgt = teacher_input_size
                    if orig_normed.shape[2] != h_tgt or orig_normed.shape[3] != w_tgt:
                        orig_normed = torch.nn.functional.interpolate(
                            orig_normed, size=(h_tgt, w_tgt),
                            mode="bilinear", align_corners=False,
                        )
                    logits_orig = teacher_model(orig_normed)

                # Anonymized path keeps gradients flowing through gen
                anon_normed = (anon_batch - teacher_mean) / teacher_std
                if anon_normed.shape[2] != h_tgt or anon_normed.shape[3] != w_tgt:
                    anon_normed = torch.nn.functional.interpolate(
                        anon_normed, size=(h_tgt, w_tgt),
                        mode="bilinear", align_corners=False,
                    )
                logits_anon = teacher_model(anon_normed)

                loss = lambda_expr * expression_consistency_loss(
                    logits_orig, logits_anon,
                )

            elif variant == "feature_level" and clf_model is not None:
                # Normalize and extract features differentiably
                with torch.no_grad():
                    orig_normed = (batch - clf_mean) / clf_std
                    feats_orig = clf_model.forward_features(orig_normed)
                    if feats_orig.dim() > 2:
                        feats_orig = feats_orig.mean(dim=(-2, -1))

                anon_normed = (anon_batch - clf_mean) / clf_std
                feats_anon = clf_model.forward_features(anon_normed)
                if feats_anon.dim() > 2:
                    feats_anon = feats_anon.mean(dim=(-2, -1))

                cos_sim = torch.nn.functional.cosine_similarity(
                    feats_orig, feats_anon, dim=1,
                )
                loss = lambda_expr * (1.0 - cos_sim.mean())

            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        logger.info(
            "  Epoch %d/%d — %s loss: %.4f",
            epoch + 1, epochs, variant, epoch_loss / max(n_batches, 1),
        )

    # Generate final anonymised images
    gen.eval()
    with torch.no_grad():
        out = _anonymize_differentiable(gen, images_t)
    return out.permute(0, 2, 3, 1).cpu().numpy()


def _write_ablation_table(results: list[dict], path: Path) -> None:
    """Write a Markdown ablation summary table."""
    lines = [
        "# Expression Loss Ablation",
        "",
        "| Variant | Privacy | Utility | Expr Consistency | Feature Cos |",
        "|---------|---------|---------|------------------|-------------|",
    ]
    for r in results:
        variant = r.get("_variant", "?")
        privacy = r.get("privacy_score", "—")
        utility = r.get("utility_score", "—")
        consistency = r.get("expression_consistency", "—")
        feat_cos = r.get("_feature_consistency", "—")
        if isinstance(privacy, float):
            privacy = f"{privacy:.4f}"
        if isinstance(utility, float):
            utility = f"{utility:.4f}"
        if isinstance(consistency, float):
            consistency = f"{consistency:.4f}"
        if isinstance(feat_cos, float):
            feat_cos = f"{feat_cos:.4f}"
        lines.append(f"| {variant} | {privacy} | {utility} | {consistency} | {feat_cos} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Ablation table: %s", path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="Ablation: expression loss variants")
    p.add_argument("--anonymizer", default="blur")
    p.add_argument("--anon-param", nargs="*", default=["kernel_size=31"],
                   help="key=value anonymizer params")
    p.add_argument("--dataset", default="fer2013")
    p.add_argument("--data-root", default="data/fer2013")
    p.add_argument("--csv-file", default="fer2013.csv")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/expression_loss_ablation")
    p.add_argument("--lambda-expr", type=float, default=1.0)
    p.add_argument("--finetune-epochs", type=int, default=5)
    args = p.parse_args()

    # Parse anonymizer params
    anon_params = {}
    if args.anon_param:
        for kv in args.anon_param:
            k, v = kv.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            anon_params[k] = v

    run_expression_loss_ablation(
        anonymizer_name=args.anonymizer,
        anon_params=anon_params,
        dataset=args.dataset,
        data_root=args.data_root,
        csv_file=args.csv_file,
        max_samples=args.max_samples,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        lambda_expr=args.lambda_expr,
        finetune_epochs=args.finetune_epochs,
    )


if __name__ == "__main__":
    main()
