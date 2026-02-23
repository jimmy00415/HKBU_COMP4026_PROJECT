"""
Ablation — Adaptive attacker robustness.

For the best anonymizer (or a user-specified one), compare:
  1. **fixed_arcface**    — Standard ArcFace evaluation (closed-set + ROC)
  2. **adaptive_attacker** — Lightweight MLP trained end-to-end on
     anonymised embeddings to re-identify subjects.

The key question: does privacy measured with a fixed ArcFace metric
*overestimate* anonymisation strength?  The adaptive attacker is a
stronger adversary that can learn to invert anonymisation-specific
artefacts.

Usage
-----
::

    python scripts/run_adaptive_attacker_ablation.py
    python scripts/run_adaptive_attacker_ablation.py --anonymizer blur --anon-param kernel_size=51
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Anonymizer configurations to probe
DEFAULT_CONFIGS: list[dict[str, Any]] = [
    {"name": "blur",     "params": {"kernel_size": 31}},
    {"name": "blur",     "params": {"kernel_size": 71}},
    {"name": "pixelate", "params": {"block_size": 8}},
    {"name": "pixelate", "params": {"block_size": 24}},
    {"name": "blackout", "params": {}},
]


def run_adaptive_attacker_ablation(
    *,
    configs: Optional[list[dict[str, Any]]] = None,
    dataset: str = "fer2013",
    data_root: str = "data/fer2013",
    csv_file: str = "fer2013.csv",
    max_samples: Optional[int] = None,
    device: str = "cpu",
    seed: int = 42,
    attacker_epochs: int = 20,
    attacker_hidden: Optional[list[int]] = None,
    attacker_dropout: float = 0.3,
    output_dir: str = "results/adaptive_attacker_ablation",
) -> list[dict]:
    """Run the adaptive attacker ablation."""
    np.random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if configs is None:
        configs = DEFAULT_CONFIGS

    from scripts.run_baseline_sweep import (
        _anonymize_dataset,
        _get_identity_embeddings,
        _load_data,
        _make_run_id,
    )

    images, expr_labels, id_labels = _load_data(
        dataset, data_root, csv_file, max_samples, seed,
    )
    N = len(images)
    logger.info("Loaded %d images.", N)

    original_embeddings = _get_identity_embeddings(images, device)
    if original_embeddings is None:
        logger.error("Identity embeddings unavailable — cannot run ablation.")
        return []

    if id_labels is None:
        logger.error("Identity labels required for adaptive attacker ablation.")
        return []

    all_results: list[dict] = []

    for cfg in configs:
        anon_name = cfg["name"]
        params = cfg.get("params", {})
        run_id = _make_run_id(anon_name, params)
        logger.info("━━━ %s ━━━", run_id)
        t0 = time.time()

        # Anonymize
        anon_images = _anonymize_dataset(images, anon_name, params)
        anon_embeddings = _get_identity_embeddings(anon_images, device)

        if anon_embeddings is None:
            logger.warning("  Embeddings unavailable for %s — skipping.", run_id)
            continue

        # ── Method A: Fixed ArcFace metrics ────────────────────────────
        fixed_metrics = _evaluate_fixed_arcface(
            original_embeddings, anon_embeddings, id_labels,
        )

        # ── Method B: Adaptive attacker ────────────────────────────────
        adaptive_metrics = _evaluate_adaptive_attacker(
            anon_embeddings, id_labels,
            epochs=attacker_epochs,
            hidden=attacker_hidden,
            dropout=attacker_dropout,
            device=device,
        )

        # ── Compare ───────────────────────────────────────────────────
        row = {
            "_run_id": run_id,
            "_anonymizer": anon_name,
            "_params": json.dumps(params),
            "fixed_top1": fixed_metrics["top1"],
            "fixed_top5": fixed_metrics["top5"],
            "fixed_auc": fixed_metrics["auc"],
            "fixed_eer": fixed_metrics["eer"],
            "adaptive_accuracy": adaptive_metrics["accuracy"],
            "adaptive_trained": adaptive_metrics["trained"],
            "privacy_gap": adaptive_metrics["accuracy"] - fixed_metrics["top1"],
            "_elapsed": time.time() - t0,
        }
        all_results.append(row)

        logger.info(
            "  Fixed top-1=%.4f | Adaptive=%.4f | Gap=%.4f",
            row["fixed_top1"], row["adaptive_accuracy"], row["privacy_gap"],
        )

    _write_comparison_table(all_results, out / "comparison_table.md")
    _write_comparison_csv(all_results, out / "comparison.csv")
    logger.info("Adaptive attacker ablation complete — %d configs.", len(all_results))
    return all_results


def _evaluate_fixed_arcface(
    original_embeddings: np.ndarray,
    anonymized_embeddings: np.ndarray,
    identity_labels: np.ndarray,
) -> dict[str, float]:
    """Evaluate privacy using fixed ArcFace metrics."""
    from src.models.identity_metrics import (
        closed_set_identification,
        compute_verification_metrics,
        generate_verification_pairs,
    )

    id_results = closed_set_identification(
        probe_embeddings=anonymized_embeddings,
        probe_labels=identity_labels,
        gallery_embeddings=original_embeddings,
        gallery_labels=identity_labels,
        top_k=5,
    )

    N = len(identity_labels)
    auc, eer = 0.5, 0.5
    try:
        emb_a, emb_b, is_same_arr = generate_verification_pairs(
            anonymized_embeddings, identity_labels,
            num_pairs=min(5000, N * 10), seed=42,
        )
        if len(emb_a) > 0:
            verif = compute_verification_metrics(
                embeddings_a=emb_a,
                embeddings_b=emb_b,
                is_same=is_same_arr,
                far_targets=(0.01,),
            )
            auc = verif.get("auc", 0.5)
            eer = verif.get("eer", 0.5)
    except ValueError as exc:
        logger.warning("Verification pairs failed: %s", exc)

    return {
        "top1": id_results.get("top1_acc", 0.0),
        "top5": id_results.get("top5_acc", 0.0),
        "auc": auc,
        "eer": eer,
    }


def _evaluate_adaptive_attacker(
    anonymized_embeddings: np.ndarray,
    identity_labels: np.ndarray,
    *,
    epochs: int = 20,
    hidden: Optional[list[int]] = None,
    dropout: float = 0.3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train an adaptive attacker MLP on anonymised embeddings."""
    N = len(identity_labels)
    if N < 20:
        return {"accuracy": 0.0, "trained": False}

    try:
        from src.models.adaptive_attacker import AdaptiveAttackerHead

        num_ids = int(identity_labels.max()) + 1
        attacker = AdaptiveAttackerHead(
            embedding_dim=anonymized_embeddings.shape[1],
            num_classes=num_ids,
            hidden_dims=hidden or [256, 128],
            dropout=dropout,
            device=device,
        )

        rng = np.random.RandomState(42)
        perm = rng.permutation(N)
        split = int(0.8 * N)
        train_idx, test_idx = perm[:split], perm[split:]

        attacker.fit(
            embeddings=anonymized_embeddings[train_idx],
            labels=identity_labels[train_idx],
            epochs=epochs,
            val_embeddings=anonymized_embeddings[test_idx],
            val_labels=identity_labels[test_idx],
            verbose=False,
        )

        accuracy = attacker.evaluate(
            anonymized_embeddings[test_idx], identity_labels[test_idx],
        )
        return {"accuracy": accuracy, "trained": True}

    except Exception as exc:
        logger.warning("Adaptive attacker failed: %s", exc)
        return {"accuracy": 0.0, "trained": False}


def _write_comparison_table(results: list[dict], path: Path) -> None:
    """Write a Markdown comparison table."""
    lines = [
        "# Adaptive Attacker Robustness Ablation",
        "",
        "Does fixed ArcFace overestimate anonymisation strength?",
        "",
        "| Anonymizer | Fixed Top-1 | Fixed AUC | Adaptive Acc | Privacy Gap |",
        "|------------|-------------|-----------|--------------|-------------|",
    ]
    for r in results:
        run_id = r.get("_run_id", "?")
        lines.append(
            f"| {run_id} "
            f"| {r.get('fixed_top1', 0):.4f} "
            f"| {r.get('fixed_auc', 0):.4f} "
            f"| {r.get('adaptive_accuracy', 0):.4f} "
            f"| {r.get('privacy_gap', 0):+.4f} |"
        )

    lines.extend([
        "",
        "**Privacy Gap** = Adaptive Accuracy − Fixed Top-1.  "
        "A *positive* gap means the fixed metric overestimates privacy "
        "(the adaptive attacker recovers more identity information).",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Comparison table: %s", path)


def _write_comparison_csv(results: list[dict], path: Path) -> None:
    """Write a CSV for downstream analysis."""
    import csv

    if not results:
        return
    fieldnames = [
        "_run_id", "_anonymizer", "_params",
        "fixed_top1", "fixed_top5", "fixed_auc", "fixed_eer",
        "adaptive_accuracy", "adaptive_trained", "privacy_gap",
        "_elapsed",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info("Comparison CSV: %s", path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="Ablation: adaptive attacker robustness")
    p.add_argument("--dataset", default="fer2013")
    p.add_argument("--data-root", default="data/fer2013")
    p.add_argument("--csv-file", default="fer2013.csv")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attacker-epochs", type=int, default=20)
    p.add_argument("--attacker-dropout", type=float, default=0.3)
    p.add_argument("--output-dir", default="results/adaptive_attacker_ablation")
    args = p.parse_args()

    run_adaptive_attacker_ablation(
        dataset=args.dataset,
        data_root=args.data_root,
        csv_file=args.csv_file,
        max_samples=args.max_samples,
        device=args.device,
        seed=args.seed,
        attacker_epochs=args.attacker_epochs,
        attacker_dropout=args.attacker_dropout,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
