"""
Unified evaluation runner.

Takes an anonymizer name + config, runs all three metric suites (privacy,
expression, realism) on a dataset, and persists results to
``results/run_{id}/``.

Output structure per run::

    results/run_{id}/
        metrics.json      — full metrics dictionary
        frontier_row.csv  — single-row CSV for frontier aggregation
        config.yaml       — Hydra config snapshot
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Aggregated result container."""

    privacy: dict = field(default_factory=dict)
    expression: dict = field(default_factory=dict)
    realism: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all evaluation results to a flat dictionary."""
        return {
            "privacy": self.privacy,
            "expression": self.expression,
            "realism": self.realism,
            "meta": self.meta,
        }


def run_evaluation(
    anonymizer_name: str,
    original_images: np.ndarray,
    anonymized_images: np.ndarray,
    *,
    expression_labels: Optional[np.ndarray] = None,
    identity_labels: Optional[np.ndarray] = None,
    original_embeddings: Optional[np.ndarray] = None,
    anonymized_embeddings: Optional[np.ndarray] = None,
    probs_original: Optional[np.ndarray] = None,
    probs_anonymized: Optional[np.ndarray] = None,
    anonymizer_params: Optional[dict] = None,
    run_privacy: bool = True,
    run_expression: bool = True,
    run_realism: bool = True,
    compute_fid: bool = False,
    compute_lpips: bool = True,
    train_adaptive: bool = True,
    device: str = "cpu",
    output_dir: str = "results",
    run_id: Optional[str] = None,
) -> EvaluationResult:
    """
    Execute the full evaluation harness.

    Parameters
    ----------
    anonymizer_name   : name of the anonymizer being evaluated.
    original_images   : (N, 256, 256, 3) float32 [0,1].
    anonymized_images : (N, 256, 256, 3) float32 [0,1].
    expression_labels : (N,) ground-truth expression labels (for utility).
    identity_labels   : (N,) identity labels (for privacy).
    original_embeddings  : (N, 512) ArcFace on original.
    anonymized_embeddings: (N, 512) ArcFace on anonymized.
    probs_original    : (N, C) expression softmax on original.
    probs_anonymized  : (N, C) expression softmax on anonymised.
    anonymizer_params : dict of anonymizer settings.
    run_privacy / run_expression / run_realism : which suites to run.
    compute_fid       : whether to compute FID (slow, saves to disk).
    compute_lpips     : whether to compute LPIPS (requires torch+lpips).
    train_adaptive    : whether to train adaptive attacker.
    device            : ``"cuda"`` or ``"cpu"``.
    output_dir        : root output directory.
    run_id            : unique identifier for this run (auto-generated if None).

    Returns
    -------
    EvaluationResult
    """
    if run_id is None:
        run_id = f"{anonymizer_name}_{int(time.time())}"

    run_dir = Path(output_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = EvaluationResult()
    result.meta = {
        "anonymizer_name": anonymizer_name,
        "anonymizer_params": anonymizer_params or {},
        "num_samples": int(original_images.shape[0]),
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    N = original_images.shape[0]

    # ── Privacy ────────────────────────────────────────────────────────
    if run_privacy and identity_labels is not None:
        if original_embeddings is None or anonymized_embeddings is None:
            logger.warning(
                "Skipping privacy metrics — embeddings not provided. "
                "Extract ArcFace embeddings first."
            )
        else:
            from src.metrics.privacy_metrics import evaluate_privacy

            privacy_report = evaluate_privacy(
                original_embeddings=original_embeddings,
                anonymized_embeddings=anonymized_embeddings,
                identity_labels=identity_labels,
                train_adaptive=train_adaptive,
                device=device,
                anonymizer_name=anonymizer_name,
                anonymizer_params=anonymizer_params,
            )
            result.privacy = privacy_report.to_dict()

    # ── Expression ─────────────────────────────────────────────────────
    if run_expression and expression_labels is not None:
        if probs_original is None or probs_anonymized is None:
            logger.warning(
                "Skipping expression metrics — softmax probabilities not provided. "
                "Run expression classifier/teacher first."
            )
        else:
            from src.metrics.expression_metrics import evaluate_expression

            expr_report = evaluate_expression(
                y_true=expression_labels,
                probs_original=probs_original,
                probs_anonymized=probs_anonymized,
                anonymizer_name=anonymizer_name,
            )
            result.expression = expr_report.to_dict()

    # ── Realism ────────────────────────────────────────────────────────
    if run_realism:
        from src.metrics.realism_metrics import evaluate_realism

        realism_report = evaluate_realism(
            images_original=original_images,
            images_anonymized=anonymized_images,
            compute_fid_score=compute_fid,
            compute_lpips_score=compute_lpips,
            device=device,
            anonymizer_name=anonymizer_name,
        )
        result.realism = realism_report.to_dict()

    # ── Persist ────────────────────────────────────────────────────────
    _save_results(result, run_dir)

    return result


def _save_results(result: EvaluationResult, run_dir: Path) -> None:
    """Write metrics.json and frontier_row.csv."""

    # ── metrics.json ───────────────────────────────────────────────────
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=_json_default)
    logger.info("Metrics saved to %s", metrics_path)

    # ── frontier_row.csv ───────────────────────────────────────────────
    csv_path = run_dir / "frontier_row.csv"
    row = _extract_frontier_row(result)
    header = ",".join(row.keys())
    values = ",".join(str(v) for v in row.values())
    with open(csv_path, "w") as f:
        f.write(header + "\n")
        f.write(values + "\n")
    logger.info("Frontier row saved to %s", csv_path)


def _extract_frontier_row(result: EvaluationResult) -> dict[str, Any]:
    """
    Extract the key scalar metrics into a flat dict for CSV aggregation.
    """
    privacy = result.privacy
    expr = result.expression
    realism = result.realism
    meta = result.meta

    return {
        "anonymizer": meta.get("anonymizer_name", ""),
        "params": json.dumps(meta.get("anonymizer_params", {})),
        "n_samples": meta.get("num_samples", 0),
        # Privacy
        "closed_set_top1": privacy.get("closed_set_top1", ""),
        "verification_auc": privacy.get("verification_auc", ""),
        "adaptive_acc": privacy.get("adaptive_attacker_accuracy", ""),
        "privacy_score": privacy.get("privacy_score", ""),
        # Expression
        "acc_original": expr.get("accuracy_original", ""),
        "acc_anonymized": expr.get("accuracy_anonymized", ""),
        "acc_delta": expr.get("accuracy_delta", ""),
        "expr_consistency": expr.get("expression_consistency", ""),
        "expr_match_rate": expr.get("expression_match_rate", ""),
        "utility_score": expr.get("utility_score", ""),
        # Realism
        "fid": realism.get("fid", ""),
        "lpips_mean": realism.get("lpips_mean", ""),
        "psnr_mean": realism.get("psnr_mean", ""),
        "ssim_mean": realism.get("ssim_mean", ""),
    }


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
