"""
Expression retention metrics.

High-level orchestrator that combines Phase-2 expression utilities to produce
an :class:`ExpressionReport` covering:

1. **FER accuracy** on original vs anonymized images.
2. **Expression consistency** — 1 − KL(teacher_orig ‖ teacher_anon).
3. **Expression match rate** — fraction where predicted labels agree.
4. **Confusion-matrix delta** — how the confusion matrix shifts.
5. **Per-class drift** — per-class recall change after anonymisation.
6. **ECE** — expected calibration error on anonymised predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExpressionReport:
    """Container for all expression-retention metrics."""

    # Accuracy
    accuracy_original: float = 0.0
    accuracy_anonymized: float = 0.0
    accuracy_delta: float = 0.0

    # Consistency (soft-label KL)
    expression_consistency: float = 0.0

    # Match rate (hard-label agreement)
    expression_match_rate: float = 0.0

    # Per-class recall
    per_class_recall_original: dict[str, float] = field(default_factory=dict)
    per_class_recall_anonymized: dict[str, float] = field(default_factory=dict)
    per_class_recall_delta: dict[str, float] = field(default_factory=dict)

    # ECE
    ece_original: float = 0.0
    ece_anonymized: float = 0.0

    # Confusion matrices  (stored as lists for JSON serialisation)
    cm_original: Optional[list] = None
    cm_anonymized: Optional[list] = None

    # Aggregate
    utility_score: float = 0.0  # higher is better

    # Metadata
    anonymizer_name: str = ""
    num_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy_original": self.accuracy_original,
            "accuracy_anonymized": self.accuracy_anonymized,
            "accuracy_delta": self.accuracy_delta,
            "expression_consistency": self.expression_consistency,
            "expression_match_rate": self.expression_match_rate,
            "per_class_recall_original": self.per_class_recall_original,
            "per_class_recall_anonymized": self.per_class_recall_anonymized,
            "per_class_recall_delta": self.per_class_recall_delta,
            "ece_original": self.ece_original,
            "ece_anonymized": self.ece_anonymized,
            "cm_original": self.cm_original,
            "cm_anonymized": self.cm_anonymized,
            "utility_score": self.utility_score,
            "anonymizer_name": self.anonymizer_name,
            "num_samples": self.num_samples,
        }


def evaluate_expression(
    y_true: np.ndarray,
    probs_original: np.ndarray,
    probs_anonymized: np.ndarray,
    *,
    num_classes: int = 7,
    class_names: Optional[Sequence[str]] = None,
    anonymizer_name: str = "",
) -> ExpressionReport:
    """
    Compute all expression-retention metrics.

    Parameters
    ----------
    y_true            : (N,) ground-truth expression labels.
    probs_original    : (N, C) teacher/classifier softmax on original images.
    probs_anonymized  : (N, C) teacher/classifier softmax on anonymized images.
    num_classes       : number of expression classes.
    class_names       : optional class name list.
    anonymizer_name   : for metadata.

    Returns
    -------
    ExpressionReport
    """
    from src.models.expression_metrics import (
        accuracy,
        confusion_matrix,
        expected_calibration_error,
        expression_consistency,
        expression_match_rate,
        per_class_recall,
    )

    N = y_true.shape[0]
    preds_orig = probs_original.argmax(axis=1)
    preds_anon = probs_anonymized.argmax(axis=1)

    report = ExpressionReport(
        anonymizer_name=anonymizer_name,
        num_samples=N,
    )

    # ── Accuracy ───────────────────────────────────────────────────────
    report.accuracy_original = accuracy(y_true, preds_orig)
    report.accuracy_anonymized = accuracy(y_true, preds_anon)
    report.accuracy_delta = report.accuracy_anonymized - report.accuracy_original
    logger.info(
        "Accuracy: orig=%.4f  anon=%.4f  Δ=%+.4f",
        report.accuracy_original, report.accuracy_anonymized, report.accuracy_delta,
    )

    # ── Expression consistency (soft-label) ────────────────────────────
    report.expression_consistency = expression_consistency(
        probs_original, probs_anonymized,
    )
    logger.info("Expression consistency=%.4f", report.expression_consistency)

    # ── Expression match rate (hard-label) ─────────────────────────────
    report.expression_match_rate = expression_match_rate(preds_orig, preds_anon)
    logger.info("Expression match rate=%.4f", report.expression_match_rate)

    # ── Per-class recall ───────────────────────────────────────────────
    report.per_class_recall_original = per_class_recall(
        y_true, preds_orig, num_classes, class_names,
    )
    report.per_class_recall_anonymized = per_class_recall(
        y_true, preds_anon, num_classes, class_names,
    )
    report.per_class_recall_delta = {
        k: report.per_class_recall_anonymized.get(k, 0.0) - v
        for k, v in report.per_class_recall_original.items()
    }

    # ── ECE ────────────────────────────────────────────────────────────
    report.ece_original = expected_calibration_error(y_true, probs_original)
    report.ece_anonymized = expected_calibration_error(y_true, probs_anonymized)

    # ── Confusion matrices ─────────────────────────────────────────────
    report.cm_original = confusion_matrix(y_true, preds_orig, num_classes).tolist()
    report.cm_anonymized = confusion_matrix(y_true, preds_anon, num_classes).tolist()

    # ── Aggregate utility ──────────────────────────────────────────────
    # Weighted combination: 60% consistency + 40% anonymised accuracy
    report.utility_score = float(np.clip(
        0.6 * report.expression_consistency + 0.4 * report.accuracy_anonymized,
        0.0, 1.0,
    ))

    return report
