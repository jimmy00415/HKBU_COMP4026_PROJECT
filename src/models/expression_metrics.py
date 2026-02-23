"""
Expression evaluation metrics.

Covers both **hard-label** classification quality and **soft-label**
consistency between original and anonymized faces.

Functions
---------
* ``accuracy``                — top-1 accuracy.
* ``per_class_recall``        — recall for each of the 7 expression classes.
* ``confusion_matrix``        — (C, C) count matrix (predicted × true).
* ``plot_confusion_matrix``   — matplotlib visualisation.
* ``expression_consistency``  — 1 − avg KL(original ‖ anonymized) logits.
* ``expected_calibration_error`` — ECE with equal-width bins.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# FER-2013 class names (kept in sync with contracts.ExpressionLabel)
_DEFAULT_NAMES: list[str] = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral",
]


# ── Hard-label metrics ─────────────────────────────────────────────────────

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 accuracy.

    Parameters
    ----------
    y_true, y_pred : (N,) int arrays.

    Returns
    -------
    float in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def per_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 7,
    class_names: Optional[Sequence[str]] = None,
) -> dict[str, float]:
    """
    Per-class recall (sensitivity / true-positive rate).

    Returns
    -------
    dict mapping class name → recall ∈ [0, 1].  Classes with zero support
    get recall = 0.0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    names = class_names or _DEFAULT_NAMES[:num_classes]

    result: dict[str, float] = {}
    for c in range(num_classes):
        mask = y_true == c
        support = int(mask.sum())
        if support == 0:
            result[names[c]] = 0.0
        else:
            result[names[c]] = float((y_pred[mask] == c).sum() / support)
    return result


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 7,
) -> np.ndarray:
    """
    Compute a (num_classes, num_classes) confusion matrix.

    ``cm[i, j]`` = number of samples with true label *i* predicted as *j*.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Expression Confusion Matrix",
    normalize: bool = True,
    save_path: Optional[str] = None,
):
    """
    Render a confusion-matrix heatmap via matplotlib.

    Parameters
    ----------
    cm : (C, C) int or float array.
    normalize : If ``True``, rows are normalised to show recall percentages.
    save_path : If given, figure is saved instead of shown.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    num_classes = cm.shape[0]
    names = class_names or _DEFAULT_NAMES[:num_classes]

    display = cm.copy().astype(float)
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        display = display / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(display, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=names,
        yticklabels=names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Print values inside cells
    fmt = ".2f" if normalize else "d"
    thresh = display.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, format(display[i, j], fmt),
                ha="center", va="center",
                color="white" if display[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)
    return fig


# ── Soft-label / consistency metrics ───────────────────────────────────────

def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    KL(p ‖ q) per sample.

    Parameters
    ----------
    p, q : (N, C) probability arrays summing to 1 along axis 1.

    Returns
    -------
    kl : (N,) float  ≥ 0.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return (p * np.log(p / q)).sum(axis=1)


def expression_consistency(
    probs_original: np.ndarray,
    probs_anonymized: np.ndarray,
) -> float:
    """
    Expression consistency score.

    Defined as::

        consistency = 1 − mean( KL(p_orig ‖ p_anon) )

    A value of 1.0 means the anonymizer perfectly preserves expression
    signals; lower values indicate degradation.  The score is clamped to
    [0, 1].

    Parameters
    ----------
    probs_original   : (N, C) teacher softmax on original faces.
    probs_anonymized : (N, C) teacher softmax on anonymized faces.

    Returns
    -------
    float in [0, 1].
    """
    kl = kl_divergence(probs_original, probs_anonymized)
    mean_kl = float(kl.mean())
    return float(np.clip(1.0 - mean_kl, 0.0, 1.0))


def expression_match_rate(
    labels_original: np.ndarray,
    labels_anonymized: np.ndarray,
) -> float:
    """
    Fraction of samples where original and anonymized images get the same
    predicted expression label.
    """
    labels_original = np.asarray(labels_original)
    labels_anonymized = np.asarray(labels_anonymized)
    if labels_original.size == 0:
        return 0.0
    return float((labels_original == labels_anonymized).mean())


# ── Calibration ────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE) with equal-width confidence bins.

    Parameters
    ----------
    y_true : (N,) int — true class labels.
    probs  : (N, C) float — predicted probabilities.
    n_bins : number of bins.

    Returns
    -------
    float — ECE ∈ [0, 1].
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    if total == 0:
        return 0.0

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Last bin is inclusive on both ends
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue
        avg_confidence = float(confidences[mask].mean())
        avg_accuracy = float(accuracies[mask].mean())
        ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)

    return ece


# ── Summary helper ─────────────────────────────────────────────────────────

def compute_expression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    num_classes: int = 7,
    class_names: Optional[Sequence[str]] = None,
) -> dict:
    """
    Produce a full expression-evaluation summary dict.

    Keys: ``accuracy``, ``per_class_recall``, ``ece``, ``confusion_matrix``.
    """
    names = class_names or _DEFAULT_NAMES[:num_classes]
    return {
        "accuracy": accuracy(y_true, y_pred),
        "per_class_recall": per_class_recall(y_true, y_pred, num_classes, names),
        "ece": expected_calibration_error(y_true, probs, n_bins=15),
        "confusion_matrix": confusion_matrix(y_true, y_pred, num_classes),
    }
