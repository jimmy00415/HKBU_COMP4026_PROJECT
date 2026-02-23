"""
Identity verification and identification metric utilities.

Provides functions to evaluate how well an anonymizer removes identity:

* **Closed-set identification** — top-1 accuracy on a gallery of known
  identities (e.g. Pins 105 classes).
* **Verification (TAR@FAR)** — given probe/gallery pairs, compute the
  true-accept rate at a fixed false-accept rate, and the full ROC curve.

All functions operate on pre-computed (N, D) embedding matrices, so they
are decoupled from the embedder backend.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Closed-set identification ──────────────────────────────────────────

def closed_set_identification(
    probe_embeddings: np.ndarray,
    probe_labels: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: int = 1,
) -> dict[str, float]:
    """
    Closed-set identification accuracy.

    For each probe, find the closest gallery embedding (cosine) and
    check whether the predicted identity matches.

    Parameters
    ----------
    probe_embeddings  : (Np, D)
    probe_labels      : (Np,) int
    gallery_embeddings: (Ng, D)
    gallery_labels    : (Ng,) int
    top_k             : report top-k accuracy (default 1)

    Returns
    -------
    dict with keys ``"top1_acc"``, ``"top5_acc"`` (if top_k >= 5), etc.
    """
    probe_embeddings = _l2_norm(probe_embeddings)
    gallery_embeddings = _l2_norm(gallery_embeddings)

    # (Np, Ng) cosine similarity matrix
    sim_matrix = probe_embeddings @ gallery_embeddings.T

    results: dict[str, float] = {}
    for k in range(1, top_k + 1):
        # Indices of top-k gallery matches for each probe
        top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]  # (Np, k)
        top_k_labels = gallery_labels[top_k_indices]             # (Np, k)
        correct = np.any(top_k_labels == probe_labels[:, None], axis=1)
        acc = float(correct.mean())
        results[f"top{k}_acc"] = acc

    return results


# ── Verification: ROC and TAR@FAR ─────────────────────────────────────

def compute_verification_metrics(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    is_same: np.ndarray,
    far_targets: tuple[float, ...] = (0.01, 0.001, 0.0001),
    num_thresholds: int = 1000,
) -> dict[str, object]:
    """
    Compute verification ROC and TAR@FAR.

    Parameters
    ----------
    embeddings_a, embeddings_b : (N, D) — paired embeddings
    is_same : (N,) bool — True if the pair is same-identity
    far_targets : target false-accept rates
    num_thresholds : number of thresholds to sweep

    Returns
    -------
    dict with:
        ``"tar_at_far"``  — { far_value: tar_value }
        ``"auc"``         — area under the ROC curve
        ``"thresholds"``  — (T,) array
        ``"tpr"``         — (T,) array
        ``"fpr"``         — (T,) array
        ``"eer"``         — equal error rate
    """
    embeddings_a = _l2_norm(embeddings_a)
    embeddings_b = _l2_norm(embeddings_b)

    # Pairwise cosine similarity
    similarities = np.sum(embeddings_a * embeddings_b, axis=1)  # (N,)
    is_same = is_same.astype(bool)

    # Sweep thresholds from min to max similarity
    thresholds = np.linspace(
        float(similarities.min()) - 0.01,
        float(similarities.max()) + 0.01,
        num_thresholds,
    )

    tpr_list: list[float] = []
    fpr_list: list[float] = []

    n_pos = is_same.sum()
    n_neg = (~is_same).sum()

    if n_pos == 0 or n_neg == 0:
        logger.warning("Degenerate verification set: n_pos=%d, n_neg=%d", n_pos, n_neg)
        return {
            "tar_at_far": {far: 0.0 for far in far_targets},
            "auc": 0.0,
            "thresholds": thresholds,
            "tpr": np.zeros_like(thresholds),
            "fpr": np.zeros_like(thresholds),
            "eer": 1.0,
        }

    for thr in thresholds:
        predicted_same = similarities >= thr
        tp = (predicted_same & is_same).sum()
        fp = (predicted_same & ~is_same).sum()
        tpr_list.append(float(tp / n_pos))
        fpr_list.append(float(fp / n_neg))

    tpr = np.array(tpr_list, dtype=np.float64)
    fpr = np.array(fpr_list, dtype=np.float64)

    # Sort by ascending FPR for clean ROC
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]

    # AUC via trapezoidal rule
    _trapz = getattr(np, "trapezoid", np.trapz)  # numpy ≥2.0 compat
    auc = float(_trapz(tpr_sorted, fpr_sorted))

    # TAR @ FAR
    tar_at_far = {}
    for target_far in far_targets:
        # Find the largest threshold where FAR <= target_far
        valid = fpr_sorted <= target_far
        if valid.any():
            tar_at_far[target_far] = float(tpr_sorted[valid][-1])
        else:
            tar_at_far[target_far] = 0.0

    # EER — where |TPR - (1 - FPR)| is minimised
    fnr_sorted = 1.0 - tpr_sorted
    abs_diff = np.abs(fpr_sorted - fnr_sorted)
    eer_idx = int(np.argmin(abs_diff))
    eer = float((fpr_sorted[eer_idx] + fnr_sorted[eer_idx]) / 2.0)

    return {
        "tar_at_far": tar_at_far,
        "auc": auc,
        "thresholds": thresholds,
        "tpr": tpr,
        "fpr": fpr,
        "eer": eer,
    }


# ── Pair generation helpers ────────────────────────────────────────────

def generate_verification_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_pairs: int = 5000,
    pos_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate balanced same / different identity pairs for verification.

    Parameters
    ----------
    embeddings : (N, D)
    labels     : (N,) int
    num_pairs  : total number of pairs
    pos_ratio  : fraction of same-identity pairs
    seed       : random seed for reproducibility

    Returns
    -------
    emb_a, emb_b : (num_pairs, D)
    is_same       : (num_pairs,) bool
    """
    rng = np.random.RandomState(seed)

    unique_labels = np.unique(labels)
    label_to_indices: dict[int, np.ndarray] = {
        int(lbl): np.where(labels == lbl)[0] for lbl in unique_labels
    }
    # Only keep labels with >= 2 samples (needed for positive pairs)
    multi_labels = [lbl for lbl, idx in label_to_indices.items() if len(idx) >= 2]

    if len(multi_labels) == 0:
        raise ValueError("Need at least one identity with >= 2 samples for positive pairs")

    n_pos = int(num_pairs * pos_ratio)
    n_neg = num_pairs - n_pos

    emb_a_list, emb_b_list, same_list = [], [], []

    # Positive pairs
    for _ in range(n_pos):
        lbl = rng.choice(multi_labels)
        i, j = rng.choice(label_to_indices[lbl], size=2, replace=False)
        emb_a_list.append(embeddings[i])
        emb_b_list.append(embeddings[j])
        same_list.append(True)

    # Negative pairs
    for _ in range(n_neg):
        lbl_a, lbl_b = rng.choice(multi_labels, size=2, replace=False)
        i = rng.choice(label_to_indices[lbl_a])
        j = rng.choice(label_to_indices[lbl_b])
        emb_a_list.append(embeddings[i])
        emb_b_list.append(embeddings[j])
        same_list.append(False)

    emb_a = np.stack(emb_a_list, axis=0).astype(np.float32)
    emb_b = np.stack(emb_b_list, axis=0).astype(np.float32)
    is_same = np.array(same_list, dtype=bool)

    # Shuffle
    perm = rng.permutation(num_pairs)
    return emb_a[perm], emb_b[perm], is_same[perm]


# ── Internal helpers ───────────────────────────────────────────────────

def _l2_norm(x: np.ndarray) -> np.ndarray:
    """L2-normalise along last axis."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-10)
