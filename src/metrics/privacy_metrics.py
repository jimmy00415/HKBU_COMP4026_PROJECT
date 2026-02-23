"""
Privacy (identity-leakage) metrics.

High-level orchestrator that combines:

* **IdentityEmbedder** — ArcFace 512-d embeddings.
* **identity_metrics** — closed-set identification, verification ROC/AUC/EER.
* **AdaptiveAttackerHead** — re-identification after domain-adaptive training.

Primary metrics
---------------
1. ``closed_set_id_accuracy``  — top-1 / top-5 accuracy using cosine
   similarity between anonymized probe embeddings and the original gallery.
2. ``verification_tar_at_far`` — True Accept Rate at specified FAR levels.
3. ``verification_auc``        — Area under the ROC curve.
4. ``adaptive_attacker_accuracy`` — accuracy of a lightweight MLP trained
   on anonymized embeddings (worst-case privacy leakage).

A *higher* leakage value means *worse* privacy.  The ``privacy_score``
convenience function returns ``1 − leakage`` so that higher is better.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class PrivacyReport:
    """Container for all privacy metric results."""

    # Closed-set identification
    closed_set_top1: float = 0.0
    closed_set_top5: float = 0.0

    # Verification
    tar_at_far: dict[str, float] = field(default_factory=dict)
    verification_auc: float = 0.0
    verification_eer: float = 0.0

    # Adaptive attacker
    adaptive_attacker_accuracy: float = 0.0
    adaptive_attacker_trained: bool = False

    # Aggregate
    privacy_score: float = 1.0  # 1 − worst leakage

    # Metadata
    anonymizer_name: str = ""
    anonymizer_params: dict = field(default_factory=dict)
    num_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "closed_set_top1": self.closed_set_top1,
            "closed_set_top5": self.closed_set_top5,
            "tar_at_far": self.tar_at_far,
            "verification_auc": self.verification_auc,
            "verification_eer": self.verification_eer,
            "adaptive_attacker_accuracy": self.adaptive_attacker_accuracy,
            "adaptive_attacker_trained": self.adaptive_attacker_trained,
            "privacy_score": self.privacy_score,
            "anonymizer_name": self.anonymizer_name,
            "anonymizer_params": self.anonymizer_params,
            "num_samples": self.num_samples,
        }


# ── Main evaluator ────────────────────────────────────────────────────────

def evaluate_privacy(
    original_embeddings: np.ndarray,
    anonymized_embeddings: np.ndarray,
    identity_labels: np.ndarray,
    *,
    far_levels: Optional[list[float]] = None,
    adaptive_attacker_epochs: int = 20,
    adaptive_attacker_hidden: Optional[list[int]] = None,
    adaptive_attacker_dropout: float = 0.3,
    train_adaptive: bool = True,
    device: str = "cpu",
    anonymizer_name: str = "",
    anonymizer_params: Optional[dict] = None,
) -> PrivacyReport:
    """
    Compute all privacy metrics.

    Parameters
    ----------
    original_embeddings  : (N, D) gallery ArcFace embeddings (non-anonymized).
    anonymized_embeddings: (N, D) probe ArcFace embeddings (anonymized).
    identity_labels      : (N,) int identity labels.
    far_levels           : FAR thresholds for TAR@FAR.
    train_adaptive       : whether to train adaptive attacker.
    device               : for adaptive attacker MLP training.

    Returns
    -------
    PrivacyReport
    """
    from src.models.identity_metrics import (
        closed_set_identification,
        compute_verification_metrics,
        generate_verification_pairs,
    )

    if far_levels is None:
        far_levels = [0.01, 0.001, 0.0001]

    N = original_embeddings.shape[0]
    report = PrivacyReport(
        anonymizer_name=anonymizer_name,
        anonymizer_params=anonymizer_params or {},
        num_samples=N,
    )

    # ── 1. Closed-set identification ───────────────────────────────────
    logger.info("Computing closed-set identification...")
    id_results = closed_set_identification(
        probe_embeddings=anonymized_embeddings,
        probe_labels=identity_labels,
        gallery_embeddings=original_embeddings,
        gallery_labels=identity_labels,
        top_k=5,
    )
    report.closed_set_top1 = id_results.get("top1_acc", 0.0)
    report.closed_set_top5 = id_results.get("top5_acc", 0.0)
    logger.info("  top-1=%.4f  top-5=%.4f", report.closed_set_top1, report.closed_set_top5)

    # ── 2. Verification ROC ────────────────────────────────────────────
    logger.info("Computing verification metrics...")
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
                far_targets=tuple(far_levels),
            )
            report.tar_at_far = verif.get("tar_at_far", {})
            report.verification_auc = verif.get("auc", 0.0)
            report.verification_eer = verif.get("eer", 0.0)
            logger.info("  AUC=%.4f  EER=%.4f", report.verification_auc, report.verification_eer)
        else:
            logger.warning("  Not enough pairs for verification.")
    except ValueError as exc:
        logger.warning("  Verification pairs failed: %s", exc)

    # ── 3. Adaptive attacker ───────────────────────────────────────────
    if train_adaptive and N >= 20:
        logger.info("Training adaptive attacker...")
        from src.models.adaptive_attacker import AdaptiveAttackerHead

        num_ids = int(identity_labels.max()) + 1
        hidden = adaptive_attacker_hidden or [256, 128]
        attacker = AdaptiveAttackerHead(
            embedding_dim=anonymized_embeddings.shape[1],
            num_classes=num_ids,
            hidden_dims=hidden,
            dropout=adaptive_attacker_dropout,
            device=device,
        )

        # Split: 80% train, 20% test
        rng = np.random.RandomState(42)
        perm = rng.permutation(N)
        split = int(0.8 * N)
        train_idx, test_idx = perm[:split], perm[split:]

        attacker.fit(
            embeddings=anonymized_embeddings[train_idx],
            labels=identity_labels[train_idx],
            epochs=adaptive_attacker_epochs,
            val_embeddings=anonymized_embeddings[test_idx],
            val_labels=identity_labels[test_idx],
            verbose=False,
        )
        report.adaptive_attacker_accuracy = attacker.evaluate(
            anonymized_embeddings[test_idx], identity_labels[test_idx],
        )
        report.adaptive_attacker_trained = True
        logger.info("  Adaptive attacker accuracy=%.4f", report.adaptive_attacker_accuracy)
    else:
        logger.info("  Skipping adaptive attacker (N=%d, train_adaptive=%s)", N, train_adaptive)

    # ── Aggregate privacy score ────────────────────────────────────────
    # Use the worst (highest) leakage as the reference
    leakage = max(
        report.closed_set_top1,
        report.adaptive_attacker_accuracy,
        report.verification_auc,
    )
    report.privacy_score = float(np.clip(1.0 - leakage, 0.0, 1.0))

    return report
