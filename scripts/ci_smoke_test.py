"""
CI smoke test — lightweight full-pipeline check (Task 8.7).

Runs on CPU with 5 synthetic images through the entire pipeline.
Designed for CI environments with no GPU, no pretrained models,
and no external datasets.

Usage:
    python scripts/ci_smoke_test.py
    python scripts/ci_smoke_test.py --verbose

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

logger = logging.getLogger(__name__)

N_SAMPLES = 5
N_CLASSES = 7
N_IDENTITIES = 5
EMB_DIM = 32  # small dim for speed

# ═══════════════════════════════════════════════════════════════════════
#  Synthetic data
# ═══════════════════════════════════════════════════════════════════════

def make_synthetic_data(n: int = N_SAMPLES, seed: int = 42):
    """Create all synthetic data needed for the smoke test."""
    from src.data.contracts import FaceCrop, FaceCropBatch, FaceCropMeta

    rng = np.random.RandomState(seed)
    faces = []
    for i in range(n):
        img = rng.rand(256, 256, 3).astype(np.float32)
        meta = FaceCropMeta(
            dataset="smoke_test", split="test", image_id=f"smoke_{i}",
            identity_label=i % N_IDENTITIES,
            expression_label=i % N_CLASSES,
        )
        faces.append(FaceCrop(image=img, meta=meta))

    batch = FaceCropBatch(crops=faces)
    expr_labels = np.array([f.meta.expression_label for f in faces])
    id_labels = np.array([f.meta.identity_label for f in faces])

    return faces, batch, expr_labels, id_labels, rng


# ═══════════════════════════════════════════════════════════════════════
#  Individual checks
# ═══════════════════════════════════════════════════════════════════════

def check_contracts() -> bool:
    """Verify data contracts work correctly."""
    logger.info("[1/8] Checking data contracts...")
    try:
        from src.data.contracts import (
            CANONICAL_RESOLUTION,
            ExpressionLabel,
            FaceCrop,
            FaceCropBatch,
            FaceCropMeta,
            float_to_uint8,
            gray_to_rgb,
            uint8_to_float,
        )

        assert len(ExpressionLabel) == 7
        assert CANONICAL_RESOLUTION == 256

        img = np.random.rand(256, 256, 3).astype(np.float32)
        meta = FaceCropMeta(dataset="test", split="test", image_id="0")
        face = FaceCrop(image=img, meta=meta)
        face.validate()

        u8 = float_to_uint8(img)
        back = uint8_to_float(u8)
        assert back.dtype == np.float32

        gray = gray_to_rgb(np.ones((10, 10), dtype=np.float32))
        assert gray.shape == (10, 10, 3)

        logger.info("  PASS")
        return True
    except Exception as e:
        logger.error("  FAIL: %s", e)
        return False


def check_anonymizers() -> bool:
    """Verify all classical anonymizers produce valid output."""
    logger.info("[2/8] Checking anonymizers...")
    try:
        from src.anonymizers import get_anonymizer, list_anonymizers
        from src.data.contracts import FaceCrop, FaceCropMeta

        names = list_anonymizers()
        assert len(names) >= 3

        img = np.random.rand(256, 256, 3).astype(np.float32)
        meta = FaceCropMeta(dataset="test", split="test", image_id="0")
        face = FaceCrop(image=img, meta=meta)

        for name in ["blur", "pixelate", "blackout"]:
            anon = get_anonymizer(name)
            result = anon.anonymize_single(face)
            assert result.image.shape == (256, 256, 3)
            assert result.image.dtype == np.float32
            assert 0.0 <= result.image.min() <= result.image.max() <= 1.0
            assert result.anonymizer_name == name

        logger.info("  PASS (tested %d anonymizers)", 3)
        return True
    except Exception as e:
        logger.error("  FAIL: %s", e)
        return False


def check_expression_metrics() -> bool:
    """Verify expression metrics."""
    logger.info("[3/8] Checking expression metrics...")
    try:
        from src.models.expression_metrics import (
            accuracy,
            confusion_matrix,
            expression_consistency,
            expression_match_rate,
            per_class_recall,
        )

        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])
        assert accuracy(y_true, y_pred) == 1.0

        cm = confusion_matrix(y_true, y_pred, num_classes=7)
        assert cm.shape == (7, 7)
        assert cm.sum() == 7

        probs = np.eye(7, dtype=np.float32)
        c = expression_consistency(probs, probs)
        assert c > 0.99

        mr = expression_match_rate(y_true, y_pred)
        assert mr == 1.0

        recall = per_class_recall(y_true, y_pred, num_classes=7)
        assert all(v == 1.0 for v in recall.values())

        logger.info("  PASS")
        return True
    except Exception as e:
        logger.error("  FAIL: %s", e)
        return False


def check_identity_metrics() -> bool:
    """Verify identity metrics with synthetic embeddings."""
    logger.info("[4/8] Checking identity metrics...")
    try:
        from src.models.identity_metrics import (
            closed_set_identification,
            compute_verification_metrics,
            generate_verification_pairs,
        )

        emb = np.eye(10, dtype=np.float32)
        labels = np.arange(10)
        result = closed_set_identification(emb, labels, emb, labels, top_k=3)
        assert result["top1_acc"] == 1.0

        # Need multiple samples per identity to generate positive pairs
        rng = np.random.RandomState(42)
        emb_multi = rng.randn(20, 10).astype(np.float32)
        labels_multi = np.array([0]*4 + [1]*4 + [2]*4 + [3]*4 + [4]*4)
        emb_a, emb_b, is_same = generate_verification_pairs(
            emb_multi, labels_multi, num_pairs=20, seed=0,
        )
        assert emb_a.shape[0] == emb_b.shape[0] == is_same.shape[0]

        vr = compute_verification_metrics(emb_a, emb_b, is_same)
        assert "auc" in vr

        logger.info("  PASS")
        return True
    except Exception as e:
        logger.error("  FAIL: %s", e)
        return False


def check_realism_metrics() -> bool:
    """Verify PSNR and SSIM."""
    logger.info("[5/8] Checking realism metrics...")
    try:
        from src.metrics.realism_metrics import psnr, ssim

        a = np.random.rand(64, 64, 3).astype(np.float32)
        assert psnr(a, a) == float("inf")
        assert ssim(a, a) > 0.99

        b = a + np.random.randn(64, 64, 3).astype(np.float32) * 0.1
        b = np.clip(b, 0, 1).astype(np.float32)
        assert psnr(a, b) > 0
        assert ssim(a, b) < 1.0

        logger.info("  PASS")
        return True
    except Exception as e:
        logger.error("  FAIL: %s", e)
        return False


def check_evaluator(tmp_dir: str) -> bool:
    """Verify the unified evaluation runner."""
    logger.info("[6/8] Checking evaluator...")
    try:
        from src.runner.evaluator import run_evaluation

        rng = np.random.RandomState(42)
        N = 5
        result = run_evaluation(
            anonymizer_name="smoke_blur",
            original_images=rng.rand(N, 256, 256, 3).astype(np.float32),
            anonymized_images=rng.rand(N, 256, 256, 3).astype(np.float32),
            expression_labels=rng.randint(0, 7, N),
            identity_labels=rng.randint(0, 5, N),
            original_embeddings=rng.randn(N, EMB_DIM).astype(np.float32),
            anonymized_embeddings=rng.randn(N, EMB_DIM).astype(np.float32),
            probs_original=rng.dirichlet(np.ones(7), N).astype(np.float32),
            probs_anonymized=rng.dirichlet(np.ones(7), N).astype(np.float32),
            run_privacy=True,
            run_expression=True,
            run_realism=True,
            compute_fid=False,
            compute_lpips=False,
            train_adaptive=False,
            device="cpu",
            output_dir=tmp_dir,
            run_id="smoke",
        )

        d = result.to_dict()
        assert "privacy" in d
        assert "expression" in d
        assert "realism" in d

        metrics_path = Path(tmp_dir) / "run_smoke" / "metrics.json"
        assert metrics_path.exists()

        logger.info("  PASS")
        return True
    except Exception as e:
        logger.error("  FAIL: %s", e)
        return False


def check_reproducibility() -> bool:
    """Verify reproducibility utilities."""
    logger.info("[7/8] Checking reproducibility...")
    try:
        from src.runner.reproducibility import get_environment_info, seed_everything

        seed_everything(42, deterministic=False)
        a = np.random.rand(5)
        seed_everything(42, deterministic=False)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

        info = get_environment_info()
        assert "python_version" in info

        logger.info("  PASS")
        return True
    except Exception as e:
        logger.error("  FAIL: %s", e)
        return False


def check_error_handling() -> bool:
    """Verify error handling utilities."""
    logger.info("[8/8] Checking error handling...")
    try:
        from src.runner.error_handling import (
            graceful_fallback,
            retry,
            safe_import,
            validate_image,
        )

        # retry
        counter = {"n": 0}

        @retry(max_retries=3, delay=0.01)
        def flaky():
            counter["n"] += 1
            if counter["n"] < 3:
                raise ValueError("not yet")
            return True

        assert flaky() is True

        # graceful_fallback
        @graceful_fallback(default=-1)
        def broken():
            raise RuntimeError("oops")

        assert broken() == -1

        # safe_import
        assert safe_import("os") is not None
        assert safe_import("nonexistent_xyz") is None

        # validate_image
        img = np.random.rand(64, 64, 3).astype(np.float32)
        out = validate_image(img)
        assert out.shape == (64, 64, 3)

        logger.info("  PASS")
        return True
    except Exception as e:
        logger.error("  FAIL: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CI smoke test")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    logger.info("=" * 60)
    logger.info("  CI Smoke Test — Anonymised FER Pipeline")
    logger.info("=" * 60)
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp_dir:
        checks = [
            ("Data Contracts", check_contracts),
            ("Anonymizers", check_anonymizers),
            ("Expression Metrics", check_expression_metrics),
            ("Identity Metrics", check_identity_metrics),
            ("Realism Metrics", check_realism_metrics),
            ("Evaluator", lambda: check_evaluator(tmp_dir)),
            ("Reproducibility", check_reproducibility),
            ("Error Handling", check_error_handling),
        ]

        results: list[tuple[str, bool]] = []
        for name, fn in checks:
            ok = fn()
            results.append((name, ok))

    elapsed = time.time() - t0
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Results: %d passed, %d failed (%.1fs)", n_pass, n_fail, elapsed)
    logger.info("=" * 60)

    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        logger.info("  [%s] %s", status, name)

    if n_fail > 0:
        logger.error("Smoke test FAILED.")
        sys.exit(1)
    else:
        logger.info("Smoke test PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
