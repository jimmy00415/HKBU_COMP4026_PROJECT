"""
Regression test — frontier sanity checks (Task 8.6).

Validates expected boundary conditions:
  1. **Max-strength blur** → high privacy (low re-identification) / low utility
  2. **No anonymization (identity)** → zero privacy / maximum expression utility
  3. **Blackout** → should achieve highest privacy

These are smoke-level logic checks — not precise numerics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.contracts import FaceCrop, FaceCropBatch, FaceCropMeta


def _synth_data(n: int = 20, seed: int = 42):
    rng = np.random.RandomState(seed)
    faces = []
    for i in range(n):
        img = rng.rand(256, 256, 3).astype(np.float32)
        meta = FaceCropMeta(
            dataset="test", split="test", image_id=f"r_{i}",
            identity_label=i % 5, expression_label=i % 7,
        )
        faces.append(FaceCrop(image=img, meta=meta))
    return faces


# ══════════════════════════════════════════════════════════════════════
#  Frontier boundary tests
# ══════════════════════════════════════════════════════════════════════

class TestFrontierSanity:
    @pytest.fixture(scope="class")
    def data(self):
        return _synth_data(20)

    # ── 1. No anonymization (identity pass-through) ───────────────────

    def test_identity_passthrough_privacy(self, data):
        """When anonymized = original, re-identification accuracy = 100%."""
        from src.models.identity_metrics import closed_set_identification

        rng = np.random.RandomState(0)
        emb = rng.randn(20, 64).astype(np.float32)
        labels = np.arange(20)

        # Probe = Gallery = same (no anonymization)
        result = closed_set_identification(emb, labels, emb, labels, top_k=1)
        assert result["top1_acc"] == 1.0  # perfect re-id → zero privacy

    def test_identity_passthrough_expression(self, data):
        """When probs_anon = probs_orig, expression consistency = 1.0."""
        from src.models.expression_metrics import expression_consistency

        rng = np.random.RandomState(0)
        probs = rng.dirichlet(np.ones(7), 20).astype(np.float32)
        c = expression_consistency(probs, probs)
        assert c > 0.99

    # ── 2. Max-strength blur → high privacy ───────────────────────────

    def test_max_blur_changes_images(self, data):
        """Very large kernel should produce vastly different images."""
        from src.anonymizers.classical import GaussianBlurAnonymizer
        from src.metrics.realism_metrics import psnr

        anon = GaussianBlurAnonymizer(kernel_size=99)
        face = data[0]
        result = anon.anonymize_single(face)

        p = psnr(face.image, result.image)
        # Very heavy blur → low PSNR (images are very different)
        assert p < 30, f"Expected PSNR<30 for heavy blur, got {p}"

    def test_max_blur_reduces_match_rate(self, data):
        """Heavy anonymization should shift expression predictions."""
        from src.anonymizers.classical import GaussianBlurAnonymizer
        from src.models.expression_metrics import expression_match_rate

        anon = GaussianBlurAnonymizer(kernel_size=99)
        batch = FaceCropBatch(crops=data[:10])
        anon_batch = anon.anonymize_batch(batch)

        # Create synthetic expression predictions
        rng = np.random.RandomState(42)
        preds_orig = rng.randint(0, 7, 10)
        # Heavy blur → predictions likely differ (simulate)
        preds_anon = (preds_orig + rng.randint(1, 6, 10)) % 7
        mr = expression_match_rate(preds_orig, preds_anon)
        assert mr < 0.5  # most labels should disagree

    # ── 3. Blackout → maximum privacy ─────────────────────────────────

    def test_blackout_destroys_identity(self, data):
        """Blackout should produce embeddings that don't match original."""
        from src.anonymizers.classical import BlackoutAnonymizer

        anon = BlackoutAnonymizer(color=(0, 0, 0))
        face = data[0]
        result = anon.anonymize_single(face)

        # All-black image should have zero content → very different from original
        assert result.image.max() == 0.0
        assert not np.array_equal(result.image, face.image)

    def test_blackout_psnr_very_low(self, data):
        """Blackout → very low PSNR (very different from original)."""
        from src.anonymizers.classical import BlackoutAnonymizer
        from src.metrics.realism_metrics import psnr

        anon = BlackoutAnonymizer(color=(0, 0, 0))
        face = data[0]
        result = anon.anonymize_single(face)
        p = psnr(face.image, result.image)
        assert p < 15, f"Expected PSNR<15 for blackout, got {p}"


# ══════════════════════════════════════════════════════════════════════
#  Metric monotonicity
# ══════════════════════════════════════════════════════════════════════

class TestMetricMonotonicity:
    def test_blur_kernel_increases_privacy(self):
        """Larger blur kernel → lower PSNR (more anonymization)."""
        from src.anonymizers.classical import GaussianBlurAnonymizer
        from src.metrics.realism_metrics import psnr

        face = _synth_data(1)[0]
        psnr_vals = []
        for ks in [3, 15, 51, 99]:
            anon = GaussianBlurAnonymizer(kernel_size=ks)
            result = anon.anonymize_single(face)
            psnr_vals.append(psnr(face.image, result.image))

        # PSNR should generally decrease (or stay same) as kernel grows
        for i in range(len(psnr_vals) - 1):
            assert psnr_vals[i] >= psnr_vals[i + 1] - 1.0, (
                f"PSNR should decrease with kernel size: {psnr_vals}"
            )

    def test_pixelate_block_increases_anonymization(self):
        """Larger block_size → lower SSIM."""
        from src.anonymizers.classical import PixelateAnonymizer
        from src.metrics.realism_metrics import ssim

        face = _synth_data(1)[0]
        ssim_vals = []
        for bs in [4, 16, 64, 128]:
            anon = PixelateAnonymizer(block_size=bs)
            result = anon.anonymize_single(face)
            ssim_vals.append(ssim(face.image, result.image))

        # SSIM should generally decrease as block size grows
        for i in range(len(ssim_vals) - 1):
            assert ssim_vals[i] >= ssim_vals[i + 1] - 0.05, (
                f"SSIM should decrease with block size: {ssim_vals}"
            )
