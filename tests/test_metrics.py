"""
Unit tests — metrics modules (Task 8.4).

Tests with known-answer synthetic data:
  • PSNR  — identical images → inf; known MSE → expected dB
  • SSIM  — identical images → 1.0
  • Expression metrics orchestrator (evaluate_expression)
  • Privacy report structure
  • Realism report structure

No external models are loaded.
"""

from __future__ import annotations

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════
#  PSNR
# ══════════════════════════════════════════════════════════════════════

class TestPSNR:
    def test_identical_is_inf(self):
        from src.metrics.realism_metrics import psnr

        img = np.random.rand(64, 64, 3).astype(np.float32)
        assert psnr(img, img) == float("inf")

    def test_known_value(self):
        from src.metrics.realism_metrics import psnr

        # Constant image; noise → known MSE
        a = np.full((64, 64, 3), 0.5, dtype=np.float32)
        # Add constant offset of 0.1 → MSE = 0.01 → PSNR = 10*log10(1/0.01) = 20 dB
        b = np.full((64, 64, 3), 0.6, dtype=np.float32)
        p = psnr(a, b, max_val=1.0)
        np.testing.assert_allclose(p, 20.0, atol=0.1)

    def test_larger_diff_lower_psnr(self):
        from src.metrics.realism_metrics import psnr

        a = np.zeros((32, 32, 3), dtype=np.float32)
        b_close = a + 0.01
        b_far = a + 0.1
        assert psnr(a, b_close) > psnr(a, b_far)


# ══════════════════════════════════════════════════════════════════════
#  SSIM
# ══════════════════════════════════════════════════════════════════════

class TestSSIM:
    def test_identical_is_one(self):
        from src.metrics.realism_metrics import ssim

        img = np.random.rand(64, 64, 3).astype(np.float32)
        s = ssim(img, img)
        np.testing.assert_allclose(s, 1.0, atol=0.01)

    def test_different_less_than_one(self):
        from src.metrics.realism_metrics import ssim

        a = np.random.rand(64, 64, 3).astype(np.float32)
        b = np.random.rand(64, 64, 3).astype(np.float32)
        s = ssim(a, b)
        assert s < 1.0

    def test_range(self):
        from src.metrics.realism_metrics import ssim

        a = np.random.rand(32, 32, 3).astype(np.float32)
        b = np.random.rand(32, 32, 3).astype(np.float32)
        s = ssim(a, b)
        assert -1.0 <= s <= 1.0


# ══════════════════════════════════════════════════════════════════════
#  RealismReport
# ══════════════════════════════════════════════════════════════════════

class TestRealismReport:
    def test_to_dict(self):
        from src.metrics.realism_metrics import RealismReport

        r = RealismReport(psnr_mean=30.0, ssim_mean=0.9, anonymizer_name="blur")
        d = r.to_dict()
        assert d["psnr_mean"] == 30.0
        assert d["ssim_mean"] == 0.9
        assert d["anonymizer_name"] == "blur"

    def test_evaluate_realism_no_fid_no_lpips(self):
        """Test evaluate_realism with only PSNR & SSIM (no optional deps)."""
        from src.metrics.realism_metrics import evaluate_realism

        rng = np.random.RandomState(42)
        orig = rng.rand(10, 64, 64, 3).astype(np.float32)
        anon = orig + rng.randn(10, 64, 64, 3).astype(np.float32) * 0.05
        anon = np.clip(anon, 0, 1).astype(np.float32)

        report = evaluate_realism(
            orig, anon,
            compute_fid_score=False,
            compute_lpips_score=False,
            anonymizer_name="test",
        )
        assert report.psnr_mean > 0
        assert 0.0 < report.ssim_mean <= 1.0
        assert report.num_samples == 10


# ══════════════════════════════════════════════════════════════════════
#  Expression metrics orchestrator
# ══════════════════════════════════════════════════════════════════════

class TestExpressionReportOrchestrator:
    def test_evaluate_expression_basic(self):
        from src.metrics.expression_metrics import evaluate_expression

        rng = np.random.RandomState(0)
        N = 50
        y_true = rng.randint(0, 7, N)

        # Build plausible softmax-ish probs
        probs_orig = rng.dirichlet(np.ones(7), N).astype(np.float32)
        # Set argmax = true label for most samples
        for i in range(N):
            probs_orig[i, y_true[i]] += 2.0
        probs_orig = probs_orig / probs_orig.sum(axis=1, keepdims=True)

        # Slightly perturbed version
        probs_anon = probs_orig + rng.randn(N, 7).astype(np.float32) * 0.05
        probs_anon = np.clip(probs_anon, 1e-6, None)
        probs_anon = probs_anon / probs_anon.sum(axis=1, keepdims=True)

        report = evaluate_expression(
            y_true, probs_orig, probs_anon, anonymizer_name="test",
        )
        assert report.num_samples == N
        assert 0.0 <= report.accuracy_original <= 1.0
        assert 0.0 <= report.accuracy_anonymized <= 1.0
        assert 0.0 <= report.expression_consistency <= 1.0
        assert 0.0 <= report.expression_match_rate <= 1.0

    def test_to_dict(self):
        from src.metrics.expression_metrics import ExpressionReport

        r = ExpressionReport(accuracy_original=0.8, accuracy_anonymized=0.7)
        d = r.to_dict()
        assert "accuracy_original" in d
        assert "accuracy_delta" in d

    def test_perfect_preservation(self):
        from src.metrics.expression_metrics import evaluate_expression

        N = 20
        y_true = np.arange(N) % 7
        probs = np.zeros((N, 7), dtype=np.float32)
        for i in range(N):
            probs[i, y_true[i]] = 1.0

        report = evaluate_expression(y_true, probs, probs)
        assert report.accuracy_delta == 0.0
        assert report.expression_match_rate == 1.0
        assert report.expression_consistency > 0.99


# ══════════════════════════════════════════════════════════════════════
#  Privacy report structure
# ══════════════════════════════════════════════════════════════════════

class TestPrivacyReport:
    def test_to_dict(self):
        from src.metrics.privacy_metrics import PrivacyReport

        r = PrivacyReport(
            closed_set_top1=0.5, closed_set_top5=0.8,
            privacy_score=0.5, anonymizer_name="blur",
        )
        d = r.to_dict()
        assert d["closed_set_top1"] == 0.5
        assert d["closed_set_top5"] == 0.8
        assert d["privacy_score"] == 0.5

    def test_privacy_score_default(self):
        from src.metrics.privacy_metrics import PrivacyReport

        r = PrivacyReport()
        assert r.privacy_score == 1.0  # no leakage by default

    def test_evaluate_privacy_with_synthetic_data(self):
        """Test evaluate_privacy with synthetic embeddings."""
        pytest.importorskip("torch", reason="torch not installed")
        from src.metrics.privacy_metrics import evaluate_privacy

        rng = np.random.RandomState(42)
        N = 60
        # Create embeddings where anonymized ≠ original (good privacy)
        orig_emb = rng.randn(N, 32).astype(np.float32)
        anon_emb = rng.randn(N, 32).astype(np.float32)  # totally random
        labels = np.arange(N) % 10

        report = evaluate_privacy(
            orig_emb, anon_emb, labels,
            train_adaptive=True,
            adaptive_attacker_epochs=3,
            adaptive_attacker_hidden=[16],
            device="cpu",
            anonymizer_name="test",
        )
        assert report.num_samples == N
        assert 0.0 <= report.privacy_score <= 1.0
        assert report.adaptive_attacker_trained


# ══════════════════════════════════════════════════════════════════════
#  Runner utilities
# ══════════════════════════════════════════════════════════════════════

class TestReproducibility:
    def test_seed_everything(self):
        from src.runner.reproducibility import seed_everything

        seed_everything(123, deterministic=False)
        a = np.random.rand(10)
        seed_everything(123, deterministic=False)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_save_and_load_manifest(self, tmp_path):
        from src.runner.reproducibility import load_manifest, save_manifest

        path = save_manifest(str(tmp_path), seed=99)
        assert path.exists()
        m = load_manifest(path)
        assert m["seed"] == 99
        assert "environment" in m

    def test_verify_reproducibility(self, tmp_path):
        from src.runner.reproducibility import save_manifest, verify_reproducibility

        save_manifest(str(tmp_path), seed=42)
        result = verify_reproducibility(tmp_path / "reproducibility.json")
        assert isinstance(result, bool)


class TestErrorHandling:
    def test_retry_succeeds_after_failures(self):
        from src.runner.error_handling import retry

        call_count = {"n": 0}

        @retry(max_retries=3, delay=0.01)
        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("not yet")
            return "ok"

        assert flaky() == "ok"
        assert call_count["n"] == 3

    def test_retry_exhausted(self):
        from src.runner.error_handling import retry

        @retry(max_retries=2, delay=0.01)
        def always_fails():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            always_fails()

    def test_graceful_fallback(self):
        from src.runner.error_handling import graceful_fallback

        @graceful_fallback(default=-1)
        def broken():
            raise ValueError("oops")

        assert broken() == -1

    def test_safe_import_missing(self):
        from src.runner.error_handling import safe_import

        mod = safe_import("nonexistent_module_xyz_abc_123")
        assert mod is None

    def test_safe_import_existing(self):
        from src.runner.error_handling import safe_import

        mod = safe_import("os")
        assert mod is not None


class TestCacheModule:
    def test_save_load_roundtrip(self, tmp_path):
        from src.runner.cache import PreprocessCache

        cache = PreprocessCache(tmp_path / "cache_test")
        arr = np.random.rand(10, 32).astype(np.float32)
        cache.save("test_key", arr)
        loaded = cache.load("test_key")
        assert loaded is not None
        np.testing.assert_array_equal(arr, loaded)

    def test_cache_miss(self, tmp_path):
        from src.runner.cache import PreprocessCache

        cache = PreprocessCache(tmp_path / "cache_test")
        assert cache.load("nonexistent") is None

    def test_invalidate_key(self, tmp_path):
        from src.runner.cache import PreprocessCache

        cache = PreprocessCache(tmp_path / "cache_test")
        cache.save("mykey", np.array([1, 2, 3]))
        assert cache.exists("mykey")
        cache.invalidate("mykey")
        assert not cache.exists("mykey")

    def test_disabled_cache(self, tmp_path):
        from src.runner.cache import PreprocessCache

        cache = PreprocessCache(tmp_path / "cache_test", enabled=False)
        cache.save("x", np.array([1]))
        assert cache.load("x") is None
        assert cache.list_keys() == []

    def test_list_keys(self, tmp_path):
        from src.runner.cache import PreprocessCache

        cache = PreprocessCache(tmp_path / "cache_test")
        cache.save("a", np.array([1]))
        cache.save("b", np.array([2]))
        keys = cache.list_keys()
        assert sorted(keys) == ["a", "b"]
