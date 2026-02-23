"""
Integration test — full pipeline end-to-end (Task 8.5).

Using 10 synthetic images:
  load → detect (mock) → anonymize (blur) → compute metrics → check output

No external datasets or pretrained models needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.contracts import (
    AnonymizedFace,
    FaceCrop,
    FaceCropBatch,
    FaceCropMeta,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _synth_faces(n: int = 10, seed: int = 42) -> list[FaceCrop]:
    """Create N synthetic 256×256 FaceCrop objects."""
    rng = np.random.RandomState(seed)
    faces = []
    for i in range(n):
        img = rng.rand(256, 256, 3).astype(np.float32)
        meta = FaceCropMeta(
            dataset="synthetic", split="test", image_id=f"img_{i}",
            identity_label=i % 5,
            expression_label=i % 7,
        )
        faces.append(FaceCrop(image=img, meta=meta))
    return faces


# ══════════════════════════════════════════════════════════════════════
#  Full pipeline integration
# ══════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """
    End-to-end: synthetic images → anonymize → evaluate → validate outputs.
    """

    @pytest.fixture
    def pipeline_data(self):
        faces = _synth_faces(10)
        batch = FaceCropBatch(crops=faces)
        return faces, batch

    def test_anonymize_batch(self, pipeline_data):
        faces, batch = pipeline_data
        from src.anonymizers.classical import GaussianBlurAnonymizer

        anon = GaussianBlurAnonymizer(kernel_size=31)
        result = anon.anonymize_batch(batch)
        assert len(result) == 10
        for r in result.faces:
            assert r.image.shape == (256, 256, 3)
            assert r.image.dtype == np.float32
            assert 0.0 <= r.image.min()
            assert r.image.max() <= 1.0

    def test_expression_metrics_pipeline(self, pipeline_data):
        """Compute expression metrics on synthetic softmax outputs."""
        from src.metrics.expression_metrics import evaluate_expression

        faces, batch = pipeline_data
        N = len(faces)
        rng = np.random.RandomState(0)

        y_true = np.array([f.meta.expression_label for f in faces])
        probs_orig = rng.dirichlet(np.ones(7), N).astype(np.float32)
        probs_anon = rng.dirichlet(np.ones(7), N).astype(np.float32)

        report = evaluate_expression(y_true, probs_orig, probs_anon)
        assert report.num_samples == N
        d = report.to_dict()
        assert "accuracy_original" in d
        assert "expression_consistency" in d

    def test_realism_metrics_pipeline(self, pipeline_data):
        """Compute realism metrics (PSNR & SSIM only)."""
        from src.metrics.realism_metrics import evaluate_realism

        faces, batch = pipeline_data
        orig = batch.images
        anon = orig * 0.9  # slight dimming

        report = evaluate_realism(
            orig, anon,
            compute_fid_score=False,
            compute_lpips_score=False,
        )
        assert report.psnr_mean > 0
        assert report.ssim_mean > 0

    def test_privacy_metrics_pipeline(self, pipeline_data):
        """Compute privacy metrics with synthetic embeddings."""
        from src.metrics.privacy_metrics import evaluate_privacy

        faces, batch = pipeline_data
        N = len(faces)
        rng = np.random.RandomState(42)

        orig_emb = rng.randn(N, 32).astype(np.float32)
        anon_emb = rng.randn(N, 32).astype(np.float32)
        id_labels = np.array([f.meta.identity_label for f in faces])

        report = evaluate_privacy(
            orig_emb, anon_emb, id_labels,
            train_adaptive=False,
            device="cpu",
        )
        assert report.num_samples == N
        assert 0 <= report.privacy_score <= 1.0

    def test_evaluator_runner(self, pipeline_data, tmp_path):
        """Test the unified evaluation runner."""
        from src.runner.evaluator import run_evaluation

        faces, batch = pipeline_data
        N = len(faces)
        rng = np.random.RandomState(42)

        orig = batch.images
        anon = orig * 0.9

        result = run_evaluation(
            anonymizer_name="test_blur",
            original_images=orig,
            anonymized_images=anon,
            expression_labels=np.array([f.meta.expression_label for f in faces]),
            identity_labels=np.array([f.meta.identity_label for f in faces]),
            original_embeddings=rng.randn(N, 32).astype(np.float32),
            anonymized_embeddings=rng.randn(N, 32).astype(np.float32),
            probs_original=rng.dirichlet(np.ones(7), N).astype(np.float32),
            probs_anonymized=rng.dirichlet(np.ones(7), N).astype(np.float32),
            run_privacy=True,
            run_expression=True,
            run_realism=True,
            compute_fid=False,
            compute_lpips=False,
            train_adaptive=False,
            device="cpu",
            output_dir=str(tmp_path),
            run_id="integration_test",
        )

        d = result.to_dict()
        assert "privacy" in d
        assert "expression" in d
        assert "realism" in d

        # Check output files
        run_dir = tmp_path / "run_integration_test"
        assert (run_dir / "metrics.json").exists()
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert "privacy" in metrics

    def test_frontier_csv_output(self, pipeline_data, tmp_path):
        """Check that frontier CSV is produced."""
        from src.runner.evaluator import run_evaluation

        faces, batch = pipeline_data
        N = len(faces)
        rng = np.random.RandomState(42)

        run_evaluation(
            anonymizer_name="blur",
            original_images=batch.images,
            anonymized_images=batch.images * 0.8,
            expression_labels=np.array([f.meta.expression_label for f in faces]),
            identity_labels=np.array([f.meta.identity_label for f in faces]),
            original_embeddings=rng.randn(N, 32).astype(np.float32),
            anonymized_embeddings=rng.randn(N, 32).astype(np.float32),
            probs_original=rng.dirichlet(np.ones(7), N).astype(np.float32),
            probs_anonymized=rng.dirichlet(np.ones(7), N).astype(np.float32),
            run_privacy=True,
            run_expression=True,
            run_realism=True,
            compute_fid=False,
            compute_lpips=False,
            train_adaptive=False,
            device="cpu",
            output_dir=str(tmp_path),
            run_id="frontier_test",
        )
        run_dir = tmp_path / "run_frontier_test"
        assert (run_dir / "frontier_row.csv").exists()


# ══════════════════════════════════════════════════════════════════════
#  Contract round-trip
# ══════════════════════════════════════════════════════════════════════

class TestContractRoundTrip:
    def test_face_survives_anonymize_and_validate(self):
        """FaceCrop → anonymize → output validates as valid image."""
        from src.anonymizers.classical import PixelateAnonymizer

        face = _synth_faces(1)[0]
        face.validate()

        anon = PixelateAnonymizer(block_size=16)
        result = anon.anonymize_single(face)

        # The anonymized image should also be valid contract
        anon_face = FaceCrop(image=result.image, meta=face.meta)
        anon_face.validate()

    def test_batch_stacking_consistent(self):
        """Batch stacking preserves image ordering."""
        faces = _synth_faces(5)
        batch = FaceCropBatch(crops=faces)
        imgs = batch.images

        for i, face in enumerate(faces):
            np.testing.assert_array_equal(imgs[i], face.image)
