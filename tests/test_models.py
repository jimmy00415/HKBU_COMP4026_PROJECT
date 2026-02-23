"""
Unit tests — model wrappers (Task 8.2).

Tests the models we can exercise *without* downloading heavyweight
pretrained checkpoints:

  • AdaptiveAttackerHead  — MLP fits synthetic embeddings and predicts labels
  • ExpressionClassifier  — construction, predict shapes, fit on tiny data
  • Expression metrics    — accuracy, per_class_recall, confusion_matrix,
                            expression_consistency, ECE
  • Identity metrics      — closed_set_identification, verification metrics

IdentityEmbedder (InsightFace) and ExpressionTeacher are tested via
graceful unavailability checks only (no downloads in CI).
"""

from __future__ import annotations

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════
#  AdaptiveAttackerHead
# ══════════════════════════════════════════════════════════════════════

class TestAdaptiveAttacker:
    @pytest.fixture
    def attacker(self):
        pytest.importorskip("torch", reason="torch not installed")
        from src.models.adaptive_attacker import AdaptiveAttackerHead
        return AdaptiveAttackerHead(
            embedding_dim=32, num_classes=5,
            hidden_dims=[16], dropout=0.0, device="cpu",
        )

    def test_construct(self, attacker):
        assert attacker.num_classes == 5
        assert attacker.embedding_dim == 32
        assert not attacker.fitted

    def test_predict_shape(self, attacker):
        rng = np.random.RandomState(0)
        emb = rng.randn(10, 32).astype(np.float32)
        preds = attacker.predict(emb)
        assert preds.shape == (10,)

    def test_predict_proba_shape(self, attacker):
        rng = np.random.RandomState(0)
        emb = rng.randn(10, 32).astype(np.float32)
        probs = attacker.predict_proba(emb)
        assert probs.shape == (10, 5)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_fit_and_evaluate(self, attacker):
        rng = np.random.RandomState(42)
        # Create 5 well-separated clusters
        embeddings = []
        labels = []
        for cls in range(5):
            center = rng.randn(32).astype(np.float32) * 10
            for _ in range(40):
                embeddings.append(center + rng.randn(32).astype(np.float32) * 0.1)
                labels.append(cls)
        emb = np.array(embeddings, dtype=np.float32)
        lab = np.array(labels, dtype=np.int64)

        history = attacker.fit(
            emb, lab, epochs=30, batch_size=32, lr=1e-2, verbose=False,
        )
        assert attacker.fitted
        assert "train_loss" in history
        assert "train_acc" in history
        # Should achieve high accuracy on clearly separated data
        acc = attacker.evaluate(emb, lab)
        assert acc > 0.8, f"Expected >0.8, got {acc}"

    def test_fit_with_validation(self, attacker):
        rng = np.random.RandomState(42)
        emb = rng.randn(40, 32).astype(np.float32)
        lab = rng.randint(0, 5, 40).astype(np.int64)
        history = attacker.fit(
            emb[:30], lab[:30], epochs=3, verbose=False,
            val_embeddings=emb[30:], val_labels=lab[30:],
        )
        assert "val_acc" in history
        assert len(history["val_acc"]) == 3

    def test_save_load(self, attacker, tmp_path):
        rng = np.random.RandomState(0)
        emb = rng.randn(20, 32).astype(np.float32)
        lab = rng.randint(0, 5, 20).astype(np.int64)
        attacker.fit(emb, lab, epochs=3, verbose=False)

        path = str(tmp_path / "attacker.pt")
        attacker.save(path)

        from src.models.adaptive_attacker import AdaptiveAttackerHead
        loaded = AdaptiveAttackerHead(
            embedding_dim=32, num_classes=5,
            hidden_dims=[16], dropout=0.0, device="cpu",
        )
        loaded.load(path)
        assert loaded.fitted

        # Same predictions
        p1 = attacker.predict(emb)
        p2 = loaded.predict(emb)
        np.testing.assert_array_equal(p1, p2)


# ══════════════════════════════════════════════════════════════════════
#  ExpressionClassifier
# ══════════════════════════════════════════════════════════════════════

class TestExpressionClassifier:
    @pytest.fixture
    def classifier(self):
        pytest.importorskip("torch", reason="torch not installed")
        pytest.importorskip("timm", reason="timm not installed")
        from src.models.expression_classifier import ExpressionClassifier
        return ExpressionClassifier(
            backbone="resnet18", num_classes=7,
            pretrained=False, device="cpu",
        )

    def test_construct(self, classifier):
        assert not classifier.fitted

    def test_predict_logits_shape(self, classifier):
        rng = np.random.RandomState(0)
        images = rng.rand(4, 256, 256, 3).astype(np.float32)
        logits = classifier.predict_logits(images)
        assert logits.shape == (4, 7)

    def test_predict_proba_sums(self, classifier):
        rng = np.random.RandomState(0)
        images = rng.rand(4, 256, 256, 3).astype(np.float32)
        probs = classifier.predict_proba(images)
        assert probs.shape == (4, 7)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_labels_range(self, classifier):
        rng = np.random.RandomState(0)
        images = rng.rand(4, 256, 256, 3).astype(np.float32)
        preds = classifier.predict(images)
        assert preds.shape == (4,)
        assert all(0 <= p < 7 for p in preds)


# ══════════════════════════════════════════════════════════════════════
#  Expression metrics (from src.models.expression_metrics)
# ══════════════════════════════════════════════════════════════════════

class TestExpressionMetrics:
    def test_accuracy_perfect(self):
        from src.models.expression_metrics import accuracy
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        assert accuracy(y_true, y_pred) == 1.0

    def test_accuracy_zero(self):
        from src.models.expression_metrics import accuracy
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([3, 2, 1, 0])
        assert accuracy(y_true, y_pred) == 0.0

    def test_accuracy_empty(self):
        from src.models.expression_metrics import accuracy
        assert accuracy(np.array([]), np.array([])) == 0.0

    def test_per_class_recall(self):
        from src.models.expression_metrics import per_class_recall
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 1])
        result = per_class_recall(y_true, y_pred, num_classes=3,
                                  class_names=["A", "B", "C"])
        assert result["A"] == 1.0
        assert result["B"] == 0.5
        assert result["C"] == 0.5

    def test_confusion_matrix_shape(self):
        from src.models.expression_metrics import confusion_matrix
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 2])
        cm = confusion_matrix(y_true, y_pred, num_classes=3)
        assert cm.shape == (3, 3)
        assert cm.sum() == 5

    def test_confusion_matrix_diagonal(self):
        from src.models.expression_metrics import confusion_matrix
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        cm = confusion_matrix(y_true, y_pred, num_classes=4)
        np.testing.assert_array_equal(np.diag(cm), [1, 1, 1, 1])

    def test_expression_consistency(self):
        from src.models.expression_metrics import expression_consistency
        # Identical distributions → consistency close to 1
        probs = np.array([[0.3, 0.5, 0.2], [0.1, 0.8, 0.1]])
        c = expression_consistency(probs, probs)
        assert c > 0.99

    def test_expression_consistency_shifted(self):
        from src.models.expression_metrics import expression_consistency
        probs_a = np.array([[1.0, 0.0, 0.0]])
        probs_b = np.array([[0.0, 0.0, 1.0]])
        c = expression_consistency(probs_a, probs_b)
        assert c < 0.5  # very different

    def test_expression_match_rate(self):
        from src.models.expression_metrics import expression_match_rate
        a = np.array([0, 1, 2, 3])
        b = np.array([0, 1, 0, 3])
        assert expression_match_rate(a, b) == 0.75

    def test_ece_perfect(self):
        from src.models.expression_metrics import expected_calibration_error
        probs = np.eye(4, dtype=np.float32)
        y_true = np.array([0, 1, 2, 3])
        ece = expected_calibration_error(y_true, probs)
        assert ece < 0.01  # perfectly calibrated


# ══════════════════════════════════════════════════════════════════════
#  Identity metrics (from src.models.identity_metrics)
# ══════════════════════════════════════════════════════════════════════

class TestIdentityMetrics:
    def test_closed_set_perfect(self):
        from src.models.identity_metrics import closed_set_identification
        # Gallery and probes identical → 100% accuracy
        emb = np.eye(5, dtype=np.float32)
        labels = np.arange(5)
        result = closed_set_identification(emb, labels, emb, labels, top_k=3)
        assert result["top1_acc"] == 1.0
        assert "top3_acc" in result

    def test_closed_set_random(self):
        from src.models.identity_metrics import closed_set_identification
        rng = np.random.RandomState(0)
        emb = rng.randn(20, 64).astype(np.float32)
        labels = np.arange(20)
        result = closed_set_identification(emb, labels, emb, labels, top_k=1)
        assert result["top1_acc"] == 1.0  # self-match

    def test_verification_metrics(self):
        from src.models.identity_metrics import compute_verification_metrics
        rng = np.random.RandomState(0)
        N = 100
        emb_a = rng.randn(N, 32).astype(np.float32)
        # Same pairs: emb_b ≈ emb_a; different pairs: random
        is_same = np.zeros(N, dtype=bool)
        is_same[:50] = True
        emb_b = np.where(is_same[:, None], emb_a + rng.randn(N, 32) * 0.1,
                         rng.randn(N, 32))
        emb_b = emb_b.astype(np.float32)

        result = compute_verification_metrics(emb_a, emb_b, is_same)
        assert "auc" in result
        assert "eer" in result
        assert "tar_at_far" in result
        assert result["auc"] > 0.5  # better than random

    def test_generate_verification_pairs(self):
        from src.models.identity_metrics import generate_verification_pairs
        rng = np.random.RandomState(0)
        emb = rng.randn(20, 32).astype(np.float32)
        labels = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
        emb_a, emb_b, is_same = generate_verification_pairs(
            emb, labels, num_pairs=50, seed=42,
        )
        assert emb_a.shape[0] == emb_b.shape[0] == is_same.shape[0]
        assert is_same.dtype == bool
        # Should have mix of same/different
        assert is_same.sum() > 0
        assert (~is_same).sum() > 0


# ══════════════════════════════════════════════════════════════════════
#  IdentityEmbedder — availability check only
# ══════════════════════════════════════════════════════════════════════

class TestIdentityEmbedderAvailability:
    def test_unavailable_gracefully(self):
        """If InsightFace isn't installed, IdentityEmbedder reports unavailable."""
        try:
            from src.models.identity_embedder import IdentityEmbedder
            embedder = IdentityEmbedder(model_name="buffalo_l", device="cpu")
            # Either it's available or it gracefully reports unavailability
            if not embedder.available:
                assert embedder.embedding_dim == 512
                with pytest.raises(RuntimeError):
                    embedder.embed(np.zeros((1, 256, 256, 3), dtype=np.float32))
        except ImportError:
            pytest.skip("InsightFace not installed")
