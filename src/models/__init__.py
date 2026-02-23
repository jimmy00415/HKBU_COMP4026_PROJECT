"""
Model wrappers and metrics.

Uses lazy imports so that heavyweight dependencies (torch, timm, insightface,
onnxruntime) are only loaded when the specific module is first accessed.
"""

from __future__ import annotations

_LAZY_MODULES = {
    # Identity
    "IdentityEmbedder":       "src.models.identity_embedder",
    "identity_embedder":      "src.models.identity_embedder",
    # Identity metrics
    "identity_metrics":       "src.models.identity_metrics",
    "closed_set_identification": "src.models.identity_metrics",
    "compute_verification_metrics": "src.models.identity_metrics",
    "generate_verification_pairs":  "src.models.identity_metrics",
    # Adaptive attacker
    "AdaptiveAttackerHead":   "src.models.adaptive_attacker",
    "adaptive_attacker":      "src.models.adaptive_attacker",
    # Expression classifier
    "ExpressionClassifier":   "src.models.expression_classifier",
    "expression_classifier":  "src.models.expression_classifier",
    # Expression teacher
    "ExpressionTeacher":      "src.models.expression_teacher",
    "expression_teacher":     "src.models.expression_teacher",
    # Expression metrics
    "expression_metrics":     "src.models.expression_metrics",
    "accuracy":               "src.models.expression_metrics",
    "per_class_recall":       "src.models.expression_metrics",
    "confusion_matrix":       "src.models.expression_metrics",
    "expression_consistency": "src.models.expression_metrics",
    "expected_calibration_error": "src.models.expression_metrics",
    "compute_expression_report":  "src.models.expression_metrics",
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        import importlib
        module = importlib.import_module(_LAZY_MODULES[name])
        attr = getattr(module, name, module)
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_MODULES.keys())
