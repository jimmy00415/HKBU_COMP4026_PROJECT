"""
Metrics package — lazy imports to avoid heavy dependencies at import time.
"""

_LAZY = {
    "evaluate_privacy":    "src.metrics.privacy_metrics",
    "PrivacyReport":       "src.metrics.privacy_metrics",
    "evaluate_expression": "src.metrics.expression_metrics",
    "ExpressionReport":    "src.metrics.expression_metrics",
    "evaluate_realism":    "src.metrics.realism_metrics",
    "RealismReport":       "src.metrics.realism_metrics",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
