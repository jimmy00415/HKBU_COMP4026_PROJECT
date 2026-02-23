"""
Runner package — lazy imports.
"""

_LAZY = {
    # evaluator
    "run_evaluation":            "src.runner.evaluator",
    "EvaluationResult":          "src.runner.evaluator",
    # frontier plot
    "collect_frontier_csv":      "src.runner.frontier_plot",
    "compute_pareto_front":      "src.runner.frontier_plot",
    "plot_frontier":             "src.runner.frontier_plot",
    "plot_metric_comparison":    "src.runner.frontier_plot",
    # report generator
    "generate_report":           "src.runner.report_generator",
    "generate_markdown_table":   "src.runner.report_generator",
    "generate_latex_table":      "src.runner.report_generator",
    # profiler (Phase 7)
    "PipelineProfiler":          "src.runner.profiler",
    "profile_pipeline":          "src.runner.profiler",
    # cache (Phase 7)
    "PreprocessCache":           "src.runner.cache",
    "get_or_compute_embeddings": "src.runner.cache",
    "get_or_compute_expression_probs": "src.runner.cache",
    # AMP utilities (Phase 7)
    "amp_inference_context":     "src.runner.amp_utils",
    "AMPTrainingContext":        "src.runner.amp_utils",
    "batch_inference":           "src.runner.amp_utils",
    "clear_gpu_cache":           "src.runner.amp_utils",
    # reproducibility (Phase 7)
    "seed_everything":           "src.runner.reproducibility",
    "save_manifest":             "src.runner.reproducibility",
    "verify_reproducibility":    "src.runner.reproducibility",
    # error handling (Phase 7)
    "retry":                     "src.runner.error_handling",
    "graceful_fallback":         "src.runner.error_handling",
    "validate_image":            "src.runner.error_handling",
    "handle_oom":                "src.runner.error_handling",
    "safe_load_image":           "src.runner.error_handling",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
