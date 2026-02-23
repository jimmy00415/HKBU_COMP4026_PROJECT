"""
Profiling & bottleneck analysis.

Provides lightweight profiling hooks for the end-to-end pipeline
(preprocessing → anonymization → evaluation).  Generates a human-readable
profiling report and optionally uses ``cProfile`` for fine-grained stats.

Usage
-----
::

    python -m src.runner.profiler --preset quick
    python -m src.runner.profiler --preset full --device cuda
"""

from __future__ import annotations

import cProfile
import io
import logging
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Timer context manager ─────────────────────────────────────────────

@dataclass
class TimingRecord:
    """One timing entry."""
    name: str
    elapsed: float  # seconds
    extra: dict = field(default_factory=dict)


class PipelineProfiler:
    """
    Lightweight profiler that wraps pipeline stages.

    Example
    -------
    >>> prof = PipelineProfiler()
    >>> with prof.stage("detect_faces"):
    ...     faces = detector.detect(image)
    >>> prof.report()
    """

    def __init__(self) -> None:
        self._records: list[TimingRecord] = []
        self._start_wall: float = time.perf_counter()

    @contextmanager
    def stage(self, name: str, **extra: Any):
        """Context manager that times a pipeline stage."""
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self._records.append(TimingRecord(name=name, elapsed=elapsed, extra=extra))
        logger.debug("  ⏱  %s: %.4fs", name, elapsed)

    def add(self, name: str, elapsed: float, **extra: Any) -> None:
        """Manually add a timing record."""
        self._records.append(TimingRecord(name=name, elapsed=elapsed, extra=extra))

    @property
    def total_elapsed(self) -> float:
        return time.perf_counter() - self._start_wall

    @property
    def records(self) -> list[TimingRecord]:
        return list(self._records)

    def summary(self) -> dict[str, float]:
        """Return {stage_name: elapsed_seconds} dict."""
        out: dict[str, float] = {}
        for r in self._records:
            out[r.name] = out.get(r.name, 0.0) + r.elapsed
        return out

    def report(self, *, top_n: int = 20) -> str:
        """
        Generate a human-readable profiling report.

        Returns
        -------
        Markdown-formatted report string.
        """
        total = self.total_elapsed
        agg = self.summary()
        sorted_stages = sorted(agg.items(), key=lambda x: x[1], reverse=True)

        lines = [
            "# Pipeline Profiling Report",
            "",
            f"**Total wall time:** {total:.2f}s",
            f"**Stages recorded:** {len(self._records)}",
            "",
            "| Stage | Time (s) | % of Total |",
            "|-------|----------|------------|",
        ]
        for name, elapsed in sorted_stages[:top_n]:
            pct = (elapsed / total * 100) if total > 0 else 0
            lines.append(f"| {name} | {elapsed:.4f} | {pct:.1f}% |")

        # Identify bottleneck
        if sorted_stages:
            bottleneck, bt_time = sorted_stages[0]
            lines.extend([
                "",
                f"**Bottleneck:** `{bottleneck}` ({bt_time:.2f}s, "
                f"{bt_time / total * 100:.1f}% of total)",
            ])

        lines.append("")
        return "\n".join(lines)

    def save_report(self, path: str | Path) -> None:
        """Write the profiling report to a file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.report(), encoding="utf-8")
        logger.info("Profiling report saved to %s", p)

    def to_dict(self) -> dict[str, Any]:
        """Serialise all records for JSON output."""
        return {
            "total_elapsed": self.total_elapsed,
            "stages": [
                {"name": r.name, "elapsed": r.elapsed, **r.extra}
                for r in self._records
            ],
            "summary": self.summary(),
        }


# ── cProfile integration ──────────────────────────────────────────────

@contextmanager
def cprofile_context(sort_by: str = "cumulative", top_n: int = 30):
    """
    Context manager that runs a ``cProfile`` session and prints results.

    Usage
    -----
    >>> with cprofile_context():
    ...     expensive_function()
    """
    pr = cProfile.Profile()
    pr.enable()
    yield pr
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
    ps.print_stats(top_n)
    logger.info("cProfile results:\n%s", s.getvalue())


# ── GPU profiling helpers ──────────────────────────────────────────────

def get_gpu_memory_mb() -> Optional[float]:
    """Return current GPU memory usage in MB, or None if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except ImportError:
        pass
    return None


def get_gpu_peak_memory_mb() -> Optional[float]:
    """Return peak GPU memory usage in MB, or None if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except ImportError:
        pass
    return None


def reset_gpu_peak_memory() -> None:
    """Reset GPU peak memory stats."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


# ── Quick benchmark runner ─────────────────────────────────────────────

def profile_pipeline(
    *,
    dataset: str = "fer2013",
    data_root: str = "data/fer2013",
    csv_file: str = "fer2013.csv",
    anonymizer: str = "blur",
    anon_params: Optional[dict] = None,
    max_samples: int = 50,
    device: str = "cpu",
    output_dir: str = "results/profiling",
) -> PipelineProfiler:
    """
    Run and profile the full pipeline on a small sample.

    Returns the populated PipelineProfiler.
    """
    prof = PipelineProfiler()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if anon_params is None:
        anon_params = {"kernel_size": 31}

    reset_gpu_peak_memory()

    # ── Stage 1: Data loading ──────────────────────────────────────
    with prof.stage("data_loading"):
        from scripts.run_baseline_sweep import _load_data
        images, expr_labels, id_labels = _load_data(
            dataset, data_root, csv_file, max_samples, seed=42,
        )
    N = len(images)

    # ── Stage 2: Identity embedding ────────────────────────────────
    with prof.stage("identity_embedding", n_samples=N):
        from scripts.run_baseline_sweep import _get_identity_embeddings
        orig_emb = _get_identity_embeddings(images, device)

    # ── Stage 3: Expression teacher ────────────────────────────────
    with prof.stage("expression_teacher", n_samples=N):
        from scripts.run_baseline_sweep import _get_expression_probs
        probs_orig = _get_expression_probs(images, device)

    # ── Stage 4: Anonymization ─────────────────────────────────────
    with prof.stage("anonymization", anonymizer=anonymizer, n_samples=N):
        from scripts.run_baseline_sweep import _anonymize_dataset
        anon_images = _anonymize_dataset(images, anonymizer, anon_params)

    # ── Stage 5: Anonymised embedding ──────────────────────────────
    with prof.stage("anonymized_embedding", n_samples=N):
        anon_emb = _get_identity_embeddings(anon_images, device)

    # ── Stage 6: Anonymised expression probs ───────────────────────
    with prof.stage("anonymized_expression", n_samples=N):
        probs_anon = _get_expression_probs(anon_images, device)

    # ── Stage 7: Evaluation ────────────────────────────────────────
    with prof.stage("evaluation", n_samples=N):
        from src.runner.evaluator import run_evaluation
        run_evaluation(
            anonymizer_name=anonymizer,
            original_images=images,
            anonymized_images=anon_images,
            expression_labels=expr_labels,
            identity_labels=id_labels,
            original_embeddings=orig_emb,
            anonymized_embeddings=anon_emb,
            probs_original=probs_orig,
            probs_anonymized=probs_anon,
            anonymizer_params=anon_params,
            run_privacy=orig_emb is not None and anon_emb is not None,
            run_expression=probs_orig is not None and expr_labels is not None,
            run_realism=True,
            compute_fid=False,
            compute_lpips=False,
            device=device,
            output_dir=str(out),
            run_id="profile_run",
        )

    # Record GPU stats
    gpu_mem = get_gpu_memory_mb()
    gpu_peak = get_gpu_peak_memory_mb()
    if gpu_mem is not None:
        prof.add("gpu_current_mb", 0.0, value=gpu_mem)
    if gpu_peak is not None:
        prof.add("gpu_peak_mb", 0.0, value=gpu_peak)

    # Save report
    prof.save_report(out / "profiling_report.md")

    import json
    (out / "profiling_data.json").write_text(
        json.dumps(prof.to_dict(), indent=2, default=str), encoding="utf-8",
    )
    logger.info("Profiling complete. Total: %.2fs", prof.total_elapsed)

    return prof


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s",
    )

    p = argparse.ArgumentParser(description="Pipeline profiler")
    p.add_argument("--preset", choices=["quick", "full"], default="quick")
    p.add_argument("--device", default="cpu")
    p.add_argument("--anonymizer", default="blur")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", default="results/profiling")
    args = p.parse_args()

    max_s = args.max_samples or (20 if args.preset == "quick" else 200)

    profile_pipeline(
        max_samples=max_s,
        device=args.device,
        anonymizer=args.anonymizer,
        output_dir=args.output_dir,
    )
