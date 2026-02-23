"""
Privacy–Utility frontier plotter.

Reads ``frontier_row.csv`` files from multiple evaluation runs,
plots privacy-score vs. utility-score scatter plots per anonymizer,
identifies the Pareto front, and saves publication-ready figures.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def collect_frontier_csv(
    results_dir: str = "results",
) -> list[dict]:
    """
    Walk *results_dir*, find every ``run_*/frontier_row.csv``, and parse
    them into a list of row dicts.

    Parameters
    ----------
    results_dir : root results directory.

    Returns
    -------
    list of dicts, one per evaluation run.
    """
    rows: list[dict] = []
    base = Path(results_dir)
    if not base.exists():
        logger.warning("Results directory %s does not exist.", results_dir)
        return rows

    for csv_path in sorted(base.rglob("frontier_row.csv")):
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    r["_source"] = str(csv_path)
                    rows.append(r)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", csv_path, exc)
    logger.info("Collected %d frontier rows from %s", len(rows), results_dir)
    return rows


def _safe_float(value: str | float, default: float = float("nan")) -> float:
    """Convert a value to float, returning *default* on failure."""
    if value == "" or value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def compute_pareto_front(
    points: list[tuple[float, float]],
) -> list[int]:
    """
    Given a list of (x, y) tuples where **both axes are to be maximised**,
    return indices of the Pareto-optimal points.

    Parameters
    ----------
    points : list of (x_utility, y_privacy) tuples.

    Returns
    -------
    Sorted list of indices on the Pareto front.
    """
    import math

    n = len(points)
    if n == 0:
        return []

    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        xi, yi = points[i]
        if math.isnan(xi) or math.isnan(yi):
            dominated[i] = True
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            xj, yj = points[j]
            if math.isnan(xj) or math.isnan(yj):
                dominated[j] = True
                continue
            # j dominates i  ↔  j ≥ i on both and strictly > on at least one
            if xj >= xi and yj >= yi and (xj > xi or yj > yi):
                dominated[i] = True
                break

    front = [i for i in range(n) if not dominated[i]]
    # sort by x for nice plotting
    front.sort(key=lambda i: points[i][0])
    return front


def plot_frontier(
    rows: list[dict],
    *,
    x_key: str = "utility_score",
    y_key: str = "privacy_score",
    x_label: str = "Utility Score",
    y_label: str = "Privacy Score",
    title: str = "Privacy–Utility Frontier",
    output_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot a privacy-vs-utility scatter chart with Pareto front highlighted.

    Parameters
    ----------
    rows        : list of frontier row dicts (from :func:`collect_frontier_csv`).
    x_key/y_key : dict keys for x/y values.
    output_path : if given, save figure to this path (PNG/PDF/SVG).
    show        : if True, call ``plt.show()``.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib is required for plotting. Install with: "
            "pip install matplotlib"
        )
        return

    if not rows:
        logger.warning("No data to plot.")
        return

    # Group by anonymizer
    groups: dict[str, list[dict]] = {}
    for r in rows:
        anon = r.get("anonymizer", "unknown")
        groups.setdefault(anon, []).append(r)

    fig, ax = plt.subplots(figsize=(8, 6))

    all_points: list[tuple[float, float]] = []
    all_labels: list[str] = []
    colours = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for idx, (anon_name, group_rows) in enumerate(sorted(groups.items())):
        colour = colours[idx % len(colours)]
        xs = [_safe_float(r.get(x_key, "")) for r in group_rows]
        ys = [_safe_float(r.get(y_key, "")) for r in group_rows]

        ax.scatter(xs, ys, label=anon_name, color=colour, s=60, alpha=0.8)

        for x, y in zip(xs, ys):
            all_points.append((x, y))
            all_labels.append(anon_name)

    # Pareto front
    if all_points:
        front_indices = compute_pareto_front(all_points)
        if len(front_indices) >= 2:
            fx = [all_points[i][0] for i in front_indices]
            fy = [all_points[i][1] for i in front_indices]
            ax.plot(
                fx,
                fy,
                "k--",
                linewidth=1.5,
                alpha=0.7,
                label="Pareto front",
            )
        # highlight Pareto points
        for i in front_indices:
            ax.scatter(
                [all_points[i][0]],
                [all_points[i][1]],
                edgecolors="black",
                facecolors="none",
                s=150,
                linewidths=2,
                zorder=5,
            )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150)
        logger.info("Frontier plot saved to %s", out)

    if show:
        plt.show()

    plt.close(fig)


def plot_metric_comparison(
    rows: list[dict],
    *,
    metric_keys: Optional[list[str]] = None,
    output_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Bar chart comparing anonymizers on multiple scalar metrics.

    Parameters
    ----------
    rows        : frontier rows.
    metric_keys : which keys to plot (default: a reasonable subset).
    output_path : save path.
    show        : display interactively.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting.")
        return

    if not rows:
        logger.warning("No data to plot.")
        return

    if metric_keys is None:
        metric_keys = [
            "privacy_score",
            "utility_score",
            "psnr_mean",
            "ssim_mean",
        ]

    # Aggregate per anonymizer (take mean if multiple runs)
    agg: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        anon = r.get("anonymizer", "unknown")
        agg.setdefault(anon, {k: [] for k in metric_keys})
        for k in metric_keys:
            val = _safe_float(r.get(k, ""))
            if not __import__("math").isnan(val):
                agg[anon][k].append(val)

    import numpy as np

    anonymizers = sorted(agg.keys())
    n_metrics = len(metric_keys)
    n_anon = len(anonymizers)

    if n_anon == 0:
        logger.warning("No anonymizers found in data.")
        return

    x = np.arange(n_anon)
    width = 0.8 / max(n_metrics, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_anon * 1.5), 5))

    for i, key in enumerate(metric_keys):
        means = []
        for anon in anonymizers:
            vals = agg[anon].get(key, [])
            means.append(float(np.mean(vals)) if vals else 0.0)
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=key.replace("_", " "))

    ax.set_xticks(x)
    ax.set_xticklabels(anonymizers, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Anonymizer Comparison")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150)
        logger.info("Comparison chart saved to %s", out)

    if show:
        plt.show()

    plt.close(fig)
