"""
Ablation / experiment report generator.

Aggregates results from multiple evaluation runs, generates:
  - Comparison tables (Markdown / CSV)
  - Summary statistics
  - Per-anonymizer cards
  - LaTeX table snippets
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────

def _safe_float(v: Any, default: float = float("nan")) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _fmt(v: float, decimals: int = 4) -> str:
    if math.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


# ── data loading ───────────────────────────────────────────────────────────

def load_all_metrics(results_dir: str = "results") -> list[dict]:
    """
    Load every ``metrics.json`` found under *results_dir*.

    Returns list of dicts (the full metrics payloads).
    """
    base = Path(results_dir)
    entries: list[dict] = []
    if not base.exists():
        logger.warning("Results directory %s not found.", results_dir)
        return entries

    for p in sorted(base.rglob("metrics.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            data["_path"] = str(p)
            entries.append(data)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", p, exc)

    logger.info("Loaded %d metrics files from %s", len(entries), results_dir)
    return entries


def load_frontier_rows(results_dir: str = "results") -> list[dict]:
    """
    Load every ``frontier_row.csv`` under *results_dir*.
    """
    from src.runner.frontier_plot import collect_frontier_csv
    return collect_frontier_csv(results_dir)


# ── Markdown report ───────────────────────────────────────────────────────

_METRIC_COLUMNS = [
    ("privacy_score",     "Privacy ↑"),
    ("closed_set_top1",   "ID-Top1 ↓"),
    ("verification_auc",  "Ver-AUC ↓"),
    ("adaptive_acc",      "Attacker ↓"),
    ("utility_score",     "Utility ↑"),
    ("acc_anonymized",    "Acc-Anon"),
    ("expr_consistency",  "Expr-Cons"),
    ("expr_match_rate",   "Match-Rate"),
    ("fid",               "FID ↓"),
    ("lpips_mean",        "LPIPS ↓"),
    ("psnr_mean",         "PSNR ↑"),
    ("ssim_mean",         "SSIM ↑"),
]


def generate_markdown_table(
    rows: list[dict],
    columns: Optional[list[tuple[str, str]]] = None,
) -> str:
    """
    Build a Markdown comparison table from frontier rows.

    Parameters
    ----------
    rows    : list of frontier row dicts.
    columns : (key, header) pairs. Defaults to :data:`_METRIC_COLUMNS`.

    Returns
    -------
    Markdown string.
    """
    if columns is None:
        columns = _METRIC_COLUMNS

    # Header
    headers = ["Anonymizer"] + [h for _, h in columns]
    sep = ["-" * max(len(h), 3) for h in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]

    for r in rows:
        anon = r.get("anonymizer", "?")
        vals = [_fmt(_safe_float(r.get(k, ""))) for k, _ in columns]
        lines.append("| " + " | ".join([anon] + vals) + " |")

    return "\n".join(lines)


def generate_latex_table(
    rows: list[dict],
    columns: Optional[list[tuple[str, str]]] = None,
    caption: str = "Anonymiser comparison.",
    label: str = "tab:comparison",
) -> str:
    """Build a LaTeX tabular snippet for the paper."""
    if columns is None:
        columns = _METRIC_COLUMNS

    n_cols = 1 + len(columns)
    col_spec = "l" + "c" * len(columns)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header_cells = ["Anonymizer"] + [h for _, h in columns]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for r in rows:
        anon = r.get("anonymizer", "?")
        vals = [_fmt(_safe_float(r.get(k, ""))) for k, _ in columns]
        lines.append(" & ".join([anon] + vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── summary statistics ─────────────────────────────────────────────────────

def compute_summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    """
    For each anonymizer, compute mean ± std of every numeric metric.

    Returns
    -------
    ``{anonymizer: {metric_mean: float, metric_std: float, ...}}``
    """
    import numpy as np  # local to keep module light

    groups: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        anon = r.get("anonymizer", "unknown")
        groups.setdefault(anon, {})
        for k, v in r.items():
            if k.startswith("_") or k in ("anonymizer", "params"):
                continue
            fv = _safe_float(v)
            if not math.isnan(fv):
                groups[anon].setdefault(k, []).append(fv)

    summary: dict[str, dict[str, float]] = {}
    for anon, metrics in groups.items():
        s: dict[str, float] = {}
        for k, vals in metrics.items():
            arr = np.array(vals)
            s[f"{k}_mean"] = float(arr.mean())
            s[f"{k}_std"] = float(arr.std())
            s[f"{k}_n"] = float(len(vals))
        summary[anon] = s
    return summary


# ── report generation entry-point ──────────────────────────────────────────

def generate_report(
    results_dir: str = "results",
    report_dir: str = "reports",
    *,
    include_latex: bool = True,
) -> Path:
    """
    Auto-generate a full comparison report.

    Outputs
    -------
    - ``reports/comparison.md``   — Markdown table + summary
    - ``reports/comparison.csv``  — Flat CSV of all runs
    - ``reports/comparison.tex``  — LaTeX table (optional)
    - ``reports/summary.json``    — Per-anonymizer summary stats

    Returns
    -------
    Path to the report directory.
    """
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_frontier_rows(results_dir)
    if not rows:
        logger.warning("No evaluation runs found in %s.", results_dir)
        (out / "comparison.md").write_text(
            "# Comparison Report\n\nNo evaluation runs found.\n"
        )
        return out

    # ── Markdown ───────────────────────────────────────────────────────
    md_parts = [
        "# Anonymiser Comparison Report\n",
        f"Generated from `{results_dir}/` — {len(rows)} run(s).\n",
        "## Results Table\n",
        generate_markdown_table(rows),
        "",
    ]
    (out / "comparison.md").write_text("\n".join(md_parts), encoding="utf-8")
    logger.info("Markdown report → %s", out / "comparison.md")

    # ── CSV ────────────────────────────────────────────────────────────
    csv_path = out / "comparison.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        logger.info("CSV export → %s", csv_path)

    # ── LaTeX ──────────────────────────────────────────────────────────
    if include_latex:
        tex = generate_latex_table(rows)
        (out / "comparison.tex").write_text(tex, encoding="utf-8")
        logger.info("LaTeX table → %s", out / "comparison.tex")

    # ── Summary stats ──────────────────────────────────────────────────
    summary = compute_summary(rows)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary stats → %s", out / "summary.json")

    return out
