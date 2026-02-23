"""
Tier A — Classical baselines sweep.

Run blur / pixelate / blackout across a range of strength parameters and
collect the privacy–utility frontier. This is the first experiment that
validates the entire evaluation harness end-to-end.

Sweep grid (from ``configs/anonymizers/*.yaml``)::

    blur      : kernel_size ∈ {11, 21, 31, 51, 71, 101}
    pixelate  : block_size  ∈ {4, 8, 12, 16, 24, 32}
    blackout  : (single operating point — no param)

Usage
-----
::

    python scripts/run_baseline_sweep.py
    python scripts/run_baseline_sweep.py --device cpu --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Sweep definitions ─────────────────────────────────────────────────────

SWEEP_GRID: dict[str, list[dict[str, Any]]] = {
    "blur": [
        {"kernel_size": k} for k in [11, 21, 31, 51, 71, 101]
    ],
    "pixelate": [
        {"block_size": b} for b in [4, 8, 12, 16, 24, 32]
    ],
    "blackout": [
        {}  # single operating point
    ],
}


def run_baseline_sweep(
    *,
    dataset: str = "fer2013",
    data_root: str = "data/fer2013",
    csv_file: str = "fer2013.csv",
    max_samples: Optional[int] = None,
    device: str = "cpu",
    seed: int = 42,
    output_dir: str = "results/baseline_sweep",
    compute_fid: bool = False,
) -> list[dict]:
    """
    Execute the classical baselines sweep experiment.

    Returns
    -------
    list of result dicts (one per anonymizer × parameter combo).
    """
    np.random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    images, expr_labels, id_labels = _load_data(
        dataset, data_root, csv_file, max_samples, seed,
    )
    N = len(images)
    logger.info("Loaded %d images for sweep.", N)

    # ── Pre-compute expression probs on originals (teacher) ────────────
    probs_original = _get_expression_probs(images, device)

    # ── Pre-compute identity embeddings on originals ───────────────────
    original_embeddings = _get_identity_embeddings(images, device)

    # ── Sweep ──────────────────────────────────────────────────────────
    all_results: list[dict] = []

    for anon_name, param_list in SWEEP_GRID.items():
        for params in param_list:
            run_id = _make_run_id(anon_name, params)
            logger.info("━━━ %s ━━━", run_id)

            t0 = time.time()

            # Anonymize
            anon_images = _anonymize_dataset(images, anon_name, params)

            # Expression probs on anonymised
            probs_anonymized = _get_expression_probs(anon_images, device)

            # Identity embeddings on anonymised
            anon_embeddings = _get_identity_embeddings(anon_images, device)

            # Run unified evaluation
            from src.runner.evaluator import run_evaluation

            result = run_evaluation(
                anonymizer_name=anon_name,
                original_images=images,
                anonymized_images=anon_images,
                expression_labels=expr_labels,
                identity_labels=id_labels,
                original_embeddings=original_embeddings,
                anonymized_embeddings=anon_embeddings,
                probs_original=probs_original,
                probs_anonymized=probs_anonymized,
                anonymizer_params=params,
                run_privacy=id_labels is not None and original_embeddings is not None,
                run_expression=expr_labels is not None and probs_original is not None,
                run_realism=True,
                compute_fid=compute_fid,
                compute_lpips=True,
                train_adaptive=id_labels is not None,
                device=device,
                output_dir=str(out),
                run_id=run_id,
            )

            row = result.to_dict()
            row["_run_id"] = run_id
            row["_elapsed"] = time.time() - t0
            all_results.append(row)

            logger.info("  Done in %.1fs", row["_elapsed"])

    # ── Aggregate frontier CSV ─────────────────────────────────────────
    _write_frontier_csv(all_results, out / "frontier.csv")

    # ── Plot frontier ──────────────────────────────────────────────────
    _plot_results(out)

    logger.info("Sweep complete — %d runs. Results in %s", len(all_results), out)
    return all_results


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_data(
    dataset: str, data_root: str, csv_file: str,
    max_samples: Optional[int], seed: int,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load images + labels, optionally subsampled."""
    if dataset == "fer2013":
        from src.data.fer2013_adapter import FER2013Dataset

        ds = FER2013Dataset(root=data_root, csv_file=csv_file, split="test")
        images = np.stack([ds[i][0] for i in range(len(ds))])
        expr_labels = np.array([ds[i][1] for i in range(len(ds))])
        # FER-2013 has no identity labels — create pseudo-IDs per sample
        id_labels = np.arange(len(images))
    elif dataset == "pins":
        from src.data.pins_adapter import PinsFaceDataset

        ds = PinsFaceDataset(root=data_root, split="test")
        images = np.stack([ds[i][0] for i in range(len(ds))])
        id_labels = np.array([ds[i][1] for i in range(len(ds))])
        expr_labels = None
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if max_samples and len(images) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(images), max_samples, replace=False)
        images = images[idx]
        if expr_labels is not None:
            expr_labels = expr_labels[idx]
        if id_labels is not None:
            id_labels = id_labels[idx]

    return images, expr_labels, id_labels


def _anonymize_dataset(
    images: np.ndarray,
    anon_name: str,
    params: dict,
) -> np.ndarray:
    """Anonymize all images and return as array."""
    from src.anonymizers import get_anonymizer
    from src.data.contracts import FaceCrop

    anon = get_anonymizer(anon_name, **params)
    results = []
    for i in range(len(images)):
        crop = FaceCrop(image=images[i])
        try:
            res = anon.anonymize_single(crop)
            results.append(res.image)
        except Exception as exc:
            logger.warning("Anonymize failed (sample %d): %s", i, exc)
            results.append(images[i])
    return np.stack(results)


def _get_expression_probs(
    images: np.ndarray, device: str,
) -> Optional[np.ndarray]:
    """Get teacher expression probabilities. Returns None on failure."""
    try:
        from src.models.expression_teacher import ExpressionTeacher

        teacher = ExpressionTeacher(backbone="resnet18", device=device)
        # Batch to avoid OOM
        probs = []
        bs = 64
        for i in range(0, len(images), bs):
            probs.append(teacher.predict_proba(images[i:i + bs]))
        return np.concatenate(probs, axis=0)
    except Exception as exc:
        logger.warning("Expression teacher unavailable: %s", exc)
        return None


def _get_identity_embeddings(
    images: np.ndarray, device: str,
) -> Optional[np.ndarray]:
    """Get ArcFace embeddings. Returns None on failure."""
    try:
        from src.models.identity_embedder import IdentityEmbedder

        embedder = IdentityEmbedder(device=device)
        if not embedder.available:
            return None
        return embedder.embed(images)
    except Exception as exc:
        logger.warning("Identity embedder unavailable: %s", exc)
        return None


def _make_run_id(name: str, params: dict) -> str:
    """Build a readable run ID."""
    if not params:
        return name
    parts = [name] + [f"{k}{v}" for k, v in sorted(params.items())]
    return "_".join(parts)


def _write_frontier_csv(results: list[dict], csv_path: Path) -> None:
    """Aggregate frontier rows into a single CSV."""
    import csv

    if not results:
        return

    # Collect all frontier_row.csv files in the output dir
    rows = []
    parent = csv_path.parent
    for p in sorted(parent.rglob("frontier_row.csv")):
        try:
            with open(p, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        except Exception:
            pass

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Frontier CSV: %s (%d rows)", csv_path, len(rows))


def _plot_results(out_dir: Path) -> None:
    """Generate frontier plot if matplotlib is available."""
    try:
        from src.runner.frontier_plot import collect_frontier_csv, plot_frontier

        rows = collect_frontier_csv(str(out_dir))
        if rows:
            plot_frontier(
                rows,
                output_path=str(out_dir / "frontier_plot.png"),
            )
    except Exception as exc:
        logger.warning("Could not generate frontier plot: %s", exc)


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Tier A: Classical baselines sweep")
    parser.add_argument("--dataset", default="fer2013")
    parser.add_argument("--data-root", default="data/fer2013")
    parser.add_argument("--csv-file", default="fer2013.csv")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/baseline_sweep")
    parser.add_argument("--compute-fid", action="store_true")
    args = parser.parse_args()

    run_baseline_sweep(
        dataset=args.dataset,
        data_root=args.data_root,
        csv_file=args.csv_file,
        max_samples=args.max_samples,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        compute_fid=args.compute_fid,
    )


if __name__ == "__main__":
    main()
