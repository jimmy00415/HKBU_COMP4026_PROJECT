"""
Tier B — k-Same averaging sweep.

Run k-Same with k ∈ {2, 5, 10, 20, 50} and collect the privacy–utility
frontier.  k-Same requires a gallery of identity embeddings, so this script
embeds the dataset first, then builds the gallery for each run.

Usage
-----
::

    python scripts/run_ksame_sweep.py
    python scripts/run_ksame_sweep.py --dataset pins --data-root data/pins
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

K_VALUES = [2, 5, 10, 20, 50]


def run_ksame_sweep(
    *,
    dataset: str = "fer2013",
    data_root: str = "data/fer2013",
    csv_file: str = "fer2013.csv",
    max_samples: Optional[int] = None,
    device: str = "cpu",
    seed: int = 42,
    output_dir: str = "results/ksame_sweep",
) -> list[dict]:
    """Execute the k-Same sweep experiment."""
    np.random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Reuse the loader from baseline sweep
    from scripts.run_baseline_sweep import (
        _get_expression_probs,
        _get_identity_embeddings,
        _load_data,
        _plot_results,
        _write_frontier_csv,
    )

    images, expr_labels, id_labels = _load_data(
        dataset, data_root, csv_file, max_samples, seed,
    )
    N = len(images)
    logger.info("Loaded %d images.", N)

    # Pre-compute originals
    probs_original = _get_expression_probs(images, device)
    original_embeddings = _get_identity_embeddings(images, device)

    if original_embeddings is None:
        logger.error(
            "k-Same requires identity embeddings. "
            "Install InsightFace or provide pre-computed embeddings."
        )
        return []

    all_results: list[dict] = []

    for k in K_VALUES:
        run_id = f"k_same_k{k}"
        logger.info("━━━ %s ━━━", run_id)
        t0 = time.time()

        # Build k-Same anonymizer and set gallery
        from src.anonymizers import get_anonymizer

        anon = get_anonymizer("k_same", k=k)
        anon.set_gallery(original_embeddings, images)

        # Anonymize
        from src.data.contracts import FaceCrop

        anon_images = []
        for i in range(N):
            crop = FaceCrop(image=images[i])
            try:
                res = anon.anonymize_single(
                    crop, query_embedding=original_embeddings[i],
                )
                anon_images.append(res.image)
            except Exception as exc:
                logger.warning("  Sample %d failed: %s", i, exc)
                anon_images.append(images[i])
        anon_images_arr = np.stack(anon_images)

        # Compute anonymised features
        probs_anonymized = _get_expression_probs(anon_images_arr, device)
        anon_embeddings = _get_identity_embeddings(anon_images_arr, device)

        # Evaluate
        from src.runner.evaluator import run_evaluation

        result = run_evaluation(
            anonymizer_name="k_same",
            original_images=images,
            anonymized_images=anon_images_arr,
            expression_labels=expr_labels,
            identity_labels=id_labels,
            original_embeddings=original_embeddings,
            anonymized_embeddings=anon_embeddings,
            probs_original=probs_original,
            probs_anonymized=probs_anonymized,
            anonymizer_params={"k": k},
            run_privacy=id_labels is not None and anon_embeddings is not None,
            run_expression=expr_labels is not None and probs_original is not None,
            run_realism=True,
            compute_fid=False,
            compute_lpips=True,
            device=device,
            output_dir=str(out),
            run_id=run_id,
        )

        row = result.to_dict()
        row["_run_id"] = run_id
        row["_elapsed"] = time.time() - t0
        all_results.append(row)
        logger.info("  k=%d done in %.1fs", k, row["_elapsed"])

    _write_frontier_csv(all_results, out / "frontier.csv")
    _plot_results(out)
    logger.info("k-Same sweep complete — %d runs.", len(all_results))
    return all_results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Tier B: k-Same sweep")
    parser.add_argument("--dataset", default="fer2013")
    parser.add_argument("--data-root", default="data/fer2013")
    parser.add_argument("--csv-file", default="fer2013.csv")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/ksame_sweep")
    args = parser.parse_args()

    run_ksame_sweep(
        dataset=args.dataset,
        data_root=args.data_root,
        csv_file=args.csv_file,
        max_samples=args.max_samples,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
