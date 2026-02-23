"""
Tier C — GAN anonymizer comparison.

Run GANonymization, DeepPrivacy2, and CIAGAN with default settings and
compare them on the privacy–utility frontier.  GAN backends must be
installed in ``third_party/``; unavailable backends are skipped gracefully.

Usage
-----
::

    python scripts/run_gan_comparison.py
    python scripts/run_gan_comparison.py --device cuda --max-samples 500
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

GAN_METHODS: list[dict] = [
    {"name": "ganonymization", "params": {"weights": "publication"}},
    {"name": "deep_privacy2",  "params": {"model": "face"}},
    {"name": "ciagan",         "params": {}},
]


def run_gan_comparison(
    *,
    dataset: str = "fer2013",
    data_root: str = "data/fer2013",
    csv_file: str = "fer2013.csv",
    max_samples: Optional[int] = None,
    device: str = "cpu",
    seed: int = 42,
    output_dir: str = "results/gan_comparison",
) -> list[dict]:
    """Run the GAN anonymizer comparison experiment."""
    np.random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    probs_original = _get_expression_probs(images, device)
    original_embeddings = _get_identity_embeddings(images, device)

    all_results: list[dict] = []

    for method in GAN_METHODS:
        anon_name = method["name"]
        params = {**method["params"], "device": device}
        run_id = anon_name
        logger.info("━━━ %s ━━━", run_id)

        # Try to instantiate — skip if third-party repo missing
        from src.anonymizers import get_anonymizer

        try:
            anon = get_anonymizer(anon_name, **params)
        except Exception as exc:
            logger.warning("Skipping %s: %s", anon_name, exc)
            continue

        t0 = time.time()

        from src.data.contracts import FaceCrop

        anon_images = []
        failed = 0
        for i in range(N):
            crop = FaceCrop(image=images[i])
            try:
                res = anon.anonymize_single(crop)
                anon_images.append(res.image)
            except Exception as exc:
                if failed < 3:
                    logger.warning("  Sample %d failed: %s", i, exc)
                elif failed == 3:
                    logger.warning("  (suppressing further warnings)")
                failed += 1
                anon_images.append(images[i])

        if failed == N:
            logger.error("  All samples failed for %s — skipping.", anon_name)
            continue

        anon_images_arr = np.stack(anon_images)

        probs_anonymized = _get_expression_probs(anon_images_arr, device)
        anon_embeddings = _get_identity_embeddings(anon_images_arr, device)

        from src.runner.evaluator import run_evaluation

        result = run_evaluation(
            anonymizer_name=anon_name,
            original_images=images,
            anonymized_images=anon_images_arr,
            expression_labels=expr_labels,
            identity_labels=id_labels,
            original_embeddings=original_embeddings,
            anonymized_embeddings=anon_embeddings,
            probs_original=probs_original,
            probs_anonymized=probs_anonymized,
            anonymizer_params=params,
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
        row["_failed_samples"] = failed
        all_results.append(row)
        logger.info("  %s done in %.1fs (%d failures)", anon_name, row["_elapsed"], failed)

    _write_frontier_csv(all_results, out / "frontier.csv")
    _plot_results(out)
    logger.info("GAN comparison complete — %d methods.", len(all_results))
    return all_results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="Tier C: GAN anonymizer comparison")
    p.add_argument("--dataset", default="fer2013")
    p.add_argument("--data-root", default="data/fer2013")
    p.add_argument("--csv-file", default="fer2013.csv")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/gan_comparison")
    args = p.parse_args()

    run_gan_comparison(
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
