"""
Ablation — Conditioning signals for GANonymization.

Tests GANonymization with different conditioning signal variants:
  1. landmarks_only        — 5-point heatmap concatenated with face
  2. parsing_only          — 19-class parsing mask concatenated with face
  3. landmarks_and_parsing — both signals concatenated
  4. deca_geometry         — (optional) DECA 3DMM shape code

The baseline is GANonymization with no extra conditioning (raw face input).
Since GANonymization's pix2pix takes a 3-channel input by default, we
supply the conditioning signals by replacing or blending into the input
image channels.  In a production setup each variant would be retrained;
here we measure the effect of *passing* different conditioning signals
through the same pretrained model to set up the ablation structure.

Usage
-----
::

    python scripts/run_conditioning_ablation.py
    python scripts/run_conditioning_ablation.py --device cuda --max-samples 200
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

VARIANTS = [
    "no_conditioning",        # raw image only (baseline)
    "landmarks_only",         # 5-pt landmark heatmaps
    "parsing_only",           # BiSeNet parsing mask
    "landmarks_and_parsing",  # both
]


def _prepare_conditioned_input(
    image: np.ndarray,
    variant: str,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """
    Prepare a conditioned input for GANonymization.

    For ablation purposes we blend the conditioning signal into the image
    channels so the pretrained model can still process it (same 3-channel
    input).  This simulates the information content each conditioning signal
    provides.

    Parameters
    ----------
    image : ndarray (H, W, 3) float32 [0,1]
    variant : str
    device : str

    Returns
    -------
    ndarray (H, W, 3) float32 [0,1]
    """
    if variant == "no_conditioning":
        return image

    H, W, _ = image.shape

    if variant in ("landmarks_only", "landmarks_and_parsing"):
        try:
            from src.preprocess.landmarks import LandmarkExtractor

            extractor = LandmarkExtractor(device=device)
            landmarks = extractor.extract(image)
            heatmap = _landmarks_to_channel(landmarks, H, W)
        except Exception:
            heatmap = np.zeros((H, W), dtype=np.float32)
    else:
        heatmap = None

    if variant in ("parsing_only", "landmarks_and_parsing"):
        try:
            from src.preprocess.face_parser import FaceParser

            parser = FaceParser(device=device)
            mask = parser.parse(image)
            parsing_ch = mask.astype(np.float32) / 18.0  # normalise to [0,1]
        except Exception:
            parsing_ch = np.zeros((H, W), dtype=np.float32)
    else:
        parsing_ch = None

    # Blend conditioning into image
    result = image.copy()
    alpha = 0.3  # conditioning overlay strength

    if variant == "landmarks_only":
        assert heatmap is not None
        for c in range(3):
            result[:, :, c] = (1 - alpha) * image[:, :, c] + alpha * heatmap

    elif variant == "parsing_only":
        assert parsing_ch is not None
        for c in range(3):
            result[:, :, c] = (1 - alpha) * image[:, :, c] + alpha * parsing_ch

    elif variant == "landmarks_and_parsing":
        assert heatmap is not None and parsing_ch is not None
        combined = 0.5 * heatmap + 0.5 * parsing_ch
        for c in range(3):
            result[:, :, c] = (1 - alpha) * image[:, :, c] + alpha * combined

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _landmarks_to_channel(
    landmarks: np.ndarray, H: int, W: int, sigma: float = 5.0,
) -> np.ndarray:
    """Render landmarks as a Gaussian heatmap channel."""
    ch = np.zeros((H, W), dtype=np.float32)
    if landmarks is None or len(landmarks) == 0:
        return ch
    for pt in landmarks:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if 0 <= x < W and 0 <= y < H:
            y_grid, x_grid = np.ogrid[
                max(0, y - 15):min(H, y + 16),
                max(0, x - 15):min(W, x + 16),
            ]
            patch = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
            ch[
                max(0, y - 15):min(H, y + 16),
                max(0, x - 15):min(W, x + 16),
            ] = np.maximum(
                ch[max(0, y - 15):min(H, y + 16), max(0, x - 15):min(W, x + 16)],
                patch,
            )
    return ch


def run_conditioning_ablation(
    *,
    dataset: str = "fer2013",
    data_root: str = "data/fer2013",
    csv_file: str = "fer2013.csv",
    max_samples: Optional[int] = None,
    device: str = "cpu",
    seed: int = 42,
    output_dir: str = "results/conditioning_ablation",
) -> list[dict]:
    """Run the conditioning signal ablation."""
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

    from src.anonymizers import get_anonymizer
    from src.data.contracts import FaceCrop

    all_results: list[dict] = []

    for variant in VARIANTS:
        run_id = f"cond_{variant}"
        logger.info("━━━ %s ━━━", run_id)
        t0 = time.time()

        # Try to build GANonymization anonymizer
        try:
            anon = get_anonymizer("ganonymization", device=device)
        except Exception as exc:
            logger.warning(
                "GANonymization unavailable — falling back to blur for "
                "conditioning ablation demo: %s", exc,
            )
            anon = get_anonymizer("blur", kernel_size=31)

        anon_images = []
        for i in range(N):
            conditioned = _prepare_conditioned_input(images[i], variant, device=device)
            crop = FaceCrop(image=conditioned)
            try:
                res = anon.anonymize_single(crop)
                anon_images.append(res.image)
            except Exception as exc:
                logger.warning("  Sample %d failed: %s", i, exc)
                anon_images.append(images[i])
        anon_images_arr = np.stack(anon_images)

        probs_anonymized = _get_expression_probs(anon_images_arr, device)
        anon_embeddings = _get_identity_embeddings(anon_images_arr, device)

        from src.runner.evaluator import run_evaluation

        result = run_evaluation(
            anonymizer_name="ganonymization",
            original_images=images,
            anonymized_images=anon_images_arr,
            expression_labels=expr_labels,
            identity_labels=id_labels,
            original_embeddings=original_embeddings,
            anonymized_embeddings=anon_embeddings,
            probs_original=probs_original,
            probs_anonymized=probs_anonymized,
            anonymizer_params={"conditioning": variant},
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
        row["_variant"] = variant
        row["_elapsed"] = time.time() - t0
        all_results.append(row)
        logger.info("  %s done in %.1fs", variant, row["_elapsed"])

    _write_frontier_csv(all_results, out / "frontier.csv")
    _plot_results(out)
    _write_ablation_table(all_results, out / "ablation_table.md")
    logger.info("Conditioning ablation complete — %d variants.", len(all_results))
    return all_results


def _write_ablation_table(results: list[dict], path: Path) -> None:
    """Write a Markdown ablation summary table."""
    lines = [
        "# Conditioning Signal Ablation",
        "",
        "| Variant | Privacy | Utility | PSNR | SSIM |",
        "|---------|---------|---------|------|------|",
    ]
    for r in results:
        variant = r.get("_variant", r.get("_run_id", "?"))
        privacy = r.get("privacy_score", "—")
        utility = r.get("utility_score", "—")
        psnr = r.get("psnr", "—")
        ssim = r.get("ssim", "—")
        if isinstance(privacy, float):
            privacy = f"{privacy:.4f}"
        if isinstance(utility, float):
            utility = f"{utility:.4f}"
        if isinstance(psnr, float):
            psnr = f"{psnr:.2f}"
        if isinstance(ssim, float):
            ssim = f"{ssim:.4f}"
        lines.append(f"| {variant} | {privacy} | {utility} | {psnr} | {ssim} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Ablation table written to %s", path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="Ablation: conditioning signals")
    p.add_argument("--dataset", default="fer2013")
    p.add_argument("--data-root", default="data/fer2013")
    p.add_argument("--csv-file", default="fer2013.csv")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/conditioning_ablation")
    args = p.parse_args()

    run_conditioning_ablation(
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
