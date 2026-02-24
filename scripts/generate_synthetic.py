"""
Synthetic anonymized dataset generator.

For each anonymizer backend: process FER-2013 / Pins → save anonymized
versions preserving expression & identity labels.

Output structure::

    data/synthetic/
        blur_k15/
            train/  val/  test/
                image_NNNNN.npy    (256×256×3 float32)
            labels.json            {split: {idx: {expression: int, identity: int}}}
        pixelate_b8/
            ...

Usage
-----
::

    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --anonymizer blur --param kernel_size=15
    python scripts/generate_synthetic.py --anonymizer all
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


def _build_param_tag(name: str, params: dict) -> str:
    """Build a filesystem-safe tag like ``blur_k15`` from name + params."""
    if not params:
        return name
    parts = [name]
    for k, v in sorted(params.items()):
        # Shorten key to first letter + value
        short_key = k[0] if len(k) > 3 else k
        parts.append(f"{short_key}{v}")
    return "_".join(parts)


def generate_synthetic_dataset(
    *,
    anonymizer_name: str = "blur",
    anonymizer_params: Optional[dict[str, Any]] = None,
    dataset: str = "fer2013",
    data_root: str = "data/fer2013",
    csv_file: str = "fer2013.csv",
    output_root: str = "data/synthetic",
    splits: Optional[list[str]] = None,
    max_samples_per_split: Optional[int] = None,
    save_format: str = "npy",  # "npy" or "png"
    device: str = "cpu",
    seed: int = 42,
) -> Path:
    """
    Anonymize a dataset and save to disk.

    Parameters
    ----------
    anonymizer_name     : registered anonymizer name.
    anonymizer_params   : dict of constructor kwargs.
    dataset             : dataset name (currently "fer2013" or "pins").
    data_root           : root of the source dataset.
    csv_file            : csv filename for FER-2013.
    output_root         : root directory for synthetic outputs.
    splits              : which splits to process (default: all).
    max_samples_per_split : cap per split (for quick testing).
    save_format         : "npy" (float32) or "png" (uint8).
    device              : for GPU-based anonymizers.
    seed                : random seed.

    Returns
    -------
    Path — directory containing the generated data.
    """
    np.random.seed(seed)
    params = anonymizer_params or {}

    # ── Build anonymizer ───────────────────────────────────────────────
    from src.anonymizers import get_anonymizer
    from src.data.contracts import FaceCrop

    anon = get_anonymizer(anonymizer_name, **params)
    tag = _build_param_tag(anonymizer_name, anon.configurable_params)
    out_dir = Path(output_root) / tag
    logger.info("Anonymizer: %s  → %s", anon.name, out_dir)

    # ── Load dataset ───────────────────────────────────────────────────
    if splits is None:
        splits = ["train", "val", "test"]

    all_labels: dict[str, dict[str, dict[str, Any]]] = {}

    for split in splits:
        logger.info("Processing split: %s", split)

        ds = _get_dataset(dataset, data_root, csv_file, split)
        n_total = len(ds)
        n_process = min(n_total, max_samples_per_split) if max_samples_per_split else n_total

        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_labels: dict[str, dict[str, Any]] = {}
        t0 = time.time()

        for i in range(n_process):
            # Load one sample at a time (lazy — avoids OOM)
            crop = ds[i]
            expr_label = _get_expression_label(crop, dataset)
            id_label = _get_identity_label(crop, dataset)

            try:
                result = anon.anonymize_single(crop)
                anon_img = result.image
            except Exception as exc:
                logger.warning("  Failed sample %d: %s", i, exc)
                anon_img = crop.image  # fallback: keep original

            # Save
            fname = f"image_{i:06d}"
            if save_format == "npy":
                np.save(str(split_dir / f"{fname}.npy"), anon_img)
            else:
                _save_png(anon_img, split_dir / f"{fname}.png")

            split_labels[str(i)] = {
                "expression": int(expr_label),
                "identity": int(id_label),
            }

            if (i + 1) % 500 == 0:
                logger.info("  %d / %d  (%.1fs)", i + 1, n_process, time.time() - t0)

            # Explicitly free memory
            del crop, anon_img

        all_labels[split] = split_labels
        logger.info("  Done split %s: %d images in %.1fs", split, n_process, time.time() - t0)

    # ── Save labels ────────────────────────────────────────────────────
    labels_path = out_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(all_labels, f, indent=2)
    logger.info("Labels saved to %s", labels_path)

    # ── Save metadata ──────────────────────────────────────────────────
    meta = {
        "anonymizer": anonymizer_name,
        "params": anon.configurable_params,
        "dataset": dataset,
        "save_format": save_format,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir


def _get_dataset(dataset: str, data_root: str, csv_file: str, split: str):
    """Return a dataset object that supports __len__ and __getitem__ (lazy)."""
    if dataset == "fer2013":
        from src.data.fer2013_adapter import FER2013Dataset
        return FER2013Dataset(root=data_root, split=split)
    elif dataset == "pins":
        from src.data.pins_adapter import PinsFaceDataset
        return PinsFaceDataset(root=data_root, split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def _get_expression_label(crop, dataset: str) -> int:
    """Extract expression label from a FaceCrop."""
    if dataset == "fer2013":
        return crop.meta.expression_label if crop.meta.expression_label is not None else -1
    return -1


def _get_identity_label(crop, dataset: str) -> int:
    """Extract identity label from a FaceCrop."""
    if dataset == "pins":
        return crop.meta.identity_label if crop.meta.identity_label is not None else -1
    return -1


def _save_png(image: np.ndarray, path: Path) -> None:
    """Save float32 [0,1] → uint8 PNG."""
    from src.data.contracts import float_to_uint8

    try:
        from PIL import Image
        pil = Image.fromarray(float_to_uint8(image))
        pil.save(str(path))
    except ImportError:
        # Fallback: save as npy with .png extension warning
        logger.warning("Pillow not installed — saving as .npy instead of .png")
        np.save(str(path).replace(".png", ".npy"), image)


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_params(param_str: str) -> dict:
    """Parse 'key1=val1,key2=val2' into a dict with auto type-cast."""
    params: dict[str, Any] = {}
    if not param_str:
        return params
    for pair in param_str.split(","):
        k, _, v = pair.partition("=")
        k = k.strip()
        v = v.strip()
        # Auto type
        if v.isdigit():
            params[k] = int(v)
        else:
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v
    return params


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate synthetic anonymized datasets.")
    parser.add_argument("--anonymizer", default="blur", help="Anonymizer name or 'all'.")
    parser.add_argument("--param", default="", help="Params: key1=val1,key2=val2")
    parser.add_argument("--dataset", default="fer2013")
    parser.add_argument("--data-root", default="data/fer2013")
    parser.add_argument("--csv-file", default="fer2013.csv")
    parser.add_argument("--output-root", default="data/synthetic")
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--format", choices=["npy", "png"], default="npy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    params = _parse_params(args.param)

    if args.anonymizer == "all":
        from src.anonymizers import list_anonymizers

        names = list_anonymizers()
        logger.info("Generating synthetic data for all anonymizers: %s", names)
        for name in names:
            try:
                generate_synthetic_dataset(
                    anonymizer_name=name,
                    anonymizer_params=params,
                    dataset=args.dataset,
                    data_root=args.data_root,
                    csv_file=args.csv_file,
                    output_root=args.output_root,
                    splits=args.splits,
                    max_samples_per_split=args.max_samples,
                    save_format=args.format,
                    device=args.device,
                    seed=args.seed,
                )
            except Exception as exc:
                logger.error("Failed for anonymizer '%s': %s", name, exc)
    else:
        generate_synthetic_dataset(
            anonymizer_name=args.anonymizer,
            anonymizer_params=params,
            dataset=args.dataset,
            data_root=args.data_root,
            csv_file=args.csv_file,
            output_root=args.output_root,
            splits=args.splits,
            max_samples_per_split=args.max_samples,
            save_format=args.format,
            device=args.device,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
