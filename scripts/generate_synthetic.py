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

        images, labels_expr, labels_id = _load_split(
            dataset, data_root, csv_file, split,
        )

        if max_samples_per_split is not None and len(images) > max_samples_per_split:
            images = images[:max_samples_per_split]
            labels_expr = labels_expr[:max_samples_per_split]
            if labels_id is not None:
                labels_id = labels_id[:max_samples_per_split]

        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_labels: dict[str, dict[str, Any]] = {}
        t0 = time.time()

        for i in range(len(images)):
            img = images[i]
            crop = FaceCrop(
                image=img,
                source_dataset=dataset,
                split=split,
                expression_label=int(labels_expr[i]) if labels_expr is not None else -1,
                identity_label=int(labels_id[i]) if labels_id is not None else -1,
            )

            try:
                result = anon.anonymize_single(crop)
                anon_img = result.image
            except Exception as exc:
                logger.warning("  Failed sample %d: %s", i, exc)
                anon_img = img  # fallback: keep original

            # Save
            fname = f"image_{i:06d}"
            if save_format == "npy":
                np.save(str(split_dir / f"{fname}.npy"), anon_img)
            else:
                _save_png(anon_img, split_dir / f"{fname}.png")

            split_labels[str(i)] = {
                "expression": int(labels_expr[i]) if labels_expr is not None else -1,
                "identity": int(labels_id[i]) if labels_id is not None else -1,
            }

            if (i + 1) % 500 == 0:
                logger.info("  %d / %d  (%.1fs)", i + 1, len(images), time.time() - t0)

        all_labels[split] = split_labels
        logger.info("  Done split %s: %d images in %.1fs", split, len(images), time.time() - t0)

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


def _load_split(
    dataset: str,
    data_root: str,
    csv_file: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load images and labels for a given split.

    Returns (images, expression_labels, identity_labels_or_None).
    """
    if dataset == "fer2013":
        from src.data.fer2013_adapter import FER2013Dataset

        ds = FER2013Dataset(root=data_root, csv_file=csv_file, split=split)
        images = np.stack([ds[i][0] for i in range(len(ds))])
        labels = np.array([ds[i][1] for i in range(len(ds))])
        return images, labels, None

    elif dataset == "pins":
        from src.data.pins_adapter import PinsFaceDataset

        ds = PinsFaceDataset(root=data_root, split=split)
        images = np.stack([ds[i][0] for i in range(len(ds))])
        id_labels = np.array([ds[i][1] for i in range(len(ds))])
        # Pins doesn't have expression labels
        expr_labels = np.full(len(ds), -1, dtype=np.int64)
        return images, expr_labels, id_labels

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


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
