#!/usr/bin/env python
"""
Download and prepare all datasets required by the project.

    FER-2013        ─  Kaggle  deadskull7/fer2013   (~48 MB)
    Pins Face       ─  Kaggle  hereisburak/pins-face-recognition (~1.5 GB)
    CelebAMask-HQ   ─  Google Drive / manual       (~6 GB, optional)

Usage:
    python scripts/download_data.py                     # all datasets
    python scripts/download_data.py --skip-celebahq     # skip CelebAMask-HQ
    python scripts/download_data.py --only fer2013      # just FER-2013
    python scripts/download_data.py --only pins         # just Pins
    python scripts/download_data.py --only celebahq     # just CelebAMask-HQ

Requires:
    pip install kaggle opendatasets gdown
    Kaggle credentials at ~/.kaggle/kaggle.json  (or %USERPROFILE%/.kaggle/kaggle.json)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

FER2013_DIR = DATA_DIR / "fer2013"
PINS_DIR = DATA_DIR / "pins_face_recognition"
CELEBAHQ_DIR = DATA_DIR / "CelebAMask-HQ"


def _green(msg: str) -> str:
    return f"\033[92m{msg}\033[0m"


def _yellow(msg: str) -> str:
    return f"\033[93m{msg}\033[0m"


def _red(msg: str) -> str:
    return f"\033[91m{msg}\033[0m"


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _ok(msg: str) -> None:
    print(_green(f"[OK]   {msg}"))


def _warn(msg: str) -> None:
    print(_yellow(f"[WARN] {msg}"))


def _err(msg: str) -> None:
    print(_red(f"[ERR]  {msg}"))


# ── Kaggle helper ─────────────────────────────────────────────────────

def _check_kaggle() -> bool:
    """Return True if kaggle API is usable."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return True
    except Exception as e:
        _err(f"Kaggle authentication failed: {e}")
        _warn(
            "Please ensure your kaggle.json is at:\n"
            "  Linux/Mac: ~/.kaggle/kaggle.json\n"
            "  Windows:   %USERPROFILE%\\.kaggle\\kaggle.json\n"
            "Get it from: https://www.kaggle.com → Profile → Settings → API → Create New Token"
        )
        return False


def _kaggle_download(dataset_slug: str, dest: Path) -> Path:
    """Download & unzip a Kaggle dataset into *dest*. Returns dest."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    dest.mkdir(parents=True, exist_ok=True)
    _info(f"Downloading kaggle dataset '{dataset_slug}' → {dest} ...")
    api.dataset_download_files(dataset_slug, path=str(dest), unzip=True)
    _ok(f"Downloaded '{dataset_slug}' to {dest}")
    return dest


# ── FER-2013 ───────────────────────────────────────────────────────────

def download_fer2013() -> bool:
    """Download FER-2013 CSV dataset from Kaggle."""
    csv_path = FER2013_DIR / "fer2013.csv"
    if csv_path.exists():
        _ok(f"FER-2013 already present at {csv_path}")
        return True

    _info("=== Downloading FER-2013 ===")
    if not _check_kaggle():
        return False

    try:
        _kaggle_download("deadskull7/fer2013", FER2013_DIR)
    except Exception:
        _warn("'deadskull7/fer2013' not found, trying 'msambare/fer2013' ...")
        try:
            _kaggle_download("msambare/fer2013", FER2013_DIR)
        except Exception as e2:
            _err(f"Failed to download FER-2013: {e2}")
            return False

    # If msambare dataset was downloaded, it has image folders instead of CSV.
    # We need to convert them to CSV format.
    if not csv_path.exists():
        # Check if image-folder format was downloaded
        train_dir = FER2013_DIR / "train"
        test_dir = FER2013_DIR / "test"
        if train_dir.exists() and test_dir.exists():
            _info("Image-folder FER-2013 detected. Converting to CSV format ...")
            _convert_fer2013_images_to_csv(FER2013_DIR, csv_path)
        else:
            # Maybe the CSV is in a subdirectory
            for sub_csv in FER2013_DIR.rglob("fer2013.csv"):
                if sub_csv != csv_path:
                    shutil.move(str(sub_csv), str(csv_path))
                    _ok(f"Moved CSV to {csv_path}")
                    break

    if csv_path.exists():
        _ok(f"FER-2013 ready at {csv_path}")
        return True
    else:
        _err("FER-2013 CSV not found after download. Please download manually.")
        return False


def _convert_fer2013_images_to_csv(root: Path, out_csv: Path) -> None:
    """Convert image-folder FER-2013 (msambare format) → CSV with pixel strings.

    The msambare dataset layout:
        root/train/<emotion_name>/*.jpg
        root/test/<emotion_name>/*.jpg
    """
    import cv2
    import numpy as np

    emotion_map = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "sad": 4,
        "surprise": 5,
        "neutral": 6,
    }

    rows: list[str] = ["emotion,pixels,Usage"]

    for split_name, usage_label in [("train", "Training"), ("test", "PublicTest")]:
        split_dir = root / split_name
        if not split_dir.exists():
            continue
        for emotion_name, emotion_idx in emotion_map.items():
            emo_dir = split_dir / emotion_name
            if not emo_dir.exists():
                continue
            img_files = sorted(emo_dir.glob("*"))
            for img_path in img_files:
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # Resize to 48x48 if needed
                if img.shape != (48, 48):
                    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
                pixel_str = " ".join(str(p) for p in img.flatten())
                rows.append(f"{emotion_idx},{pixel_str},{usage_label}")

    # Split PublicTest into PublicTest and PrivateTest (50/50)
    # to match original FER-2013 format
    public_rows = [r for r in rows[1:] if r.endswith(",PublicTest")]
    # Take half as PrivateTest
    mid = len(public_rows) // 2
    final_rows = [rows[0]]  # header
    pt_count = 0
    for r in rows[1:]:
        if r.endswith(",PublicTest"):
            if pt_count >= mid:
                r = r[:-len("PublicTest")] + "PrivateTest"
            pt_count += 1
        final_rows.append(r)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(final_rows) + "\n")
    _ok(f"Converted {len(final_rows) - 1} images → {out_csv}")


# ── Pins Face Recognition ────────────────────────────────────────────

def download_pins() -> bool:
    """Download Pins Face Recognition dataset from Kaggle."""
    if PINS_DIR.exists() and any(PINS_DIR.iterdir()):
        n_dirs = sum(1 for p in PINS_DIR.iterdir() if p.is_dir())
        if n_dirs > 0:
            _ok(f"Pins dataset already present at {PINS_DIR} ({n_dirs} identities)")
            return True

    _info("=== Downloading Pins Face Recognition ===")
    if not _check_kaggle():
        return False

    try:
        # The Kaggle dataset extracts to a subfolder
        tmp_dir = DATA_DIR / "_pins_tmp"
        _kaggle_download("hereisburak/pins-face-recognition", tmp_dir)

        # Find the actual identity folder root
        # Kaggle may extract into tmp_dir/105_classes_pins_dataset/ or similar
        candidate = None
        for p in tmp_dir.rglob("pins_*"):
            if p.is_dir():
                candidate = p.parent
                break

        if candidate and candidate != PINS_DIR:
            PINS_DIR.mkdir(parents=True, exist_ok=True)
            for d in sorted(candidate.iterdir()):
                if d.is_dir() and d.name.startswith("pins_"):
                    target = PINS_DIR / d.name
                    if not target.exists():
                        shutil.move(str(d), str(target))
            # Clean up
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif candidate is None:
            # Maybe it extracted directly
            if tmp_dir != PINS_DIR:
                if PINS_DIR.exists():
                    shutil.rmtree(PINS_DIR)
                shutil.move(str(tmp_dir), str(PINS_DIR))

    except Exception as e:
        _err(f"Failed to download Pins: {e}")
        return False

    n_dirs = sum(1 for p in PINS_DIR.iterdir() if p.is_dir()) if PINS_DIR.exists() else 0
    if n_dirs > 0:
        _ok(f"Pins dataset ready at {PINS_DIR} ({n_dirs} identities)")
        return True
    else:
        _err("Pins dataset folder appears empty after download.")
        return False


# ── CelebAMask-HQ ────────────────────────────────────────────────────

def download_celebahq() -> bool:
    """Download CelebAMask-HQ dataset from Kaggle mirror (ipythonx/celebamaskhq).

    Falls back to Google Drive if Kaggle fails.
    """
    img_dir = CELEBAHQ_DIR / "CelebA-HQ-img"
    mask_dir = CELEBAHQ_DIR / "CelebAMask-HQ-mask-anno"

    if img_dir.exists():
        n_imgs = sum(1 for _ in img_dir.glob("*.jpg")) + sum(1 for _ in img_dir.glob("*.png"))
        if n_imgs > 1000:
            _ok(f"CelebAMask-HQ already present at {CELEBAHQ_DIR} ({n_imgs} images)")
            return True

    _info("=== Downloading CelebAMask-HQ ===")
    _info("Attempting download from Kaggle mirror (ipythonx/celebamaskhq) ...")
    _info("This is ~6 GB and may take a while.")

    if not _check_kaggle():
        return False

    try:
        tmp_dir = DATA_DIR / "_celebahq_tmp"
        _kaggle_download("ipythonx/celebamaskhq", tmp_dir)

        # The Kaggle dataset extracts as CelebAMask-HQ/ subfolder
        extracted = tmp_dir / "CelebAMask-HQ"
        if extracted.exists() and extracted.is_dir():
            if CELEBAHQ_DIR.exists():
                shutil.rmtree(CELEBAHQ_DIR)
            shutil.move(str(extracted), str(CELEBAHQ_DIR))
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif (tmp_dir / "CelebA-HQ-img").exists():
            # Extracted directly
            if CELEBAHQ_DIR.exists():
                shutil.rmtree(CELEBAHQ_DIR)
            shutil.move(str(tmp_dir), str(CELEBAHQ_DIR))
        else:
            # Check if any matching subfolder exists
            for candidate in tmp_dir.rglob("CelebA-HQ-img"):
                if candidate.is_dir():
                    parent = candidate.parent
                    if CELEBAHQ_DIR.exists():
                        shutil.rmtree(CELEBAHQ_DIR)
                    shutil.move(str(parent), str(CELEBAHQ_DIR))
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    break

    except Exception as e:
        _err(f"Failed to download CelebAMask-HQ from Kaggle: {e}")
        _warn(
            "Please download manually from:\n"
            "  https://github.com/switchablenorms/CelebAMask-HQ\n"
            "And place files at:\n"
            f"  {img_dir}\n"
            f"  {mask_dir}"
        )
        return False

    # Verify
    if img_dir.exists():
        n_imgs = sum(1 for _ in img_dir.glob("*.jpg")) + sum(1 for _ in img_dir.glob("*.png"))
        _ok(f"CelebAMask-HQ ready at {CELEBAHQ_DIR} ({n_imgs} images)")
        return True
    else:
        _err(f"CelebA-HQ-img directory not found at {img_dir} after extraction.")
        return False


# ── Verification ──────────────────────────────────────────────────────

def verify_datasets(include_celebahq: bool = True) -> None:
    """Print summary of dataset availability."""
    print("\n" + "=" * 60)
    print("Dataset Verification Summary")
    print("=" * 60)

    # FER-2013
    csv_path = FER2013_DIR / "fer2013.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        n = len(df)
        cols = list(df.columns)
        _ok(f"FER-2013: {n:,} samples, columns={cols}")
        for usage in ["Training", "PublicTest", "PrivateTest"]:
            count = len(df[df["Usage"] == usage])
            print(f"          {usage}: {count:,}")
    else:
        _err(f"FER-2013: NOT FOUND at {csv_path}")

    # Pins
    if PINS_DIR.exists():
        id_dirs = [d for d in PINS_DIR.iterdir() if d.is_dir()]
        n_images = sum(
            sum(1 for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
            for d in id_dirs
        )
        _ok(f"Pins: {len(id_dirs)} identities, {n_images:,} images")
    else:
        _err(f"Pins: NOT FOUND at {PINS_DIR}")

    # CelebAMask-HQ
    if include_celebahq:
        img_dir = CELEBAHQ_DIR / "CelebA-HQ-img"
        mask_dir = CELEBAHQ_DIR / "CelebAMask-HQ-mask-anno"
        if img_dir.exists():
            n_imgs = sum(1 for _ in img_dir.glob("*.*"))
            _ok(f"CelebAMask-HQ: {n_imgs:,} images")
        else:
            _err(f"CelebAMask-HQ images: NOT FOUND at {img_dir}")
        if mask_dir.exists():
            n_mask_dirs = sum(1 for d in mask_dir.iterdir() if d.is_dir())
            _ok(f"CelebAMask-HQ masks: {n_mask_dirs} annotation folders")
        else:
            _err(f"CelebAMask-HQ masks: NOT FOUND at {mask_dir}")

    print("=" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download project datasets")
    parser.add_argument(
        "--skip-celebahq",
        action="store_true",
        help="Skip CelebAMask-HQ (large, optional)",
    )
    parser.add_argument(
        "--only",
        choices=["fer2013", "pins", "celebahq"],
        help="Download only a specific dataset",
    )
    args = parser.parse_args()

    results: dict[str, bool] = {}

    if args.only:
        if args.only == "fer2013":
            results["FER-2013"] = download_fer2013()
        elif args.only == "pins":
            results["Pins"] = download_pins()
        elif args.only == "celebahq":
            results["CelebAMask-HQ"] = download_celebahq()
    else:
        results["FER-2013"] = download_fer2013()
        results["Pins"] = download_pins()
        if not args.skip_celebahq:
            results["CelebAMask-HQ"] = download_celebahq()

    include_celebahq = "CelebAMask-HQ" in results or (
        not args.only and not args.skip_celebahq
    )
    verify_datasets(include_celebahq=include_celebahq)

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        _err(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        _ok("All requested datasets ready!")


if __name__ == "__main__":
    main()
