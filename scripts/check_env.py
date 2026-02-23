#!/usr/bin/env python3
"""
Environment verification script.
Run:  python scripts/check_env.py
Checks Python version, CUDA availability, and importability of core packages.
"""

import sys
import importlib
import platform


def _check(name: str, import_name: str | None = None):
    """Try importing a package and print its version."""
    mod_name = import_name or name
    try:
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", "installed (version unknown)")
        print(f"  [OK]  {name:30s} {ver}")
        return True
    except ImportError:
        print(f"  [!!]  {name:30s} NOT FOUND")
        return False


def main():
    print("=" * 60)
    print("COMP4026 — Environment Check")
    print("=" * 60)

    # ── Python ──────────────────────────────────────────
    print(f"\nPython:   {sys.version}")
    print(f"Platform: {platform.platform()}")
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print("  [!!]  Python >= 3.9 required")

    # ── Core packages ───────────────────────────────────
    print("\nCore packages:")
    pkgs = [
        ("torch", None),
        ("torchvision", None),
        ("hydra-core", "hydra"),
        ("omegaconf", None),
        ("numpy", None),
        ("pandas", None),
        ("Pillow", "PIL"),
        ("opencv-python", "cv2"),
        ("matplotlib", None),
        ("seaborn", None),
        ("tqdm", None),
        ("scikit-learn", "sklearn"),
        ("scipy", None),
        ("timm", None),
        ("kornia", None),
    ]
    for name, imp in pkgs:
        _check(name, imp)

    # ── Face / metric packages ──────────────────────────
    print("\nFace & metric packages:")
    face_pkgs = [
        ("insightface", None),
        ("onnxruntime", "onnxruntime"),
        ("facenet-pytorch", "facenet_pytorch"),
        ("pytorch-fid", "pytorch_fid"),
        ("clean-fid", "cleanfid"),
        ("lpips", None),
    ]
    for name, imp in face_pkgs:
        _check(name, imp)

    # ── CUDA ────────────────────────────────────────────
    print("\nGPU / CUDA:")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  [OK]  CUDA available — {torch.cuda.get_device_name(0)}")
            print(f"        CUDA version:  {torch.version.cuda}")
            print(f"        cuDNN version: {torch.backends.cudnn.version()}")
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"        GPU memory:    {mem:.1f} GB")
        else:
            print("  [!!]  CUDA NOT available — will fall back to CPU (slow)")
    except Exception as e:
        print(f"  [!!]  Could not check CUDA: {e}")

    print("\n" + "=" * 60)
    print("Environment check complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
