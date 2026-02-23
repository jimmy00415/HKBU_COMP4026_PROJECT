"""
Unit tests — data contracts & adapters (Task 8.1).

Tests:
  • FaceCrop / FaceCropMeta dataclass construction and validation
  • FaceCropBatch stacking & property accessors
  • AnonymizedFace / AnonymizedBatch construction
  • Helper functions (uint8_to_float, float_to_uint8, gray_to_rgb)
  • ExpressionLabel enum values
  • FER-2013 adapter (mock CSV — no real data needed)
  • Pins adapter split hashing (deterministic, correct ratios)

All tests use synthetic data.  No external datasets required.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Contract imports ──────────────────────────────────────────────────

from src.data.contracts import (
    CANONICAL_ALIGNMENT,
    CANONICAL_COLOR_SPACE,
    CANONICAL_RESOLUTION,
    AnonymizedBatch,
    AnonymizedFace,
    ExpressionLabel,
    FaceCrop,
    FaceCropBatch,
    FaceCropMeta,
    float_to_uint8,
    gray_to_rgb,
    uint8_to_float,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _random_image(
    *, h: int = 256, w: int = 256, c: int = 3,
    dtype: type = np.float32, seed: int = 0,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if dtype == np.float32:
        return rng.rand(h, w, c).astype(np.float32)
    return rng.randint(0, 256, (h, w, c), dtype=np.uint8)


def _make_face(seed: int = 0, **meta_kw) -> FaceCrop:
    defaults = dict(
        dataset="test", split="test", image_id=f"img_{seed}",
        identity_label=seed % 5, expression_label=seed % 7,
    )
    defaults.update(meta_kw)
    return FaceCrop(
        image=_random_image(seed=seed),
        meta=FaceCropMeta(**defaults),
    )


# ══════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_canonical_resolution(self):
        assert CANONICAL_RESOLUTION == 256

    def test_canonical_color_space(self):
        assert CANONICAL_COLOR_SPACE == "RGB"

    def test_canonical_alignment(self):
        assert CANONICAL_ALIGNMENT == "arcface_5pt"


# ══════════════════════════════════════════════════════════════════════
#  ExpressionLabel
# ══════════════════════════════════════════════════════════════════════

class TestExpressionLabel:
    def test_num_classes(self):
        assert len(ExpressionLabel) == 7

    def test_values_contiguous(self):
        values = sorted(e.value for e in ExpressionLabel)
        assert values == list(range(7))

    def test_names(self):
        expected = {"ANGRY", "DISGUST", "FEAR", "HAPPY", "SAD", "SURPRISE", "NEUTRAL"}
        assert {e.name for e in ExpressionLabel} == expected


# ══════════════════════════════════════════════════════════════════════
#  FaceCropMeta
# ══════════════════════════════════════════════════════════════════════

class TestFaceCropMeta:
    def test_defaults(self):
        meta = FaceCropMeta(dataset="fer2013", split="test", image_id="0")
        assert meta.alignment_method == CANONICAL_ALIGNMENT
        assert meta.identity_label is None
        assert meta.expression_label is None
        assert meta.detection_score == 0.0

    def test_with_labels(self):
        meta = FaceCropMeta(
            dataset="pins", split="train", image_id="x",
            identity_label=42, expression_label=3,
        )
        assert meta.identity_label == 42
        assert meta.expression_label == 3


# ══════════════════════════════════════════════════════════════════════
#  FaceCrop
# ══════════════════════════════════════════════════════════════════════

class TestFaceCrop:
    def test_construct_valid(self):
        face = _make_face()
        assert face.image.shape == (256, 256, 3)
        assert face.image.dtype == np.float32

    def test_validate_passes(self):
        face = _make_face()
        face.validate()  # should not raise

    def test_validate_wrong_shape(self):
        face = _make_face()
        face.image = np.zeros((128, 128, 3), dtype=np.float32)
        with pytest.raises(AssertionError, match="128"):
            face.validate()

    def test_validate_wrong_dtype(self):
        face = _make_face()
        face.image = (face.image * 255).astype(np.uint8)
        with pytest.raises(AssertionError, match="float32"):
            face.validate()

    def test_validate_out_of_range(self):
        face = _make_face()
        face.image[0, 0, 0] = 1.5
        with pytest.raises(AssertionError):
            face.validate()

    def test_validate_wrong_channels(self):
        face = _make_face()
        face.image = face.image[:, :, :1]
        with pytest.raises(AssertionError, match="3 channels"):
            face.validate()

    def test_optional_landmarks(self):
        face = _make_face()
        assert face.landmarks_5pt is None
        face.landmarks_5pt = np.zeros((5, 2), dtype=np.float32)
        assert face.landmarks_5pt.shape == (5, 2)


# ══════════════════════════════════════════════════════════════════════
#  FaceCropBatch
# ══════════════════════════════════════════════════════════════════════

class TestFaceCropBatch:
    def test_len_and_index(self):
        faces = [_make_face(i) for i in range(5)]
        batch = FaceCropBatch(crops=faces)
        assert len(batch) == 5
        assert batch[0].meta.image_id == "img_0"

    def test_images_stacking(self):
        batch = FaceCropBatch(crops=[_make_face(i) for i in range(3)])
        imgs = batch.images
        assert imgs.shape == (3, 256, 256, 3)
        assert imgs.dtype == np.float32

    def test_expression_labels(self):
        batch = FaceCropBatch(crops=[_make_face(i) for i in range(7)])
        labels = batch.expression_labels
        assert len(labels) == 7
        assert labels[0] == 0  # 0 % 7

    def test_identity_labels(self):
        batch = FaceCropBatch(crops=[_make_face(i) for i in range(5)])
        labels = batch.identity_labels
        assert labels == [0, 1, 2, 3, 4]


# ══════════════════════════════════════════════════════════════════════
#  AnonymizedFace / Batch
# ══════════════════════════════════════════════════════════════════════

class TestAnonymizedFace:
    def test_construct(self):
        face = _make_face()
        anon = AnonymizedFace(
            image=face.image, original=face,
            anonymizer_name="blur", anonymizer_params={"k": 31},
        )
        assert anon.anonymizer_name == "blur"
        assert anon.anonymizer_params["k"] == 31

    def test_batch_images(self):
        faces = [_make_face(i) for i in range(4)]
        anons = [
            AnonymizedFace(image=f.image, original=f, anonymizer_name="test")
            for f in faces
        ]
        batch = AnonymizedBatch(faces=anons)
        assert batch.images.shape == (4, 256, 256, 3)
        assert batch.original_images.shape == (4, 256, 256, 3)
        assert len(batch) == 4


# ══════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_uint8_to_float(self):
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)
        out = uint8_to_float(img)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out[0, 0], [0.0, 128 / 255, 1.0], atol=1e-5)

    def test_float_to_uint8(self):
        img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        out = float_to_uint8(img)
        assert out.dtype == np.uint8
        assert out[0, 0, 0] == 0
        assert out[0, 0, 2] == 255

    def test_float_to_uint8_clips(self):
        img = np.array([[[-0.5, 1.5]]], dtype=np.float32)
        out = float_to_uint8(img)
        assert out[0, 0, 0] == 0
        assert out[0, 0, 1] == 255

    def test_gray_to_rgb_2d(self):
        g = np.ones((10, 10), dtype=np.float32)
        rgb = gray_to_rgb(g)
        assert rgb.shape == (10, 10, 3)

    def test_gray_to_rgb_3d_single(self):
        g = np.ones((10, 10, 1), dtype=np.float32)
        rgb = gray_to_rgb(g)
        assert rgb.shape == (10, 10, 3)

    def test_gray_to_rgb_already_rgb(self):
        rgb = np.ones((10, 10, 3), dtype=np.float32)
        out = gray_to_rgb(rgb)
        assert out.shape == (10, 10, 3)
        np.testing.assert_array_equal(out, rgb)


# ══════════════════════════════════════════════════════════════════════
#  FER-2013 adapter  (mock CSV, no real data needed)
# ══════════════════════════════════════════════════════════════════════

class TestFER2013Adapter:
    @pytest.fixture
    def mock_csv(self, tmp_path: Path) -> Path:
        """Create a minimal FER-2013 CSV with 6 rows (2 per split)."""
        pytest.importorskip("torch", reason="torch not installed")
        pytest.importorskip("pandas", reason="pandas not installed")
        rng = np.random.RandomState(0)
        rows = []
        for usage in ["Training", "PublicTest", "PrivateTest"]:
            for i in range(2):
                emotion = i % 7
                pixels = " ".join(str(x) for x in rng.randint(0, 256, 48 * 48))
                rows.append(f"{emotion},{pixels},{usage}")

        csv_path = tmp_path / "fer2013.csv"
        csv_path.write_text(
            "emotion,pixels,Usage\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        return csv_path

    def test_loads_train_split(self, mock_csv: Path):
        from src.data.fer2013_adapter import FER2013Dataset

        ds = FER2013Dataset(root=mock_csv.parent, split="train")
        assert len(ds) == 2

    def test_loads_test_split(self, mock_csv: Path):
        from src.data.fer2013_adapter import FER2013Dataset

        ds = FER2013Dataset(root=mock_csv.parent, split="test")
        assert len(ds) == 2

    def test_output_contract(self, mock_csv: Path):
        from src.data.fer2013_adapter import FER2013Dataset

        ds = FER2013Dataset(root=mock_csv.parent, split="train")
        face = ds[0]
        assert isinstance(face, FaceCrop)
        assert face.image.shape == (256, 256, 3)
        assert face.image.dtype == np.float32
        assert 0.0 <= face.image.min()
        assert face.image.max() <= 1.0
        face.validate()

    def test_expression_label_set(self, mock_csv: Path):
        from src.data.fer2013_adapter import FER2013Dataset

        ds = FER2013Dataset(root=mock_csv.parent, split="train")
        face = ds[0]
        assert face.meta.expression_label is not None
        assert 0 <= face.meta.expression_label < 7

    def test_dataset_field(self, mock_csv: Path):
        from src.data.fer2013_adapter import FER2013Dataset

        ds = FER2013Dataset(root=mock_csv.parent, split="val")
        face = ds[0]
        assert face.meta.dataset == "fer2013"
        assert face.meta.split == "val"


# ══════════════════════════════════════════════════════════════════════
#  Pins adapter split determinism
# ══════════════════════════════════════════════════════════════════════

class TestPinsSplitHash:
    def test_hash_deterministic(self):
        """Same filename → same hash → same split every time."""
        import hashlib

        filename = "pins_Adam Sandler/image_001.jpg"
        h1 = int(hashlib.sha256(filename.encode()).hexdigest(), 16) % 100
        h2 = int(hashlib.sha256(filename.encode()).hexdigest(), 16) % 100
        assert h1 == h2

    def test_split_ratios_approximate(self):
        """Over many random names, 70/15/15 split should hold (±5%)."""
        import hashlib

        rng = np.random.RandomState(0)
        counts = {"train": 0, "val": 0, "test": 0}
        N = 10000
        for _ in range(N):
            name = f"person_{rng.randint(1e9)}/img_{rng.randint(1e9)}.jpg"
            h = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 100
            if h < 70:
                counts["train"] += 1
            elif h < 85:
                counts["val"] += 1
            else:
                counts["test"] += 1

        assert abs(counts["train"] / N - 0.70) < 0.05
        assert abs(counts["val"] / N - 0.15) < 0.05
        assert abs(counts["test"] / N - 0.15) < 0.05


# ══════════════════════════════════════════════════════════════════════
#  Error handling integration (validate_image)
# ══════════════════════════════════════════════════════════════════════

class TestValidateImage:
    def test_valid_float32(self):
        from src.runner.error_handling import validate_image

        img = np.random.rand(256, 256, 3).astype(np.float32)
        out = validate_image(img)
        assert out.shape == (256, 256, 3)

    def test_valid_uint8(self):
        from src.runner.error_handling import validate_image

        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        out = validate_image(img)
        assert out.shape == (256, 256, 3)

    def test_rejects_empty(self):
        from src.runner.error_handling import ImageValidationError, validate_image

        with pytest.raises(ImageValidationError, match="empty"):
            validate_image(np.array([]))

    def test_rejects_nan(self):
        from src.runner.error_handling import ImageValidationError, validate_image

        img = np.zeros((256, 256, 3), dtype=np.float32)
        img[0, 0, 0] = float("nan")
        with pytest.raises(ImageValidationError, match="NaN"):
            validate_image(img)

    def test_clips_high_values(self):
        from src.runner.error_handling import validate_image

        img = np.full((10, 10, 3), 5.0, dtype=np.float32)
        out = validate_image(img)
        assert out.max() <= 1.0

    def test_grayscale_conversion(self):
        from src.runner.error_handling import validate_image

        img = np.random.rand(10, 10).astype(np.float32)
        out = validate_image(img, allow_grayscale=True)
        assert out.shape == (10, 10, 3)
